import os
import tempfile
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Import controlled dataset helpers
from dataset import dataset

app = FastAPI(title="Luminang Semantic NLP Backend", version="1.0.0")

# Enable CORS for Unity web/local requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and precomputed embeddings
model = None
whisper_model = None
phrase_embeddings = {}  # Cache: (lang, phrase) -> embedding

# Optimize PyTorch CPU memory usage (essential for 512MB RAM environments like Render Free Tier)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Setup CPU/GPU device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Using multilingual-e5-small as it is extremely lightweight, fast on CPU, and high quality
MODEL_NAME = "intfloat/multilingual-e5-small"
WHISPER_MODEL_NAME = "tiny"  # Tiny model is 75MB (half the RAM of base) and extremely fast/accurate for short phrases

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Strip punctuation and lowercase
    cleaned = text.lower()
    cleaned = re_sub_punc.sub(" ", cleaned)
    # Phonetic normalize if needed (kept minimal to avoid changing semantics)
    cleaned = re_sub_space.sub(" ", cleaned).strip()
    return cleaned

# Compiled Regex for cleaning text
import re
re_sub_punc = re.compile(r'[.,?!:;"\'()\-_\/]')
re_sub_space = re.compile(r'\s+')

@app.on_event("startup")
def startup_event():
    global model, whisper_model
    import whisper
    from sentence_transformers import SentenceTransformer
    import gc
    
    print(f"Loading Sentence Transformer model '{MODEL_NAME}' on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    
    print(f"Loading Whisper model '{WHISPER_MODEL_NAME}' on {DEVICE}...")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)
    
    print("Pre-encoding controlled dataset target phrases...")
    precompute_dataset_embeddings()
    
    # Run garbage collection to clean up loading artifacts from RAM
    gc.collect()
    print("Startup complete! Server is ready.")

def precompute_dataset_embeddings():
    global phrase_embeddings
    phrase_embeddings.clear()
    
    unique_phrases = set()
    for entry in dataset.phrases:
        for lang in ["english", "ilokano", "cebuano"]:
            phrase = entry.get(lang, "")
            if phrase and phrase != "___":
                unique_phrases.add(phrase)
                
    if not unique_phrases:
        return
        
    phrases_list = list(unique_phrases)
    # Multilingual-e5 expects "query: " or "passage: " prefix depending on task
    # For semantic similarity, prefixing with "query: " is recommended for retrieval-like tasks
    prefixed_phrases = [f"query: {p}" for p in phrases_list]
    
    embeddings = model.encode(prefixed_phrases, show_progress_bar=False, convert_to_numpy=True)
    
    for phrase, embedding in zip(phrases_list, embeddings):
        # Normalize the embedding to unit length for easy cosine similarity (dot product)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        phrase_embeddings[phrase] = embedding

def get_cosine_similarity(text1: str, text2: str) -> float:
    # Clean inputs
    t1 = clean_text(text1)
    t2 = clean_text(text2)
    
    if not t1 or not t2:
        return 0.0
        
    # Check if we have precomputed embedding for text2 (target phrase)
    emb2 = phrase_embeddings.get(text2)
    if emb2 is None:
        emb2_raw = model.encode(f"query: {t2}", convert_to_numpy=True)
        norm = np.linalg.norm(emb2_raw)
        emb2 = emb2_raw / norm if norm > 0 else emb2_raw
        
    # Check if we have precomputed embedding for text1 (input phrase)
    emb1 = phrase_embeddings.get(text1)
    if emb1 is None:
        emb1_raw = model.encode(f"query: {t1}", convert_to_numpy=True)
        norm = np.linalg.norm(emb1_raw)
        emb1 = emb1_raw / norm if norm > 0 else emb1_raw
        
    # Dot product of normalized vectors is the cosine similarity
    similarity = float(np.dot(emb1, emb2))
    return similarity

def transcribe_audio_file(audio_bytes: bytes) -> str:
    # Save audio bytes to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        # Transcribe using Whisper
        # We specify beam_size=5 for higher quality, and try to assist with language if possible
        result = whisper_model.transcribe(temp_path, temperature=0.0)
        transcript = result.get("text", "").strip()
        return transcript
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/evaluate")
async def evaluate(
    expected_phrase: str = Form(...),
    transcribed_text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Evaluates player's speech input against the expected phrase.
    Supports either pre-transcribed text or direct audio upload.
    """
    transcript = ""
    if audio:
        audio_bytes = await audio.read()
        transcript = transcribe_audio_file(audio_bytes)
    elif transcribed_text:
        transcript = transcribed_text
    else:
        raise HTTPException(status_code=400, detail="Either audio or transcribed_text must be provided")

    print(f"Expected: '{expected_phrase}' | Heard: '{transcript}'")

    # 1. Preprocess texts
    expected_clean = clean_text(expected_phrase)
    transcript_clean = clean_text(transcript)


    # Find the expected entry in the dataset
    expected_entry = None
    for entry in dataset.phrases:
        phrases_list = [
            clean_text(entry.get("english")),
            clean_text(entry.get("ilokano")),
            clean_text(entry.get("cebuano"))
        ]
        if expected_clean in phrases_list:
            expected_entry = entry
            break

    if expected_entry is None:
        print(f"Warning: Expected phrase '{expected_phrase}' is not in the controlled dataset!")
        return {
            "transcript": transcript,
            "score": 0.0,
            "result": "try_again"
        }

    # 3. Compute cosine similarity between embeddings
    score = get_cosine_similarity(transcript_clean, expected_clean)
    score = max(0.0, min(1.0, score))

    # Cross-reference: Find the absolute best matching phrase in the dataset for player response
    best_entry = None
    best_score = -1.0
    for entry in dataset.phrases:
        for lang in ["english", "ilokano", "cebuano"]:
            phrase = entry.get(lang)
            if phrase and phrase != "___":
                phrase_clean = clean_text(phrase)
                sim = get_cosine_similarity(transcript_clean, phrase_clean)
                if sim > best_score:
                    best_score = sim
                    best_entry = entry

    # 4. Apply threshold scoring
    # If score is >= 0.80 and player's response matches expected better or comparably to any other phrase
    result = "try_again"
    if score >= 0.80:
        if best_entry == expected_entry or (best_score - score < 0.05):
            result = "correct"
        else:
            print(f"Rejected greeting mismatch: player said '{transcript}', which matches '{best_entry.get('english')}' (score {best_score:.4f}) better than expected '{expected_phrase}' (score {score:.4f})")

    print(f"Evaluation: score = {score:.4f}, result = {result}")

    return {
        "transcript": transcript,
        "score": round(score, 4),
        "result": result
    }

@app.post("/find_best_match")
async def find_best_match(
    region: str = Form(...),
    transcribed_text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Searches the controlled dataset for the closest matching phrase.
    """
    transcript = ""
    if audio:
        audio_bytes = await audio.read()
        transcript = transcribe_audio_file(audio_bytes)
    elif transcribed_text:
        transcript = transcribed_text
    else:
        raise HTTPException(status_code=400, detail="Either audio or transcribed_text must be provided")

    transcript_clean = clean_text(transcript)
    

    # Get valid target phrases for the active region
    targets = dataset.get_all_targets(region)
    if not targets:
        return {
            "transcript": transcript,
            "best_entry": None,
            "language": "",
            "score": 0.0,
            "is_english": False
        }

    # Find closest match using semantic embeddings
    best_entry = None
    best_lang = ""
    max_score = -1.0

    for entry, lang, phrase in targets:
        if not phrase or phrase == "___":
            continue
        score = get_cosine_similarity(transcript_clean, phrase)
        if score > max_score:
            max_score = score
            best_entry = entry
            best_lang = lang

    # Check English ONLY to detect if they are speaking English instead of regional
    best_english_score = 0.0
    english_targets = [(e, "english", e.get("english")) for e in dataset.phrases]
    for entry, lang, phrase in english_targets:
        if phrase:
            score = get_cosine_similarity(transcript_clean, phrase)
            if score > best_english_score:
                best_english_score = score

    # If English is a much better match than regional, flag it (like the original C# code)
    matched_english = False
    if best_english_score > 0.85 and best_english_score > (max_score + 0.15):
        matched_english = True

    return {
        "transcript": transcript,
        "best_entry": best_entry,
        "language": best_lang,
        "score": round(max_score, 4),
        "is_english": matched_english
    }

@app.post("/find_all_matches")
async def find_all_matches(
    region: str = Form(...),
    transcribed_text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Searches the dataset for all matching phrases above a similarity threshold.
    """
    transcript = ""
    if audio:
        audio_bytes = await audio.read()
        transcript = transcribe_audio_file(audio_bytes)
    elif transcribed_text:
        transcript = transcribed_text
    else:
        raise HTTPException(status_code=400, detail="Either audio or transcribed_text must be provided")

    transcript_clean = clean_text(transcript)

    targets = dataset.get_all_targets(region)
    matches = []

    for entry, lang, phrase in targets:
        if not phrase or phrase == "___":
            continue
        score = get_cosine_similarity(transcript_clean, phrase)
        # 65% is the original threshold in PhraseEvaluator.cs
        if score >= 0.65:
            # We return matches in the structure expected by Unity
            matches.append({
                "entry": entry,
                "language": lang,
                "score": round(score * 100.0, 2)  # C# expects 0-100 scale
            })

    # Sort matches by position of phrase in transcription, or score descending
    # Let's sort by score descending for best results
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)

    return {
        "transcript": transcript,
        "matches": matches
    }

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcribes uploaded audio file and returns transcription.
    """
    audio_bytes = await audio.read()
    transcript = transcribe_audio_file(audio_bytes)
    return {"text": transcript}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

