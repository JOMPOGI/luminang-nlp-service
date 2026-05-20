import os
import tempfile
import requests
import uvicorn
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import difflib
from huggingface_hub import InferenceClient

# Import controlled dataset helpers
from dataset import dataset

app = FastAPI(title="Luminang Cloud-API NLP Backend", version="2.0.0")

# Enable CORS for Unity web/local requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read API keys from Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
MODEL_NAME = "intfloat/multilingual-e5-small"

# Compiled Regex for cleaning text
import re
re_sub_punc = re.compile(r'[.,?!:;"\'()\-_\/]')
re_sub_space = re.compile(r'\s+')

def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = re_sub_punc.sub(" ", cleaned)
    cleaned = re_sub_space.sub(" ", cleaned).strip()
    return cleaned

@app.on_event("startup")
def startup_event():
    print("Checking API Keys...")
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY environment variable is not set!")
    else:
        print("GROQ_API_KEY is configured.")
        
    if not HF_API_TOKEN:
        print("WARNING: HF_API_TOKEN environment variable is not set!")
    else:
        print("HF_API_TOKEN is configured.")
        
    print(f"Successfully loaded {len(dataset.phrases)} phrases from dataset.")
    print("Startup complete! Server is running in Serverless API mode.")

def transcribe_audio_file(audio_bytes: bytes) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY environment variable is missing on the server. Please set it to enable Whisper STT."
        )
        
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    # Write bytes to temporary file for upload
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name
        
    try:
        with open(temp_file_path, "rb") as f:
            files = {
                "file": ("audio.wav", f, "audio/wav")
            }
            data = {
                "model": "whisper-large-v3",
                "response_format": "json"
            }
            response = requests.post(url, headers=headers, files=files, data=data)
            
        if response.status_code != 200:
            print(f"Groq API Error: {response.status_code} | {response.text}")
            raise HTTPException(status_code=502, detail="Failed to transcribe audio via Groq API.")
            
        result = response.json()
        return result.get("text", "").strip()
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_similarities_batch(source_text: str, target_texts: List[str]) -> List[float]:
    if not HF_API_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="HF_API_TOKEN environment variable is missing on the server. Please set it to enable NLP evaluation."
        )
        
    s_clean = clean_text(source_text)
    targets_clean = [clean_text(t) for t in target_texts]
    
    if not s_clean or not targets_clean:
        return [0.0] * len(target_texts)
        
    try:
        # Use Hugging Face SDK client (which handles task routing and automatic load retries under the hood)
        client = InferenceClient(token=HF_API_TOKEN)
        scores = client.sentence_similarity(
            sentence=f"query: {s_clean}",
            other_sentences=[f"passage: {t}" for t in targets_clean],
            model=MODEL_NAME
        )
        return [float(score) for score in scores]
    except Exception as e:
        print(f"Hugging Face API Error via SDK: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to calculate semantic similarity: {str(e)}")

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

    # Cross-reference check: We compile every valid phrase in the dataset
    phrase_targets = []  # List of tuples: (entry, lang, phrase)
    for entry in dataset.phrases:
        for lang in ["english", "ilokano", "cebuano"]:
            phrase = entry.get(lang)
            if phrase and phrase != "___":
                phrase_targets.append((entry, lang, phrase))

    # Send a single batch request to Hugging Face API
    target_phrases = [item[2] for item in phrase_targets]
    scores = get_similarities_batch(transcript_clean, target_phrases)

    # Find the expected score and the overall best matching phrase
    score = 0.0
    best_score = -1.0
    best_entry = None

    for (entry, lang, phrase), sim_score in zip(phrase_targets, scores):
        sim_score = max(0.0, min(1.0, sim_score))
        
        # Anti-Hallucination Filter: Penalize if lexical overlap is extremely low
        lexical_score = difflib.SequenceMatcher(None, transcript_clean, clean_text(phrase)).ratio()
        if lexical_score < 0.35:
            sim_score = max(0.0, sim_score - 0.15)
            
        # If this is our expected phrase, record the score
        if clean_text(phrase) == expected_clean:
            score = sim_score
            
        if sim_score > best_score:
            best_score = sim_score
            best_entry = entry

    # 4. Apply threshold scoring
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
    
    # Get regional targets
    regional_targets = []
    for entry, lang, phrase in dataset.get_all_targets(region):
        if phrase and phrase != "___":
            regional_targets.append((entry, lang, phrase))
            
    # Get English targets
    english_targets = []
    for entry in dataset.phrases:
        phrase = entry.get("english")
        if phrase:
            english_targets.append((entry, "english", phrase))
            
    # Combine all targets to do a single batch call to Hugging Face
    all_targets = regional_targets + english_targets
    target_phrases = [item[2] for item in all_targets]
    
    scores = get_similarities_batch(transcript_clean, target_phrases)
    
    # Determine best regional match
    best_entry = None
    best_lang = ""
    max_score = -1.0
    
    # Determine best English match
    best_english_score = 0.0
    
    for (entry, lang, phrase), sim_score in zip(all_targets, scores):
        sim_score = max(0.0, min(1.0, sim_score))
        
        # Anti-Hallucination Filter: Penalize if lexical overlap is extremely low
        lexical_score = difflib.SequenceMatcher(None, transcript_clean, clean_text(phrase)).ratio()
        if lexical_score < 0.35:
            sim_score = max(0.0, sim_score - 0.15)
            
        # If it is part of the regional targets
        if lang != "english" or (entry, lang, phrase) in regional_targets:
            if sim_score > max_score:
                max_score = sim_score
                best_entry = entry
                best_lang = lang
                
        # If it is part of the English targets
        if lang == "english":
            if sim_score > best_english_score:
                best_english_score = sim_score
                
    # If English is a much better match than regional, flag it
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

    regional_targets = []
    for entry, lang, phrase in dataset.get_all_targets(region):
        if phrase and phrase != "___":
            regional_targets.append((entry, lang, phrase))
            
    if not regional_targets:
        return {
            "transcript": transcript,
            "matches": []
        }
        
    target_phrases = [item[2] for item in regional_targets]
    scores = get_similarities_batch(transcript_clean, target_phrases)
    
    matches = []
    for (entry, lang, phrase), sim_score in zip(regional_targets, scores):
        sim_score = max(0.0, min(1.0, sim_score))
        
        # Anti-Hallucination Filter: Penalize if lexical overlap is extremely low
        lexical_score = difflib.SequenceMatcher(None, transcript_clean, clean_text(phrase)).ratio()
        if lexical_score < 0.35:
            sim_score = max(0.0, sim_score - 0.15)
            
        if sim_score >= 0.80:
            matches.append({
                "entry": entry,
                "language": lang,
                "score": round(sim_score * 100.0, 2)  # C# expects 0-100 scale
            })
            
    # Sort matches by score descending
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)

    # Filter out weak/cluttered matches if we have a strong candidate
    if matches:
        max_match_score = matches[0]["score"]
        matches = [m for m in matches if m["score"] >= max_match_score - 3.0]

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
