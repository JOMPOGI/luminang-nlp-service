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
from phonetic import get_phonetic_similarity


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

from evaluator import evaluate_phrase, clean_text

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

def is_valid_language(detected_lang: str, region: str) -> bool:
    """
    Returns True if the Whisper detected language aligns with the expected regional context.
    Whisper often misclassifies Ilokano as Tagalog, Indonesian, or Malay due to lack of a dedicated Ilokano token.
    """
    if region == "English":
        return detected_lang == "english"
        
    valid_regional_langs = {"cebuano", "tagalog", "filipino", "indonesian", "malay"}
    return detected_lang in valid_regional_langs

def transcribe_audio_file(audio_bytes: bytes, region: str = None) -> Dict[str, str]:
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
                "response_format": "verbose_json"
            }
            if region:
                # Add prompt hint for better ASR
                data["prompt"] = f"This is speech in {region} language."
            response = requests.post(url, headers=headers, files=files, data=data)
            
        if response.status_code != 200:
            print(f"Groq API Error: {response.status_code} | {response.text}")
            raise HTTPException(status_code=502, detail="Failed to transcribe audio via Groq API.")
            
        result = response.json()
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "").lower()
        }
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
    audio: Optional[UploadFile] = File(None),
    region: str = Form("Default"),
    category: Optional[str] = Form(None)
):
    """
    Evaluates player's speech input against the expected phrase.
    Supports either pre-transcribed text or direct audio upload.
    """
    transcript = ""
    detected_lang = ""
    if audio:
        audio_bytes = await audio.read()
        res = transcribe_audio_file(audio_bytes)
        transcript = res["text"]
        detected_lang = res["language"]
        
        if not is_valid_language(detected_lang, region):
            print(f"Warning: Rejected audio because Whisper detected '{detected_lang}' (expected {region})")
            return {
                "transcript": transcript,
                "score": 0.0,
                "result": "try_again",
                "exact_score": 0.0,
                "lexical_score": 0.0,
                "phonetic_score": 0.0,
                "semantic_score": 0.0,
                "template_score": 0.0,
                "final_confidence": 0.0,
                "code_switched": False
            }
    elif transcribed_text:
        transcript = transcribed_text
    else:
        raise HTTPException(status_code=400, detail="Either audio or transcribed_text must be provided")

    print(f"Expected: '{expected_phrase}' | Heard: '{transcript}'")

    expected_clean = clean_text(expected_phrase)
    transcript_clean = clean_text(transcript)

    # 1. Gather all potential targets to find expected entry and for cross-referencing
    all_targets = dataset.get_all_targets("BossBattle") # All languages
    
    expected_entry = None
    expected_lang = ""
    for entry, lang, phrase in all_targets:
        # Check against template structures too
        if clean_text(phrase) == expected_clean or ( "{" in phrase and expected_clean.startswith(clean_text(phrase.split("{")[0])) ):
            expected_entry = entry
            expected_lang = lang
            expected_phrase = phrase # use the exact template from dataset
            break
            
    if expected_entry is None:
        print(f"Warning: Expected phrase '{expected_phrase}' is not in the controlled dataset!")
        return {
            "transcript": transcript,
            "score": 0.0,
            "result": "try_again",
            "exact_score": 0.0,
            "lexical_score": 0.0,
            "phonetic_score": 0.0,
            "semantic_score": 0.0,
            "template_score": 0.0,
            "final_confidence": 0.0,
        }

    # 2. Get dataset targets (filtered by region/category if specified)
    context_targets = dataset.get_all_targets(region, category)
    if not context_targets:
        context_targets = all_targets
        
    target_phrases = [item[2] for item in context_targets]
    
    # 3. Get Semantic Similarities
    scores = get_similarities_batch(transcript_clean, target_phrases)

    # 4. Evaluate each target
    best_match_eval = None
    expected_eval = None

    for (entry, lang, phrase), sim_score in zip(context_targets, scores):
        sim_score = max(0.0, min(1.0, sim_score))
        
        eval_result = evaluate_phrase(
            transcript=transcript_clean,
            target_phrase=phrase,
            category=entry.get("category", ""),
            lang=lang,
            semantic_score=sim_score
        )
        
        if phrase == expected_phrase:
            expected_eval = eval_result
            
        if not best_match_eval or eval_result["final_confidence"] > best_match_eval["final_confidence"]:
            best_match_eval = eval_result

    # If the expected phrase wasn't in context targets, evaluate it explicitly
    if not expected_eval:
        sim_scores = get_similarities_batch(transcript_clean, [expected_phrase])
        expected_eval = evaluate_phrase(
            transcript=transcript_clean,
            target_phrase=expected_phrase,
            category=expected_entry.get("category", ""),
            lang=expected_lang,
            semantic_score=max(0.0, min(1.0, sim_scores[0])) if sim_scores else 0.0
        )

    # 5. Determine Result
    final_conf = expected_eval["final_confidence"]
    result = "try_again"
    
    if final_conf >= 0.80:
        if best_match_eval and (best_match_eval["final_confidence"] - final_conf > 0.15):
            print(f"Rejected: player said '{transcript}' matching another phrase better than expected '{expected_phrase}'")
        else:
            result = "correct"
    elif final_conf >= 0.50:
        result = "uncertain"
        
    print(f"Evaluation: conf = {final_conf:.4f}, result = {result}")

    return {
        "transcript": transcript,
        "score": round(final_conf, 4), # for backwards compatibility
        "result": result,
        "exact_score": round(expected_eval["exact_score"], 4),
        "lexical_score": round(expected_eval["lexical_score"], 4),
        "phonetic_score": round(expected_eval["phonetic_score"], 4),
        "semantic_score": round(expected_eval["semantic_score"], 4),
        "template_score": round(expected_eval["template_score"], 4),
        "final_confidence": round(final_conf, 4),
        "code_switched": expected_eval.get("code_switched", False)
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
    detected_lang = ""
    if audio:
        audio_bytes = await audio.read()
        res = transcribe_audio_file(audio_bytes)
        transcript = res["text"]
        detected_lang = res["language"]
        
        if not is_valid_language(detected_lang, region):
            print(f"Warning: Rejected audio because Whisper detected '{detected_lang}' (expected {region})")
            return {
                "transcript": transcript,
                "best_entry": None,
                "language": detected_lang,
                "score": 0.0,
                "is_english": True
            }
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
        phonetic_score = get_phonetic_similarity(transcript_clean, clean_text(phrase))
        combined_score = 0.8 * sim_score + 0.2 * phonetic_score
        
        # Anti-Hallucination Filter: Penalize if lexical overlap is extremely low
        lexical_score = difflib.SequenceMatcher(None, transcript_clean, clean_text(phrase)).ratio()
        if lexical_score < 0.35:
            combined_score = max(0.0, combined_score - 0.15)
            
        # If it is part of the regional targets
        if lang != "english" or (entry, lang, phrase) in regional_targets:
            if combined_score > max_score:
                max_score = combined_score
                best_entry = entry
                best_lang = lang
                
        # If it is part of the English targets
        if lang == "english":
            if combined_score > best_english_score:
                best_english_score = combined_score
                
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
    detected_lang = ""
    if audio:
        audio_bytes = await audio.read()
        res = transcribe_audio_file(audio_bytes)
        transcript = res["text"]
        detected_lang = res["language"]
        
        if not is_valid_language(detected_lang, region):
            print(f"Warning: Rejected audio because Whisper detected '{detected_lang}' (expected {region})")
            return {
                "transcript": transcript,
                "matches": []
            }
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
        phonetic_score = get_phonetic_similarity(transcript_clean, clean_text(phrase))
        combined_score = 0.8 * sim_score + 0.2 * phonetic_score
        
        # Anti-Hallucination Filter: Penalize if lexical overlap is extremely low
        lexical_score = difflib.SequenceMatcher(None, transcript_clean, clean_text(phrase)).ratio()
        if lexical_score < 0.35:
            combined_score = max(0.0, combined_score - 0.15)
            
        if combined_score >= 0.80:
            matches.append({
                "entry": entry,
                "language": lang,
                "score": round(combined_score * 100.0, 2)  # C# expects 0-100 scale
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
    res = transcribe_audio_file(audio_bytes)
    return {"text": res["text"], "language": res["language"]}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
