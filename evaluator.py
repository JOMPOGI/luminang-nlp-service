import re
import difflib
from phonetic import get_phonetic_similarity

re_sub_punc = re.compile(r'[.,?!:;"\'()\-_\/]')
re_sub_space = re.compile(r'\s+')

def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = re_sub_punc.sub(" ", cleaned)
    cleaned = re_sub_space.sub(" ", cleaned).strip()
    return cleaned

def evaluate_phrase(transcript: str, target_phrase: str, category: str, lang: str, semantic_score: float) -> dict:
    """
    Evaluates a transcribed phrase against a target phrase using hybrid matching logic.
    Returns a dictionary of individual scores and a final confidence score.
    """
    transcript_clean = clean_text(transcript)
    target_clean = clean_text(target_phrase)
    
    # Check if template (e.g. "ako si {name}", "taga {place} ak")
    is_template = "{" in target_phrase and "}" in target_phrase
    
    # 1. Exact / Lexical Score
    if is_template:
        # --- FIX: Use word-level matching instead of chunk-level ---
        # Replace all {slot} placeholders with a space, then extract individual words.
        # This means "ti nagan ko ket {name}" yields fixed_words = ["ti", "nagan", "ko", "ket"]
        # and "ako si {name}" yields fixed_words = ["ako", "si"]
        # Both are then matched word-by-word against the transcript, which is resilient
        # to STT reordering and partial recognition errors.
        fixed_string = re.sub(r'\{.*?\}', ' ', target_clean)
        fixed_words = [w for w in fixed_string.split() if w.strip()]

        lexical_score = 0.0
        template_score = 0.0

        if fixed_words:
            # Count how many required fixed words appear in the transcript
            transcript_words = set(transcript_clean.split())
            matched = sum(1 for w in fixed_words if w in transcript_words)
            template_score = matched / len(fixed_words)
            base_target = " ".join(fixed_words)
        else:
            # No fixed words at all (e.g. pure slot like "{name}") — always pass
            template_score = 1.0
            base_target = target_clean

        # Lexical / phonetic compare against the fixed structural skeleton only,
        # so a name/place value in the transcript doesn't drag the score down.
        matcher = difflib.SequenceMatcher(None, transcript_clean, base_target)
        lexical_score = matcher.ratio() if base_target else 0.0

        exact_score = 1.0 if template_score >= 0.8 else 0.0
        phonetic_score = get_phonetic_similarity(transcript_clean, base_target)

    else:
        template_score = 1.0  # Not a template
        exact_score = 1.0 if transcript_clean == target_clean else 0.0
        matcher = difflib.SequenceMatcher(None, transcript_clean, target_clean)
        lexical_score = matcher.ratio()
        phonetic_score = get_phonetic_similarity(transcript_clean, target_clean)

    # 2. Short word handling (Count, Pronouns, Responses, Interrogatives)
    short_word_categories = ["Count", "Pronouns", "Responses", "Interrogatives"]
    is_short_word = category in short_word_categories or len(target_clean.split()) <= 2

    # 3. Final Confidence Calculation
    if is_template:
        # Template scoring: structure match is critical (60%), phonetic (20%), semantic (20%)
        final_confidence = (template_score * 0.6) + (semantic_score * 0.2) + (phonetic_score * 0.2)
        if template_score < 0.5:
            final_confidence = 0.0  # Reject if fixed structure is mostly missed
    elif is_short_word:
        # Short words rely strictly on exact/lexical/phonetic. Semantic is ignored.
        final_confidence = (lexical_score * 0.5) + (phonetic_score * 0.4) + (exact_score * 0.1)
    else:
        # Long fixed phrases use hybrid approach
        final_confidence = (lexical_score * 0.4) + (phonetic_score * 0.3) + (semantic_score * 0.3)

    # 4. Anti-Hallucination (only for non-templates)
    if not is_template and lexical_score < 0.35:
        final_confidence = max(0.0, final_confidence - 0.2)

    return {
        "exact_score": exact_score,
        "lexical_score": lexical_score,
        "phonetic_score": phonetic_score,
        "semantic_score": semantic_score,
        "template_score": template_score,
        "final_confidence": final_confidence,
        "code_switched": False
    }
