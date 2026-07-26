# Luminang NLP Service

A controlled, dataset-grounded hybrid speech evaluation pipeline that combines Speech-to-Text transcription, lexical matching, phonetic/ASR variation handling, template validation, target-language constraints, and semantic similarity.

## System Overview

This backend is specifically designed for controlled Ilokano and Cebuano learning scenarios. It evaluates a user's spoken phrase against expected phrases in a dataset.

- **Speech-to-Text:** Whisper (via Groq) performs the initial audio transcription.
- **Authoritative Dataset:** The Luminang dataset (`LuminangPhrases.json`) is the authoritative source of accepted language content.
- **Lexical Priority:** Exact and lexical matching are prioritized over loose semantic matching to prevent hallucinations and false positives.
- **Dynamic Templates:** Template matching (e.g., `{name}`, `{location}`) handles dynamic phrases securely without penalizing the dynamic slot.
- **ASR Variation Handling:** Phonetic matching helps compensate for Whisper transcription variations or minor mispronunciations.
- **Semantic Support:** Semantic similarity (via Hugging Face `multilingual-e5-small`) is a supporting NLP signal, not the sole authority. It helps with minor wording variations in longer sentences but is penalized heavily for short grammatical words (like numbers and pronouns).
- **Contextual Filtering:** Language and category context filtering reduce false positives by narrowing the search space.

## Architecture

1. Unity game sends audio to the `/evaluate` endpoint.
2. `server.py` transcribes audio using Whisper STT (with language region hints).
3. `dataset.py` fetches relevant target phrases based on region and category.
4. `evaluator.py` scores the transcription using a hybrid approach (Template, Lexical, Phonetic, Semantic).
5. The API returns a detailed breakdown of scores, a final confidence, and a status of `correct`, `try_again`, or `uncertain`.
