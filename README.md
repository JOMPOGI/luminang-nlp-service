# Luminang NLP Service 
A fast, scalable, and intelligent Natural Language Processing (NLP) backend designed specifically for the Luminang game. Built with **FastAPI**, it handles Speech-to-Text (STT) transcription, semantic speech evaluation, and dynamic slot-filling (wildcards like `{name}` and `{place}`).

## Features 
- **Transcription**: Uses Groq's API and `whisper-large-v3` to transcribe player speech almost instantly.
- **Semantic Evaluation**: Uses Hugging Face's `intfloat/multilingual-e5-small` model to calculate semantic similarities between what the player said and what they were expected to say.
- **Dynamic Wildcard Support**: Automatically handles dynamic template tags like `{name}` and `{place}` in expected phrases without requiring strict exact matches.
- **Multi-regional Support**: Validates and restricts language contexts based on region (e.g., Cebuano, Ilokano, Tagalog).
- **Anti-Hallucination Filters**: Penalizes and prevents the AI from falsely approving hallucinated words that have zero phonetic or lexical overlap.

## Tech Stack 🛠️
- **Framework**: FastAPI (Python)
- **Server**: Uvicorn
- **Speech-to-Text (STT)**: Whisper (`whisper-large-v3`) via Groq API
- **Embeddings / NLP**: Hugging Face Inference API (`intfloat/multilingual-e5-small`)
---

##  How It Works
The NLP Service acts as the brain behind the game's speech recognition. When a player speaks into the microphone, the following happens:
1. **Transcription (STT)**: The audio is sent to the backend and instantly converted into text using the Whisper model.
2. **Semantic Comparison**: Instead of just doing a basic exact-match comparison, the backend takes the transcribed text and compares it *semantically* to the expected phrase using Hugging Face embeddings. This means it understands the *meaning* of the sentence, so minor mispronunciations or slightly different phrasings can still be accepted.
3. **Scoring**: It calculates a final confidence score based on semantic similarity, lexical overlap, and phonetic similarity, then sends a result (`correct`, `uncertain`, or `try_again`) back to Unity.

###  Handling Dynamic Names & Places
If an expected phrase contains a wildcard (e.g., `"ako si {name}"` or `"taga {place} ako"`), the backend skips the strict full-sentence evaluation. 
Instead, it dynamically checks if the player's transcribed speech **starts with** the exact prefix before the wildcard (e.g., `"ako si "`). This allows the system to accept literally *any* name or place the player decides to say, as long as they get the core sentence structure righ
