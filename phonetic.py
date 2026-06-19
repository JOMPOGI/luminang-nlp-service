import re

def clean_text_local(text: str) -> str:
    if not text:
        return ""
    # Lowercase and keep only letters and spaces
    cleaned = text.lower()
    cleaned = re.sub(r'[^a-z\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def phonetic_encode_word(word: str) -> str:
    if not word:
        return ""
    
    # Keep the first character (sometimes helpful for soundex/prefix matching)
    # but process it to maintain phonetics. Let's do a direct replacement pipeline.
    
    # 1. Phonetic mappings for English/Philippine languages
    w = word
    
    # Replace standard digraphs
    w = w.replace("ng", "n")
    w = w.replace("ch", "ts")
    w = w.replace("sh", "s")
    w = w.replace("ph", "p")  # F/P are interchangeable
    w = w.replace("th", "t")
    
    # Single letter mappings
    # Map 'c' to 'k' or 's'
    new_w = []
    for i, char in enumerate(w):
        if char == 'c':
            # Check next char
            if i + 1 < len(w) and w[i+1] in ['e', 'i', 'y']:
                new_w.append('s')
            else:
                new_w.append('k')
        elif char == 'q':
            new_w.append('k')
        elif char == 'x':
            new_w.append('ks')
        elif char == 'z':
            new_w.append('s')
        elif char == 'f':
            new_w.append('p')
        elif char == 'v':
            new_w.append('b')
        elif char == 'j':
            new_w.append('d')  # sounds like dy / d
        elif char == 'h':
            # Keep h only if it's the very first letter of the word, else omit (often silent or dialect variation)
            if i == 0:
                new_w.append('h')
        else:
            new_w.append(char)
            
    w = "".join(new_w)
    
    # Interchangeable vowels
    # In Tagalog/Ilokano/Cebuano: e/i are highly interchangeable, o/u are highly interchangeable.
    # Map y -> i, w -> u to represent semivowels phonetically
    w = w.replace("e", "i")
    w = w.replace("y", "i")
    w = w.replace("o", "u")
    w = w.replace("w", "u")
    
    # Remove consecutive duplicate characters
    collapsed = []
    prev = None
    for char in w:
        if char != prev:
            collapsed.append(char)
            prev = char
            
    return "".join(collapsed)

def phonetic_encode(text: str) -> str:
    cleaned = clean_text_local(text)
    words = cleaned.split(" ")
    encoded_words = [phonetic_encode_word(w) for w in words if w]
    return " ".join(encoded_words)

def jaro_similarity(s1: str, s2: str) -> float:
    # If the strings are equal
    if s1 == s2:
        return 1.0

    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    # Maximum distance limit for matching
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    # Hash for matches
    hash_s1 = [0] * len1
    hash_s2 = [0] * len2

    match = 0

    # Traverse through the first string
    for i in range(len1):
        # Check if there is any matches
        start = max(0, i - max_dist)
        end = min(len2, i + max_dist + 1)
        for j in range(start, end):
            # If there is a match
            if s1[i] == s2[j] and hash_s2[j] == 0:
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    # If no match
    if match == 0:
        return 0.0

    # Number of transpositions
    t = 0
    point = 0

    # Traverse to find transpositions
    for i in range(len1):
        if hash_s1[i]:
            # Find the next matched character in second string
            while hash_s2[point] == 0:
                point += 1

            if s1[i] != s2[point]:
                t += 1
            point += 1

    t = t / 2.0

    # Return the Jaro Similarity
    return (match / len1 + match / len2 + (match - t) / match) / 3.0

def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    jaro_sim = jaro_similarity(s1, s2)
    
    # Calculate prefix length (up to 4 characters)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
            
    # Calculate Jaro-Winkler similarity
    jw_sim = jaro_sim + prefix_len * p * (1.0 - jaro_sim)
    return jw_sim

def get_phonetic_similarity(text1: str, text2: str) -> float:
    # 1. Get phonetic codes
    code1 = phonetic_encode(text1)
    code2 = phonetic_encode(text2)
    
    # If both empty
    if not code1 and not code2:
        return 1.0
    if not code1 or not code2:
        return 0.0
        
    return jaro_winkler_similarity(code1, code2)
