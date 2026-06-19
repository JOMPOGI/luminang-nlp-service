from phonetic import phonetic_encode, get_phonetic_similarity

test_cases = [
    # (Text 1, Text 2, expected to be highly similar or dissimilar)
    ("naimbag ah bigat", "naimbag a bigat", True),  # Ilocano variations (ah vs a)
    ("tinagan kuket jerom", "ti nagan ko ket jerome", True),  # spelling differences
    ("kumusta ka", "kamusta ka", True),  # Cebuano/Tagalog vowel variations
    ("sa duhatulo", "magpabilin", False),  # completely different phrases
    ("usa duha tulo", "duha", False),
    ("hello", "helo", True),
    ("lugar", "logar", True),
]

print("=== Phonetic Encoder Test ===")
for t1, t2, should_match in test_cases:
    code1 = phonetic_encode(t1)
    code2 = phonetic_encode(t2)
    score = get_phonetic_similarity(t1, t2)
    print(f"'{t1}' -> '{code1}'")
    print(f"'{t2}' -> '{code2}'")
    print(f"Similarity: {score:.4f} (Expected highly similar: {should_match})\n")
