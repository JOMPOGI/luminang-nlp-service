import sys
from evaluator import evaluate_phrase

def run_test(name, transcript, target, category, lang, expected_result):
    # Dummy semantic score for test isolation (assume HF gives decent semantic score)
    semantic_score = 0.85
    result = evaluate_phrase(transcript, target, category, lang, semantic_score)
    print(f"Test: {name}")
    print(f"Transcript: '{transcript}'")
    print(f"Target: '{target}'")
    print(f"Result: {result}")
    
    passed = False
    if expected_result == "correct" and result["final_confidence"] >= 0.8:
        passed = True
    elif expected_result == "uncertain" and 0.5 <= result["final_confidence"] < 0.8:
        passed = True
    elif expected_result == "wrong" and result["final_confidence"] < 0.5:
        passed = True
        
    print(f"Status: {'PASS' if passed else 'FAIL'}\n")
    return passed

tests = [
    ("Correct Ilokano", "agyamanak", "agyamanak", "Gratitude", "ilokano", "correct"),
    ("Correct Cebuano", "salamat", "salamat", "Gratitude", "cebuano", "correct"),
    ("Correct dynamic Ilokano", "ti nagan ko ket Jom", "ti nagan ko ket {name}", "Identity", "ilokano", "correct"),
    ("Correct dynamic Cebuano", "ako si Jom", "ako si {name}", "Identity", "cebuano", "correct"),
    ("Correct location template", "taga Dasmarinas ak", "taga {location} ak", "Identity", "ilokano", "correct"),
    ("Code-switched response", "ti nagan ko is Jom", "ti nagan ko ket {name}", "Identity", "ilokano", "wrong"),
    ("Wrong phrase", "mangan", "maturog", "Action Verbs", "ilokano", "wrong"),
    ("Short vocabulary exactly", "maysa", "maysa", "Count", "ilokano", "correct"),
    ("Short vocabulary wrong lang", "usa", "maysa", "Count", "ilokano", "wrong"),
    ("ASR Variation", "naiimbag a bigat", "naimbag a bigat", "Greetings", "ilokano", "correct")
]

all_pass = True
for t in tests:
    if not run_test(*t):
        all_pass = False
        
if not all_pass:
    print("Some tests failed!")
    sys.exit(1)
print("All evaluator logic tests passed!")
