"""
test_complexity_classifier.py
"""

from src.complexity import QuestionComplexityClassifier

def test_classifier():
    classifier = QuestionComplexityClassifier()
    
    # Test cases with expected classifications
    test_questions = [
        # Simple (factual, single-concept)
        ("What is O-RAN?", "simple"),
        ("Define the E2 interface.", "simple"),
        ("What does RIC stand for?", "simple"),
        
        # Medium (require some explanation)
        ("Explain how the A1 interface works in O-RAN.", "medium"),
        ("What are the main components of O-RAN architecture?", "medium"),
        ("How does O-RAN support network slicing?", "medium"),
        
        # Complex (multi-hop, comparisons, trade-offs)
        ("Compare the functional split options between O-DU and O-CU, explaining latency and centralization trade-offs.", "complex"),
        ("Describe the interaction between Near-RT RIC and Non-RT RIC, and how they coordinate for network optimization.", "complex"),
        ("What is the relationship between E2 service models and xApp deployment in O-RAN?", "complex"),
    ]
    
    print("="*80)
    print("Complexity Classifier Validation")
    print("="*80)
    
    correct = 0
    for question, expected in test_questions:
        actual = classifier.classify(question)
        match = "✓" if actual == expected else "✗"
        
        print(f"\n{match} Question: {question[:60]}...")
        print(f"   Expected: {expected:8s} | Actual: {actual:8s}")
        
        if actual == expected:
            correct += 1
    
    accuracy = correct / len(test_questions) * 100
    print(f"\n{'='*80}")
    print(f"Accuracy: {correct}/{len(test_questions)} ({accuracy:.1f}%)")
    print(f"{'='*80}")
    
    if accuracy >= 80:
        print("✓ Classifier performs well!")
    else:
        print("⚠️  Classifier needs tuning")

if __name__ == "__main__":
    test_classifier()
