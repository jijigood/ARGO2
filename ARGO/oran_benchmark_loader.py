"""
ORAN-Bench-13K Loader
Load and manage ORAN benchmark questions with three difficulty levels
"""
import json
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class ORANBenchmark:
    """
    Load and manage ORAN-Bench-13K multiple choice questions
    
    File structure: [question_text, [option1, option2, option3, option4], correct_index]
    Example: ["What is O-RAN?", ["1. Option A", "2. Option B", ...], "3"]
    """
    
    def __init__(self, benchmark_dir: str = "ORAN-Bench-13K/Benchmark", use_cleaned: bool = True):
        """
        Initialize benchmark loader
        
        Args:
            benchmark_dir: Path to benchmark directory
            use_cleaned: 是否使用清洗后的数据集 (默认True)
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.use_cleaned = use_cleaned
        
        # Load all difficulty levels
        # Hard难度使用清洗后的数据集 (移除了19个有问题的题目)
        hard_filename = 'fin_H_clean.json' if use_cleaned else 'fin_H.json'
        
        self.questions = {
            'easy': self._load_questions('fin_E.json'),
            'medium': self._load_questions('fin_M.json'),
            'hard': self._load_questions(hard_filename)  # 使用清洗后的数据
        }
        
        # Statistics
        self.stats = {
            'easy': len(self.questions['easy']),
            'medium': len(self.questions['medium']),
            'hard': len(self.questions['hard']),
            'total': sum(len(q) for q in self.questions.values())
        }
        
        dataset_type = "(清洗后)" if use_cleaned and hard_filename == 'fin_H_clean.json' else ""
        print(f"Loaded ORAN-Bench-13K {dataset_type}:")
        print(f"  Easy: {self.stats['easy']} questions")
        print(f"  Medium: {self.stats['medium']} questions")
        print(f"  Hard: {self.stats['hard']} questions {dataset_type}")
        print(f"  Total: {self.stats['total']} questions")
    
    def _load_questions(self, filename: str) -> List[Dict]:
        """Load questions from JSONL file (JSON Lines format)"""
        filepath = self.benchmark_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return []
        
        # Load JSONL format (each line is a JSON array)
        questions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Malformed JSON at line {line_num + 1} in {filename}")
                    continue
                
                if len(item) != 3:
                    print(f"Warning: Invalid format at line {line_num + 1} in {filename}")
                    continue
                
                question_text, options, correct_answer = item
                
                # Parse options (format: "1. Option text", "2. Option text", ...")
                parsed_options = []
                for opt in options:
                    # Remove leading number and dot (e.g., "1. " -> "")
                    if opt.startswith(('1.', '2.', '3.', '4.')):
                        parsed_options.append(opt[3:].strip())
                    else:
                        parsed_options.append(opt.strip())
                
                questions.append({
                    'id': line_num,
                    'question': question_text,
                    'options': parsed_options,
                    'correct_answer': int(correct_answer),  # 1, 2, 3, or 4
                'options_formatted': options  # Keep original format
            })
        
        return questions
    
    def get_question(self, difficulty: str, question_id: int) -> Optional[Dict]:
        """Get a specific question by ID"""
        if difficulty not in self.questions:
            raise ValueError(f"Invalid difficulty: {difficulty}")
        
        questions = self.questions[difficulty]
        if 0 <= question_id < len(questions):
            return questions[question_id]
        return None
    
    def sample_questions(self, 
                        n: int = 100, 
                        difficulty: Optional[str] = None,
                        seed: Optional[int] = None) -> List[Dict]:
        """
        Sample n questions from the benchmark
        
        Args:
            n: Number of questions to sample
            difficulty: 'easy', 'medium', 'hard', or None for mixed
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled questions with metadata
        """
        if seed is not None:
            random.seed(seed)
        
        if difficulty:
            # Sample from specific difficulty
            if difficulty not in self.questions:
                raise ValueError(f"Invalid difficulty: {difficulty}")
            
            questions = self.questions[difficulty]
            n_sample = min(n, len(questions))
            sampled = random.sample(questions, n_sample)
            
            # Add difficulty label
            for q in sampled:
                q['difficulty'] = difficulty
            
            return sampled
        else:
            # Sample from all difficulties
            all_questions = []
            for diff, questions in self.questions.items():
                for q in questions:
                    q_copy = q.copy()
                    q_copy['difficulty'] = diff
                    all_questions.append(q_copy)
            
            n_sample = min(n, len(all_questions))
            return random.sample(all_questions, n_sample)
    
    def format_question_for_llm(self, question: Dict) -> str:
        """
        Format question for LLM prompt
        
        Returns:
            Formatted question string with options
        """
        formatted = f"{question['question']}\n\n"
        for i, opt in enumerate(question['options'], 1):
            formatted += f"{i}. {opt}\n"
        formatted += "\nAnswer (1-4):"
        
        return formatted
    
    def check_answer(self, question: Dict, predicted: int) -> bool:
        """
        Check if predicted answer is correct
        
        Args:
            question: Question dictionary
            predicted: Predicted answer (1-4)
            
        Returns:
            True if correct, False otherwise
        """
        return predicted == question['correct_answer']


# Example usage
if __name__ == "__main__":
    # Load benchmark
    benchmark = ORANBenchmark()
    
    print("\n" + "=" * 80)
    print("Sample Questions from Each Difficulty Level")
    print("=" * 80)
    
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n[{difficulty.upper()}]")
        questions = benchmark.sample_questions(n=2, difficulty=difficulty, seed=42)
        
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}:")
            print(benchmark.format_question_for_llm(q))
            print(f"Correct Answer: {q['correct_answer']}")
            print("-" * 60)
    
    # Test mixed sampling
    print("\n" + "=" * 80)
    print("Mixed Difficulty Sample (10 questions)")
    print("=" * 80)
    
    mixed = benchmark.sample_questions(n=10, seed=42)
    difficulty_count = {'easy': 0, 'medium': 0, 'hard': 0}
    for q in mixed:
        difficulty_count[q['difficulty']] += 1
    
    print(f"Sampled distribution: {difficulty_count}")
