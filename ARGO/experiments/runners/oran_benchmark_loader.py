#!/usr/bin/env python
"""
ORAN Benchmark Loader
简化版数据加载器，用于Pareto实验
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


class ORANBenchmark:
    """ORAN-Bench-13K 数据加载器"""
    
    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: 数据集路径
        """
        if data_path is None:
            # 默认路径
            data_path = "/data/user/huangxiaolin/ARGO/ORAN-Bench-13K/Benchmark"
        
        self.data_path = Path(data_path)
        self.questions = []
        self._load_data()
    
    def _load_data(self):
        """加载所有数据"""
        self.questions = []
        
        # 加载不同难度的数据 (实际文件名)
        difficulty_files = {
            'easy': 'fin_E.json',
            'medium': 'fin_M.json', 
            'hard': 'fin_H_clean.json'  # 使用清洗后的版本
        }
        
        for difficulty, filename in difficulty_files.items():
            filepath = self.data_path / filename
            
            # 尝试备选文件名
            if not filepath.exists():
                filepath = self.data_path / filename.replace('_clean', '')
            
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        # JSONL格式：每行是一个JSON数组 [question, options, answer]
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                                # 格式: [question, options, answer]
                                if isinstance(item, list) and len(item) >= 3:
                                    self.questions.append({
                                        'question': item[0],
                                        'options': item[1],
                                        'answer': str(item[2]),
                                        'difficulty': difficulty,
                                        'id': f"{difficulty}_{line_num}"
                                    })
                                elif isinstance(item, dict):
                                    item['difficulty'] = difficulty
                                    item['id'] = f"{difficulty}_{line_num}"
                                    self.questions.append(item)
                            except json.JSONDecodeError as e:
                                pass  # 跳过解析失败的行
                except Exception as e:
                    print(f"⚠ Error loading {filename}: {e}")
        
        print(f"✓ Loaded {len(self.questions)} questions from ORAN-Bench-13K")
        
        # 打印各难度分布
        for diff in ['easy', 'medium', 'hard']:
            count = len([q for q in self.questions if q.get('difficulty') == diff])
            print(f"  {diff}: {count} questions")
    
    def sample_questions(
        self,
        n: int = 100,
        difficulty: str = "medium",
        seed: int = 42
    ) -> List[Dict]:
        """
        采样问题
        
        Args:
            n: 采样数量
            difficulty: 难度级别 (easy/medium/hard/mixed)
            seed: 随机种子
        
        Returns:
            问题列表
        """
        random.seed(seed)
        
        if difficulty == "mixed":
            pool = self.questions
        else:
            pool = [q for q in self.questions if q.get('difficulty') == difficulty]
        
        if len(pool) == 0:
            print(f"⚠ No questions found for difficulty={difficulty}, using all questions")
            pool = self.questions
        
        n = min(n, len(pool))
        sampled = random.sample(pool, n)
        
        return sampled
    
    def get_all_questions(self, difficulty: str = None) -> List[Dict]:
        """获取所有问题"""
        if difficulty is None:
            return self.questions
        return [q for q in self.questions if q.get('difficulty') == difficulty]


if __name__ == "__main__":
    # 测试
    benchmark = ORANBenchmark()
    
    if len(benchmark.questions) > 0:
        questions = benchmark.sample_questions(n=3, difficulty="hard", seed=42)
        print(f"\nSampled {len(questions)} hard questions:")
        for i, q in enumerate(questions, 1):
            q_text = q['question'][:70] if q['question'] else 'N/A'
            print(f"  {i}. {q_text}...")
            print(f"     Answer: {q['answer']}")
