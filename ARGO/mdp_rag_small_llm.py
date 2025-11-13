"""
MDP-Guided RAG with Small LLM (Qwen2.5-1.5B/3B)
适用于 CPU 或受限 GPU 环境

推荐使用:
- Qwen2.5-1.5B-Instruct (CPU 可用)
- Qwen2.5-3B-Instruct (CPU 稍慢但更准)
- Qwen2.5-7B-Instruct (需要 GPU 或量化)
"""

import os
import sys
import torch

# 检查可用设备（GTX 1080 Ti 不兼容 PyTorch 2.x，强制使用 CPU）
FORCE_CPU = True  # 由于 GTX 1080 Ti (CC 6.1) 与 PyTorch 2.x 不兼容

if FORCE_CPU:
    device = "cpu"
    print("✓ Using CPU (forced, GPU compatibility issue)")
elif torch.cuda.is_available():
    try:
        # 测试 CUDA 是否真的可用
        torch.cuda.init()
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    except:
        device = "cpu"
        print("⚠ GPU detected but not usable, falling back to CPU")
else:
    device = "cpu"
    print("✓ Using CPU")

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from typing import Dict, List
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oran_benchmark_loader import ORANBenchmark

# 导入 MDP
sys.path.insert(0, '../ARGO_MDP/src')
try:
    from mdp_solver import MDPSolver
    MDP_AVAILABLE = True
except:
    MDP_AVAILABLE = False


class SmallLLM_MDP_RAG:
    """
    使用小型 LLM 的 MDP-Guided RAG
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        use_mdp: bool = True,
        device: str = "auto"
    ):
        """
        Args:
            model_name: 模型名称或路径
                - "Qwen/Qwen2.5-1.5B-Instruct" (推荐 CPU)
                - "Qwen/Qwen2.5-3B-Instruct" (推荐 CPU/GPU)
                - 本地路径: "/path/to/model"
            use_mdp: 是否使用 MDP 策略
            device: "cpu", "cuda", 或 "auto"
        """
        # GTX 1080 Ti 不兼容 PyTorch 2.x，强制使用 CPU
        if device == "auto":
            self.device = "cpu"
        else:
            self.device = device
        
        print(f"\n{'='*80}")
        print(f"Initializing Small LLM MDP-RAG")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Use MDP: {use_mdp}")
        print(f"{'='*80}\n")
        
        # 加载 MDP 策略
        if use_mdp and MDP_AVAILABLE:
            import yaml
            config_path = "../ARGO_MDP/configs/base.yaml"
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                self.mdp_solver = MDPSolver(config)
                self.mdp_solver.solve()
                
                self.theta_star = self.mdp_solver.theta_star
                self.theta_cont = self.mdp_solver.theta_cont
                
                print(f"✓ MDP Policy Loaded: θ*={self.theta_star:.3f}, θ_cont={self.theta_cont:.3f}")
            else:
                self.theta_star, self.theta_cont = 0.5, 0.2
        else:
            self.theta_star, self.theta_cont = 0.5, 0.2
        
        # 加载 LLM
        print(f"\nLoading model {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 根据设备选择数据类型
        if self.device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # 减少内存占用
        )
        
        print(f"✓ Model loaded on {self.device}")
        
        if self.device == "cuda":
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def get_action(self, uncertainty: float) -> str:
        """MDP 决策"""
        if uncertainty >= self.theta_star:
            return 'retrieve'
        elif uncertainty >= self.theta_cont:
            return 'reason'
        else:
            return 'terminate'
    
    def simulate_retrieve(self, iteration: int) -> float:
        """模拟检索（收益递减）"""
        base = 0.15
        diminishing = 0.85 ** iteration
        return base * diminishing
    
    def reason_with_llm(
        self,
        question: dict,
        num_docs_retrieved: int
    ) -> tuple:
        """
        使用 LLM 推理
        
        Returns:
            (answer, confidence)
        """
        # 构建提示（简化版，不需要真实文档）
        prompt = f"""Answer the following multiple choice question. Output ONLY the number (1, 2, 3, or 4).

Question: {question['question']}

Options:
1. {question['options'][0]}
2. {question['options'][1]}
3. {question['options'][2]}
4. {question['options'][3]}

Answer (1-4):"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_text = generated_text[len(prompt):].strip()
        
        # 提取答案
        answer = self._extract_answer(answer_text)
        
        # 估计置信度（简化）
        confidence = 0.7 if num_docs_retrieved >= 3 else 0.5
        
        return answer, confidence
    
    def _extract_answer(self, text: str) -> int:
        """从文本提取答案数字"""
        import re
        
        text = text.strip()
        
        # 直接匹配
        if text in ['1', '2', '3', '4']:
            return int(text)
        
        # 查找第一个数字
        match = re.search(r'[1-4]', text)
        if match:
            return int(match.group())
        
        # 无法解析，随机猜
        return np.random.choice([1, 2, 3, 4])
    
    def answer_question(
        self,
        question: dict,
        max_iterations: int = 10,
        verbose: bool = False
    ) -> Dict:
        """使用 MDP 策略回答问题"""
        U = 1.0
        C = 0.0
        num_retrievals = 0
        num_reasons = 0
        current_answer = None
        history = []
        
        if verbose:
            print(f"\n  Q: {question['question'][:60]}...")
        
        for iteration in range(max_iterations):
            action = self.get_action(U)
            
            if verbose:
                print(f"    Iter {iteration+1}: U={U:.3f}, Action={action}")
            
            if action == 'terminate':
                break
            
            elif action == 'retrieve':
                delta_u = self.simulate_retrieve(num_retrievals)
                U = max(0, U - delta_u)
                C += 0.1
                num_retrievals += 1
            
            elif action == 'reason':
                answer, confidence = self.reason_with_llm(question, num_retrievals)
                current_answer = answer
                
                delta_u = confidence * 0.15
                U = max(0, U - delta_u)
                C += 0.05
                num_reasons += 1
                
                if verbose:
                    print(f"      → Answer: {answer}, Confidence: {confidence:.2f}")
            
            history.append({
                'iteration': iteration + 1,
                'action': action,
                'uncertainty': float(U),
                'cost': float(C)
            })
        
        if current_answer is None:
            current_answer, _ = self.reason_with_llm(question, num_retrievals)
        
        is_correct = (current_answer == question['correct_answer'])
        
        if verbose:
            icon = "✓" if is_correct else "✗"
            print(f"  {icon} Predicted: {current_answer}, Correct: {question['correct_answer']}")
        
        return {
            'question_id': question['id'],
            'predicted': current_answer,
            'correct': question['correct_answer'],
            'is_correct': is_correct,
            'iterations': len(history),
            'num_retrievals': num_retrievals,
            'num_reasons': num_reasons,
            'total_cost': float(C),
            'history': history
        }


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    n_questions: int = 20,
    difficulty: str = "easy",
    seed: int = 42
):
    """
    运行实验
    
    Args:
        model_name: 模型名称
        n_questions: 问题数量
        difficulty: 难度
        seed: 随机种子
    """
    print("="*80)
    print(f"MDP-Guided RAG Experiment with Small LLM")
    print("="*80)
    print(f"Questions: {n_questions} ({difficulty})")
    print(f"Seed: {seed}")
    print("="*80)
    
    # 加载问题
    np.random.seed(seed)
    benchmark = ORANBenchmark()
    questions = benchmark.sample_questions(n=n_questions, difficulty=difficulty, seed=seed)
    
    # 初始化系统
    rag = SmallLLM_MDP_RAG(model_name=model_name, use_mdp=True)
    
    # 评估
    results = []
    correct = 0
    
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{n_questions}]", end=" ")
        result = rag.answer_question(q, verbose=True)
        results.append(result)
        
        if result['is_correct']:
            correct += 1
    
    # 统计
    accuracy = correct / n_questions
    avg_cost = np.mean([r['total_cost'] for r in results])
    avg_iters = np.mean([r['iterations'] for r in results])
    
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    print(f"Accuracy: {accuracy:.3f} ({correct}/{n_questions})")
    print(f"Avg Cost: {avg_cost:.3f}")
    print(f"Avg Iterations: {avg_iters:.2f}")
    print("="*80)
    
    # 保存
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model': model_name,
        'config': {
            'n_questions': n_questions,
            'difficulty': difficulty,
            'seed': seed
        },
        'metrics': {
            'accuracy': float(accuracy),
            'avg_cost': float(avg_cost),
            'avg_iterations': float(avg_iters)
        },
        'results': results
    }
    
    os.makedirs('results/small_llm', exist_ok=True)
    output_file = f"results/small_llm/{model_name.split('/')[-1]}_{difficulty}_{n_questions}q.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                       default="Qwen/Qwen2.5-1.5B-Instruct",
                       help='Model name or path')
    parser.add_argument('-n', '--num_questions', type=int, default=20)
    parser.add_argument('-d', '--difficulty', type=str, 
                       choices=['easy', 'medium', 'hard'], default='easy')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    run_experiment(
        model_name=args.model,
        n_questions=args.num_questions,
        difficulty=args.difficulty,
        seed=args.seed
    )
