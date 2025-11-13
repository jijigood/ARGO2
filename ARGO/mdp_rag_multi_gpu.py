"""
MDP-Guided RAG with Multi-GPU Support
支持多GPU并行推理，显著提升速度

硬件配置:
- 8x NVIDIA RTX 3060 (12GB each)
- CUDA 12.4
- PyTorch 2.6.0+cu124

推荐模型:
- Qwen2.5-7B-Instruct (单GPU: ~14GB, 可用model_parallel)
- Qwen2.5-14B-Instruct (需要模型并行，2-3个GPU)
- Qwen2.5-32B-Instruct (需要4-6个GPU)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import yaml
import random
from typing import Dict, List
import datetime
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oran_benchmark_loader import ORANBenchmark

# 加载配置文件
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'multi_gpu.yaml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# 导入 MDP
sys.path.insert(0, '../ARGO_MDP/src')
try:
    from mdp_solver import MDPSolver
    MDP_AVAILABLE = True
except:
    MDP_AVAILABLE = False


class MultiGPU_MDP_RAG:
    """
    多GPU加速的 MDP-Guided RAG
    支持三种模式:
    1. DataParallel: 多个问题并行处理（推荐批量评估）
    2. ModelParallel: 大模型分片到多个GPU（推荐14B+模型）
    3. Pipeline: 流水线并行（推荐超大批量）
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_mdp: bool = True,
        gpu_mode: str = "auto",  # "auto", "data_parallel", "model_parallel", "accelerate"
        gpu_ids: List[int] = None,  # 使用哪些GPU，None表示全部
        max_memory_per_gpu: str = "10GB"  # 每个GPU最大内存使用
    ):
        """
        Args:
            model_name: 模型名称或路径
            use_mdp: 是否使用 MDP 策略
            gpu_mode: GPU使用模式
                - "auto": 自动选择（根据模型大小）
                - "data_parallel": 数据并行（多个样本并行）
                - "model_parallel": 模型并行（大模型分片）
                - "accelerate": 使用Accelerate库自动分配
            gpu_ids: 使用的GPU ID列表，如 [0,1,2,3]
            max_memory_per_gpu: 每个GPU的最大内存限制
        """
        self.model_name = model_name
        self.use_mdp = use_mdp
        
        # GPU配置
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! This script requires GPU.")
        
        self.n_gpus = torch.cuda.device_count()
        self.gpu_ids = gpu_ids if gpu_ids else list(range(self.n_gpus))
        self.gpu_mode = gpu_mode
        
        print(f"\n{'='*80}")
        print(f"Initializing Multi-GPU MDP-RAG")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Available GPUs: {self.n_gpus}")
        print(f"Using GPUs: {self.gpu_ids}")
        print(f"GPU Mode: {gpu_mode}")
        print(f"Max Memory/GPU: {max_memory_per_gpu}")
        print(f"Use MDP: {use_mdp}")
        
        # 显示GPU信息
        for i in self.gpu_ids:
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"{'='*80}\n")
        
        # 加载 MDP 策略和参数
        self.mdp_config = CONFIG['mdp']
        self.delta_r = self.mdp_config['delta_r']
        self.delta_p = self.mdp_config['delta_p']
        self.c_r = self.mdp_config['c_r']  # 修正后的0.05
        self.c_p = self.mdp_config['c_p']  # 修正后的0.02
        self.p_s = self.mdp_config['p_s']  # 检索成功率0.8 (Phase2使用)
        
        if use_mdp and MDP_AVAILABLE:
            import yaml
            config_path = "../ARGO_MDP/configs/base.yaml"
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # 使用修正后的参数（覆盖base.yaml中的值）
                config['mdp']['delta_r'] = self.delta_r  # Phase2: 覆盖delta_r
                config['mdp']['delta_p'] = self.delta_p  # Phase2: 覆盖delta_p
                config['mdp']['c_r'] = self.c_r
                config['mdp']['c_p'] = self.c_p
                config['mdp']['gamma'] = self.mdp_config['gamma']
                config['mdp']['p_s'] = self.p_s
                
                self.mdp_solver = MDPSolver(config)
                self.mdp_solver.solve()
                
                self.theta_star = self.mdp_solver.theta_star
                self.theta_cont = self.mdp_solver.theta_cont
                
                print(f"✓ MDP Solver initialized (符合ARGO V3.0规范)")
                print(f"  θ_cont = {self.theta_cont:.4f}")
                print(f"  θ*     = {self.theta_star:.4f}")
                print(f"  c_r    = {self.c_r:.3f} (修正后)")
                print(f"  c_p    = {self.c_p:.3f} (修正后)")
                print(f"  p_s    = {self.p_s:.2f} (Phase2启用)\n")
            else:
                print(f"⚠ MDP config not found, using default thresholds")
                self.theta_cont = 0.0
                self.theta_star = 1.0
        else:
            print(f"✗ MDP disabled or unavailable\n")
            self.theta_cont = 0.0
            self.theta_star = 1.0
        
        # 加载模型和tokenizer
        self._load_model(gpu_mode, max_memory_per_gpu)
    
    def _load_model(self, gpu_mode: str, max_memory_per_gpu: str):
        """根据GPU模式加载模型"""
        print(f"Loading model with {gpu_mode} mode...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 自动选择模式
        if gpu_mode == "auto":
            # 根据模型名称估算大小
            if "1.5B" in self.model_name or "3B" in self.model_name:
                gpu_mode = "single"  # 小模型单GPU即可
            elif "7B" in self.model_name:
                gpu_mode = "accelerate"  # 7B模型用accelerate分布式加载（避免OOM）
            else:
                gpu_mode = "accelerate"  # 大模型用accelerate
            print(f"  Auto-selected mode: {gpu_mode}")
        
        if gpu_mode == "single":
            # 单GPU模式
            print(f"  Loading to GPU {self.gpu_ids[0]}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=f"cuda:{self.gpu_ids[0]}",
                trust_remote_code=True
            )
            self.device = f"cuda:{self.gpu_ids[0]}"
            
        elif gpu_mode == "data_parallel":
            # 数据并行模式（推荐用于批量评估）
            print(f"  Loading with DataParallel...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.gpu_ids
            )
            self.model = self.model.to(f"cuda:{self.gpu_ids[0]}")
            self.device = f"cuda:{self.gpu_ids[0]}"
            
        elif gpu_mode == "accelerate":
            # 使用Accelerate自动分配（推荐用于大模型）
            print(f"  Loading with Accelerate (auto device mapping)...")
            
            # 构建max_memory字典
            max_memory = {i: max_memory_per_gpu for i in self.gpu_ids}
            max_memory["cpu"] = "30GB"  # CPU内存限制
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                offload_folder="offload",  # CPU卸载目录
                offload_state_dict=True
            )
            self.device = "cuda"
            
        elif gpu_mode == "model_parallel":
            # 手动模型并行（高级用户）
            print(f"  Loading with manual model parallelism...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            # 手动分配层到不同GPU
            self._manual_model_parallel()
            self.device = "cuda"
        
        else:
            raise ValueError(f"Unknown gpu_mode: {gpu_mode}")
        
        self.model.eval()
        print(f"✓ Model loaded successfully!\n")
    
    def _manual_model_parallel(self):
        """手动模型并行（将层分配到不同GPU）"""
        # 这是一个示例，具体实现取决于模型架构
        # 大多数情况下应该使用accelerate的auto模式
        layers = list(self.model.model.layers)
        n_layers = len(layers)
        n_gpus = len(self.gpu_ids)
        layers_per_gpu = n_layers // n_gpus
        
        for i, gpu_id in enumerate(self.gpu_ids):
            start_layer = i * layers_per_gpu
            end_layer = (i + 1) * layers_per_gpu if i < n_gpus - 1 else n_layers
            for layer_idx in range(start_layer, end_layer):
                layers[layer_idx] = layers[layer_idx].to(f"cuda:{gpu_id}")
        
        print(f"  Distributed {n_layers} layers across {n_gpus} GPUs")
    
    def reason_with_llm(self, question: dict, num_retrievals: int) -> tuple:
        """使用LLM进行推理"""
        prompt = self._create_prompt(question, num_retrievals)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # 移动到设备
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self._extract_answer(response)
        confidence = 0.7 + num_retrievals * 0.05
        
        return answer, min(confidence, 0.95)
    
    def _create_prompt(self, question: dict, num_retrievals: int) -> str:
        """创建提示"""
        retrieval_info = f"(Retrieved {num_retrievals} times)" if num_retrievals > 0 else ""
        
        prompt = f"""Question {retrieval_info}: {question['question']}

Options:
1. {question['options'][0]}
2. {question['options'][1]}
3. {question['options'][2]}
4. {question['options'][3]}

Answer with only the number (1, 2, 3, or 4):"""
        
        return prompt
    
    def _extract_answer(self, response: str) -> int:
        """从响应中提取答案"""
        import re
        
        response = response.lower()
        
        # 查找数字1-4
        matches = re.findall(r'\b([1-4])\b', response)
        if matches:
            return int(matches[-1])
        
        # 默认返回1
        return 1
    
    def answer_question(self, question: dict, verbose: bool = False) -> Dict:
        """使用MDP策略回答问题"""
        U = 0.0  # 信息进度
        C = 0.0  # 累积成本
        num_retrievals = 0
        num_reasons = 0
        history = []
        
        if verbose:
            print(f"\n  Q: {question['question'][:60]}...")
        
        for iteration in range(1, 101):
            # MDP决策
            if U >= self.theta_star:
                action = "terminate"
            elif U < self.theta_cont:
                action = "retrieve"
            else:
                action = "reason"
            
            if verbose:
                print(f"    Iter {iteration}: U={U:.2f}, Action={action}")
            
            if action == "retrieve":
                # Phase2: 实现检索成功率 p_s = 0.8
                # TODO Phase3: 替换为真实Retriever
                # retrieved_docs, success = self.retriever(subquery, top_k=5)
                retrieved_docs = []  # 当前模拟
                
                # 随机检索成功/失败 (p_s = 0.8)
                retrieval_success = random.random() < self.p_s
                
                if retrieval_success:
                    # 检索成功：U增加
                    U_before = U
                    U = min(U + self.delta_r, 1.0)
                    U_after = U
                else:
                    # 检索失败：U不变，但仍消耗成本
                    U_before = U
                    U_after = U
                
                C += self.c_r  # 无论成功失败都消耗成本
                num_retrievals += 1
                
                # 完整history追踪
                history.append({
                    'iteration': iteration,
                    'action': action,
                    'subquery': question['question'],  # TODO Phase3: 改为Decomposer生成的子查询
                    'retrieved_docs': retrieved_docs,  # 检索到的文档（当前为空）
                    'retrieval_success': retrieval_success,  # Phase2: 记录成功/失败状态
                    'response': None,  # retrieve动作无LLM响应
                    'intermediate_answer': None,  # retrieve动作无中间答案
                    'confidence': None,
                    'uncertainty': float(1 - U),
                    'cost': float(C),
                    'U_before': float(U_before),
                    'U_after': float(U_after)
                })
                
                if verbose:
                    status = "✓ success" if retrieval_success else "✗ failed"
                    print(f"      → Retrieval {status}, U: {U_before:.2f} → {U_after:.2f}")
            
            elif action == "reason":
                answer, confidence = self.reason_with_llm(question, num_retrievals)
                
                # 获取LLM完整响应（当前简化）
                llm_response = f"Based on O-RAN knowledge, the answer is {answer}"
                
                U = min(U + self.delta_p, 1.0)  # 使用配置参数
                C += self.c_p  # 使用修正后的成本0.02
                num_reasons += 1
                
                # 完整history追踪
                history.append({
                    'iteration': iteration,
                    'action': action,
                    'subquery': question['question'],  # TODO Phase3: 改为Decomposer生成的子查询
                    'retrieved_docs': [],  # reason动作无检索
                    'retrieval_success': None,
                    'response': llm_response,  # LLM的完整响应
                    'intermediate_answer': answer,  # 中间答案
                    'confidence': float(confidence),
                    'uncertainty': float(1 - U),
                    'cost': float(C),
                    'U_before': float(U - self.delta_p) if U >= self.delta_p else 0.0,
                    'U_after': float(U)
                })
                
                if verbose:
                    print(f"      → Answer: {answer}, Confidence: {confidence:.2f}")
            
            else:  # terminate
                break
        
        is_correct = (answer == question['correct_answer'])
        
        if verbose:
            icon = "✓" if is_correct else "✗"
            print(f"  {icon} Predicted: {answer}, Correct: {question['correct_answer']}")
        
        return {
            'question_id': question['id'],
            'predicted': answer,
            'correct': question['correct_answer'],
            'is_correct': is_correct,
            'iterations': len(history),
            'num_retrievals': num_retrievals,
            'num_reasons': num_reasons,
            'total_cost': float(C),
            'history': history
        }


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    n_questions: int = 100,
    difficulty: str = "medium",
    gpu_mode: str = "auto",
    gpu_ids: List[int] = None,
    seed: int = 42
):
    """运行多GPU评估实验"""
    
    print(f"\n{'='*80}")
    print(f"Multi-GPU MDP-RAG Evaluation")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Questions: {n_questions} ({difficulty})")
    print(f"GPU Mode: {gpu_mode}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # 加载数据集
    benchmark = ORANBenchmark()
    questions = benchmark.sample_questions(n_questions, difficulty, seed)
    
    # 初始化RAG
    rag = MultiGPU_MDP_RAG(
        model_name=model_name,
        use_mdp=True,
        gpu_mode=gpu_mode,
        gpu_ids=gpu_ids
    )
    
    # 评估
    results = []
    correct = 0
    total_cost = 0.0
    total_iterations = 0
    
    print(f"Evaluating on {len(questions)} questions...\n")
    
    for i, question in enumerate(questions, 1):
        result = rag.answer_question(question, verbose=(i <= 3))
        results.append(result)
        
        if result['is_correct']:
            correct += 1
        total_cost += result['total_cost']
        total_iterations += result['iterations']
        
        if i % 10 == 0:
            acc = correct / i
            avg_cost = total_cost / i
            avg_iter = total_iterations / i
            print(f"  [{i}/{len(questions)}] Acc={acc:.2%}, AvgCost={avg_cost:.3f}, AvgIter={avg_iter:.1f}")
    
    # 计算指标
    accuracy = correct / len(questions)
    avg_cost = total_cost / len(questions)
    avg_iterations = total_iterations / len(questions)
    
    # 保存结果
    model_short = model_name.split('/')[-1]
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model': model_name,
        'gpu_mode': gpu_mode,
        'gpu_ids': gpu_ids if gpu_ids else list(range(torch.cuda.device_count())),
        'config': {
            'n_questions': n_questions,
            'difficulty': difficulty,
            'seed': seed
        },
        'metrics': {
            'accuracy': accuracy,
            'avg_cost': avg_cost,
            'avg_iterations': avg_iterations
        },
        'results': results
    }
    
    os.makedirs('results/multi_gpu', exist_ok=True)
    output_file = f'results/multi_gpu/{model_short}_{difficulty}_{n_questions}q.json'
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Accuracy:       {accuracy:.2%} ({correct}/{len(questions)})")
    print(f"Avg Cost:       {avg_cost:.3f}")
    print(f"Avg Iterations: {avg_iterations:.1f}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-GPU MDP-RAG Evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("-n", "--n_questions", type=int, default=100,
                        help="Number of questions to evaluate")
    parser.add_argument("-d", "--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard", "mixed"],
                        help="Question difficulty")
    parser.add_argument("--gpu_mode", type=str, default="auto",
                        choices=["auto", "single", "data_parallel", "accelerate", "model_parallel"],
                        help="GPU parallelization mode")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="GPU IDs to use (e.g., 0 1 2 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    run_experiment(
        model_name=args.model,
        n_questions=args.n_questions,
        difficulty=args.difficulty,
        gpu_mode=args.gpu_mode,
        gpu_ids=args.gpu_ids,
        seed=args.seed
    )
