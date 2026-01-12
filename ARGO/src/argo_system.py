"""
ARGO System - Phase 3.4 (Enhanced with ARGO Prompts V2.0)
==========================================================

完整的4组件架构 MDP-Guided RAG系统（使用标准化高质量Prompts）

组件集成：
1. QueryDecomposer: 动态子查询生成（带进度追踪）
2. Retriever: 向量检索（Chroma）+ 答案生成
3. Reasoner: MDP引导的推理（使用增强prompts）
4. AnswerSynthesizer: 最终答案合成（格式化输出）

核心流程：
    while U_t < θ* and t < T_max:
        action = MDP.get_action(U_t)
        if action == 'retrieve':
            q_t = Decomposer(x, H_t, U_t)
            r_t = Retriever(q_t, k)
            intermediate_answer = Retriever.generate_answer(q_t, r_t)
            U_{t+1} = U_t + delta_r (if success) else U_t
        else:  # reason
            intermediate_answer = Reasoner(x, H_t)
            U_{t+1} = U_t + delta_p
        H_{t+1} = H_t + [(action, ...)]
    
    O = Synthesizer(x, H_T)

Author: ARGO Team
Version: 3.1 (Enhanced Prompts)
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
from .prompts import ARGOPrompts, PromptConfig
from .progress import ProgressTracker, StationaryProgressTracker, HybridProgressTracker
from .fixed_progress import FixedProgressTracker, BoundedConfidenceTracker
from .complexity import QuestionComplexityClassifier
from .complexity_v2 import ORANComplexityClassifier, ComplexityProfile
from .threshold_table import ThresholdTable

logger = logging.getLogger(__name__)

# 导入组件
from .decomposer import QueryDecomposer
from .retriever import Retriever, MockRetriever
from .synthesizer import AnswerSynthesizer

# 导入MDP Solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ARGO_MDP/src'))
try:
    from mdp_solver import MDPSolver
    MDP_AVAILABLE = True
except:
    logger.warning("MDPSolver not available, will use fixed strategy")
    MDP_AVAILABLE = False


class ARGO_System:
    """
    ARGO V3.0 - 完整的4组件架构系统
    
    核心改进：
    1. 模块化设计：每个组件独立，易于测试和替换
    2. MDP引导：动态决策Retrieve vs Reason
    3. 渐进式推理：根据历史调整查询策略
    4. 可追溯性：完整记录推理链
    """

    @staticmethod
    def _deep_update(base: Dict, updates: Dict) -> Dict:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                ARGO_System._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def _build_policy_config(self, overrides: Optional[Dict], default_max_steps: int) -> Dict:
        simple_steps = max(3, int(round(default_max_steps * 0.5)))
        medium_steps = max(5, int(round(default_max_steps * 0.8)))
        complex_steps = max(medium_steps + 2, int(round(default_max_steps * 1.2)))

        config: Dict = {
            'theta_star': None,
            'theta_cont': None,
            'theta_star_by_complexity': {
                'simple': 0.75,
                'medium': 0.80,
                'complex': 0.85
            },
            'theta_cont_by_complexity': {
                'simple': 0.35,
                'medium': 0.45,
                'complex': 0.50
            },
            'max_steps': max(5, default_max_steps),
            'max_steps_by_complexity': {
                'simple': simple_steps,
                'medium': medium_steps,
                'complex': complex_steps
            },
            'hard_cap_steps': default_max_steps + 5,
            'progress': {
                'enabled': True,
                'base_retrieval_gain': None,
                'base_reason_gain': None,
                'coverage_weight': 0.5,
                'novelty_weight': 0.3,
                'confidence_weight': 0.4,
                'min_gain': 0.015,
                'max_gain': 0.35,
                'gain_multipliers': {
                    'simple': 1.15,
                    'medium': 1.0,
                    'complex': 0.85
                }
            },
            'complexity': {}
        }

        if overrides:
            config = self._deep_update(config, overrides)

        config['max_steps'] = max(3, int(config['max_steps']))
        config['hard_cap_steps'] = max(
            config['max_steps'], int(config.get('hard_cap_steps', config['max_steps']))
        )

        for label, value in list(config.get('max_steps_by_complexity', {}).items()):
            config['max_steps_by_complexity'][label] = max(
                1, min(int(value), config['hard_cap_steps'])
            )

        for label, value in list(config.get('theta_star_by_complexity', {}).items()):
            config['theta_star_by_complexity'][label] = max(0.4, min(float(value), 0.99))

        for label, value in list(config.get('theta_cont_by_complexity', {}).items()):
            config['theta_cont_by_complexity'][label] = max(0.0, min(float(value), 0.95))

        return config

    def _resolve_thresholds_for_umax(self, question_umax: float) -> Tuple[float, float, float]:
        """
        Resolve thresholds using pre-computed lookup table.
        
        This fixes Problem 1: Threshold Scaling Breaks Optimality
        Instead of linear scaling, we use thresholds computed via 
        full value iteration for each U_max bucket.
        
        Args:
            question_umax: Estimated U_max for the question
            
        Returns:
            (theta_cont, theta_star, actual_umax): Optimal thresholds and bucket used
        """
        if self.threshold_table is not None:
            # Use pre-computed optimal thresholds (O(1) lookup)
            theta_cont, theta_star, bucket_umax = self.threshold_table.lookup(
                question_umax, 
                strategy='ceiling'  # Conservative: use higher bucket
            )
            
            # FIX: Scale thresholds to match question_umax
            # Problem: bucket_umax > question_umax (due to ceiling)
            #          but U_t is capped at question_umax
            #          So if theta_star > question_umax, termination is impossible!
            # Solution: Scale thresholds proportionally
            if bucket_umax > 0 and bucket_umax > question_umax:
                scale = question_umax / bucket_umax
                theta_cont = theta_cont * scale
                theta_star = theta_star * scale
            
            return theta_cont, theta_star, question_umax  # Return actual question_umax, not bucket
        
        # Fallback: use MDP solver's base thresholds (legacy behavior)
        base_theta_star = self.mdp_solver.theta_star if self.use_mdp else 0.75
        base_theta_cont = self.mdp_solver.theta_cont if self.use_mdp else 0.35
        
        return base_theta_cont, base_theta_star, 1.0
    
    def _resolve_thresholds(self, complexity: str) -> Tuple[float, float]:
        """Legacy method for backward compatibility."""
        base_theta_star = self.mdp_solver.theta_star if self.use_mdp else 0.75
        base_theta_cont = self.mdp_solver.theta_cont if self.use_mdp else 0.35

        theta_star = self.policy_config.get('theta_star', base_theta_star)
        theta_cont = self.policy_config.get('theta_cont', base_theta_cont)

        theta_star = self.policy_config.get('theta_star_by_complexity', {}).get(
            complexity,
            theta_star
        )
        theta_cont = self.policy_config.get('theta_cont_by_complexity', {}).get(
            complexity,
            theta_cont
        )

        theta_star = max(0.4, min(theta_star, 0.99))
        theta_cont = max(0.0, min(theta_cont, theta_star - 0.05))

        return theta_cont, theta_star

    def _resolve_max_steps(self, complexity: str) -> int:
        base_steps = int(self.policy_config.get('max_steps', self.base_max_steps))
        comp_override = self.policy_config.get('max_steps_by_complexity', {}).get(
            complexity
        )
        if comp_override is not None:
            base_steps = comp_override
        hard_cap = int(max(base_steps, self.policy_config.get('hard_cap_steps', base_steps)))
        return max(1, min(base_steps, hard_cap))

    def _build_progress_tracker(
        self,
        question: str,
        complexity: str,
        question_umax: float = 1.0
    ):
        """
        Build progress tracker based on configuration.
        
        Supports four modes:
        - 'fixed': Pure fixed gains, exactly implements Eq(2) (RECOMMENDED)
        - 'bounded': Fixed gains + bounded confidence scaling (practical)
        - 'stationary': Fixed gains (alias for 'fixed')
        - 'dynamic': Content-based gains (legacy, violates Assumption 1)
        
        Args:
            question: The question being processed
            complexity: Question complexity label
            question_umax: Estimated maximum progress for this question
            
        Returns:
            ProgressTracker instance or None if disabled
        """
        if not self.progress_enabled:
            return None

        cfg = self.progress_config
        tracker_mode = cfg.get('mode', 'fixed')  # Default to theory-aligned
        
        if tracker_mode in ('fixed', 'stationary'):
            # MDP-compliant: fixed gains (Solves Problem 2)
            # Implements Equation (2) exactly
            # Apply complexity-based gain multipliers for simple questions
            # This allows simple questions to reach termination faster (1-2 steps)
            gain_multiplier = cfg.get('gain_multipliers', {}).get(complexity, 1.0)
            adjusted_delta_r = self.delta_r * gain_multiplier
            adjusted_delta_p = self.delta_p * gain_multiplier
            
            if self.verbose and gain_multiplier != 1.0:
                print(f"  Applying gain multiplier for {complexity}: {gain_multiplier:.2f}x")
                print(f"    delta_r: {self.delta_r:.3f} → {adjusted_delta_r:.3f}")
                print(f"    delta_p: {self.delta_p:.3f} → {adjusted_delta_p:.3f}")
            
            return FixedProgressTracker(
                delta_r=adjusted_delta_r,
                delta_p=adjusted_delta_p,
                u_max=question_umax,
                initial_progress=0.0
            )
        
        elif tracker_mode == 'bounded':
            # Fixed gains + bounded confidence scaling (practical compromise)
            # Also apply complexity-based gain multipliers
            gain_multiplier = cfg.get('gain_multipliers', {}).get(complexity, 1.0)
            adjusted_delta_r = self.delta_r * gain_multiplier
            adjusted_delta_p = self.delta_p * gain_multiplier
            confidence_scale = cfg.get('confidence_scale', 0.3)
            
            if self.verbose and gain_multiplier != 1.0:
                print(f"  Applying gain multiplier for {complexity}: {gain_multiplier:.2f}x")
            
            return BoundedConfidenceTracker(
                delta_r=adjusted_delta_r,
                delta_p=adjusted_delta_p,
                u_max=question_umax,
                confidence_scale=confidence_scale,
                initial_progress=0.0
            )
        
        elif tracker_mode == 'hybrid':
            # Legacy hybrid mode (uses old HybridProgressTracker)
            return HybridProgressTracker(
                question=question,
                delta_r=self.delta_r,
                delta_p=self.delta_p,
                U_max=question_umax,
                track_confidence=cfg.get('track_confidence', True)
            )
        
        else:  # 'dynamic' - legacy mode
            # WARNING: This violates Assumption 1
            gain_multiplier = cfg.get('gain_multipliers', {}).get(complexity, 1.0)
            base_retrieval_gain = cfg.get('base_retrieval_gain', self.delta_r)
            base_reason_gain = cfg.get('base_reason_gain', self.delta_p)

            return ProgressTracker(
                question=question,
                question_umax=question_umax,
                base_retrieval_gain=base_retrieval_gain,
                base_reason_gain=base_reason_gain,
                coverage_weight=cfg.get('coverage_weight', 0.5),
                novelty_weight=cfg.get('novelty_weight', 0.3),
                confidence_weight=cfg.get('confidence_weight', 0.4),
                min_gain=cfg.get('min_gain', 0.015),
                max_gain=cfg.get('max_gain', 0.35),
                gain_multiplier=gain_multiplier
            )
    
    def __init__(
        self,
        model,
        tokenizer,
        use_mdp: bool = True,
        mdp_config: Optional[Dict] = None,
        policy_config: Optional[Dict] = None,
        retriever_mode: str = "mock",  # "mock" or "chroma"
        chroma_dir: Optional[str] = None,
        collection_name: str = "oran_specs",
        max_steps: int = 10,
        verbose: bool = True
    ):
        """
        Args:
            model: 预训练的因果语言模型
            tokenizer: 对应的tokenizer
            use_mdp: 是否使用MDP策略（否则使用固定策略）
            mdp_config: MDP配置字典
            policy_config: 自适应终止策略与进度跟踪配置
            retriever_mode: "mock"（测试用）或"chroma"（真实检索）
            chroma_dir: Chroma数据库路径
            collection_name: Chroma集合名称
            max_steps: 最大推理步数
            verbose: 是否打印详细日志
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_mdp = use_mdp and MDP_AVAILABLE
        self.base_max_steps = max_steps
        self.policy_config = self._build_policy_config(policy_config, max_steps)
        self.max_steps = int(self.policy_config['max_steps'])
        self.hard_cap_steps = int(max(self.max_steps, self.policy_config.get('hard_cap_steps', self.max_steps)))
        self.verbose = verbose
        self.progress_config = self.policy_config.get('progress', {})
        self.progress_enabled = self.progress_config.get('enabled', True)
        # 选择分类器版本: 'v2' 使用 O-RAN 领域感知分类器
        classifier_version = self.policy_config.get('classifier_version', 'v2')
        if classifier_version == 'v2':
            # V2: 领域感知 + 与 ThresholdTable buckets 对齐
            self.complexity_classifier = ORANComplexityClassifier(
                umax_buckets=self.policy_config.get('umax_buckets'),
                config=self.policy_config.get('complexity')
            )
            logger.info("Using ORANComplexityClassifier (V2, domain-aware)")
        else:
            # V1: 原始通用分类器
            self.complexity_classifier = QuestionComplexityClassifier(
                self.policy_config.get('complexity')
            )
            logger.info("Using QuestionComplexityClassifier (V1, generic)")
        
        # 设备
        self.device = next(model.parameters()).device
        
        logger.info(f"Initializing ARGO System on {self.device}")
        
        # 初始化组件
        self._init_components(
            mdp_config,
            retriever_mode,
            chroma_dir,
            collection_name
        )
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'total_steps': 0,
            'retrieve_actions': 0,
            'reason_actions': 0,
            'successful_retrievals': 0,
            'total_time': 0.0
        }
        
        logger.info("✅ ARGO System initialized successfully")
    
    def _init_components(
        self,
        mdp_config: Optional[Dict],
        retriever_mode: str,
        chroma_dir: Optional[str],
        collection_name: str
    ):
        """初始化所有组件"""
        
        # 1. Query Decomposer
        self.decomposer = QueryDecomposer(
            self.model,
            self.tokenizer,
            max_subquery_length=128,
            temperature=0.7,
            top_p=0.9
        )
        logger.info("✅ QueryDecomposer initialized")
        
        # 2. Retriever
        if retriever_mode == "mock":
            self.retriever = MockRetriever(p_s_value=0.8)
            logger.info("✅ MockRetriever initialized (test mode)")
        elif retriever_mode == "chroma":
            if chroma_dir is None:
                raise ValueError("chroma_dir must be provided for chroma mode")
            # 从MDP配置读取p_s值（Retriever在MDP Solver之前初始化，所以从mdp_config读取）
            p_s_value = 0.8  # 默认值
            if self.use_mdp and mdp_config and 'mdp' in mdp_config:
                p_s_value = mdp_config['mdp'].get('p_s', 0.8)
                logger.info(f"✅ Using p_s={p_s_value:.3f} from MDP config")
            else:
                logger.info(f"⚠️  Using default p_s={p_s_value:.3f} (MDP config not available)")
            
            self.retriever = Retriever(
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                similarity_threshold=0.5,  # 提高阈值：从0.3到0.5，只接受高质量检索
                p_s_mode="threshold",
                p_s_value=p_s_value,  # 从MDP配置读取，而不是硬编码
                use_reranker=True,  # 启用Reranking改进检索质量
                # 使用本地已下载的bge-reranker-base模型（278M参数，1.1GB）
                reranker_model_path="/data/user/huangxiaolin/.cache/huggingface/hub/models--BAAI--bge-reranker-base",
                reranker_device=None  # None=自动选择（优先GPU，失败时回退到CPU）；'cuda'=强制GPU；'cpu'=强制CPU
            )
            logger.info(f"✅ Retriever initialized (Chroma: {chroma_dir}, p_s={p_s_value:.3f})")
        else:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")
        
        # 3. MDP Solver (if enabled)
        if self.use_mdp:
            # 默认MDP配置（符合MDPSolver的config格式）
            default_mdp_config = {
                'mdp': {
                    'U_max': 1.0,
                    'delta_r': 0.25,
                    'delta_p': 0.08,
                    'c_r': 0.05,
                    'c_p': 0.02,
                    'p_s': 0.8,
                    'mu': 0.0,
                    'gamma': 0.98,
                    'U_grid_size': 1000
                },
                'quality': {
                    'mode': 'linear',
                    'k': 1.0
                },
                'reward_shaping': {
                    'enabled': False,
                    'k': 0.0
                },
                'solver': {
                    'max_iterations': 1000,
                    'convergence_threshold': 1e-6,
                    'verbose': False
                }
            }
            
            # 合并用户配置
            if mdp_config:
                # 深度合并
                for key in mdp_config:
                    if key in default_mdp_config and isinstance(default_mdp_config[key], dict):
                        default_mdp_config[key].update(mdp_config[key])
                    else:
                        default_mdp_config[key] = mdp_config[key]
            
            self.mdp_solver = MDPSolver(default_mdp_config)
            logger.info("✅ MDPSolver initialized")
            
            # Issue 3 Fix: 记录关键 MDP 参数，确保 mu 值正确传递
            logger.info(f"MDP关键参数: c_r={self.mdp_solver.c_r:.4f}, c_p={self.mdp_solver.c_p:.4f}, "
                        f"μ={self.mdp_solver.mu:.4f}, δ_r={self.mdp_solver.delta_r:.4f}, "
                        f"δ_p={self.mdp_solver.delta_p:.4f}, p_s={self.mdp_solver.p_s:.3f}")
            if self.mdp_solver.mu == 0.0:
                logger.warning("⚠️ μ=0 检测到! MDP 将无法感知成本差异，成本扫描可能无效。请检查配置文件中 mdp.mu 的值。")
            
            # 预计算策略（使用solve方法完成value iteration和阈值计算）
            result = self.mdp_solver.solve()
            logger.info(f"MDP策略已收敛: θ*={result['theta_star']:.4f}, θ_cont={result['theta_cont']:.4f}")
        else:
            self.mdp_solver = None
            logger.info("⚠️  MDP disabled, using fixed strategy")
        
        # 4. Answer Synthesizer
        self.synthesizer = AnswerSynthesizer(
            self.model,
            self.tokenizer,
            max_answer_length=512,
            temperature=0.3,
            top_p=0.95,
            include_sources=True
        )
        logger.info("✅ AnswerSynthesizer initialized")

        if self.use_mdp:
            self.delta_r = getattr(self.mdp_solver, 'delta_r', 0.25)
            self.delta_p = getattr(self.mdp_solver, 'delta_p', 0.08)
        else:
            progress_cfg = self.policy_config.get('progress', {})
            self.delta_r = progress_cfg.get('base_retrieval_gain') or 0.25
            self.delta_p = progress_cfg.get('base_reason_gain') or 0.08
        
        # 5. Initialize Threshold Lookup Table (Fixes Problem 1)
        # Pre-computes optimal thresholds for each U_max bucket
        self.threshold_table = None
        if self.use_mdp:
            try:
                # Get base config from MDP solver
                table_config = {
                    'mdp': {
                        'U_max': 1.0,  # Will be overridden per bucket
                        'delta_r': self.delta_r,
                        'delta_p': self.delta_p,
                        'c_r': getattr(self.mdp_solver, 'c_r', 0.05),
                        'c_p': getattr(self.mdp_solver, 'c_p', 0.02),
                        'p_s': getattr(self.mdp_solver, 'p_s', 0.8),
                        'mu': getattr(self.mdp_solver, 'mu', 0.0),
                        'gamma': getattr(self.mdp_solver, 'gamma', 0.98),
                        'U_grid_size': 100  # Smaller for faster computation
                    },
                    'quality': {
                        'mode': getattr(self.mdp_solver, 'quality_mode', 'linear'),
                        'k': getattr(self.mdp_solver, 'quality_k', 1.0)
                    },
                    'reward_shaping': {'enabled': False, 'k': 0.0},
                    'solver': {
                        'max_iterations': 1000,
                        'convergence_threshold': 1e-6,
                        'verbose': False
                    }
                }
                
                # Cache path for threshold table
                # Issue 1 Fix: 在缓存文件名中包含 c_r 和 μ 值，防止成本扫描和μ扫描时使用过期缓存
                cache_dir = os.path.join(os.path.dirname(__file__), '../configs')
                c_r_value = table_config['mdp']['c_r']
                mu_value = table_config['mdp']['mu']
                cache_filename = f'threshold_cache_cr{c_r_value:.4f}_mu{mu_value:.4f}.json'
                cache_path = os.path.join(cache_dir, cache_filename)
                logger.info(f"ThresholdTable 缓存路径: {cache_path}")
                
                self.threshold_table = ThresholdTable(
                    mdp_base_config=table_config,
                    cache_path=cache_path
                )
                logger.info("✅ ThresholdTable initialized (pre-computed optimal thresholds)")
                
                if self.verbose:
                    self.threshold_table.print_table()
                    
            except Exception as e:
                logger.warning(f"Failed to initialize ThresholdTable: {e}")
                logger.warning("Falling back to legacy threshold scaling")
                self.threshold_table = None
    
    def answer_question(
        self,
        question: str,
        return_history: bool = True,
        options: Optional[List[str]] = None,
        question_metadata: Optional[Dict] = None
    ) -> Tuple[str, Optional[str], Optional[List[Dict]], Optional[Dict]]:
        """
        回答问题（主入口）
        
        Args:
            question: 输入问题
            return_history: 是否返回推理历史
            options: 选择题的选项列表（可选，格式为 ["选项1文本", "选项2文本", ...]）
            question_metadata: 问题元数据（可选），包含 'difficulty' 字段用于复杂度分类
            question_metadata: 问题元数据（可选），包含 'difficulty' 字段用于复杂度分类
        
        Returns:
            (answer, choice, history, metadata):
                - answer: 最终答案（详细解释）
                - choice: 选择的选项编号（"1"/"2"/"3"/"4"，仅当提供options时）
                - history: 推理历史（如果return_history=True）
                - metadata: 元数据（步数、时间等）
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ARGO System Processing Question")
            print(f"{'='*80}")
            print(f"Question: {question}")
            print(f"Base Max Steps: {self.max_steps}")
            print(f"Strategy: {'MDP-Guided' if self.use_mdp else 'Fixed'}")
            print(f"{'='*80}\n")
        
        # ============ Question-Adaptive Setup ============
        complexity_profile = self.complexity_classifier.classify(question, question_metadata)
        complexity_label = complexity_profile.label if isinstance(complexity_profile, ComplexityProfile) else complexity_profile
        question_umax = self.complexity_classifier.estimate_umax(question, question_metadata)
        adaptive_cap = self.complexity_classifier.get_adaptive_max_steps(
            question,
            base_max_steps=self.base_max_steps
        )
        policy_cap = self._resolve_max_steps(complexity_label)
        per_question_max_steps = min(policy_cap, adaptive_cap)

        # Use pre-computed thresholds from lookup table (Fixes Problem 1)
        theta_cont, theta_star, actual_umax = self._resolve_thresholds_for_umax(question_umax)
        # Note: thresholds are already correctly computed for this U_max bucket
        # No linear scaling needed - that would break optimality!
        
        # Apply complexity-based threshold scaling if configured
        # This allows reducing thresholds for medium/complex questions to reduce steps
        if self.policy_config and 'theta_star_by_complexity' in self.policy_config:
            complexity_threshold_base = self.policy_config['theta_star_by_complexity'].get(
                complexity_label,
                None
            )
            if complexity_threshold_base is not None:
                # Apply configured threshold: theta_star = base × question_umax
                target_theta_star = complexity_threshold_base * question_umax
                # Only apply if it's different from threshold_table value
                if abs(target_theta_star - theta_star) > 0.01:  # Allow small floating point differences
                    if self.verbose:
                        print(f"  Applying complexity-based threshold override:")
                        print(f"    θ* from threshold_table: {theta_star:.3f}")
                        print(f"    θ* from config ({complexity_label}, base={complexity_threshold_base:.2f}): {target_theta_star:.3f}")
                    theta_star = target_theta_star
                    
                    # Also adjust theta_cont proportionally to maintain the ratio
                    if theta_star > 0 and theta_cont > 0:
                        original_ratio = theta_cont / theta_star
                        theta_cont = target_theta_star * original_ratio
                        
                        # Apply theta_cont_by_complexity if configured
                        if 'theta_cont_by_complexity' in self.policy_config:
                            cont_base = self.policy_config['theta_cont_by_complexity'].get(
                                complexity_label,
                                None
                            )
                            if cont_base is not None:
                                target_theta_cont = cont_base * question_umax
                                if abs(target_theta_cont - theta_cont) > 0.01:
                                    if self.verbose:
                                        print(f"    θ_cont from config ({complexity_label}, base={cont_base:.2f}): {target_theta_cont:.3f}")
                                    theta_cont = target_theta_cont

        progress_tracker = self._build_progress_tracker(
            question,
            complexity_label,
            question_umax=question_umax
        )

        history: List[Dict] = []
        t = 0
        U_t = progress_tracker.current_progress if progress_tracker else 0.0
        
        if self.verbose:
            print(f"Complexity: {complexity_label}")
            print(f"Estimated U_max(x): {question_umax:.3f} (bucket: {actual_umax:.3f})")
            print(f"Optimal thresholds: θ_cont={theta_cont:.3f}, θ*={theta_star:.3f}")
            print(f"Max Steps (adaptive): {per_question_max_steps}")
            print(f"Progress tracker: {type(progress_tracker).__name__ if progress_tracker else 'None'}")
        
        # 主循环
        while U_t < theta_star and t < per_question_max_steps:
            t += 1
            
            if self.verbose:
                print(f"\n--- Step {t} ---")
                print(f"Current U_t: {U_t:.3f} / {question_umax:.3f} (target: {theta_star:.3f})")
            
            # 决定动作
            if self.use_mdp:
                action = self._mdp_get_action(U_t)
            else:
                # 固定策略
                action = 'retrieve' if U_t < theta_cont else 'reason'
            
            if self.verbose:
                print(f"Action: {action.upper()}")
            
            # 执行动作
            if action == 'retrieve':
                step_data = self._execute_retrieve(question, history, U_t, options=options)
                
                if self.verbose and step_data['retrieval_success']:
                    print("✓ Retrieval successful")
                elif self.verbose and not step_data['retrieval_success']:
                    print("✗ Retrieval failed")

                self.stats['retrieve_actions'] += 1
                if step_data['retrieval_success']:
                    self.stats['successful_retrievals'] += 1
            
            else:  # reason
                step_data = self._execute_reason(question, history, U_t, options=options)
                
                self.stats['reason_actions'] += 1
            
            # 记录到历史
            step_data['progress_before'] = U_t
            if progress_tracker:
                U_t = progress_tracker.update(action, step_data)
            else:
                if action == 'retrieve' and step_data.get('retrieval_success'):
                    U_t += self.delta_r
                elif action == 'reason':
                    U_t += self.delta_p
            U_t = min(U_t, question_umax)
            step_data['progress_after'] = U_t
            step_data['progress'] = U_t
            history.append(step_data)
            
            if self.verbose:
                print(f"Updated U_t → {U_t:.3f}")
            
            if U_t >= theta_star:
                if self.verbose:
                    print(f"✓ Termination threshold reached")
                break
        
        # 合成最终答案
        terminated_early = U_t >= theta_star
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Synthesizing Final Answer")
            print(f"{'='*80}")
            print(f"Final U_t: {U_t:.3f} / {question_umax:.3f}")
            print(f"Total Steps: {t}")
            if terminated_early:
                print("Terminated early (reached threshold)")
            elif t >= per_question_max_steps:
                print("Reached max_steps cap")
        
        final_answer, choice, sources = self.synthesizer.synthesize(
            question, 
            history,
            options=options
        )
        
        # 计算统计
        elapsed_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['total_steps'] += t
        self.stats['total_time'] += elapsed_time
        
        metadata = {
            'total_steps': t,
            'final_progress': U_t,
            'final_uncertainty': U_t,
            'question_umax': question_umax,
            'retrieve_count': sum(1 for s in history if s['action'] == 'retrieve'),
            'reason_count': sum(1 for s in history if s['action'] == 'reason'),
            'successful_retrievals': sum(1 for s in history 
                                        if s['action'] == 'retrieve' and s.get('retrieval_success', False)),
            'elapsed_time': elapsed_time,
            'sources': sources,
            'theta_star': theta_star,
            'theta_star_base': self.mdp_solver.theta_star if self.mdp_solver else theta_star,
            'theta_cont': theta_cont,
            'theta_cont_base': self.mdp_solver.theta_cont if self.mdp_solver else theta_cont,
            'max_steps_cap': per_question_max_steps,
            'complexity': complexity_label,
            'terminated_early': terminated_early,
            'step_cap_hit': (not terminated_early and t >= per_question_max_steps),
            'progress_mode': 'dynamic' if progress_tracker else 'static',
            'progress_efficiency': U_t / max(1, t)
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"COMPLETED")
            print(f"{'='*80}")
            print(f"Answer: {final_answer[:200]}...")
            if choice:
                print(f"Selected Choice: {choice}")
            print(f"Steps: {metadata['total_steps']}")
            print(f"Time: {metadata['elapsed_time']:.2f}s")
            if sources:
                print(f"Sources: {', '.join(sources[:3])}")
            print(f"{'='*80}\n")
        
        if return_history:
            return final_answer, choice, history, metadata
        else:
            return final_answer, choice, None, metadata
    
    def _execute_retrieve(
        self,
        original_question: str,
        history: List[Dict],
        U_t: float,
        options: Optional[List[str]] = None
    ) -> Dict:
        """
        执行检索动作（使用增强的答案生成）
        
        Args:
            original_question: 原始问题
            history: 推理历史
            U_t: 当前进度
            options: 选择题选项列表（数据集全部是选择题，直接传递）
        """
        
        # 1. 生成子查询
        uncertainty = 1.0 - U_t
        subquery = self.decomposer.generate_subquery(
            original_question,
            history,
            uncertainty
        )
        
        if self.verbose:
            print(f"Subquery: {subquery}")
        
        # 2. 检索
        docs, success, scores = self.retriever.retrieve(
            subquery,
            k=5,  # 增加检索数量：从3到5，提高找到相关文档的概率
            return_scores=True
        )
        
        if self.verbose:
            if success:
                print(f"Retrieved {len(docs)} documents")
            else:
                print(f"Retrieval failed")
        
        # 3. 生成基于检索的中间答案（新增）
        intermediate_answer = ""
        confidence = 0.0
        
        # Track if LLM found useful information
        effective_retrieval = success  # Start with technical success
        
        if success and docs:
            # 使用检索器的答案生成方法（基于ARGO V2.0 prompts）
            # 数据集全部是选择题，直接传递原始题目和选项（即使为None也传递）
            intermediate_answer = self.retriever.generate_answer_from_docs(
                question=subquery,
                docs=docs,
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=PromptConfig.REASONER_MAX_LENGTH,
                temperature=PromptConfig.REASONER_TEMPERATURE,
                top_p=PromptConfig.REASONER_TOP_P,
                original_question=original_question,
                options=options  # 直接传递，prompt内部会判断是否使用
            )
            
            # Check if LLM found useful information
            # Improved logic: check if answer contains useful information, not just exact "No information found" string
            if not intermediate_answer or intermediate_answer.strip() == "":
                effective_retrieval = False
                if self.verbose:
                    print("⚠ Retrieval returned empty answer (LLM found no useful information)")
            else:
                # Check if answer is effective (contains useful information)
                effective_retrieval = self._is_answer_effective(intermediate_answer)
                if not effective_retrieval:
                    if self.verbose:
                        print("⚠ LLM indicated no information found (retrieval ineffective)")
            
            # 估算置信度（基于检索分数和答案质量）
            if intermediate_answer:
                avg_score = np.mean(scores) if scores else 0.5
                confidence = min(0.95, avg_score * 0.8 + 0.2)
            else:
                confidence = 0.0  # No confidence if no answer
            
            if self.verbose and intermediate_answer:
                print(f"Intermediate Answer: {intermediate_answer[:150]}...")
                print(f"Confidence: {confidence:.2f}")
            elif self.verbose and not intermediate_answer:
                print("⚠ No intermediate answer generated (retrieval ineffective)")
        
        # 4. 构造步骤数据
        # Use effective_retrieval for progress tracking: only count as successful if we got useful content
        step_data = {
            'action': 'retrieve',
            'subquery': subquery,
            'retrieval_success': effective_retrieval,  # Use effective_retrieval instead of technical success
            'technical_retrieval_success': success,  # Keep track of technical success separately
            'retrieved_docs': docs,
            'retrieval_scores': scores,
            'intermediate_answer': intermediate_answer,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'progress': U_t
        }
        
        return step_data
    
    def _is_answer_effective(self, answer: str) -> bool:
        """
        判断答案是否有效（包含有用信息）
        
        改进的判断逻辑：
        - 如果答案主要是"No information found"但没有其他有用信息，认为无效
        - 如果答案包含选项编号、数值、技术术语等，认为有效
        - 如果答案较长（>100字符），可能包含解释，认为有效
        """
        if not answer or answer.strip() == "":
            return False
        
        # 检查是否主要是"No information found"
        no_info_patterns = [
            "[No information found in O-RAN specs]",
            "No information found in O-RAN specs",
            "no information found in O-RAN specs",
            "```no information found in O-RAN specs```",
            "[No information found",
            "No information found"
        ]
        
        answer_lower = answer.lower()
        has_no_info = any(pattern.lower() in answer_lower for pattern in no_info_patterns)
        
        if has_no_info:
            # 检查是否包含有用信息（选项编号、数值、技术术语等）
            import re
            useful_indicators = [
                r'\d+\.',      # 选项编号（如"1.", "2."）
                r'\d+°C',     # 温度
                r'\d+%',      # 百分比
                r'\d+[A-Z]',  # 技术规格（如"5G", "FR1"）
                r'[A-Z]{2,}', # 缩写（如"O-RAN", "SSH"）
            ]
            
            # 检查是否有数字（但排除纯"No information found"的情况）
            has_numbers = bool(re.search(r'\d+', answer))
            # 如果答案很短且只有"No information found"，认为无效
            if len(answer.strip()) < 50 and not has_numbers:
                return False
            
            has_useful_info = any(re.search(pattern, answer) for pattern in useful_indicators)
            
            # 如果答案较长（>100字符），可能包含解释，认为有效
            # 或者包含有用信息，认为有效
            if len(answer.strip()) > 100 or has_useful_info or has_numbers:
                return True  # 有有用信息，认为检索有效
            else:
                return False  # 主要是"No information found"，认为无效
        
        return True  # 没有"No information found"，认为有效
    
    def _execute_reason(
        self,
        original_question: str,
        history: List[Dict],
        U_t: float,
        options: Optional[List[str]] = None
    ) -> Dict:
        """
        执行推理动作（使用增强的推理prompt）
        
        Args:
            original_question: 原始问题
            history: 推理历史
            U_t: 当前进度
            options: 选择题选项列表（数据集全部是选择题，直接传递）
        """
        
        # 构建推理提示词（使用ARGO V2.0标准模板，传递原始题目和选项）
        prompt = ARGOPrompts.build_reasoning_prompt(
            original_question=original_question,
            history=history,
            options=options
        )
        
        # 生成中间答案
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096  # 增加以容纳完整历史
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=PromptConfig.REASONER_MAX_LENGTH,
                temperature=PromptConfig.REASONER_TEMPERATURE,
                top_p=PromptConfig.REASONER_TOP_P,
                do_sample=True if PromptConfig.REASONER_TEMPERATURE > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        intermediate_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        intermediate_answer = intermediate_answer.strip()
        
        # 简单的置信度估计（基于答案长度和U_t）
        confidence = min(0.9, 0.5 + U_t * 0.4)
        
        if self.verbose:
            print(f"Intermediate Answer: {intermediate_answer[:150]}...")
            print(f"Confidence: {confidence:.2f}")
        
        step_data = {
            'action': 'reason',
            'intermediate_answer': intermediate_answer,
            'confidence': confidence,
            'uncertainty': 1.0 - U_t,
            'progress': U_t
        }
        
        return step_data
    
    def _mdp_get_action(self, U_t: float) -> str:
        """
        从MDP的Q函数中获取最优动作
        
        Args:
            U_t: 当前状态（知识完整度）
        
        Returns:
            "retrieve" or "reason"
        """
        # 获取状态索引
        idx = self.mdp_solver.get_state_index(U_t)
        
        # 获取Q值（忽略terminate动作，因为我们在主循环中已经检查了theta_star）
        Q_retrieve = self.mdp_solver.Q[idx, 0]
        Q_reason = self.mdp_solver.Q[idx, 1]
        
        # 选择Q值更高的动作
        if Q_retrieve >= Q_reason:
            return 'retrieve'
        else:
            return 'reason'
    
    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        stats = self.stats.copy()
        
        if stats['total_queries'] > 0:
            stats['avg_steps_per_query'] = stats['total_steps'] / stats['total_queries']
            stats['avg_time_per_query'] = stats['total_time'] / stats['total_queries']
            stats['retrieval_success_rate'] = (
                stats['successful_retrievals'] / stats['retrieve_actions']
                if stats['retrieve_actions'] > 0 else 0.0
            )
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_queries': 0,
            'total_steps': 0,
            'retrieve_actions': 0,
            'reason_actions': 0,
            'successful_retrievals': 0,
            'total_time': 0.0
        }
        logger.info("Statistics reset")
