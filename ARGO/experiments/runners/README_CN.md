# 实验运行器与入口点

此目录是跟踪*启动*实验的脚本的规范位置：

- Python 驱动程序（`Exp_real_cost_impact_v2.py`、`Exp_RAG_benchmark.py`、`Exp_7B_optimized.py` 等）。
- Shell 包装器（`run_exp1_full.sh`、`run_multi_gpu.sh`、`start_experiment.sh`）。
- 辅助提示，如 `START_HERE.sh` 或 `quickstart_prompts_v2.sh`。

对于每个运行器，请记录：
1. 预期的数据集（例如，`ORAN-Bench-13K`、`results/multi_gpu` 缓存）。
2. GPU/内存需求（链接到 `docs/strategy/ACCELERATION_PLAN.md`）。
3. 输出位置（指向 `artifacts/results`）。

从根目录移动脚本时，保留其原始名称并添加每个脚本的 README 或文档字符串来总结 CLI 参数。
