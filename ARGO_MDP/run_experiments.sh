#!/bin/bash
# ARGO Experiment Runner Script
# Quick commands for running different experiments

PYTHON_PATH="/root/miniconda/envs/ARGO/bin/python"
PROJECT_DIR="/data/user/huangxiaolin/ARGO2/ARGO_MDP"

echo "======================================"
echo "ARGO MDP Experiment Runner"
echo "======================================"
echo ""

# Function to run experiment
run_experiment() {
    local config=$1
    local extra_args=$2
    echo "Running: $config $extra_args"
    $PYTHON_PATH $PROJECT_DIR/scripts/run_single.py --config $PROJECT_DIR/configs/$config $extra_args
}

# Parse arguments
case "$1" in
    "test")
        echo "Running basic tests..."
        $PYTHON_PATH $PROJECT_DIR/scripts/test_basic.py
        ;;
    
    "quick")
        echo "Running quick experiment (base config)..."
        run_experiment "base.yaml"
        ;;
    
    "sensitivity")
        echo "Running experiment with sensitivity analysis..."
        run_experiment "base.yaml" "--sensitivity"
        ;;
    
    "visualize")
        echo "Generating visualizations..."
        $PYTHON_PATH $PROJECT_DIR/draw_figs/plot_value_function.py
        $PYTHON_PATH $PROJECT_DIR/draw_figs/plot_comparison.py
        echo "Figures saved to $PROJECT_DIR/figs/"
        ;;
    
    "full")
        echo "Running full pipeline..."
        run_experiment "base.yaml" "--sensitivity"
        echo ""
        echo "Generating visualizations..."
        $PYTHON_PATH $PROJECT_DIR/draw_figs/plot_value_function.py
        $PYTHON_PATH $PROJECT_DIR/draw_figs/plot_comparison.py
        echo ""
        echo "✓ Full pipeline completed!"
        echo "Results in: $PROJECT_DIR/results/"
        echo "Figures in: $PROJECT_DIR/figs/"
        ;;
    
    "clean")
        echo "Cleaning results and figures..."
        rm -rf $PROJECT_DIR/results/*.csv
        rm -rf $PROJECT_DIR/results/*.txt
        rm -rf $PROJECT_DIR/figs/*.png
        echo "✓ Cleaned"
        ;;
    
    "help"|*)
        echo "Usage: bash run_experiments.sh [command]"
        echo ""
        echo "Commands:"
        echo "  test        - Run unit tests"
        echo "  quick       - Run basic experiment"
        echo "  sensitivity - Run with sensitivity analysis"
        echo "  visualize   - Generate plots from existing results"
        echo "  full        - Run complete pipeline (experiment + viz)"
        echo "  clean       - Remove results and figures"
        echo "  help        - Show this message"
        echo ""
        echo "Examples:"
        echo "  bash run_experiments.sh test"
        echo "  bash run_experiments.sh full"
        ;;
esac
