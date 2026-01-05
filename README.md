# MLIP Finetuning Strategies

A systematic evaluation framework for continual learning strategies in Machine Learning Interatomic Potentials (MLIP) finetuning.

## Overview

When finetuning pretrained MLIP models on new datasets, catastrophic forgetting poses a significant challenge: the model loses previously learned knowledge while adapting to new data. This library implements and evaluates several continual learning strategies to mitigate this problem.

## Features

### Implemented Strategies

| Strategy | Description | Key Mechanism |
|----------|-------------|---------------|
| **Naive** | Standard finetuning without any forgetting prevention | Baseline for comparison |
| **EWC** | Elastic Weight Consolidation | Penalizes changes to important parameters using Fisher Information Matrix |
| **Replay** | Experience Replay | Retains subset of previous data and mixes with new data during training |
| **P&C** | Progress & Compress | Separates learning (active column) from knowledge preservation (knowledge base) |

### Supported MLIP Architectures

- **NequIP**: Neural Equivariant Interatomic Potentials
- **Allegro** (coming soon)
- **MACE** (coming soon)

### Analysis Tools

- **Memory Retention Analysis**: Evaluate how well models retain pretrained knowledge during finetuning
- **Catastrophic Forgetting Metrics**: Quantify performance degradation on original tasks
- **Training Progress Visualization**: Track metrics across epochs with automatic plotting

## Installation

### From Source

```bash
git clone https://github.com/yourusername/mlip-finetuning-strategies.git
cd mlip-finetuning-strategies
pip install -e .
```

### Dependencies

Core dependencies:
- PyTorch >= 1.12.0
- ASE >= 3.22.0
- NumPy >= 1.21.0
- PyYAML >= 6.0
- Weights & Biases >= 0.12.0

Optional model dependencies:
- NequIP >= 0.5.0 (for NequIP support)
- Allegro >= 0.2.0 (for Allegro support)
- MACE-torch >= 0.2.0 (for MACE support)

## Quick Start

### 1. Basic Usage

```bash
# Run naive finetuning (baseline)
mlip-finetune --config configs/naive/naive_default.yaml --run-name my_experiment

# Run EWC finetuning
mlip-finetune --config configs/ewc/ewc_default.yaml --run-name ewc_experiment
```

### 2. Configuration

Create a configuration file (e.g., `my_config.yaml`):

```yaml
# Model configuration
model:
  architecture: nequip
  pretrained_path: /path/to/pretrained/model.nequip

# Data configuration
data:
  pretrain_data: /path/to/pretrain/dataset.xyz
  finetune_data: /path/to/finetune/dataset.xyz
  batch_size: 32

# Training configuration
training:
  epochs: 100
  lr: 1.0e-4
  optimizer: adam

# Strategy configuration
strategy:
  name: ewc
  lambda: 1000.0
  fisher_samples: 500

# Evaluation metrics
evaluation:
  metrics: [energy_mae, force_mae, stress_mae]

# Logging
wandb_project: mlip-finetuning
```

### 3. Python API

```python
from mlip_finetuning_strategies import MLIPModel, EWCStrategy
from mlip_finetuning_strategies.data import MLIPDataset, create_dataloader

# Load model
model = MLIPModel(
    architecture="nequip",
    model_path="/path/to/pretrained/model.nequip"
)

# Load data
dataset = MLIPDataset("/path/to/data.xyz")
dataloader = create_dataloader(dataset, batch_size=32)

# Create strategy
strategy = EWCStrategy(model, {"lambda": 1000.0})

# Train (simplified example)
for batch in dataloader:
    predictions = model(batch)
    loss = strategy.compute_loss(predictions, batch, task_id=0)
    # ... optimization step
```

## Memory Retention Analysis

Analyze how well your finetuned models retain knowledge from pretraining.

### 1. CLI Analysis Tool

```bash
# Basic memory retention analysis
mlip-analyze --experiment-dir experiments/20250105_120000_ewc_experiment \
             --reference-datasets pretrain:/path/to/pretrain/data.xyz \
             --wandb-project memory-retention-analysis

# Multiple reference datasets
mlip-analyze --experiment-dir experiments/20250105_120000_ewc_experiment \
             --reference-datasets \
                 pretrain:/path/to/pretrain/data.xyz \
                 validation:/path/to/validation/data.xyz \
                 test:/path/to/test/data.xyz

# Advanced options
mlip-analyze --experiment-dir experiments/my_experiment \
             --reference-datasets pretrain:/path/to/data.xyz \
             --batch-size 16 \
             --epoch-step 5 \
             --max-epochs 100 \
             --subset-size 1000 \
             --device cuda
```

### 2. Python API for Analysis

```python
from mlip_finetuning_strategies.analysis import analyze_memory_retention

# Define reference datasets
reference_datasets = {
    "pretrain": "/path/to/pretrain/data.xyz",
    "validation": "/path/to/validation/data.xyz"
}

# Model configuration
model_config = {
    "architecture": "nequip",
    "pretrained_path": "/path/to/pretrained/model.nequip"
}

# Run analysis
results = analyze_memory_retention(
    experiment_dir="experiments/my_experiment",
    reference_datasets=reference_datasets,
    model_config=model_config,
    wandb_project="memory-retention",
    batch_size=32,
    max_epochs=100,
    epoch_step=5
)

# Access results
for dataset_name, data in results.items():
    epochs = data["epochs"]
    metrics = data["metrics"]
    print(f"{dataset_name}: {len(epochs)} epochs analyzed")
```

### 3. Configuration for Automatic Analysis

Add memory retention configuration to your training config:

```yaml
# In your training config.yaml
memory_retention:
  reference_datasets:
    pretrain_general: /path/to/pretrain/general.xyz
    pretrain_specific: /path/to/pretrain/specific.xyz
    validation: /path/to/validation.xyz

  analysis:
    batch_size: 32
    epoch_step: 10  # Analyze every 10 epochs
    subset_size: 1000  # Use subset for faster analysis
```

### 4. Understanding the Results

The analysis generates several visualizations:

- **Energy/Force Error Evolution**: Track how errors change during training
- **Normalized Metrics**: Compare relative performance (1.0 = pretrained baseline)
- **Forgetting Rate**: Percentage increase in error from pretrained baseline
- **Multi-Dataset Comparison**: Compare retention across different datasets

**Key Metrics:**
- `energy_mae`: Mean Absolute Error for energy predictions
- `force_mae`: Mean Absolute Error for force predictions
- `energy_forgetting_pct`: Percentage increase in energy error
- `force_forgetting_pct`: Percentage increase in force error

**Interpreting Results:**
- Values > 1.0 in normalized plots indicate worse performance than pretrained
- Positive forgetting rates indicate catastrophic forgetting
- Stable or decreasing trends show good retention

## Configuration Reference

### Strategy-Specific Parameters

#### EWC (Elastic Weight Consolidation)
```yaml
strategy:
  name: ewc
  lambda: 1000.0           # Regularization strength
  fisher_samples: 500      # Number of samples for Fisher computation
  diagonal_only: true      # Use diagonal Fisher approximation
```

#### Replay
```yaml
strategy:
  name: replay
  buffer_size: 1000       # Size of replay buffer
  replay_ratio: 0.2       # Fraction of replay data in each batch
```

#### Progress & Compress
```yaml
strategy:
  name: pnc
  compress_ratio: 0.1     # Compression ratio for knowledge base
  active_size: 0.1        # Size of active column
```

## Evaluation Metrics

The framework computes several metrics to evaluate both performance and forgetting:

### Retention Metrics
- **Pretrain MAE**: Error on original pretrain dataset after finetuning
- **Forgetting Rate**: (Post-finetune error - Pre-finetune error) / Pre-finetune error

### Transfer Metrics
- **Finetune MAE**: Error on finetuning target dataset
- **Forward Transfer**: Learning efficiency compared to training from scratch

### Property-Specific Metrics
- Energy MAE/RMSE (eV)
- Force MAE/RMSE (eV/Å)
- Stress MAE/RMSE (eV/Å³)

## Project Structure

```
mlip-finetuning-strategies/
├── src/mlip_finetuning_strategies/
│   ├── strategies/          # Continual learning strategies
│   ├── models/              # MLIP model wrappers
│   ├── data/                # Data loading and processing
│   ├── utils/               # Utility functions
│   ├── cli.py               # Command-line interface
│   └── runner.py            # Main experiment runner
├── configs/                 # Example configurations
├── examples/                # Example scripts and notebooks
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Adding New Strategies

To add a new continual learning strategy:

1. Create a new file in `src/mlip_finetuning_strategies/strategies/`
2. Inherit from `BaseStrategy` and implement the required methods
3. Add the strategy to the `create_strategy` function in `runner.py`
4. Create configuration templates in `configs/`

### Adding New MLIP Architectures

To add support for a new MLIP architecture:

1. Extend the `MLIPModel` class in `src/mlip_finetuning_strategies/models/mlip.py`
2. Implement model loading and forward pass methods
3. Add dependencies to `pyproject.toml`

## Citation

If you use this library in your research, please cite:

```bibtex
@software{mlip_finetuning_strategies,
  title={MLIP Finetuning Strategies: A Framework for Continual Learning in Interatomic Potentials},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/mlip-finetuning-strategies}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting in neural networks." PNAS.
2. Schwarz et al. (2018). "Progress & Compress: A scalable framework for continual learning." ICML.
3. Batzner et al. (2022). "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials." Nature Communications.