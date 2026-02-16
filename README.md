# DeepBridge Distillation

[![Tests](https://github.com/DeepBridge-Validation/deepbridge-distillation/actions/workflows/tests.yml/badge.svg)](https://github.com/DeepBridge-Validation/deepbridge-distillation/actions)
[![codecov](https://codecov.io/gh/DeepBridge-Validation/deepbridge-distillation/branch/main/graph/badge.svg)](https://codecov.io/gh/DeepBridge-Validation/deepbridge-distillation)
[![PyPI version](https://badge.fury.io/py/deepbridge-distillation.svg)](https://badge.fury.io/py/deepbridge-distillation)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Model Compression and Knowledge Distillation Toolkit - Extension for [DeepBridge](https://github.com/DeepBridge-Validation/DeepBridge)

> **Part of the DeepBridge v2.0 Ecosystem**
>
> This package was extracted from DeepBridge v1.x to provide focused model compression capabilities.
> See [Migration Guide](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md) if migrating from v1.x.

## Installation

```bash
pip install deepbridge-distillation
```

This will automatically install `deepbridge>=2.0.0` as a dependency.

## Quick Start

```python
from deepbridge import DBDataset
from deepbridge_distillation import AutoDistiller

# Create dataset with teacher model predictions
dataset = DBDataset(
    data=df,
    target_column='target',
    features=features,
    prob_cols=['prob_0', 'prob_1']
)

# Run automated distillation
distiller = AutoDistiller(
    dataset=dataset,
    output_dir='results',
    n_trials=10
)
results = distiller.run(use_probabilities=True)
```

## Features

- **Automated Distillation**: AutoDistiller with hyperparameter optimization
- **Knowledge Distillation**: Transfer knowledge from teacher to student models
- **Surrogate Models**: Create efficient surrogate models
- **HPM Knowledge Distillation**: Hierarchical Prototype-based Method
- **Multi-framework Support**: Works with scikit-learn, XGBoost, PyTorch

## Documentation

Full documentation: https://deepbridge.readthedocs.io/en/latest/distillation/

## Related Projects

- [deepbridge](https://github.com/DeepBridge-Validation/deepbridge) - Model Validation Toolkit (core)
- [deepbridge-synthetic](https://github.com/DeepBridge-Validation/deepbridge-synthetic) - Synthetic Data Generation

## License

MIT License - see [LICENSE](LICENSE)
