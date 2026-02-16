# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-alpha.1] - 2026-02-16

### Added

**Initial release as a standalone package!**

This package was extracted from DeepBridge v1.x to provide focused model compression and distillation capabilities.

#### Features
- **AutoDistiller**: Automated model distillation with hyperparameter optimization (Optuna)
- **Knowledge Distillation**: Transfer knowledge from complex teacher models to simpler student models
- **Surrogate Models**: Create efficient surrogate models for expensive black-box models
- **HPM Knowledge Distillation**: Hierarchical Prototype-based Method for knowledge distillation
- **Multi-framework Support**: Works with scikit-learn, XGBoost, and PyTorch models
- **Experiment Runner**: Manage and track distillation experiments
- **Comprehensive Testing**: Full test suite with >70% coverage

#### Examples
- `basic_distillation.py`: Complete end-to-end distillation example

#### Documentation
- Comprehensive README with quick start guide
- Migration guide from DeepBridge v1.x
- API documentation
- Examples directory

### Changed
- **Package name**: `deepbridge.distillation` → `deepbridge_distillation`
- **Import path**: Updated to use new package structure
- **Dependencies**: Now explicitly depends on `deepbridge>=2.0.0` for core functionality
- **Structure**: Reorganized as standalone package with own CI/CD

### Migration from DeepBridge v1.x

If you were using `deepbridge.distillation` in v1.x:

**Before (v1.x):**
```python
from deepbridge.distillation import AutoDistiller
from deepbridge.db_data import DBDataset
```

**After (v2.0):**
```python
from deepbridge_distillation import AutoDistiller
from deepbridge import DBDataset
```

**Installation:**
```bash
# v1.x
pip install deepbridge  # Includes everything

# v2.0
pip install deepbridge deepbridge-distillation
```

See the full [Migration Guide](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md) for details.

---

## Related Projects

- **[deepbridge](https://github.com/DeepBridge-Validation/deepbridge)** - Model Validation Toolkit (core)
  - Required dependency for this package
  - Provides DBDataset, Experiment, and MetricsEvaluator

- **[deepbridge-synthetic](https://github.com/DeepBridge-Validation/deepbridge-synthetic)** - Synthetic Data Generation
  - Standalone package for synthetic data generation
  - Can be used independently or with deepbridge

---

## Support

- **Issues**: [GitHub Issues](https://github.com/DeepBridge-Validation/deepbridge-distillation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DeepBridge-Validation/deepbridge-distillation/discussions)
- **Documentation**: https://deepbridge.readthedocs.io/en/latest/distillation/
- **Email**: gustavo.haase@gmail.com

---

**Maintainers**: Gustavo Haase, Paulo Dourado
**License**: MIT
