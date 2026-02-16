"""
DeepBridge Distillation - Model Compression and Knowledge Distillation

This package provides automated knowledge distillation and model compression
tools for machine learning models.

Requires: deepbridge>=2.0.0
"""

__version__ = '2.0.0-alpha.1'
__author__ = 'Team DeepBridge'

from deepbridge_distillation.auto_distiller import AutoDistiller
from deepbridge_distillation.experiment_runner import ExperimentRunner
from deepbridge_distillation.hpmkd_wrapper import HPMKD

try:
    from deepbridge_distillation.techniques.knowledge_distillation import (
        KnowledgeDistillation,
    )
    from deepbridge_distillation.techniques.surrogate import SurrogateModel
    from deepbridge_distillation.techniques.ensemble import EnsembleDistillation
except ImportError as e:
    # Pode falhar se dependências não estiverem instaladas
    pass

__all__ = [
    'AutoDistiller',
    'ExperimentRunner',
    'HPMKD',
    'KnowledgeDistillation',
    'SurrogateModel',
    'EnsembleDistillation',
]
