"""
Specific distillation techniques implementations.
"""

from deepbridge_distillation.techniques.ensemble import EnsembleDistillation
from deepbridge_distillation.techniques.knowledge_distillation import (
    KnowledgeDistillation,
)
from deepbridge_distillation.techniques.surrogate import SurrogateModel

__all__ = ['KnowledgeDistillation', 'SurrogateModel', 'EnsembleDistillation']
