"""
Optimization techniques for model distillation.
"""

from deepbridge_distillation.techniques.optimization.pruning import Pruning
from deepbridge_distillation.techniques.optimization.quantization import (
    Quantization,
)
from deepbridge_distillation.techniques.optimization.temperature_scaling import (
    TemperatureScaling,
)

__all__ = ['Pruning', 'Quantization', 'TemperatureScaling']
