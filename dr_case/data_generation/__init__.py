"""
Dr.Case — Data Generation Module

Генерація навчальних даних на основі реальних частот симптомів.
"""

from .frequency_sampler import (
    FrequencySampler,
    SamplerConfig,
    GeneratedSample,
)

from .two_branch_generator import (
    TwoBranchDataGenerator,
    TwoBranchSamplerConfig,
)

__all__ = [
    "FrequencySampler",
    "SamplerConfig", 
    "GeneratedSample",
    "TwoBranchDataGenerator",
    "TwoBranchSamplerConfig",
]
