"""isort:skip_file
"""
# Basics
from .base import CostBase
from .gradient_magnitude import GradientMagnitude
from .image_variance import ImageVariance

# from .zhu_average_timestamp import ZhuAverageTimestamp
# from .paredes_average_timestamp import ParedesAverageTimestamp

# Flow related
from .total_variation import TotalVariation

# Normalized ~
from .normalized_image_variance import NormalizedImageVariance
from .normalized_gradient_magnitude import NormalizedGradientMagnitude

# Multi-reference ~
from .multi_focal_normalized_image_variance import MultiFocalNormalizedImageVariance
from .multi_focal_normalized_gradient_magnitude import MultiFocalNormalizedGradientMagnitude


def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


functions = {k.name: k for k in inheritors(CostBase)}

# For hybrid loss
from .hybrid import HybridCost
