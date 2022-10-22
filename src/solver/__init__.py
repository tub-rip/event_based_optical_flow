"""isort:skip_file
"""

# Non DNN
from .base import SolverBase

# from .contrast_maximization import ContrastMaximization
from .patch_contrast_mixed import MixedPatchContrastMaximization
from .time_aware_patch_contrast import TimeAwarePatchContrastMaximization
from .patch_contrast_pyramid import PyramidalPatchContrastMaximization


# List of supported solver - non DNN
collections = {
    # CMax and variants
    # "contrast_maximization": ContrastMaximization,
    "pyramidal_patch_contrast_maximization": PyramidalPatchContrastMaximization,
    "time_aware_mixed_patch_contrast_maximization": TimeAwarePatchContrastMaximization,
}
