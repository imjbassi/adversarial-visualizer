"""Adversarial attack implementations.

All attacks operate on image tensors in [0, 1] pixel space. Wrap your model
with `utils.model_utils.NormalizedModel` (or apply normalization inside the
model's forward pass) so that gradients flow through the normalization.

Each attack accepts an optional ``callback(iteration, loss, confidence)``
used for live progress reporting.
"""

from .fgsm import fgsm_attack
from .pgd import pgd_attack
from .deepfool import deepfool_attack
from .cw import cw_attack

ATTACKS = {
    "FGSM": fgsm_attack,
    "PGD": pgd_attack,
    "DeepFool": deepfool_attack,
    "CW": cw_attack,
}

__all__ = ["fgsm_attack", "pgd_attack", "deepfool_attack", "cw_attack", "ATTACKS"]
