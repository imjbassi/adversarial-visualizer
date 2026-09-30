import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizedModel(nn.Module):
    """Wraps a classifier so it accepts [0, 1] pixel-space inputs.

    Normalization happens inside the forward pass, so attacks can operate
    directly on pixel values: gradients flow through the normalization, the
    epsilon budget is expressed in true pixel units, and clamping to [0, 1]
    is valid.
    """

    def __init__(self, model, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def load_model(device=None):
    """Load a pretrained ResNet-18 wrapped for pixel-space inputs.

    Returns:
        (model, categories): the wrapped model in eval mode on `device`,
        and the list of ImageNet class names indexed by class id.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = NormalizedModel(models.resnet18(weights=weights))
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, list(weights.meta['categories'])


def build_transform(size=224):
    """Preprocessing for pixel-space models (no normalize — the model wrapper
    handles that). Resize the short side then center-crop, the standard
    ImageNet evaluation transform, so images are never stretched."""
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
