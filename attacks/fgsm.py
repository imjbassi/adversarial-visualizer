import torch
import torch.nn.functional as F


def fgsm_attack(model, image, label, epsilon=0.03, callback=None, **kwargs):
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    Single-step attack: x_adv = clip(x + epsilon * sign(grad_x L), 0, 1).

    Args:
        model: classifier taking [0, 1] pixel-space tensors (1, C, H, W)
        image: input tensor in [0, 1]
        label: true label tensor of shape (1,)
        epsilon: L-inf perturbation budget in pixel units
        callback: optional callable(iteration, loss, confidence)
    Returns:
        adversarial image tensor in [0, 1]
    """
    image = image.clone().detach()

    if callback is not None:
        with torch.no_grad():
            output = model(image)
            callback(0, F.cross_entropy(output, label).item(),
                     torch.softmax(output, dim=1).max().item())

    image.requires_grad_(True)
    output = model(image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()

    perturbed = torch.clamp(image + epsilon * image.grad.sign(), 0, 1).detach()

    if callback is not None:
        with torch.no_grad():
            output = model(perturbed)
            callback(1, F.cross_entropy(output, label).item(),
                     torch.softmax(output, dim=1).max().item())

    return perturbed
