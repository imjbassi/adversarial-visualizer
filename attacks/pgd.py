import torch
import torch.nn.functional as F


def pgd_attack(model, image, label, epsilon=0.03, alpha=None, iters=40,
               momentum=0.9, random_start=True, callback=None, **kwargs):
    """Projected Gradient Descent with momentum (Madry et al., 2018 / MI-FGSM).

    Args:
        model: classifier taking [0, 1] pixel-space tensors (1, C, H, W)
        image: input tensor in [0, 1]
        label: true label tensor of shape (1,)
        epsilon: L-inf perturbation budget in pixel units
        alpha: step size (defaults to 2.5 * epsilon / iters)
        iters: number of iterations
        momentum: gradient momentum decay factor
        random_start: start from a random point inside the epsilon ball
        callback: optional callable(iteration, loss, confidence)
    Returns:
        adversarial image tensor in [0, 1]
    """
    ori_image = image.clone().detach()
    if alpha is None:
        alpha = max(2.5 * epsilon / max(iters, 1), 1e-4)

    if random_start:
        image = ori_image + torch.empty_like(ori_image).uniform_(-epsilon, epsilon)
        image = torch.clamp(image, 0, 1).detach()
    else:
        image = ori_image.clone().detach()

    grad_accum = torch.zeros_like(image)

    if callback is not None:
        with torch.no_grad():
            output = model(image)
            callback(0, F.cross_entropy(output, label).item(),
                     torch.softmax(output, dim=1).max().item())

    for i in range(iters):
        image.requires_grad_(True)
        output = model(image)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()

        if callback is not None:
            with torch.no_grad():
                callback(i + 1, loss.item(),
                         torch.softmax(output, dim=1).max().item())

        # Normalize gradient by its L1 norm before accumulating (MI-FGSM)
        grad = image.grad / (image.grad.abs().mean() + 1e-12)
        grad_accum = momentum * grad_accum + grad
        adv_image = image + alpha * grad_accum.sign()

        eta = torch.clamp(adv_image - ori_image, -epsilon, epsilon)
        image = torch.clamp(ori_image + eta, 0, 1).detach()

    return image
