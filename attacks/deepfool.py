import torch
import torch.nn.functional as F


def deepfool_attack(model, image, label=None, num_classes=10, overshoot=0.02,
                    max_iter=50, callback=None, **kwargs):
    """DeepFool minimal-perturbation attack (Moosavi-Dezfooli et al., 2016).

    Iteratively moves the input toward the nearest decision boundary among
    the top-``num_classes`` candidate classes of the original prediction.
    Only the candidate classes are differentiated, so the attack stays fast
    even for 1000-class models.

    Args:
        model: classifier taking [0, 1] pixel-space tensors (1, C, H, W)
        image: input tensor in [0, 1]
        label: true label tensor of shape (1,); inferred if None
        num_classes: number of candidate classes to consider
        overshoot: final perturbation scaling to cross the boundary
        max_iter: max iterations
        callback: optional callable(iteration, loss, confidence)
    Returns:
        adversarial image tensor in [0, 1]
    """
    image = image.clone().detach()

    with torch.no_grad():
        output = model(image)
    orig_label = label if label is not None else output.argmax(dim=1)
    orig_class = orig_label.item()

    # Candidate classes: highest-logit classes of the clean input
    k = min(num_classes, output.shape[1])
    candidates = output[0].topk(k).indices.tolist()
    if orig_class not in candidates:
        candidates.append(orig_class)

    pert_image = image.clone().detach()

    for loops in range(max_iter):
        pert_image = pert_image.detach().requires_grad_(True)
        output = model(pert_image)
        logits = output[0]

        if callback is not None:
            callback(loops, F.cross_entropy(output, orig_label).item(),
                     torch.softmax(output, dim=1).max().item())

        if logits.argmax().item() != orig_class:
            break

        grad_orig = torch.autograd.grad(logits[orig_class], pert_image,
                                        retain_graph=True)[0]

        min_dist = float('inf')
        best_w = None
        for c in candidates:
            if c == orig_class:
                continue
            grad_c = torch.autograd.grad(logits[c], pert_image,
                                         retain_graph=True)[0]
            w_c = grad_c - grad_orig
            f_c = (logits[c] - logits[orig_class]).item()
            norm_w = w_c.flatten().norm().item() + 1e-8
            dist = abs(f_c) / norm_w
            if dist < min_dist:
                min_dist = dist
                best_w = w_c / norm_w

        if best_w is None:
            break

        r_i = (min_dist + 1e-4) * best_w
        pert_image = torch.clamp(pert_image + (1 + overshoot) * r_i, 0, 1).detach()

    pert_image = pert_image.detach()

    if callback is not None:
        with torch.no_grad():
            output = model(pert_image)
            callback(max_iter, F.cross_entropy(output, orig_label).item(),
                     torch.softmax(output, dim=1).max().item())

    return pert_image
