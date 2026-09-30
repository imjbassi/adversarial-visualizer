import torch
import torch.nn.functional as F


def cw_attack(model, image, label=None, targeted=False, c=1.0, kappa=0,
              lr=0.01, max_iter=100, callback=None, **kwargs):
    """Carlini & Wagner L2 attack (Carlini & Wagner, 2017).

    Optimizes in tanh space so the adversarial image stays in [0, 1] without
    clipping. Tracks the best (lowest-L2 successful) adversarial found.

    Args:
        model: classifier taking [0, 1] pixel-space tensors (1, C, H, W)
        image: input tensor in [0, 1]
        label: true label (untargeted) or target label (targeted); inferred
            from the model's prediction if None
        targeted: whether to run a targeted attack
        c: trade-off constant between L2 distance and classification loss
        kappa: confidence margin
        lr: Adam learning rate
        max_iter: optimization steps
        callback: optional callable(iteration, loss, confidence)
    Returns:
        adversarial image tensor in [0, 1]
    """
    image = image.clone().detach()
    device = image.device

    if label is not None:
        target = label.item()
    else:
        with torch.no_grad():
            target = model(image).argmax(dim=1).item()

    # Initialize w so that tanh(w) reproduces the original image
    image_clamped = torch.clamp(image, 1e-3, 1 - 1e-3)
    w = torch.atanh(2 * image_clamped - 1).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = image.clone()
    best_l2 = float('inf')

    for i in range(max_iter):
        adv_image = (torch.tanh(w) + 1) / 2
        output = model(adv_image)

        target_score = output[0, target]
        mask = torch.ones(output.shape[1], dtype=torch.bool, device=device)
        mask[target] = False
        best_other = output[0, mask].max()

        if targeted:
            f = torch.clamp(best_other - target_score + kappa, min=0)
        else:
            f = torch.clamp(target_score - best_other + kappa, min=0)

        l2 = torch.sum((adv_image - image) ** 2)
        loss = l2 + c * f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            adv_eval = (torch.tanh(w) + 1) / 2
            eval_output = model(adv_eval)
            pred = eval_output.argmax(dim=1).item()
            success = (pred == target) if targeted else (pred != target)
            cur_l2 = torch.sum((adv_eval - image) ** 2).item()
            if success and cur_l2 < best_l2:
                best_l2 = cur_l2
                best_adv = adv_eval.clone()

            if callback is not None and (i % max(1, max_iter // 20) == 0
                                         or i == max_iter - 1):
                callback(i, loss.item(),
                         torch.softmax(eval_output, dim=1).max().item())

    if best_l2 == float('inf'):
        # No successful adversarial found; return the final iterate
        with torch.no_grad():
            best_adv = (torch.tanh(w) + 1) / 2
    return best_adv.detach()
