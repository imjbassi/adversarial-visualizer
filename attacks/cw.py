import torch
import torch.nn.functional as F

def cw_attack(model, image, label, targeted=False, c=1e-2, kappa=0, lr=0.01, max_iter=100):
    """
    Minimal Carlini-Wagner L2 attack implementation for image classification models.
    Args:
        model: PyTorch model
        image: input tensor (1, C, H, W)
        label: true label tensor (1,)
        targeted: whether to perform a targeted attack
        c: regularization constant
        kappa: confidence parameter
        lr: learning rate
        max_iter: max iterations
    Returns:
        perturbed image tensor
    """
    device = image.device
    image = image.clone().detach()
    
    # Avoid extreme values that cause problems with atanh
    image_clamp = torch.clamp(image, 0.001, 0.999)
    
    # Initialize w with a safe value
    w = torch.atanh(2 * image_clamp - 1)
    w = w.clone().detach().requires_grad_(True)
    
    # Use a smaller learning rate for stability
    optimizer = torch.optim.Adam([w], lr=min(lr, 0.005))
    
    if label is not None:
        target = label.item() 
    else:
        with torch.no_grad():
            target = model(image).argmax().item()
    
    for i in range(max_iter):
        # Convert w to image space with bounds
        adv_image = torch.tanh(w) * 0.5 + 0.5
        adv_image = torch.clamp(adv_image, 0, 1)
        
        output = model(adv_image)
        
        # Handle out-of-bounds target
        if target >= output.size(1):
            target = output.argmax(dim=1).item()
        
        # Compute the attack objective safely
        target_score = output[0, target]
        
        # Create a mask for all classes except the target
        mask = torch.ones(output.size(1), dtype=torch.bool, device=device)
        mask[target] = False
        other_scores = output[0, mask]
        
        if targeted:
            if len(other_scores) > 0:
                best_other_score = torch.max(other_scores)
                f = torch.clamp(best_other_score - target_score + kappa, min=0)
            else:
                f = torch.tensor(0.0, device=device)
        else:
            if len(other_scores) > 0:
                best_other_score = torch.max(other_scores)
                f = torch.clamp(target_score - best_other_score + kappa, min=0)
            else:
                f = torch.tensor(0.0, device=device)
        
        # Calculate the L2 distance
        l2 = torch.sum((adv_image - image) ** 2)
        
        # Total loss
        loss = l2 + c * f
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check for NaN in gradients and fix
        if torch.isnan(w.grad).any() or torch.isinf(w.grad).any():
            w.grad[torch.isnan(w.grad) | torch.isinf(w.grad)] = 0.0
        
        optimizer.step()
        
        # Clamp w to prevent extreme values
        with torch.no_grad():
            w.data = torch.clamp(w.data, -10, 10)
        
        # Check for successful attack
        if i % 10 == 0:
            with torch.no_grad():
                current_output = model(adv_image)
                current_pred = current_output.argmax(dim=1).item()
                if (targeted and current_pred == target) or (not targeted and current_pred != target):
                    if not torch.isnan(current_output).any():
                        break
    
    with torch.no_grad():
        adv_image = torch.tanh(w) * 0.5 + 0.5
        adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image.detach()
