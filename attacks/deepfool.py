import torch
import torch.nn.functional as F

def deepfool_attack(model, image, label, epsilon=None, random_start=None, num_classes=10, overshoot=0.02, max_iter=50):
    """
    Minimal DeepFool implementation for image classification models.
    Args:
        model: PyTorch model
        image: input tensor (1, C, H, W)
        label: true label tensor (1,)
        num_classes: number of classes to consider
        overshoot: final perturbation scaling
        max_iter: max iterations
    Returns:
        perturbed image tensor
    """
    image = image.clone().detach().requires_grad_(True)
    pert_image = image.clone().detach()
    output = model(image)
    _, orig_label = output.max(1)
    if label is not None:
        orig_label = label
    
    # Limit num_classes if needed
    num_classes = min(num_classes, output.shape[1])
    
    loops = 0
    while loops < max_iter:
        output = model(pert_image)
        logits = output[0]
        orig_class = orig_label.item()
        pert_image.requires_grad = True
        
        # Get gradient for original class
        try:
            grad_orig = torch.autograd.grad(logits[orig_class], pert_image, retain_graph=True, allow_unused=True)[0]
            if grad_orig is None:
                grad_orig = torch.zeros_like(pert_image)
        except Exception:
            grad_orig = torch.zeros_like(pert_image)
        
        min_dist = float('inf')
        w = None
        valid_grad_found = False
        
        for k in range(num_classes):
            if k == orig_class:
                continue
                
            try:
                grad_k = torch.autograd.grad(logits[k], pert_image, retain_graph=True, allow_unused=True)[0]
                if grad_k is None:
                    continue
                    
                w_k = grad_k - grad_orig
                norm_w_k = torch.norm(w_k.flatten()) + 1e-8
                
                f_k = (logits[k] - logits[orig_class]).item()
                if abs(f_k) < 1e-8:
                    continue
                    
                dist = abs(f_k) / norm_w_k
                
                if dist < min_dist:
                    min_dist = dist
                    w = w_k
                    valid_grad_found = True
            except Exception:
                continue
        
        if not valid_grad_found or w is None:
            noise = torch.randn_like(pert_image) * 0.01
            pert_image = torch.clamp(pert_image + noise, 0, 1).detach().requires_grad_(True)
            loops += 1
            continue
            
        r_i = min_dist * w / (torch.norm(w.flatten()) + 1e-8)
        pert_image = pert_image + (1 + overshoot) * r_i
        pert_image = torch.clamp(pert_image, 0, 1).detach().requires_grad_(True)
        
        with torch.no_grad():
            new_output = model(pert_image)
            new_label = new_output.max(1)[1].item()
            if new_label != orig_class:
                break
                
        loops += 1
    
    return pert_image.detach()
