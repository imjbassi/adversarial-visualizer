import torch

def pgd_attack(model, image, label, epsilon=0.03, alpha=0.01, iters=40, momentum=0.9):
    ori_image = image.clone().detach()
    grad_accum = torch.zeros_like(image)  # Accumulate gradients
    
    for _ in range(iters):
        image.requires_grad = True
        output = model(image)
        loss = torch.nn.functional.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        
        grad = image.grad.sign()
        grad_accum = momentum * grad_accum + grad  # Apply momentum
        adv_image = image + alpha * grad_accum.sign()
        
        eta = torch.clamp(adv_image - ori_image, -epsilon, epsilon)
        image = torch.clamp(ori_image + eta, 0, 1).detach()
    
    return image
