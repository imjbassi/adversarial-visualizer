import torch

def fgsm_attack(model, image, label, epsilon, random_start=True):
    if random_start:
        image = image + torch.empty_like(image).uniform_(-epsilon, epsilon)
        image = torch.clamp(image, 0, 1)
    
    image.requires_grad = True
    output = model(image)
    loss = torch.nn.functional.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    perturbed = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed, 0, 1)
