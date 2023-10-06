import torch
import torchvision

from torch import nn


def create_vit_model(num_classes:int=101, 
                          seed:int=42):
    """Creates an ViT feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): ViT feature extractor model. 
        transforms (torchvision.transforms): ViT image transforms.
    """
    
    # Instantiate model to load saved state dict()
    
    # Create ViT pretrained weights, transforms and model
    vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT # 'DEFAULT' = best available
    transforms = vit_weights.transforms()
    model = torchvision.models.vit_b_16(weights = vit_weights)

    # Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.heads = nn.Linear(in_features = 768,
                        out_features = 101)
    
    return model, transforms
