import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle




class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')

    def forward(self, inputs, targets):
        logpt = self.ce_fn(inputs, targets)
        pt = torch.exp(-logpt)
        loss = self.alpha * (1 - pt) ** self.gamma * logpt
        return loss.mean()
    


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


def fast_reconstruct_from_patches(patches, num_patches_height, num_patches_width, patch_height, patch_width):

    if patches.dim() == 4:
        channels = patches.shape[1]
        patches = patches.view(num_patches_height, num_patches_width, channels, patch_height, patch_width)
    else:
        channels = 1
        patches = patches.view(num_patches_height, num_patches_width, 1, patch_height, patch_width)
    
    output_height = num_patches_height * patch_height
    output_width = num_patches_width * patch_width
    
    reconstructed = patches.permute(0, 3, 1, 4, 2).contiguous()
    reconstructed = reconstructed.view(output_height, output_width, channels)
    
    if channels == 1:
        reconstructed = reconstructed.squeeze(-1)
    
    return reconstructed

def convert_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return np.array([convert_to_numpy(t) for t in tensor])
    else:
        return np.array(tensor)
    


def safe_pickle_dump(obj, filename):
    temp_filename = filename + '.temp'
    with open(temp_filename, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(temp_filename, filename)


def safe_pickle_load(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        print(f"Error loading {filename}. Starting with empty history.")
        return {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_iou': [], 'val_iou': [], 'train_dice': [], 'val_dice': []}