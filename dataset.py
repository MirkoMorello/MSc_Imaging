import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90, RandomResizedCrop, RandomBrightnessContrast, ShiftScaleRotate, Resize
from torchvision import transforms
from torch.nn import functional as F




class ImagePatchDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, patch_size=(512, 512), overlap=0, transform=None, use_torch=True, stitch_levels=0):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.patch_size = tuple(patch_size)
        self.overlap = overlap
        self.transform = transform
        self.use_torch = use_torch
        self.stitch_levels = stitch_levels
        self.image_files = df['name'].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_name}_lab.png")

        img, mask = self.load_image_and_mask(img_path, mask_path)
        img_patches, mask_patches = self.generate_patches(img, mask)

        if self.transform:
            img_patches, mask_patches = self.apply_transform(img_patches, mask_patches)

        return img_patches, mask_patches

    def load_image_and_mask(self, img_path, mask_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (4500, 3000))
        mask = cv2.resize(mask, (4500, 3000), interpolation=cv2.INTER_NEAREST)
        
        return img, mask

    def generate_patches(self, img, mask):
        if self.use_torch:
            return self.create_patches_torch(img, mask)
        else:
            return self.create_patches_numpy(img, mask)

    def create_patches_torch(self, img, mask):
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        all_img_patches = []
        all_mask_patches = []

        for level in range(self.stitch_levels + 1):
            scale_factor = 2 ** level
            current_patch_size = (self.patch_size[0] * scale_factor, self.patch_size[1] * scale_factor)
            
            img_patches = img_tensor.unfold(2, current_patch_size[0], current_patch_size[0] - self.overlap)
            img_patches = img_patches.unfold(3, current_patch_size[1], current_patch_size[1] - self.overlap)
            img_patches = img_patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, current_patch_size[0], current_patch_size[1])

            mask_patches = mask_tensor.unfold(2, current_patch_size[0], current_patch_size[0] - self.overlap)
            mask_patches = mask_patches.unfold(3, current_patch_size[1], current_patch_size[1] - self.overlap)
            mask_patches = mask_patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, current_patch_size[0], current_patch_size[1])

            if level > 0:
                img_patches = F.interpolate(img_patches, size=self.patch_size, mode='bilinear', align_corners=False)
                mask_patches = F.interpolate(mask_patches.float(), size=self.patch_size, mode='nearest').long()

            all_img_patches.append(img_patches)
            all_mask_patches.append(mask_patches)

        img_patches = torch.cat(all_img_patches, dim=0)
        mask_patches = torch.cat(all_mask_patches, dim=0).long() 

        return img_patches, mask_patches

    def create_patches_numpy(self, img, mask):
        all_img_patches = []
        all_mask_patches = []

        for level in range(self.stitch_levels + 1):
            scale_factor = 2 ** level
            current_patch_size = (self.patch_size[0] * scale_factor, self.patch_size[1] * scale_factor)
            
            stride_height = current_patch_size[0] - self.overlap
            stride_width = current_patch_size[1] - self.overlap

            img_patches = self.extract_patches(img, current_patch_size[0], current_patch_size[1], stride_height, stride_width)
            mask_patches = self.extract_patches(mask, current_patch_size[0], current_patch_size[1], stride_height, stride_width, is_mask=True)

            if level > 0:
                img_patches = np.array([cv2.resize(patch, self.patch_size[::-1]) for patch in img_patches])
                mask_patches = np.array([cv2.resize(patch, self.patch_size[::-1], interpolation=cv2.INTER_NEAREST) for patch in mask_patches])

            all_img_patches.append(img_patches)
            all_mask_patches.append(mask_patches)

        img_patches = np.concatenate(all_img_patches, axis=0)
        mask_patches = np.concatenate(all_mask_patches, axis=0)

        return img_patches, mask_patches.astype(np.int64)

    def extract_patches(self, img, patch_height, patch_width, stride_height, stride_width, is_mask=False):
        height, width = img.shape[:2]
        pad_height = (patch_height - height % stride_height) % patch_height
        pad_width = (patch_width - width % stride_width) % patch_width

        if is_mask:
            img_padded = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        else:
            img_padded = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

        patches = []
        for y in range(0, height + pad_height - patch_height + 1, stride_height):
            for x in range(0, width + pad_width - patch_width + 1, stride_width):
                if is_mask:
                    patch = img_padded[y:y+patch_height, x:x+patch_width]
                else:
                    patch = img_padded[y:y+patch_height, x:x+patch_width, :]
                patches.append(patch)

        return np.array(patches)

    def apply_transform(self, img_patches, mask_patches):
        transformed_img_patches = []
        transformed_mask_patches = []

        for img_patch, mask_patch in zip(img_patches, mask_patches):
            if torch.is_tensor(img_patch):
                img_patch = img_patch.permute(1, 2, 0).numpy()
            if torch.is_tensor(mask_patch):
                mask_patch = mask_patch.squeeze().numpy()

            # apply the transformation
            transformed = self.transform(image=img_patch, mask=mask_patch)
            transformed_img = transformed['image']
            transformed_mask = transformed['mask']

            # add to list (no need to convert back to tensor, as ToTensorV2 does this)
            transformed_img_patches.append(transformed_img)
            transformed_mask_patches.append(transformed_mask)

        # final stack
        transformed_img_patches = torch.stack(transformed_img_patches)
        transformed_mask_patches = torch.stack(transformed_mask_patches).long()

        return transformed_img_patches, transformed_mask_patches

    def get_original_image(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img