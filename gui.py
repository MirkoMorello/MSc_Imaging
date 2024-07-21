import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import io
import segmentation_models_pytorch as smp
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
# import functionals of torch.nn as functional
import torch.nn.functional as F
from plotly import express as px
from plotly import graph_objects as go

def extract_patches(image, patch_size, overlap):
    h, w = image.shape[:2]
    stride = patch_size - overlap
    patches = []
    locations = []
    for y in range(0, h-patch_size+1, stride):
        for x in range(0, w-patch_size+1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            locations.append((y, x))
    return np.array(patches), locations

def stitch_patches(patches, locations, original_size):
    h, w = original_size[:2]
    if len(patches.shape) == 4:
        c = patches.shape[3]
        stitched = np.zeros((h, w, c), dtype=np.float32)
    else:
        stitched = np.zeros((h, w), dtype=np.float32)
    
    counts = np.zeros((h, w), dtype=np.float32)
    
    for patch, (y, x) in zip(patches, locations):
        patch_h, patch_w = patch.shape[:2]
        stitched[y:y+patch_h, x:x+patch_w] += patch
        counts[y:y+patch_h, x:x+patch_w] += 1
    
    # Avoid division by zero
    mask = counts > 0
    if len(stitched.shape) == 3:
        stitched[mask] /= counts[mask, np.newaxis]
    else:
        stitched[mask] /= counts[mask]
    
    return stitched

from utils import fast_reconstruct_from_patches

def perform_segmentation(image, model, device, patch_size=1500, overlap=0, batch_size=24):
    # Resize image to 3000x4500
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (3000, 4500))
    
    h, w = img_array.shape[:2]
    
    # Create patches
    transform = A.Compose([
        A.Resize(256, 256),  # Resize all patches to 256x256
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    all_patches = []
    
    stride = patch_size - overlap
    num_patches_height = (h - patch_size + stride) // stride
    num_patches_width = (w - patch_size + stride) // stride
    
    print(f"Image shape: {img_array.shape}")
    print(f"Number of patches: {num_patches_height}x{num_patches_width}")
    
    for y in range(0, num_patches_height * stride, stride):
        for x in range(0, num_patches_width * stride, stride):
            patch = img_array[y:y+patch_size, x:x+patch_size]
            transformed = transform(image=patch)
            all_patches.append(transformed['image'])
    
    all_patches = torch.stack(all_patches)
    print(f"Total patches: {len(all_patches)}")
    
    # Perform inference in batches
    model.eval()
    pred_patches = []
    with torch.no_grad():
        for i in range(0, len(all_patches), batch_size):
            batch = all_patches[i:i+batch_size].to(device)
            pred = model(batch)
            pred_probs = torch.softmax(pred, dim=1)
            pred_patches.append(pred_probs.cpu())
    
    pred_patches = torch.cat(pred_patches, dim=0)
    print(f"Predicted patches shape: {pred_patches.shape}")
    
    # Determine number of classes from the model output
    num_classes = pred_patches.shape[1]
    
    # Resize predictions back to patch_size
    resized_pred_patches = F.interpolate(pred_patches, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
    print(f"Resized predicted patches shape: {resized_pred_patches.shape}")
    
    # Reconstruct the full image prediction using fast_reconstruct_from_patches
    reconstructed_pred = fast_reconstruct_from_patches(resized_pred_patches, num_patches_height, num_patches_width, patch_size, patch_size)
    print(f"Reconstructed prediction shape: {reconstructed_pred.shape}")
    
    # Handle potential dimension mismatch
    if reconstructed_pred.shape[0] > h or reconstructed_pred.shape[1] > w:
        reconstructed_pred = reconstructed_pred[:h, :w]
    
    pred_mask = torch.argmax(reconstructed_pred, dim=2).numpy()
    stitched_probs = reconstructed_pred.numpy()
    
    print(f"Final pred_mask shape: {pred_mask.shape}")
    print(f"Final stitched_probs shape: {stitched_probs.shape}")
    
    return pred_mask, stitched_probs

def create_confidence_heatmap(pred_probs):
    return np.max(pred_probs, axis=2)

def create_entropy_map(pred_probs):
    epsilon = 1e-10
    entropy = -np.sum(pred_probs * np.log(pred_probs + epsilon), axis=2)
    entropy_normalized = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))
    return entropy_normalized

def create_overlay(original_img, pred_mask, colors):
    # Ensure pred_mask has the same shape as original_img
    if original_img.shape[:2] != pred_mask.shape:
        pred_mask = cv2.resize(pred_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    overlay = original_img.copy()
    for i in range(len(colors)):
        mask = pred_mask == i
        overlay[mask] = overlay[mask] * 0.5 + np.array(colors[i] * 255) * 0.5
    return overlay.astype(np.uint8)

def create_colored_mask(pred_mask, colors):
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        colored_mask[pred_mask == i] = (np.array(color) * 255).astype(np.uint8)
    return colored_mask

def apply_colormap(image, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    colored_image = cmap(image)
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

def add_colorbar(fig, cmap_name='viridis', title=''):
    cmap = plt.get_cmap(cmap_name)
    fig.add_trace(go.Heatmap(z=[[0, 1]], colorscale=cmap_name, showscale=True, visible=False))
    fig.data[-1].colorbar.title = title
    fig.data[-1].colorbar.titleside = 'right'
    fig.data[-1].colorbar.x = 1.05
    fig.data[-1].colorbar.thickness = 15


def main():
    st.set_page_config(layout="wide")
    st.title("Advanced Image Segmentation App")
    
    classes = pd.read_excel('dataset/ColorPalette-Values.xlsx', usecols='G:H', skiprows=8)
    classes.columns = ['name', 'color']
    classes[['R', 'G', 'B']] = classes['color'].str.split(expand=True)
    classes['R'] = classes['R'].str.extract(r'(\d+)')
    classes['G'] = classes['G'].str.extract(r'(\d+)')
    classes['B'] = classes['B'].str.extract(r'(\d+)')
    classes['R'] = pd.to_numeric(classes['R'])
    classes['G'] = pd.to_numeric(classes['G'])
    classes['B'] = pd.to_numeric(classes['B'])
    classes.drop('color', axis=1, inplace=True)
    
    model = smp.create_model(arch='DeepLabV3Plus',
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 classes=len(classes),
                 in_channels=3)
    
    model_name = 'DeepLabV3Plus_resnet34'
    model.load_state_dict(torch.load(f'models/{model_name}_best.pth'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        with col2:
            pred_mask, pred_probs = perform_segmentation(image, model, device)
            
            confidence_heatmap = create_confidence_heatmap(pred_probs)
            entropy_map = create_entropy_map(pred_probs)
            colors = classes[['R', 'G', 'B']].values / 255.0
            overlay = create_overlay(np.array(image), pred_mask, colors)
            
            # Apply colormaps
            confidence_heatmap_colored = apply_colormap(confidence_heatmap, 'viridis')
            entropy_map_colored = apply_colormap(entropy_map, 'hot')
            
            # Convert numpy arrays to PIL Images for proper resizing
            confidence_heatmap_img = Image.fromarray(confidence_heatmap_colored)
            entropy_map_img = Image.fromarray(entropy_map_colored)
            overlay_img = Image.fromarray(overlay)
            
            # Resize images to match original image size while maintaining aspect ratio
            original_size = image.size
            confidence_heatmap_resized = confidence_heatmap_img.resize(original_size, Image.LANCZOS)
            entropy_map_resized = entropy_map_img.resize(original_size, Image.LANCZOS)
            overlay_resized = overlay_img.resize(original_size, Image.LANCZOS)
            
            # Create two columns for heatmap and entropy map
            heatmap_col, entropy_col = st.columns(2)
            
            with heatmap_col:
                fig_heatmap = go.Figure(data=go.Image(z=confidence_heatmap_resized))
                add_colorbar(fig_heatmap, 'viridis', 'Confidence')
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.caption("Confidence Heatmap")
            
            with entropy_col:
                fig_entropy = go.Figure(data=go.Image(z=entropy_map_resized))
                add_colorbar(fig_entropy, 'hot', 'Entropy')
                st.plotly_chart(fig_entropy, use_container_width=True)
                st.caption("Entropy Map")
            
            # Display overlay below heatmap and entropy map
            st.image(overlay_resized, caption="Predicted Mask Overlay", use_column_width=True)
        
        st.subheader("Interactive Segmentation Map")
        fig = create_interactive_plot(np.array(overlay_resized), pred_mask, classes)
        st.plotly_chart(fig, use_container_width=True)

        
        st.subheader("Interactive Segmentation Map")
        fig = create_interactive_plot(np.array(overlay_resized), pred_mask, classes)
        st.plotly_chart(fig, use_container_width=True)

def create_interactive_plot(overlay, pred_mask, classes):
    fig = go.Figure()

    fig.add_trace(go.Image(z=overlay))

    fig.update_layout(
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # Ensure pred_mask has the same dimensions as overlay
    pred_mask_resized = cv2.resize(pred_mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)

    hover_text = np.empty(pred_mask_resized.shape, dtype=object)
    for i in range(pred_mask_resized.shape[0]):
        for j in range(pred_mask_resized.shape[1]):
            class_id = pred_mask_resized[i, j]
            class_name = classes['name'].iloc[class_id]
            color = classes.iloc[class_id][['R', 'G', 'B']].values
            hover_text[i, j] = f"Class: {class_name}<br>RGB: {color}"

    fig.data[0].customdata = hover_text
    fig.data[0].hovertemplate = "%{customdata}<extra></extra>"

    fig.update_layout(
        hoverdistance=-1,
        hovermode='closest',
    )

    return fig

if __name__ == "__main__":
    main()