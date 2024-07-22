import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import segmentation_models_pytorch as smp
from utils import fast_reconstruct_from_patches

# ------------------ Load classes and create transform ------------------

classes = pd.read_excel('dataset/ColorPalette-Values.xlsx', usecols='G:H', skiprows=8)
classes.columns = ['name', 'color']
classes[['R', 'G', 'B']] = classes['color'].str.split(expand=True)
classes['R'] = classes['R'].str.extract(r'(\d+)')
classes['G'] = classes['G'].str.extract(r'(\d+)')
classes['B'] = classes['B'].str.extract(r'(\d+)')
classes[['R', 'G', 'B']] = classes[['R', 'G', 'B']].apply(pd.to_numeric)
classes.drop('color', axis=1, inplace=True)


transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ------------------ Helper functions ------------------

def create_visualizations(image, reconstructed_pred_mask, prob_maps):
    colors = classes[['R', 'G', 'B']].values / 255.0
    height, width = image.shape[:2]
    reconstructed_pred_mask = cv2.resize(reconstructed_pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # --- Overlay ---
    overlay = image.copy()
    for i in range(len(classes)):
        mask = reconstructed_pred_mask == i
        overlay[mask] = overlay[mask] * 0.5 + np.array(colors[i] * 255) * 0.5
    
    # --- Max probability map ---
    max_prob_map = np.max(prob_maps, axis=0)
    max_prob_map = cv2.resize(max_prob_map, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # --- Entropy map ---
    epsilon = 1e-10
    prob_maps_resized = [cv2.resize(prob_map, (width, height), interpolation=cv2.INTER_LINEAR) for prob_map in prob_maps]
    entropy_map = -np.sum(np.array(prob_maps_resized) * np.log(np.array(prob_maps_resized) + epsilon), axis=0)
    
    return overlay, max_prob_map, entropy_map

def process_image(image, model, device):
    orig_height, orig_width = image.shape[:2]
    img = cv2.resize(image, (4500, 3000))
    
    patches = []
    for i in range(2):
        for j in range(3):
            patch = img[i*1500:(i+1)*1500, j*1500:(j+1)*1500]
            patch_tensor = transform(patch).unsqueeze(0)  # add batch dimension
            patches.append(patch_tensor)
    
    # stack all patches into a single tensor
    patches = torch.cat(patches, dim=0).to(device)
    
    # --- Forward pass ---
    with torch.no_grad():
        pred_mask_patches = model(patches)
        pred_probs = torch.softmax(pred_mask_patches, dim=1)
        pred_mask_patches = torch.argmax(pred_mask_patches, dim=1)
    
    reconstructed_pred_mask = fast_reconstruct_from_patches(pred_mask_patches, 2, 3, 256, 256)
    prob_maps = [fast_reconstruct_from_patches(pred_probs[:, i], 2, 3, 256, 256) for i in range(len(classes))] # list of probability maps for each class
    
    reconstructed_pred_mask = cv2.resize(reconstructed_pred_mask.cpu().numpy(), (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    prob_maps = [cv2.resize(prob_map.cpu().numpy(), (orig_width, orig_height), interpolation=cv2.INTER_LINEAR) for prob_map in prob_maps]
    
    return reconstructed_pred_mask, prob_maps

def main():
    st.set_page_config(layout="wide")
    st.title("Flooding Visualization Tool")

    # File uploader in the sidebar
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Load and process the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load the model
        model = smp.create_model(arch='DeepLabV3Plus',
                    encoder_name='resnet34',
                    encoder_weights='imagenet',
                    classes=10,
                    in_channels=3)
        
        experiment_name = 'DeepLabV3Plus_resnet34'
        model.load_state_dict(torch.load(f'models/{experiment_name}_best.pth'))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        reconstructed_pred_mask, prob_maps = process_image(image, model, device)
        
        overlay, max_prob_map, entropy_map = create_visualizations(image, reconstructed_pred_mask, prob_maps)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
            st.subheader("Predicted Mask Overlay")
            st.image(overlay, use_column_width=True)

        with col2:
            st.subheader("Confidence Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(max_prob_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            
            st.subheader("Entropy Map")
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(entropy_map, cmap='viridis', interpolation='nearest')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

        st.sidebar.subheader("Class Legend")
        legend_cols = st.sidebar.columns(2)
        for i, (_, row) in enumerate(classes.iterrows()):
            legend_cols[i % 2].color_picker(row['name'], f"#{row['R']:02x}{row['G']:02x}{row['B']:02x}", disabled=True)

if __name__ == '__main__':
    main()
