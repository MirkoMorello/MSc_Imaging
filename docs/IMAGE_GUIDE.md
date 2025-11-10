# Image Guide for README

This guide explains what images should be added to the README and where to place them.

## Directory Structure

All images should be placed in the `docs/images/` directory:

```
docs/
└── images/
    ├── banner.png
    ├── dataset_samples.png
    ├── architecture_diagram.png
    ├── gui_screenshot.png
    ├── patch_size_comparison.png
    ├── training_metrics.png
    └── prediction_visualization.png
```

## Required Images and How to Create Them

### 1. `banner.png`
**Location**: `docs/images/banner.png`

**Description**: A banner image for the README header showing the project's purpose

**How to create**:
- Take a sample aerial image from your dataset
- Add the predicted segmentation overlay
- Add text: "Flood Detection using Deep Learning"
- Recommended size: 1600×400 pixels
- Can be created using the visualization from cell 18 in main.ipynb

**From notebook**: Use the output from `visualize_predictions_with_heatmap()` and crop/resize appropriately


### 2. `dataset_samples.png`
**Location**: `docs/images/dataset_samples.png`

**Description**: Shows example images with ground truth annotations

**How to create**:
- Run cell 8 in main.ipynb: `visualize_single_image_with_mask()`
- Take screenshots of 2-3 different examples
- Arrange them in a grid (2×2 or 1×3 layout)
- Show diversity in scenes (urban, suburban, different flood levels)
- Recommended size: 1600×800 pixels

**Code to run**:
```python
# In main.ipynb
for i in [0, 10, 20]:  # Different samples
    visualize_single_image_with_mask(train_df, path_img_train, path_mask_train, classes, rnd=i)
    # Save each figure
```


### 3. `architecture_diagram.png`
**Location**: `docs/images/architecture_diagram.png`

**Description**: Diagram showing the DeepLabV3Plus architecture and patch-based processing

**How to create**:
- Option 1: Use a diagramming tool (draw.io, Lucidchart, PowerPoint)
- Option 2: Use torchviz to visualize the model
- Option 3: Create manually showing:
  - Input: Aerial image (4500×3000)
  - Patch extraction: 6 patches of 1500×1500
  - Resize to 256×256
  - Model: DeepLabV3Plus (ResNet34 + ASPP)
  - Output: Segmentation maps
  - Reconstruction: Stitch back to full resolution

**Code for torchviz**:
```python
from torchviz import make_dot
dummy_input = torch.randn(1, 3, 256, 256).to(device)
output = model(dummy_input)
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("architecture_diagram", format="png")
```


### 4. `gui_screenshot.png`
**Location**: `docs/images/gui_screenshot.png`

**Description**: Screenshot of the Streamlit GUI in action

**How to create**:
- Run the Streamlit app: `streamlit run gui.py`
- Upload a test image
- Take a full-screen screenshot showing:
  - Uploaded image
  - Predicted segmentation overlay
  - Confidence heatmap
  - Entropy map
  - Class legend on the side
- Recommended size: 1920×1080 pixels

**Steps**:
1. `cd Final_Project`
2. `streamlit run gui.py`
3. Upload an image from `dataset/test/test-org-img/`
4. Take screenshot
5. Crop and save as PNG


### 5. `patch_size_comparison.png`
**Location**: `docs/images/patch_size_comparison.png`

**Description**: Comparison of validation metrics for different patch sizes

**How to create**:
- Run cell 17 in main.ipynb
- This creates a 2×2 grid of plots comparing:
  - Loss
  - Accuracy
  - IoU
  - Dice Score
- Save the matplotlib figure

**Code to save**:
```python
# At the end of cell 17, add:
plt.savefig('docs/images/patch_size_comparison.png', dpi=150, bbox_inches='tight')
```


### 6. `training_metrics.png`
**Location**: `docs/images/training_metrics.png`

**Description**: Training and validation metrics over epochs

**How to create**:
- Run cell 16 in main.ipynb
- Save the output figure showing 4 subplots:
  - Training/validation loss
  - Training/validation accuracy
  - Training/validation IoU
  - Training/validation Dice

**Code to save**:
```python
# Modify cell 16 to save the figure:
plot_metrics_from_pickle(f'models/pickles/{experiment_name}_history.pkl', experiment_name)
plt.savefig('docs/images/training_metrics.png', dpi=150, bbox_inches='tight')
```


### 7. `prediction_visualization.png`
**Location**: `docs/images/prediction_visualization.png`

**Description**: Multi-panel visualization showing all prediction aspects

**How to create**:
- Run cell 18 in main.ipynb: `visualize_predictions_with_heatmap()`
- Save the full figure showing 6 panels:
  1. Original image
  2. Ground truth mask
  3. Predicted mask
  4. Confidence heatmap
  5. Entropy map
  6. Prediction overlay
- Recommended size: 1800×1200 pixels

**Code to save**:
```python
# Add at the end of visualize_predictions_with_heatmap():
plt.savefig('docs/images/prediction_visualization.png', dpi=150, bbox_inches='tight')
```


## Quick Script to Generate All Images

Create a file `generate_readme_images.py` in `Final_Project/`:

```python
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load model and data
# ... (your existing code)

# 1. Dataset samples
for i, sample_idx in enumerate([0, 10, 20]):
    plt.figure(figsize=(8, 8))
    visualize_single_image_with_mask(train_df, path_img_train, path_mask_train, classes, rnd=sample_idx)
    plt.savefig(f'../docs/images/dataset_sample_{i}.png', dpi=150, bbox_inches='tight')
    plt.close()

# 2. Training metrics
plot_metrics_from_pickle(f'models/pickles/{experiment_name}_history.pkl', experiment_name)
plt.savefig('../docs/images/training_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Patch size comparison
# ... (code from cell 17)
plt.savefig('../docs/images/patch_size_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Prediction visualization
visualize_predictions_with_heatmap(model, test_dataset, 31, device, classes)
plt.savefig('../docs/images/prediction_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

print("All images generated successfully!")
```

Then run:
```bash
cd Final_Project
python generate_readme_images.py
```


## Image Specifications

### General Guidelines
- **Format**: PNG (for screenshots and plots) or JPEG (for photos)
- **Resolution**: At least 150 DPI
- **Width**: 1600-1920 pixels for full-width images
- **Compression**: Optimize images to keep file sizes reasonable (<500KB each)
- **Color**: RGB color space

### Tools for Image Editing
- **Screenshots**: Native OS tools (Cmd+Shift+4 on Mac, Snipping Tool on Windows)
- **Plots**: Matplotlib with `dpi=150` and `bbox_inches='tight'`
- **Diagrams**: draw.io, Lucidchart, Microsoft PowerPoint, Adobe Illustrator
- **Optimization**: TinyPNG, ImageOptim, or `imagemagick`

### Image Optimization
After creating images, optimize them:

```bash
# Install imagemagick
sudo apt-get install imagemagick  # Linux
brew install imagemagick          # Mac

# Optimize all images
cd docs/images
for img in *.png; do
    convert "$img" -resize 1600x -quality 85 "${img%.png}_optimized.png"
    mv "${img%.png}_optimized.png" "$img"
done
```


## Alternative: Using Placeholder Images

If you want to publish the README before creating all images, you can:

1. Use placeholder images from https://via.placeholder.com/
2. Replace image tags with descriptive text
3. Add a note at the top: "📸 *Images will be added soon*"

Example:
```markdown
<!-- Temporary placeholder -->
<p align="center">
  <strong>[Image placeholder: Aerial image with flood segmentation overlay]</strong>
</p>
```


## Checklist

Before publishing:
- [ ] Create `docs/images/` directory
- [ ] Generate `banner.png`
- [ ] Generate `dataset_samples.png`
- [ ] Generate `architecture_diagram.png`
- [ ] Generate `gui_screenshot.png`
- [ ] Generate `patch_size_comparison.png`
- [ ] Generate `training_metrics.png`
- [ ] Generate `prediction_visualization.png`
- [ ] Optimize all images for web
- [ ] Verify all images display correctly in README
- [ ] Push images to repository


## Tips

1. **High-Quality Figures**: Use `plt.savefig()` with `dpi=150` or higher
2. **Consistent Style**: Use the same color scheme and fonts across all images
3. **Clear Labels**: Ensure all plots have readable axis labels and titles
4. **Annotations**: Add arrows or text to highlight important features
5. **Backgrounds**: Use white or transparent backgrounds for diagrams
6. **File Size**: Keep images under 500KB each for faster loading


## Questions?

If you need help creating any of these images:
1. Check the corresponding notebook cells
2. Review matplotlib documentation
3. Use the code snippets provided above
4. Consider using online diagramming tools for architecture diagrams
