# Flood Detection and Segmentation using Deep Learning

<p align="center">
  <img src="docs/images/banner.png" alt="Flood Detection Banner" width="800"/>
  <br>
  <em>Semantic segmentation of flooded areas in aerial imagery using deep learning</em>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Visualization](#visualization)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project implements a **semantic segmentation system** for identifying and classifying flooded areas in high-resolution aerial imagery. Using state-of-the-art deep learning models, the system can distinguish between flooded and non-flooded buildings, roads, vegetation, and other environmental features to support disaster response and flood risk assessment.

### Problem Statement

During flood disasters, rapid assessment of affected areas is crucial for emergency response. Manual analysis of aerial imagery is time-consuming and error-prone. This project automates the detection and segmentation of flooded regions using deep learning, providing:

- **Real-time flood extent mapping**
- **Classification of flooded vs. non-flooded infrastructure**
- **Quantitative assessment of affected areas**
- **Confidence and uncertainty visualization**

### Applications

- 🚨 Emergency response and disaster management
- 🏗️ Infrastructure damage assessment
- 📊 Flood risk mapping and analysis
- 🌍 Climate change impact studies
- 📈 Insurance claim verification

---

## Key Features

✨ **Multi-class Segmentation**: 10 distinct classes including flooded/non-flooded buildings, roads, water bodies, vegetation, and more

🎯 **High Accuracy**: Achieves **90.62% pixel accuracy** and **86.45% IoU** on test data

⚡ **Real-time Performance**: Processes images at **83.74 FPS** on GPU

🔍 **Uncertainty Quantification**: Provides confidence heatmaps and entropy maps for prediction uncertainty

🎨 **Interactive Visualization**: Streamlit-based GUI for easy model deployment and visualization

📊 **Comprehensive Metrics**: IoU, Dice score, and per-class performance metrics

🧩 **Patch-based Processing**: Handles large images (4500×3000 pixels) efficiently using patch extraction

---

## Dataset

### Dataset Description

The dataset consists of high-resolution aerial imagery with pixel-level annotations for flood detection tasks. Images are captured at **4500×3000 pixels** and contain diverse environmental features affected by flooding.

### Class Distribution

| Class ID | Class Name | RGB Color | Description |
|----------|------------|-----------|-------------|
| 0 | Background | (0, 0, 0) | Unlabeled or void regions |
| 1 | Building (Flooded) | (255, 0, 0) | Buildings submerged in water |
| 2 | Building (Non-flooded) | (180, 120, 120) | Buildings above water level |
| 3 | Road (Flooded) | (160, 150, 20) | Roads covered by water |
| 4 | Road (Non-flooded) | (140, 140, 140) | Accessible roads |
| 5 | Water | (61, 230, 250) | Natural water bodies |
| 6 | Tree | (0, 82, 255) | Vegetation and trees |
| 7 | Vehicle | (255, 0, 245) | Cars and vehicles |
| 8 | Pool | (255, 235, 0) | Swimming pools |
| 9 | Grass | (4, 250, 7) | Grass and lawns |

### Dataset Split

- **Training**: 1,445 images
- **Validation**: Dataset split
- **Testing**: 448 images

### Data Augmentation

The training pipeline includes extensive data augmentation:

- Random rotation (90°)
- Horizontal/vertical flipping
- Shift, scale, and rotation transformations
- Gaussian noise and blur
- Brightness/contrast adjustments
- HSV color space variations

<p align="center">
  <img src="docs/images/dataset_samples.png" alt="Dataset Samples" width="800"/>
  <br>
  <em>Example images with ground truth annotations</em>
</p>

---

## Architecture

### Model Selection

The project explores multiple architectures:

#### 1. **DeepLabV3Plus with ResNet34** (Primary Model)

- **Encoder**: ResNet34 pre-trained on ImageNet
- **Decoder**: ASPP (Atrous Spatial Pyramid Pooling) module
- **Parameters**: ~21M parameters
- **Input Size**: 256×256 patches
- **Best Performance**: 90.62% accuracy, 86.45% IoU

**Why DeepLabV3Plus?**
- Excellent performance on semantic segmentation tasks
- ASPP captures multi-scale contextual information
- Efficient for deployment with reasonable parameter count

#### 2. **Custom U-Net Architecture** (CustomNet)

A custom implementation featuring:
- 4 encoder-decoder blocks
- Skip connections for fine-grained details
- GELU activation functions
- Batch normalization

#### 3. **Custom Segmentation Model with Attention**

Advanced custom architecture with:
- **Attention blocks** for feature refinement
- **ASPP module** at the bottleneck
- Multi-scale feature extraction
- Lighter weight for faster inference

<p align="center">
  <img src="docs/images/architecture_diagram.png" alt="Architecture Diagram" width="800"/>
  <br>
  <em>DeepLabV3Plus architecture with patch-based processing</em>
</p>

### Training Strategy

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: AdamW with fused operations
- **Learning Rate**: 1e-4 with Cosine Annealing
- **Gradient Accumulation**: 2 steps
- **Mixed Precision Training**: FP16 for faster training
- **Early Stopping**: Patience of 14 epochs
- **Batch Size**: 8 for training, 1 for validation/testing

---

## Project Structure

```
MSc_Imaging/
├── Final_Project/
│   ├── main.ipynb              # Main training and visualization notebook
│   ├── dataset.py              # Custom Dataset class with patch extraction
│   ├── models.py               # Custom model architectures
│   ├── utils.py                # Helper functions (losses, reconstruction, etc.)
│   ├── gui.py                  # Streamlit GUI application
│   └── models/                 # Saved model checkpoints
│       ├── DeepLabV3Plus_resnet34_best.pth
│       └── pickles/            # Training history
├── Lessons_notes/              # Course materials and notes
├── dataset/                    # Dataset directory
│   ├── train/
│   │   ├── train-org-img/     # Training images
│   │   └── train-label-img/   # Training masks
│   ├── val/
│   │   ├── val-org-img/       # Validation images
│   │   └── val-label-img/     # Validation masks
│   └── test/
│       ├── test-org-img/      # Test images
│       └── test-label-img/    # Test masks
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install streamlit
pip install openpyxl
pip install tqdm
pip install kaggle
```

### Dataset Setup

1. Download the dataset (instructions specific to your dataset source)
2. Organize into the structure shown above
3. Ensure `ColorPalette-Values.xlsx` is in the `dataset/` directory

---

## Usage

### 1. Training

Open and run `Final_Project/main.ipynb` to train models:

```python
# Configure model
model = smp.create_model(
    arch='DeepLabV3Plus',
    encoder_name='resnet34',
    encoder_weights='imagenet',
    classes=10,
    in_channels=3
)

# Train
history = train_model(
    epochs=100,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    experiment_name='DeepLabV3Plus_resnet34',
    accumulation_steps=2
)
```

### 2. Evaluation

```python
# Test the model
test_results = test_model(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device=device,
    classes_df=classes
)
```

### 3. Visualization

```python
# Visualize predictions
visualize_predictions_with_heatmap(
    model=model,
    dataset=test_dataset,
    idx=31,
    device=device,
    classes=classes
)
```

### 4. Interactive GUI

Launch the Streamlit application:

```bash
cd Final_Project
streamlit run gui.py
```

The GUI provides:
- Image upload functionality
- Real-time flood segmentation
- Confidence heatmaps
- Entropy-based uncertainty visualization
- Color-coded class legend

<p align="center">
  <img src="docs/images/gui_screenshot.png" alt="GUI Screenshot" width="800"/>
  <br>
  <em>Streamlit GUI for flood detection visualization</em>
</p>

---

## Results

### Overall Performance (Test Set)

| Metric | Value |
|--------|-------|
| **Test Loss** | 0.2885 |
| **Pixel Accuracy** | 90.62% |
| **Mean IoU** | 86.45% |
| **Mean Dice Score** | 91.33% |
| **Inference Speed** | 83.74 FPS |
| **Avg. Time per Image** | 0.0119 seconds |

### Per-Class IoU Performance

| Class | IoU Score |
|-------|-----------|
| Background | 40.85% |
| Building (Flooded) | 69.01% |
| Building (Non-flooded) | 76.88% |
| Road (Flooded) | 58.00% |
| Road (Non-flooded) | 82.00% |
| Water | 75.55% |
| Tree | 82.61% |
| Vehicle | 36.18% |
| Pool | 50.16% |
| Grass | 88.78% |

### Key Insights

- **Strong performance** on large, clearly defined classes (grass, trees, non-flooded roads)
- **Challenging classes**: vehicles (small objects) and background (ambiguous regions)
- **Flooded regions**: Good detection with room for improvement (58-69% IoU)
- **Real-time capable**: 83+ FPS enables live video stream processing

### Model Comparison

Experiments with different patch sizes:

<p align="center">
  <img src="docs/images/patch_size_comparison.png" alt="Patch Size Comparison" width="800"/>
  <br>
  <em>Validation metrics for different patch sizes (750×750, 1500×1500, full image)</em>
</p>

**Findings**:
- 1500×1500 patches provide best balance between context and memory efficiency
- Full images overfit more quickly
- Smaller patches lose important spatial context

---

## Technical Details

### Patch-Based Processing

Large aerial images (4500×3000) are processed using a patch-based approach:

1. **Patch Extraction**: Images divided into 256×256 patches (6 patches per image)
2. **Overlap Handling**: Configurable overlap to reduce boundary artifacts
3. **Multi-scale Support**: Optional stitch levels for hierarchical feature extraction
4. **Efficient Reconstruction**: GPU-accelerated patch reassembly

```python
# Patch extraction configuration
patch_size = (1500, 1500)  # Original patch size
overlap = 0                 # No overlap
resize_to = (256, 256)     # Model input size
```

### Advanced Features

#### 1. **Gradient Accumulation**

Simulates larger batch sizes on limited GPU memory:
- Accumulation steps: 2
- Effective batch size: 16

#### 2. **Mixed Precision Training**

Uses PyTorch AMP (Automatic Mixed Precision):
- Faster training (up to 2-3x speedup)
- Reduced memory usage
- Maintained model accuracy

#### 3. **Batch Normalization Accumulation**

Custom BN statistics accumulation for gradient accumulation:
```python
class AccumulateBN:
    def update(self, module):
        # Accumulates running mean/var across micro-batches
        ...
```

#### 4. **Confidence and Uncertainty Estimation**

- **Confidence Heatmap**: Max probability across classes
- **Entropy Map**: Predictive uncertainty visualization
- **Error Visualization**: Pixel-level prediction errors with class-specific highlighting

<p align="center">
  <img src="docs/images/prediction_visualization.png" alt="Prediction Visualization" width="800"/>
  <br>
  <em>From left to right: Original image, ground truth, prediction, confidence heatmap, entropy map, overlay</em>
</p>

### Memory Optimization

- **Non-blocking transfers**: Asynchronous CPU-GPU data movement
- **Pin memory**: Faster host-to-device transfers
- **Prefetch factor**: 2 for DataLoader
- **Number of workers**: 8 for parallel data loading

---

## Visualization

The project includes comprehensive visualization tools:

### 1. **Dataset Visualization**

- Original images with mask overlays
- Patch extraction visualization
- Reconstructed images with border highlighting

### 2. **Training Progress**

Training metrics plotted in real-time:
- Loss curves (train/val)
- Accuracy trends
- IoU score progression
- Dice score evolution

<p align="center">
  <img src="docs/images/training_metrics.png" alt="Training Metrics" width="800"/>
  <br>
  <em>Training and validation metrics over epochs</em>
</p>

### 3. **Prediction Analysis**

Multi-panel visualization:
- **Original Image**: Input aerial imagery
- **Ground Truth**: Pixel-level annotations
- **Prediction**: Model output
- **Confidence Heatmap**: Prediction confidence (0-1)
- **Entropy Map**: Uncertainty estimation
- **Overlay**: Predictions overlaid on original image
- **Error Visualization**: Highlighting misclassified pixels

### 4. **Interactive GUI**

Streamlit application features:
- Drag-and-drop image upload
- Real-time inference
- Multiple visualization modes
- Color-coded class legend
- High-resolution output

---

## Future Work

### Short-term Improvements

- [ ] Implement test-time augmentation (TTA) for improved accuracy
- [ ] Add multi-scale inference for better boundary delineation
- [ ] Integrate focal loss for handling class imbalance
- [ ] Optimize small object detection (vehicles, pools)
- [ ] Add batch processing for multiple images

### Long-term Goals

- [ ] **Temporal Analysis**: Video stream processing for real-time flood monitoring
- [ ] **3D Reconstruction**: Combine with elevation data for flood depth estimation
- [ ] **Change Detection**: Pre/post-flood comparison
- [ ] **Mobile Deployment**: Optimize for edge devices (TensorRT, ONNX)
- [ ] **Active Learning**: Incorporate user feedback for model improvement
- [ ] **Multi-modal Fusion**: Integrate SAR, thermal, and optical imagery

### Research Directions

- Transformer-based architectures (Segformer, SegNeXt)
- Self-supervised pre-training on unlabeled aerial imagery
- Domain adaptation for different geographical regions
- Explainable AI for disaster response decision support

---

## Implementation Details

### Core Components

#### `dataset.py` - ImagePatchDataset

Custom PyTorch Dataset class featuring:
- Automatic patch extraction with configurable size/overlap
- Support for both NumPy and PyTorch tensor operations
- Multi-scale patch generation with stitch levels
- On-the-fly data augmentation
- Original image retrieval for visualization

Key methods:
```python
def __getitem__(self, idx):
    # Returns patches for training
    img_patches, mask_patches = ...
    return img_patches, mask_patches

def get_original_image(self, idx):
    # Returns full-resolution original image
    return original_img
```

#### `models.py` - Neural Network Architectures

Three model implementations:
1. **CustomNet**: Basic U-Net with GELU activations
2. **CustomSegmentationModel**: Advanced U-Net with attention and ASPP
3. **DeepLabV3Plus**: Imported from `segmentation_models_pytorch`

#### `utils.py` - Helper Functions

- `FocalLoss`: Alternative loss function for class imbalance
- `DiceLoss`: Dice coefficient loss
- `fast_reconstruct_from_patches`: GPU-accelerated image reconstruction
- `convert_to_numpy`: Tensor/array conversion utilities
- `safe_pickle_dump/load`: Robust checkpoint saving/loading

#### `gui.py` - Streamlit Application

Interactive web interface:
- Image upload and preprocessing
- Model loading and inference
- Multi-panel visualization
- Real-time performance metrics

### Training Pipeline

1. **Data Preparation**
   - Load image/mask pairs
   - Extract patches with overlap
   - Apply augmentation transforms

2. **Model Training**
   - Forward pass with mixed precision
   - Compute loss and backpropagate
   - Accumulate gradients
   - Update BN statistics
   - Clip gradients (optional)

3. **Validation**
   - Evaluate on validation set
   - Compute metrics (accuracy, IoU, Dice)
   - Save best model checkpoint

4. **Testing**
   - Per-class IoU calculation
   - Confidence and uncertainty visualization
   - Performance benchmarking

---

## Performance Tips

### Training Optimization

1. **Use mixed precision training** for 2-3x speedup
2. **Increase num_workers** if CPU is underutilized
3. **Tune batch size** to maximize GPU utilization
4. **Enable gradient accumulation** for larger effective batch sizes
5. **Use fused optimizer** (AdamW with `fused=True`)

### Inference Optimization

1. **Batch inference** when processing multiple images
2. **TensorRT optimization** for deployment
3. **ONNX export** for cross-platform compatibility
4. **Dynamic batching** for variable input sizes
5. **Model quantization** (INT8) for edge devices

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce batch size
- Enable gradient accumulation
- Decrease patch size
- Use mixed precision training

**Slow Training**
- Increase num_workers in DataLoader
- Enable pin_memory and non_blocking transfers
- Use fused optimizer
- Profile with PyTorch profiler

**Poor Convergence**
- Adjust learning rate
- Increase data augmentation
- Try different loss functions (Focal, Dice)
- Check data normalization

**Low Accuracy**
- Verify data preprocessing
- Check class distribution
- Experiment with different architectures
- Increase model capacity

---

## Citation

If you use this code or find it helpful, please consider citing:

```bibtex
@misc{flood-detection-segmentation,
  author = {Your Name},
  title = {Flood Detection and Segmentation using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/MSc_Imaging}
}
```

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## Acknowledgments

- **segmentation_models_pytorch**: For pre-trained model architectures
- **Albumentations**: For powerful data augmentation
- **PyTorch**: For the deep learning framework
- Course materials from MSc Imaging program

---

## Contact

For questions, issues, or collaborations:

- Open an issue on GitHub
- Email: [your.email@example.com]

---

<p align="center">
  Made with ❤️ for disaster response and flood risk assessment
</p>
