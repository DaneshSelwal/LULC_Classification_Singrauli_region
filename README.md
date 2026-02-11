# Singrauli LULC Classification

![Singrauli LULC](https://img.shields.io/badge/Project-LULC%20Classification-green) ![GEE](https://img.shields.io/badge/GEE-Sentinel--2-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/ML-TensorFlow%20%7C%20Optuna-orange)

Welcome to the **Singrauli Land-Use Land-Cover (LULC) Classification** repository! This project leverages the power of Google Earth Engine (GEE), Classical Machine Learning, and Deep Convolutional Neural Networks (CNNs) to analyze and classify the landscape of the Singrauli district using Sentinel-2 satellite imagery.

This is the **Updated Version 1** of the project, featuring refined CNN models (9x9 and 15x15 patch sizes), improved data pipelines, and a comprehensive accuracy assessment workflow.

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Getting Started & Prerequisites](#-getting-started--prerequisites)
- [Data Access](#-data-access)
- [Methodology](#-methodology)
- [Results](#-results)
- [Workflow Guide](#-workflow-guide)
- [Contributors](#-contributors)

---

## 🛰️ Project Overview

Singrauli is a region known for its rich coal reserves and power plants. Monitoring its Land Use and Land Cover is crucial for environmental assessment and planning. This project aims to accurately classify the region into 6 distinct classes:
1. **Water** 💧
2. **Agriculture** 🌾
3. **Settlement** 🏙️
4. **Mining** ⛏️
5. **Barren/Scrubland** 🍂
6. **Forest** 🌳

We employ a multi-modal approach, comparing traditional Random Forest classifiers (implemented in GEE) against patch-based CNN models (implemented in Python/TensorFlow) that leverage spatial context.

---

## 📂 Repository Structure

```text
.
├── 📁 Accuracy Assessment and Visualisation
│   ├── 📄 Accuracy_assessment.ipynb    # Interactive tool for ground truth validation
│   ├── 📄 model_accuracy_report.docx   # Generated report
│   └── 📄 reference_data.csv           # Ground truth points
├── 📁 CNN_15*15                        # CNN Model (15x15 Input)
│   ├── 📄 cnn_training_15.ipynb        # Training notebook
│   ├── 📄 cnn-prediction-15.ipynb      # Prediction notebook
│   └── 📁 model/                       # Saved .keras model and scalers
├── 📁 CNN_9*9                          # CNN Model (9x9 Input)
│   ├── 📄 cnn_training_9.ipynb         # Training notebook
│   ├── 📄 cnn-prediction-9.ipynb       # Prediction notebook
│   └── 📁 model/                       # Saved .keras model and scalers
├── 📁 Data
│   ├── 📁 CNN Patch Data               # Extraction scripts & patch info
│   │   ├── 📄 Data_Extraction.ipynb    # Script to create .npz from GeoTIFF
│   │   └── 📄 readme.md                # Data download links
│   └── 📁 CSV format                   # Tabular data (train/test splits)
├── 📁 Google Earth Engine Code Editor
│   ├── 📄 Random_Forest_LULC.js        # Main GEE script for RF classification
│   └── 📄 LULC_comparison.js           # GEE script for comparing results
├── 📁 Results
│   ├── 📁 Classified Images            # Final GeoTIFF outputs
│   └── 📁 Comparison Results           # Charts and area statistics
└── 📄 README.md                        # Project documentation
```

---

## 🚀 Getting Started & Prerequisites

### Environment
The Python notebooks in this repository are optimized for **Google Colab**. They assume data is mounted via Google Drive.
If running locally, you will need to adjust file paths and ensure the following libraries are installed:

*   `numpy`, `pandas`, `matplotlib`, `seaborn`
*   `scikit-learn`
*   `tensorflow` (for CNN models)
*   `rasterio` (for geospatial image handling)
*   `folium`, `ipywidgets` (for accuracy assessment maps)

### Installation
Clone this repository:
```bash
git clone https://github.com/your-username/Singrauli-LULC.git
cd Singrauli-LULC
```

---

## 💾 Data Access

Due to GitHub's file size limits, the high-resolution satellite imagery and pre-processed training datasets (patches) are hosted externally.

**⚠️ Important:** Download the following files and place them in your Google Drive (e.g., in a folder named `Internship_project`) to run the notebooks seamlessly.

| File Asset | Description | Download Link |
| :--- | :--- | :--- |
| **`Singrauli_Merged_Image.tif`** | **(Input)** Base Sentinel-2 composite image (11 bands) used for patch extraction. | [Download from Google Drive](https://drive.google.com/file/d/1uIC0iZzhQ5EUYnMRkReEbGNvEtgJUDE0/view?usp=sharing) |
| **`cnn_prepared_patches_9.npz`** | **(Output)** Pre-processed 9x9 pixel image patches with labels. | [Download from Google Drive](https://drive.google.com/file/d/1-aSA4gISWprNtQGOb4O2sxUqgosA3lus/view?usp=sharing) |
| **`cnn_prepared_patches_15.npz`** | **(Output)** Pre-processed 15x15 pixel image patches with labels. | [Download from Google Drive](https://drive.google.com/file/d/1em-NAvo3nfZ5FVeBJDVBsoGpIXis6sfg/view?usp=sharing) |

---

## 🔬 Methodology

### 1. Google Earth Engine (Random Forest)
*   **Script**: `Google Earth Engine Code Editor/Random_Forest_LULC.js`
*   **Process**: Acquires Sentinel-2 imagery, computes indices (NDVI, NDWI, etc.), and runs a Random Forest classifier directly on the cloud.
*   **Output**: Pixel-based classification.

### 2. Deep Learning (CNN)
*   **Architecture**: Custom Sequential CNNs with varying input patch sizes (9x9 and 15x15) to capture different scales of spatial context.
*   **Process**:
    *   Extract patches around labeled points.
    *   Train CNNs to classify the center pixel based on the surrounding patch.
*   **Performance**: CNNs generally offer smoother classification and reduce "salt-and-pepper" noise compared to pixel-based RF.

---

## 📊 Results

Final outputs are stored in the `Results/` directory:
*   **Classified Images**: GeoTIFFs for RF, CNN-9, and CNN-15 predictions.
*   **Comparison**: Area statistics (`LULC_class_area.csv`) and visual comparisons showing the distribution of classes across different models.

---

## 🛠️ Workflow Guide

Follow this step-by-step guide to reproduce the project results:

### Step 1: Data Acquisition (GEE)
1.  Open the `Google Earth Engine Code Editor/Random_Forest_LULC.js` script in the GEE Code Editor.
2.  Run the script to process Sentinel-2 data.
3.  Export the **Raw Composite Image** (11 bands) and the **Labeled Point Data** (CSV) to your Google Drive.

### Step 2: Patch Extraction
1.  Open `Data/CNN Patch Data/Data_Extraction.ipynb` in Google Colab.
2.  Mount your Google Drive containing the exported GEE image and CSV.
3.  Run the notebook to extract 9x9 and 15x15 pixel patches.
4.  **Output**: This will generate `.npz` files (e.g., `cnn_prepared_patches_9.npz`) saved to your Drive.

### Step 3: Model Training
1.  Navigate to `CNN_9*9/` or `CNN_15*15/`.
2.  Open `cnn_training_9.ipynb` (or the 15x15 equivalent).
3.  Ensure the path to the `.npz` file matches your Drive location.
4.  Run all cells to train the model. The notebook handles data splitting, normalization, training, and saving the model (`.keras`) and scaler (`.joblib`).

### Step 4: Prediction
1.  Open `cnn-prediction-9.ipynb`.
2.  Load the saved model and scaler.
3.  Run the prediction on the full satellite image (or test set) to generate the final Classified LULC Map.

### Step 5: Accuracy Assessment
1.  Open `Accuracy Assessment and Visualisation/Accuracy_assessment.ipynb`.
2.  Load a classified image (RF, CNN-9, or CNN-15).
3.  Use the interactive widget to generate stratified random points and validate them against high-resolution basemaps.
4.  **Output**: The notebook calculates Confusion Matrices, Kappa Statistics, and F1-Scores, generating a downloadable report (`model_accuracy_report.docx`).

---

## 👥 Contributors

*   *Data Science & Modeling*
*   *Remote Sensing Specialist*

---
*Happy Mapping!* 🗺️
