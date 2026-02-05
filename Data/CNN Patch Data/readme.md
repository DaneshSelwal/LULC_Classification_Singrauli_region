
## 💾 Data Assets & Downloads

Due to GitHub's file size limits, the high-resolution satellite imagery and processed training datasets are hosted externally. You can access the raw inputs, the processing logic, and the generated outputs via the links below:

| File Asset | Description | Download Link |
| :--- | :--- | :--- |
| **`Singrauli_Merged_Image.tif`** | **(Input)** Base Sentinel-2 composite image (11 bands) used for patch extraction. | [Download from Google Drive](https://drive.google.com/file/d/1uIC0iZzhQ5EUYnMRkReEbGNvEtgJUDE0/view?usp=sharing) |
| **`cnn_prepared_patches_9.npz`** | **(Output)** Pre-processed 9x9 pixel image patches with labels, ready for CNN training. | [Download from Google Drive](https://drive.google.com/file/d/1-aSA4gISWprNtQGOb4O2sxUqgosA3lus/view?usp=sharing) |
| **`cnn_prepared_patches_15.npz`** | **(Output)** Pre-processed 15x15 pixel image patches with labels, ready for CNN training. | [Download from Google Drive](https://drive.google.com/file/d/1em-NAvo3nfZ5FVeBJDVBsoGpIXis6sfg/view?usp=sharing) |

> **⚠️ Setup Note:** After downloading, please place `Singrauli_Merged_Image.tif` and `cnn_prepared_patches_9.npz` in your project's root directory (or update the file paths in the notebook) to reproduce the results.
