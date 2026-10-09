# MRI Brain Tumor Detection System
A  coded in Python.

## 🛠️ Features & Tech Stack
**Language:**
- Python

**Libraries:**
- numpy and matplotlib.pyplot
- From PIL: Image
- From skimage: img_as_float, morphology, measure, filters
- From scipy: ndimage
- From mri_pipeline: denoise_image, enhance_contrast, estimate_noise_sigma

**Key Features:** 
1. Regular threshold segmentation	mri_pipeline.py
2. Apply to real images	real_mri_pipeline.py
3. Pixel-level supervised learning (Random Forest)	random_forest_segmentation.py, Dice ≈ 0.49
4. Deep Learning (U-Net)	unet_reference.py	⚠️ The architecture is complete, but it lacks PyTorch and a GPU.

## 📖 How to View the Sample Image
1. Type this into a terminal or command prompt: `git clone https://github.com/happyjasondev/mri-brain-tumor-detection-system.git`
2. Open `index.html` in any modern web browser.
