# MRI Brain Tumor Detection System
A practice program that finds tumors in MRI brain images, coded in Python.
The results may be false positives, due to its reliance on simple threshold-based segmentation and simply looking at brightness levels. 
Real-world imaging technology may use deep learning models trained on vast amounts of annotated data based on shape, location, and texture.

## 📖 How to View the Sample Image
In a browser, copy and paste this URL on the search bar: `https://mri-brain-tumor-detection-system.netlify.app/`

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
This script directly reuses the four functional steps defined in `mri_pipeline.py`:
- denoise_image()      -- Denoising
- enhance_contrast()   -- Contrast enhancement
- segment_lesion()     -- Segmentation / Annotation of abnormally bright regions
- visualize_pipeline() -- Visualization of output

**Disclaimer:**
The "automatic annotation" here is purely a demonstration of image processing techniques
(identifying regions with abnormally high signal intensity) and does not constitute a medical diagnosis.
Actual clinical interpretation must still be performed by a professional physician.
