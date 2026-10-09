"""
Applying the "MRI Image Cleaning / Enhancement / Annotation" Pipeline to Real Uploaded Images
==========================================================================================
This script directly reuses the four functional steps defined in `mri_pipeline.py`:
denoise_image()      -- Denoising
enhance_contrast()   -- Contrast enhancement
segment_lesion()     -- Segmentation / Annotation of abnormally bright regions
visualize_pipeline() -- Visualization of output

Only the "image loading" step differs: here, we load a real JPG image uploaded by the user
instead of using a synthetic Shepp-Logan phantom. This demonstrates that
as long as the input NumPy array is formatted correctly (single channel, values ​​between 0 and 1),
the entire subsequent pipeline can be applied to real medical images without any modifications.

[Disclaimer]
The "automatic annotation" here is purely a demonstration of image processing techniques
(identifying regions with abnormally high signal intensity) and does not constitute a medical diagnosis.
Actual clinical interpretation must still be performed by a professional radiologist.
"""

import sys
sys.path.insert(0, "/home/claude")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import img_as_float, morphology, measure, filters
from scipy import ndimage as ndi

from mri_pipeline import denoise_image, enhance_contrast, estimate_noise_sigma


# ---------------------------------------------------------------
# Step 1: Load the real uploaded image
# ---------------------------------------------------------------
def load_real_mri(path):
    """
    Load a real MRI image (standard formats like JPG/PNG).
    Real MRI images in JPG/PNG formats have typically already been compressed into 8-bit grayscale images; 
    here, we simply convert the image to grayscale and normalize the values ​​to the 0–1 range,
    allowing it to be fed directly into the existing pipeline. 
    
    If handling standard medical imaging formats in the future:
    - DICOM (.dcm)      -> Read using pydicom
    - NIfTI (.nii/.gz)  -> Read using nibabel
    """
    img = Image.open(path).convert("L") # Convert to grayscale
    arr = img_as_float(np.array(img))
    return arr


# ---------------------------------------------------------------
# Segmentation function fine-tuned for real images
# ---------------------------------------------------------------
def segment_bright_region(img, skull_erosion=6, min_size=120):
    """
    The logic is identical to mri_pipeline.segment_lesion(),
    but the skull erosion pixel count (skull_erosion) has been readjusted
    based on the actual image dimensions, as this image has a much lower
    resolution than the previous synthetic demonstration images. 
    
    [Important] Here, we intentionally pass an image that has undergone
    noise reduction but *not* CLAHE (local contrast enhancement) for
    threshold-based segmentation. CLAHE performs local normalization,
    stretching the contrast in every small region; this causes normal
    cerebral gyri to appear "brighter" locally, which interferes with
    global brightness assessment. Global brightness information yields
    more accurate segmentation, while the enhanced image is reserved
    for visualization purposes.
    """
    head_mask = img > filters.threshold_otsu(img)
    head_mask = ndi.binary_fill_holes(head_mask)
    brain_mask = morphology.erosion(head_mask, morphology.disk(skull_erosion))

    brain_pixels = img[brain_mask]
    local_thresh = filters.threshold_otsu(brain_pixels)
    bright_thresh = local_thresh + (brain_pixels.max() - local_thresh) * 0.55
    binary = (img > bright_thresh) & brain_mask

    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.closing(binary, morphology.disk(2))

    labeled = measure.label(binary)
    regions = measure.regionprops(labeled, intensity_image=img)
    return binary, regions, brain_mask


def visualize_real_mri(raw, denoised, enhanced, mask, regions, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), facecolor="white")

    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("1. Original MRI", fontsize=12)

    axes[1].imshow(denoised, cmap="gray")
    axes[1].set_title("2. Denoised", fontsize=12)

    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("3. Contrast enhanced", fontsize=12)

    axes[3].imshow(enhanced, cmap="gray")
    overlay = np.ma.masked_where(~mask, mask)
    axes[3].imshow(overlay, cmap="autumn", alpha=0.55)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle(
            (minc - 2, minr - 2),
            (maxc - minc) + 4,
            (maxr - minr) + 4,
            fill=False, edgecolor="cyan", linewidth=1.6,
        )
        axes[3].add_patch(rect)
    axes[3].set_title(f"4. Highlighted bright region(s): {len(regions)}", fontsize=12)

    for ax in axes:
        ax.axis("off")

    # Disclaimer: plain text, enlarged
    plt.tight_layout(rect=[0, 0.3, 1, 1])
    fig.text(
        0.5, 0.21,
        "NOT A MEDICAL DIAGNOSIS",
        ha="center", va="center", fontsize=24, fontweight="bold", color="black",
    )
    fig.text(
        0.5, 0.08,
        "Image-processing demo only. Highlighted areas are automatic brightness-based detections;\n"
        "they may include normal structures or miss real abnormalities.\n"
        "Clinical interpretation must be performed by a qualified radiologist.",
        ha="center", va="center", fontsize=15, color="#222222", linespacing=1.5,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"The result image has been saved to: {save_path}")


if __name__ == "__main__":
    raw = load_real_mri("/mnt/user-data/uploads/Y36.JPG")
    denoised = denoise_image(raw)
    enhanced = enhance_contrast(denoised)
    mask, regions, brain_mask = segment_bright_region(denoised)

    visualize_real_mri(
        raw, denoised, enhanced, mask, regions,
        "/mnt/user-data/outputs/real_mri_result.png",
    )

    print(f"\nDetected a total of {len(regions)} abnormally bright regions:")
    for i, r in enumerate(regions, 1):
        cy, cx = r.centroid
        print(
            f"  Region {i}: Center coordinates=({cy:.1f}, {cx:.1f}), "
            f"  Area={r.area:.0f} px, Mean intensity={r.intensity_mean:.3f}"
        )
