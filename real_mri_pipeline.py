import sys
sys.path.insert(0, "/home/claude")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import img_as_float, morphology, measure, filters
from scipy import ndimage as ndi

from mri_pipeline import denoise_image, enhance_contrast, estimate_noise_sigma

def load_real_mri(path):
    img = Image.open(path).convert("L")
    arr = img_as_float(np.array(img))
    return arr


# ---------------------------------------------------------------
# 針對真實影像微調過的分割函式
# ---------------------------------------------------------------
def segment_bright_region(img, skull_erosion=6, min_size=120):
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

    fig.suptitle(
        "Automated image-processing demo only — not a medical diagnosis",
        fontsize=10, color="gray", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"結果影像已儲存至: {save_path}")


if __name__ == "__main__":
    raw = load_real_mri("/mnt/user-data/uploads/Y36.JPG")
    denoised = denoise_image(raw)
    enhanced = enhance_contrast(denoised)
    mask, regions, brain_mask = segment_bright_region(denoised)

    visualize_real_mri(
        raw, denoised, enhanced, mask, regions,
        "/mnt/user-data/outputs/real_mri_result.png",
    )

    print(f"\n共標註出 {len(regions)} 個異常明亮區域：")
    for i, r in enumerate(regions, 1):
        cy, cx = r.centroid
        print(
            f"  區域 {i}: 中心座標=({cy:.1f}, {cx:.1f}), "
            f"面積={r.area:.0f} px, 平均亮度={r.intensity_mean:.3f}"
        )
