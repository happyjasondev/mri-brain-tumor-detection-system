"""
套用「MRI 影像清洗 / 增強 / 標註」流程到真實上傳的影像
======================================================
這支程式直接重用 mri_pipeline.py 裡面已經寫好的四個步驟函式：
    denoise_image()      -- 降噪
    enhance_contrast()   -- 對比度增強
    segment_lesion()     -- 分割 / 標註異常明亮區域
    visualize_pipeline() -- 視覺化輸出

只有「讀取影像」這一步不一樣：這裡改成讀取使用者上傳的真實 JPG 影像，
而不是用合成的 Shepp-Logan phantom。這正好示範了：
只要把讀進來的 numpy array 格式對了 (單通道、數值介於 0~1 之間)，
後面整套 pipeline 完全不需要更動就能套用在真實醫學影像上。

【免責聲明】
這裡的「自動標註」純粹是影像處理技術演示（找出訊號異常明亮的區域），
不構成醫療診斷。真實臨床判讀仍需由專業放射科醫師執行。
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
# 步驟 1：讀取真實上傳的影像
# ---------------------------------------------------------------
def load_real_mri(path):
    """
    讀取真實 MRI 影像（JPG/PNG 等一般圖檔格式）。
    真實 MRI 若是 JPG/PNG 格式，通常已經被壓縮成 8-bit 灰階圖，
    這裡簡單轉成灰階 + 正規化到 0~1 之間，就可以直接接上原本的 pipeline。

    若未來要處理正規的醫學影像格式：
        - DICOM (.dcm)      -> 用 pydicom 讀取
        - NIfTI (.nii/.gz)  -> 用 nibabel 讀取
    """
    img = Image.open(path).convert("L")  # 轉灰階
    arr = img_as_float(np.array(img))
    return arr


# ---------------------------------------------------------------
# 針對真實影像微調過的分割函式
# ---------------------------------------------------------------
def segment_bright_region(img, skull_erosion=6, min_size=120):
    """
    與 mri_pipeline.segment_lesion() 邏輯相同，
    但侵蝕像素數 (skull_erosion) 依真實影像尺寸重新調整過，
    因為這張影像解析度比之前的合成示範影像小很多。

    【重要】這裡刻意傳入「降噪後、但尚未做 CLAHE 局部對比增強」的影像來做閾值分割。
    CLAHE 是局部正規化，會讓每一小塊區域的對比都被拉開，
    導致正常的大腦皮質迴紋 (gyri) 在局部也會顯得「偏亮」，反而干擾全域亮度判斷。
    分割用全域亮度資訊比較準確，增強後的影像則保留給視覺化顯示使用。
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
