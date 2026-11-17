from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def split_blocks(image):
    """Chia ảnh xám thành các khối 8 x 8, trả về mảng khối và hai kích thước ảnh."""
    # Tính số hàng, cột cần đệm để hai chiều chia hết cho 8.
    height, width = image.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    # Lặp lại giá trị biên ở phía dưới và bên phải ảnh.
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")

    # Lấy các khối theo thứ tự từ trái sang phải, từ trên xuống dưới.
    blocks = []
    for y in range(0, padded.shape[0], 8):
        for x in range(0, padded.shape[1], 8):
            blocks.append(padded[y:y + 8, x:x + 8])

    return np.array(blocks), image.shape, padded.shape


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11})
    folder = Path(__file__).resolve().parent
    output = folder / "ket_qua_week8"
    output.mkdir(exist_ok=True)

    # Câu 1a: Đọc ảnh ở chế độ xám. Kích thước ảnh có thứ tự (H, W).
    image = cv2.imread(str(folder / "RaceHorses_416x240_frame001.png"), 0)
    if image is None:
        raise FileNotFoundError("Không đọc được ảnh đầu vào.")

    # Câu 1b: Gọi hàm chia khối và lấy kích thước trước, sau khi đệm.
    blocks, original_shape, padded_shape = split_blocks(image)
    print("Kích thước ảnh gốc (H, W):", original_shape)
    print("Kích thước sau đệm:", padded_shape)
    print("Số khối:", len(blocks))
    print("Kích thước mảng blocks:", blocks.shape)

    # Câu 1c: Chọn khối gần giữa ảnh làm khối mẫu chung cho cả ba câu.
    y = (image.shape[0] // 16) * 8
    x = (image.shape[1] // 16) * 8
    # Chỉ số khối = số hàng khối phía trước * số khối mỗi hàng + số cột khối.
    index = (y // 8) * (padded_shape[1] // 8) + x // 8
    print(f"Khối mẫu: chỉ số {index}, góc trên trái (y, x) = ({y}, {x})")
    print(blocks[index])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title(f"(a) Ảnh gốc: {image.shape[0]} × {image.shape[1]}")
    axes[0].axis("off")
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    # Dịch đường lưới 0.5 để đường nằm giữa các điểm ảnh.
    for col in range(0, padded_shape[1] + 1, 8):
        axes[1].axvline(col - 0.5, color="red", linewidth=0.25, alpha=0.5)
    for row in range(0, padded_shape[0] + 1, 8):
        axes[1].axhline(row - 0.5, color="red", linewidth=0.25, alpha=0.5)
    axes[1].add_patch(Rectangle((x - 0.5, y - 0.5), 8, 8,
                                fill=False, edgecolor="lime", linewidth=1.5))
    axes[1].annotate(f"Khối {index}", xy=(x + 3.5, y + 3.5),
                     xytext=(x + 40, y - 55), color="black", fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"),
                     arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))
    axes[1].set_title("(b) Chia khối 8 × 8")
    axes[1].axis("off")
    fig.savefig(output / "cau1_chia_khoi.png", dpi=300, bbox_inches="tight", pad_inches=0.08)

    # Tách ma trận thành hình riêng để các giá trị vẫn rõ khi chèn báo cáo.
    fig_sample, ax_sample = plt.subplots(figsize=(4.8, 4.8), layout="constrained")
    ax_sample.imshow(blocks[index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_sample.set_title(f"Ma trận điểm ảnh của khối {index}\nGóc trên trái: (y, x) = ({y}, {x})")
    ax_sample.set_xlabel("Chỉ số cột")
    ax_sample.set_ylabel("Chỉ số hàng")
    ax_sample.set_xticks(range(8))
    ax_sample.set_yticks(range(8))
    ax_sample.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax_sample.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax_sample.grid(which="minor", color="gray", linewidth=0.5)
    ax_sample.tick_params(which="minor", bottom=False, left=False)
    for row in range(8):
        for col in range(8):
            value = int(blocks[index, row, col])
            ax_sample.text(col, row, str(value), ha="center", va="center",
                           fontsize=11, color="white" if value < 128 else "black")
    fig_sample.savefig(output / "cau1_khoi_mau.png", dpi=300, bbox_inches="tight", pad_inches=0.08)

    # Thử ảnh cắt 13 x 19 để kiểm tra trường hợp cần đệm cả hai chiều.
    cropped = image[:13, :19]
    small_blocks, small_original, small_padded = split_blocks(cropped)
    print("\nKiểm tra ảnh cắt 13 x 19:")
    print("Trước đệm:", small_original, "- Sau đệm:", small_padded)
    print("Số khối:", len(small_blocks))
    # Ảnh 13 x 19 cần thêm 3 hàng và 5 cột để thành 16 x 24.
    padded_crop = np.pad(cropped, ((0, 3), (0, 5)), mode="edge")
    fig2, axes2 = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for ax, data, title in zip(axes2, [cropped, padded_crop],
                               ["(a) Ảnh cắt 13 × 19", "(b) Sau đệm: 16 × 24"]):
        ax.imshow(data, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Chỉ số cột")
        ax.set_ylabel("Chỉ số hàng")
        ax.set_xticks(np.arange(0, data.shape[1], 4))
        ax.set_yticks(np.arange(0, data.shape[0], 4))
    # Đường nét đứt đánh dấu ranh giới giữa ảnh cắt và phần đệm.
    boundary = axes2[1].axhline(12.5, color="red", linestyle="--", linewidth=1.2,
                               label="Ranh giới ảnh trước đệm")
    axes2[1].axvline(18.5, color="red", linestyle="--", linewidth=1.2)
    fig2.legend(handles=[boundary], loc="lower center", frameon=False)
    fig2.tight_layout(rect=(0, 0.1, 1, 1))
    fig2.savefig(output / "cau1_dem_bien.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    if "--no-show" not in sys.argv:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
