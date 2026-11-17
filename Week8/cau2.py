from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cau1 import split_blocks


def dct_blocks(blocks):
    """Tính DCT cho từng khối, trả về mảng float32 có dạng (N, 8, 8)."""
    coefficients = []
    for block in blocks:
        # Đổi sang float32 trước khi trừ 128 để biểu diễn được giá trị âm.
        shifted = block.astype(np.float32) - 128.0
        # Tính DCT hai chiều cho khối đã dịch mức sáng.
        coefficients.append(cv2.dct(shifted))
    return np.array(coefficients, dtype=np.float32)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11})
    folder = Path(__file__).resolve().parent
    output = folder / "ket_qua_week8"
    output.mkdir(exist_ok=True)
    image = cv2.imread(str(folder / "RaceHorses_416x240_frame001.png"), 0)
    if image is None:
        raise FileNotFoundError("Không đọc được ảnh đầu vào.")

    # Câu 2a: Chia ảnh thành các khối 8 x 8 rồi tính DCT cho từng khối.
    blocks, _, padded_shape = split_blocks(image)
    coefficients = dct_blocks(blocks)
    print("Kích thước mảng DCT:", coefficients.shape)
    print("Kiểu dữ liệu:", coefficients.dtype)

    # Câu 2b: In ma trận DCT của cùng khối mẫu đã chọn ở Câu 1.
    y = (image.shape[0] // 16) * 8
    x = (image.shape[1] // 16) * 8
    index = (y // 8) * (padded_shape[1] // 8) + x // 8
    sample = coefficients[index]
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(f"\nMa trận DCT của khối {index}:")
    print(sample)
    # Phần tử (0, 0) là hệ số DC; 63 phần tử còn lại là các hệ số AC.
    print("Hệ số DC tại (0, 0):", sample[0, 0])

    # Câu 2c: Vẽ log(1 + |DCT|) để giảm chênh lệch độ lớn khi hiển thị.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), layout="constrained")
    axes[0].imshow(blocks[index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title(f"(a) Khối điểm ảnh {index}")
    axes[0].set_xlabel("Chỉ số cột")
    axes[0].set_ylabel("Chỉ số hàng")
    heatmap = axes[1].imshow(np.log1p(np.abs(sample)), cmap="hot", interpolation="nearest")
    axes[1].set_title("(b) log(1 + |DCT|)")
    axes[1].set_xlabel("Chỉ số tần số v")
    axes[1].set_ylabel("Chỉ số tần số u")
    for ax in axes:
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
    fig.colorbar(heatmap, ax=axes[1], shrink=0.8, label="log(1 + |DCT|)")
    fig.savefig(output / "cau2_dct.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    if "--no-show" not in sys.argv:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
