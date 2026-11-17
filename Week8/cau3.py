import csv
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cau1 import split_blocks
from cau2 import dct_blocks
from jpeg_support import encode, decode, decode_quantized


def quantize(coefficients, quant_scale):
    """Lượng tử hóa hệ số DCT, trả về mảng số nguyên và bảng Q kiểu float32."""
    if not np.isfinite(quant_scale) or quant_scale <= 0:
        raise ValueError("quant_scale phải là số dương hữu hạn.")
    # Câu 3a: Tạo bảng Q0 theo công thức Q0[u, v] = 8 + 4 * (u + v).
    Q0 = np.zeros((8, 8), dtype=np.float32)
    for u in range(8):
        for v in range(8):
            Q0[u, v] = 8 + 4 * (u + v)
    # Nhân với scale, làm tròn và giới hạn từng phần tử của Q không nhỏ hơn 1.
    Q = np.maximum(1, np.rint(Q0 * quant_scale)).astype(np.float32)
    # Chia từng hệ số cho phần tử tương ứng của Q rồi làm tròn về số nguyên.
    quantized = np.rint(coefficients / Q).astype(np.int32)
    return quantized, Q


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11})
    folder = Path(__file__).resolve().parent
    output = folder / "ket_qua_week8"
    output.mkdir(exist_ok=True)
    image = cv2.imread(str(folder / "RaceHorses_416x240_frame001.png"), 0)
    if image is None:
        raise FileNotFoundError("Không đọc được ảnh đầu vào.")
    blocks, original_shape, padded_shape = split_blocks(image)
    coefficients = dct_blocks(blocks)

    # Câu 3b: In bảng Q, khối mẫu và tỉ lệ hệ số 0 khi quant_scale = 1.0.
    quantized, Q = quantize(coefficients, 1.0)
    y = (image.shape[0] // 16) * 8
    x = (image.shape[1] // 16) * 8
    index = (y // 8) * (padded_shape[1] // 8) + x // 8
    print("Bảng Q ở scale = 1.0:")
    print(Q)
    print(f"\nKhối mẫu {index} sau lượng tử hóa:")
    print(quantized[index])
    print(f"Tỉ lệ hệ số 0 toàn ảnh: {np.mean(quantized == 0):.2%}")

    # Câu 3c: So sánh ảnh dựng lại ở ba mức quant_scale: 0.5, 1.0 và 3.0.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), layout="constrained")
    axes = axes.ravel()
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("(a) Ảnh gốc")
    rows = []
    print("\nScale | MSE | Số bit dữ liệu | Tỉ lệ hệ số 0")
    for i, scale in enumerate([0.5, 1.0, 3.0], start=1):
        quantized, Q = quantize(coefficients, scale)
        # Dùng các hàm mã hóa và giải mã được cung cấp trong jpeg_support.py.
        bits, codes = encode(quantized)
        restored = decode(bits, codes, Q, original_shape, padded_shape)
        assert restored.shape == image.shape
        assert restored.dtype == np.uint8
        # Kiểm tra các hệ số sau mã hóa và giải mã khớp với hệ số đã lượng tử hóa.
        assert np.array_equal(decode_quantized(bits, codes, padded_shape), quantized)
        # Đổi sang số thực trước khi tính sai số giữa hai ảnh.
        error = image.astype(np.float32) - restored.astype(np.float32)
        mse = float(np.mean(error ** 2))
        zero_ratio = float(np.mean(quantized == 0))
        print(f"{scale:.1f} | {mse:.2f} | {len(bits)} | {zero_ratio:.2%}")
        rows.append([scale, mse, len(bits), zero_ratio])
        axes[i].imshow(restored, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[i].set_title(f"({chr(97 + i)}) Scale {scale} | MSE = {mse:.2f}")
        cv2.imwrite(str(output / f"cau3_anh_scale_{scale:.1f}.png"), restored)

    for ax in axes:
        ax.axis("off")
    fig.savefig(output / "cau3_so_sanh.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    # Lưu các số liệu của ba mức lượng tử hóa vào bảng CSV.
    with (output / "cau3_bang_ket_qua.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["quant_scale", "MSE", "so_bit_du_lieu", "ti_le_he_so_0"])
        writer.writerows(rows)

    # Kiểm tra ảnh cắt 13 x 19 được dựng lại đúng kích thước ở cả ba mức.
    cropped = image[:13, :19]
    crop_blocks, crop_original, crop_padded = split_blocks(cropped)
    for scale in [0.5, 1.0, 3.0]:
        q_crop, Q_crop = quantize(dct_blocks(crop_blocks), scale)
        crop_bits, crop_codes = encode(q_crop)
        crop_restored = decode(crop_bits, crop_codes, Q_crop, crop_original, crop_padded)
        assert crop_restored.shape == cropped.shape
        assert crop_restored.dtype == np.uint8
        print(f"Ảnh cắt, scale {scale}: {crop_original} -> {crop_padded} -> {crop_restored.shape}: OK")

    if "--no-show" not in sys.argv:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
