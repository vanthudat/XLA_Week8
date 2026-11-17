# Bài thực hành số 8: Mô phỏng nén JPEG ảnh xám

Bài thực hành sử dụng Python để chia ảnh thành các khối 8×8, biến đổi DCT và lượng tử hóa. Sau đó, chương trình dùng phần mã hóa và giải mã được cung cấp trong đề để dựng lại ảnh, so sánh chất lượng và số bit ở ba mức lượng tử hóa.

## 1. Nội dung từng câu

| Tệp | Nội dung |
| --- | --- |
| `cau1.py` | Đọc ảnh xám, viết hàm `split_blocks`, hiển thị lưới 8×8 và thử đệm ảnh cắt 13×19. |
| `cau2.py` | Viết hàm `dct_blocks`, in hệ số DCT của khối mẫu và hiển thị `log(1 + |DCT|)` kèm thang màu. |
| `cau3.py` | Viết hàm `quantize`, chạy mã hóa và dựng lại ảnh ở ba mức `0.5`, `1.0`, `3.0`; tính MSE, số bit và tỉ lệ hệ số 0. |
| `jpeg_support.py` | Code hỗ trợ từ đề bài: zigzag, RLE, Huffman, giải mã, giải lượng tử, IDCT và ghép khối. |
| `main.tex` | Mã LaTeX của báo cáo, sử dụng các hình trong `ket_qua_week8/`. |

Chỉ ba hàm `split_blocks`, `dct_blocks` và `quantize` là các module cần tự xây dựng theo đề. Các câu được tách thành từng tệp; Câu 2 và Câu 3 nhập lại hàm ở câu trước để sử dụng.

## 2. Chuẩn bị và cài thư viện

Máy cần có **Python 3.10 trở lên**. Mở Terminal hoặc PowerShell, tải mã nguồn rồi chuyển vào thư mục dự án:

```bash
git clone https://github.com/vanthudat/XLA_Week8.git
cd XLA_Week8
```

Nếu không dùng Git, chọn **Code → Download ZIP** trên GitHub, giải nén rồi mở Terminal tại thư mục chứa `cau1.py`.

Cài các thư viện:

```bash
python -m pip install -r requirements.txt
```

Nếu máy dùng lệnh `py` hoặc `python3`, thay `python` bằng lệnh tương ứng trong các hướng dẫn dưới đây.

Giữ các tệp sau trong cùng thư mục:

```text
XLA_Week8/
├── README.md
├── requirements.txt
├── RaceHorses_416x240_frame001.png
├── cau1.py
├── cau2.py
├── cau3.py
├── jpeg_support.py
├── main.tex
└── ket_qua_week8/                 # Chương trình tự tạo nếu chưa có
```

Ảnh đầu vào phải giữ đúng tên `RaceHorses_416x240_frame001.png`. Chương trình tìm ảnh theo vị trí của tệp Python, nên không cần sửa đường dẫn riêng theo từng máy.

## 3. Cách chạy

### Câu 1: Chia khối và đệm biên

```bash
python cau1.py
```

Chương trình in kích thước ảnh, số khối, ma trận điểm ảnh của khối mẫu và kết quả đệm ảnh 13×19. Các cửa sổ hình thể hiện ảnh gốc, lưới chia khối, khối mẫu và phần đệm.

Với ảnh được cung cấp, kích thước `(H, W)` là `(240, 416)`, tạo **1.560 khối**. Ảnh cắt 13×19 được đệm thành 16×24, tạo **6 khối**.

### Câu 2: Biến đổi DCT

```bash
python cau2.py
```

Chương trình in ma trận DCT và hệ số DC của cùng khối mẫu ở Câu 1, đồng thời hiển thị bản đồ độ lớn hệ số. Trước khi tính DCT, mỗi khối được chuyển sang `float32` rồi trừ 128.

### Câu 3: Lượng tử hóa, mã hóa và dựng lại ảnh

```bash
python cau3.py
```

Chương trình in bảng lượng tử, khối mẫu sau lượng tử hóa tại `quant_scale = 1.0` và bảng số liệu của cả ba mức. Hình so sánh gồm ảnh gốc và ba ảnh dựng lại, sắp xếp theo bố cục 2×2.

Các phép kiểm tra trong chương trình xác nhận ảnh dựng lại có đúng kích thước và kiểu `uint8`, kể cả trường hợp ảnh cần đệm. Chương trình cũng kiểm tra bộ mã hóa và giải mã bảo toàn các hệ số nguyên đã lượng tử hóa của ảnh đầy đủ.

**Có thể chạy riêng từng câu**, không cần chạy Câu 1 để tạo dữ liệu trước khi chạy Câu 2 hoặc Câu 3. Tuy nhiên, cần giữ đầy đủ các tệp Python vì chúng có nhập hàm của nhau. Khi chạy bình thường, đóng các cửa sổ hình để chương trình kết thúc và Terminal nhận lệnh tiếp theo.

### Chạy không mở cửa sổ hình

Nếu chỉ muốn in số liệu và lưu ảnh kết quả:

```bash
python cau1.py --no-show
python cau2.py --no-show
python cau3.py --no-show
```

Trên môi trường không có màn hình, có thể đặt thêm biến `MPLBACKEND=Agg` trước khi chạy. Ví dụ với PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
python cau3.py --no-show
```

## 4. Kết quả được lưu ở đâu?

Mỗi câu tự tạo thư mục `ket_qua_week8/` và lưu các kết quả tương ứng:

| Tệp kết quả | Nội dung |
| --- | --- |
| `cau1_chia_khoi.png` | Ảnh gốc và lưới chia khối, có đánh dấu khối mẫu. |
| `cau1_khoi_mau.png` | Ma trận điểm ảnh 8×8 của khối mẫu. |
| `cau1_dem_bien.png` | Ảnh cắt trước và sau đệm biên. |
| `cau2_dct.png` | Khối mẫu và bản đồ độ lớn hệ số DCT. |
| `cau3_so_sanh.png` | So sánh ảnh gốc với ba ảnh dựng lại. |
| `cau3_anh_scale_0.5.png` | Ảnh dựng lại ở mức 0.5. |
| `cau3_anh_scale_1.0.png` | Ảnh dựng lại ở mức 1.0. |
| `cau3_anh_scale_3.0.png` | Ảnh dựng lại ở mức 3.0. |
| `cau3_bang_ket_qua.csv` | Bảng số liệu để xem bằng Excel hoặc trình đọc CSV. |

Chạy lại chương trình sẽ ghi đè các tệp kết quả cùng tên. Số liệu và ma trận được in trực tiếp ra Terminal; chương trình không tạo tệp `.txt`.

## 5. Kết quả tham khảo

Kết quả chạy trên ảnh đi kèm:

| quant_scale | MSE | Số bit dữ liệu | Tỉ lệ hệ số 0 |
| ---: | ---: | ---: | ---: |
| 0.5 | 12.34 | 166473 | 70.82% |
| 1.0 | 30.48 | 108115 | 81.15% |
| 3.0 | 97.46 | 47347 | 91.87% |

![So sánh ảnh gốc và ảnh dựng lại](ket_qua_week8/cau3_so_sanh.png)

Khi tăng `quant_scale`, kết quả thử nghiệm cho thấy số bit giảm, tỉ lệ hệ số 0 tăng và MSE tăng. Ảnh ở mức 3.0 mất chi tiết và xuất hiện các ô khối rõ hơn.

Đây là **mô phỏng quy trình JPEG trong bộ nhớ**, không tạo tệp JPEG chuẩn. Cột số bit chỉ tính độ dài chuỗi Huffman, chưa tính bảng mã, bảng lượng tử và metadata. Các ảnh PNG được lưu là ảnh kết quả để xem, không phải dữ liệu nén dùng để đo số bit trong bảng.

## 6. Một số lỗi thường gặp

- **`ModuleNotFoundError`:** cài lại thư viện bằng `python -m pip install -r requirements.txt` trong đúng môi trường Python đang dùng để chạy bài.
- **Không đọc được ảnh đầu vào:** kiểm tra ảnh PNG có nằm cùng thư mục với các tệp Python và tên ảnh có đúng như trên không.
- **Không tìm thấy `cau1`, `cau2` hoặc `jpeg_support`:** giữ các tệp Python trong cùng thư mục và chạy các lệnh từ thư mục dự án.
- **Không thấy cửa sổ hình:** kiểm tra có đang dùng `--no-show` hoặc backend `Agg` hay không. Nếu chạy trên máy chủ, xem ảnh đã lưu trong `ket_qua_week8/`.

Báo cáo được viết trong `main.tex`. Nếu muốn biên dịch báo cáo, dùng XeLaTeX trên môi trường có font Times New Roman và giữ thư mục ảnh kết quả đi kèm.
