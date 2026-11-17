"""Mã hỗ trợ được cung cấp trong đề để mã hóa và giải mã ảnh xám.

Ba hàm tự viết được đặt trong cau1.py, cau2.py và cau3.py.
"""
from collections import Counter
import heapq
import numpy as np
import cv2

# Tạo thứ tự duyệt zigzag dùng chung cho mã hóa và giải mã.
ZIGZAG = []
for diagonal in range(15):
    positions = [(row, diagonal-row) for row in range(8) if 0 <= diagonal-row < 8]
    ZIGZAG.extend(positions[::-1] if diagonal % 2 == 0 else positions)
EOB = "EOB"


def zigzag(block):
    return [int(block[row, col]) for row, col in ZIGZAG]


def inverse_zigzag(values):
    if len(values) != 64:
        raise ValueError("Một khối phải có 64 hệ số.")
    block = np.zeros((8, 8), dtype=np.int32)
    for (row, col), value in zip(ZIGZAG, values):
        block[row, col] = value
    return block


def rle_encode(values):
    tokens, run = [], 0
    for value in values:
        if value == 0:
            run += 1
        else:
            tokens.append((run, int(value)))
            run = 0
    return tokens + [EOB]


def rle_decode(tokens):
    values = []
    for token in tokens:
        if token == EOB:
            return values + [0] * (64 - len(values))
        run, value = token
        if run < 0 or value == 0 or len(values) + run + 1 > 64:
            raise ValueError("Token RLE không hợp lệ.")
        values.extend([0] * run)
        values.append(value)
    raise ValueError("Thiếu EOB.")


def build_huffman_table(tokens):
    # Dùng số thứ tự để phân biệt các nút có cùng tần suất trong heap.
    heap = [(count, order, token)
            for order, (token, count) in enumerate(Counter(tokens).items())]
    heapq.heapify(heap)
    order = len(heap)
    while len(heap) > 1:
        count_a, _, left = heapq.heappop(heap)
        count_b, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (count_a + count_b, order, [left, right]))
        order += 1
    table = {}

    def visit(node, prefix):
        if isinstance(node, list):
            visit(node[0], prefix + "0")
            visit(node[1], prefix + "1")
        else:
            table[node] = prefix or "0"

    visit(heap[0][2], "")
    return table


def huffman_encode(tokens, table):
    return "".join(table[token] for token in tokens)


def huffman_decode(bits, table):
    reverse_table = {bits: token for token, bits in table.items()}
    tokens, prefix = [], ""
    for bit in bits:
        prefix += bit
        if prefix in reverse_table:
            tokens.append(reverse_table[prefix])
            prefix = ""
    if prefix:
        raise ValueError("Chuỗi bit kết thúc giữa một mã Huffman.")
    return tokens


def encode(quantized):
    """Các khối nguyên (N, 8, 8) -> chuỗi bit và bảng Huffman."""
    tokens = [t for block in quantized for t in rle_encode(zigzag(block))]
    codes = build_huffman_table(tokens)
    return huffman_encode(tokens, codes), codes


def decode_quantized(bits, codes, padded_shape):
    """Giải mã để lấy lại chính xác các hệ số nguyên đã lượng tử hóa."""
    tokens = huffman_decode(bits, codes)
    blocks, current = [], []
    for token in tokens:
        current.append(token)
        if token == EOB:
            values = rle_decode(current)
            blocks.append(inverse_zigzag(values))
            current = []
    expected = (padded_shape[0] // 8) * (padded_shape[1] // 8)
    if current or len(blocks) != expected:
        raise ValueError("Số khối giải mã không hợp lệ.")
    return np.array(blocks, dtype=np.int32)


def decode(bits, codes, Q, original_shape, padded_shape):
    """Giải mã, giải lượng tử, IDCT, ghép khối và cắt phần đệm."""
    quantized = decode_quantized(bits, codes, padded_shape)
    blocks = [cv2.idct(b.astype(np.float32) * Q) + 128.0 for b in quantized]
    padded = np.empty(padded_shape, dtype=np.float32)
    index = 0
    for y in range(0, padded_shape[0], 8):
        for x in range(0, padded_shape[1], 8):
            padded[y:y+8, x:x+8] = blocks[index]
            index += 1
    height, width = original_shape
    restored = padded[:height, :width]
    return np.clip(np.rint(restored), 0, 255).astype(np.uint8)
