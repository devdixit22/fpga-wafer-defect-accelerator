import numpy as np
import cv2


def preprocess_image(image_path, size=32):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (size, size))

    img = img.astype(np.float32)

    img = img / 255.0

    return img


def im2col(image, kernel_size=3, stride=1):

    H, W = image.shape

    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    cols = []

    for y in range(out_h):
        for x in range(out_w):

            patch = image[y:y+kernel_size, x:x+kernel_size]

            cols.append(patch.flatten())

    cols = np.array(cols)

    return cols


def quantize_int8(x):

    max_val = np.max(np.abs(x))

    if max_val == 0:
        return x.astype(np.int8)

    scale = 127.0 / max_val

    q = (x * scale).astype(np.int8)

    return q


def export_activation_mem(cols):

    q_cols = quantize_int8(cols)

    flat = q_cols.flatten()

    np.savetxt("activation.mem", flat, fmt="%d")

    return q_cols
