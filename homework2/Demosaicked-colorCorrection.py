import rawpy
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Read the RAW DNG
# ---------------------------------------------------------

filename = "inputDNG.dng"

with rawpy.imread(filename) as raw:

    # RAW sensor data
    raw_image = raw.raw_image_visible.copy().astype(np.float32)

    print("RAW shape:", raw_image.shape)
    print("RAW minimum:", raw_image.min())
    print("RAW maximum:", raw_image.max())

    # Bayer pattern
    #print("Camera:", raw.camera_make, raw.camera_model)
    print("Bayer pattern:", raw.raw_pattern)

    # -----------------------------------------------------
    # 2. Black-level and white-level information
    # -----------------------------------------------------

    print("Black level:", raw.black_level_per_channel)
    print("White level:", raw.white_level)

    black = np.array(raw.black_level_per_channel,
                     dtype=np.float32)

    white = float(raw.white_level)

    # -----------------------------------------------------
    # 3. Demosaicing
    # -----------------------------------------------------

    # rawpy performs demosaicing using the selected algorithm.
    #
    # output_bps=16 keeps high precision.
    #
    # use_camera_wb=True uses the camera's white-balance
    # information.

    rgb = raw.postprocess(
        use_camera_wb=True,
        use_auto_wb=False,
        output_color=rawpy.ColorSpace.sRGB,
        output_bps=16,
        no_auto_bright=True,
        gamma=(1, 1),
        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
    )

# ---------------------------------------------------------
# 4. Convert to floating point [0,1]
# ---------------------------------------------------------

rgb = rgb.astype(np.float32) / 65535.0

rgb = np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------
# 5. Color correction matrix
# ---------------------------------------------------------

CCM = np.array([
    [1.50, -0.30, -0.20],
    [-0.20, 1.30, -0.10],
    [-0.10, -0.20, 1.30]
], dtype=np.float32)


height, width, channels = rgb.shape

rgb_vector = rgb.reshape(-1, 3)

rgb_corrected = rgb_vector @ CCM.T

rgb_corrected = rgb_corrected.reshape(
    height, width, 3
)

rgb_corrected = np.clip(
    rgb_corrected,
    0.0,
    1.0
)


# ---------------------------------------------------------
# 6. sRGB gamma correction
# ---------------------------------------------------------

rgb_srgb = np.where(
    rgb_corrected <= 0.0031308,
    12.92 * rgb_corrected,
    1.055 * np.power(rgb_corrected, 1 / 2.4) - 0.055
)

rgb_srgb = np.clip(rgb_srgb, 0.0, 1.0)


# ---------------------------------------------------------
# 7. Display
# ---------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.imshow(rgb_srgb)

plt.title("Final RGB Image")

plt.axis("off")

plt.show()


# ---------------------------------------------------------
# 8. Save RGB image
# ---------------------------------------------------------

rgb_8bit = (rgb_srgb * 255).astype(np.uint8)

cv2.imwrite(
    "output_RGB.png",
    cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR)
)

print("Saved: output_RGB.png")


