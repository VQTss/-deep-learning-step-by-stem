import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../assets/dog.jpg")
img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Shape:", img_array.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img_array)
axes[0].set_title("Original")
axes[0].axis("off")

img_transposed_array = cv2.transpose(img_array)
axes[1].imshow(img_transposed_array)
axes[1].set_title("After cv2.transpose (algebraic)")
axes[1].axis("off")

plt.show()

img_rotated = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
fig2, ax2 = plt.subplots(figsize=(5, 6))
ax2.imshow(img_rotated)
ax2.set_title("90° rotation")
ax2.axis("off")
plt.show()
