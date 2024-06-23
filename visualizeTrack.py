from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Open an image file
img = Image.open('track.png')

# Resize the image
img_resized = img.resize((800, 600))

line_coordinates = np.array([[184, 550, 200, 470],
                             [165, 461, 102, 507],
                             [160, 441, 105, 385],
                             [207, 450, 220, 371],
                             [237, 365, 315, 377],
                             [197, 325, 250, 271],
                             [148, 203, 224, 220],
                             [286, 208, 329, 141],
                             [379, 301, 428, 241],
                             [482, 265, 546, 310],
                             [451, 166, 527, 163],
                             [555, 129, 566, 51],
                             [605, 172, 676, 148],
                             [622, 288, 697, 284],
                             [628, 411, 703, 414],
                             [595, 464, 633, 532],
                             [485, 472, 489, 547],
                             [314, 480, 312, 558]])

# Plot the image
plt.figure(figsize=(10, 7.5))  # Adjust the figure size to match the aspect ratio
plt.imshow(img_resized)

for (x1, y1, x2, y2) in line_coordinates:
    plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)

plt.show()