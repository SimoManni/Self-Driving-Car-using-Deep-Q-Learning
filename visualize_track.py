from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Open an image file
img = Image.open('track.png')

# Resize the image
img_resized = img.resize((800, 600))

line_coordinates = np.array([[396, 555, 392, 477],
                         [314, 480, 312, 558],
                         [253, 480, 244, 559],
                         [184, 550, 200, 470],
                         [165, 461, 102, 507],
                         [160, 441, 91, 412],
                         [176, 442, 189, 364],
                         [229, 373, 245, 450],
                         [237, 365, 310, 388],
                         [229, 349, 281, 291],
                         [175, 305, 237, 259],
                         [144, 226, 224, 229],
                         [232, 209, 176, 155],
                         [256, 199, 260, 123],
                         [286, 208, 329, 141],
                         [331, 247, 385, 194],
                         [379, 301, 428, 241],
                         [455, 261, 434, 337],
                         [482, 268, 517, 336],
                         [482, 255, 558, 268],
                         [470, 225, 547, 205],
                         [451, 166, 527, 163],
                         [531, 139, 463, 97],
                         [555, 129, 566, 51],
                         [582, 141, 628, 78],
                         [605, 172, 676, 148],
                         [618, 229, 691, 217],
                         [622, 288, 697, 284],
                         [624, 349, 701, 354],
                         [628, 411, 703, 414],
                         [621, 446, 684, 481],
                         [595, 464, 633, 532],
                         [539, 468, 550, 547],
                         [485, 472, 489, 547]])

# Plot the image
plt.figure(figsize=(10, 7.5))  # Adjust the figure size to match the aspect ratio
plt.imshow(img_resized)

for i, (x1, y1, x2, y2) in enumerate(line_coordinates):
    plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)
    # Add annotation next to the line
    plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(i), color='blue', fontsize=12, ha='center', va='center')


plt.show()