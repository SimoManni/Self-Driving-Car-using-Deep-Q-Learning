from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from settings import *

# Open an image file
img = Image.open('track.png')

# Resize the image
img_resized = img.resize((800, 600))

line_coordinates = CHECKPOINTS

# Compute mean points
starting_points = STARTING_POINTS

# Plot the image
plt.figure(figsize=(10, 7.5))  # Adjust the figure size to match the aspect ratio
plt.imshow(img_resized)

perpendicular_lines = []
for i, (x_m, y_m) in enumerate(zip(x_middle, y_middle)):
    index = (i + 1) % (len(x_middle))
    perpendicular_lines.append([x_m, y_m, x_middle[index], y_middle[index]])

perpendicular_lines = np.roll(np.array(perpendicular_lines), 1, axis=0)

angles = STARTING_ANGLES


for i, ((x1, y1, x2, y2), x_m, y_m, (x1_p, y1_p, x2_p, y2_p)) in enumerate(zip(line_coordinates, x_middle, y_middle, perpendicular_lines)):
    plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)
    plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(i), color='blue', fontsize=12, ha='center', va='center')

    if i < 1:
        plt.plot(starting_points[i][0], starting_points[i][1], 'o', color='yellow')

        plt.plot([x1_p, x2_p], [y1_p, y2_p], color='green', linewidth=2)
        plt.text((x1_p + x2_p) / 2, (y1_p + y2_p) / 2, str(angles[i]), color='white', fontsize=12, ha='center', va='center')



plt.show()