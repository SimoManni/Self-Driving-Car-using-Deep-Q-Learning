import cv2
import numpy as np
import matplotlib.pyplot as plt

## Script to discretize the track contour as a collection of lines

# Load the image
image = cv2.imread('track.png')
image = cv2.resize(image, (800, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Extract contours with minimum length to prevent other lines except for the contours of the track to be selected
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
min_contour_length = 200
long_contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_contour_length]

# Outer and inner contour of the track
epsilon_outer = 0.001 * cv2.arcLength(long_contours[0], True)
approx_contours_outer = cv2.approxPolyDP(long_contours[0], epsilon_outer, True)
approx_contours_outer = np.squeeze(approx_contours_outer)

epsilon_inner = 0.001 * cv2.arcLength(long_contours[-1], True)
approx_contours_inner = cv2.approxPolyDP(long_contours[-1], epsilon_outer, True)
approx_contours_inner = np.squeeze(approx_contours_inner)

# Define line segments from contiguous points
np.savetxt('contour_points_outer.txt', approx_contours_outer, fmt='%d')
np.savetxt('contour_points_inner.txt', approx_contours_inner, fmt='%d')

# def extract_line_segments(contours):
#     contours = np.squeeze(contours)
#     line_segments = []
#     for i in range(len(contours)-1):
#         line_segments.append(np.concatenate((contours[i], contours[i+1])))
#     line_segments.append(np.concatenate((contours[-1], contours[0])))
#     return np.array(line_segments)
#
# line_segments_outer = extract_line_segments(approx_contours_outer)
# line_segments_inner = extract_line_segments(approx_contours_inner)
# print(f'Number of line segments (outer): {len(line_segments_outer)}')
# print(f'Number of line segments (inner): {len(line_segments_inner)}')
# def define_lines(line_segments):
#     lines_coeff = []
#     for (x1, y1, x2, y2) in line_segments:
#         if x2 - x1 != 0:
#             m = (y2 - y1) / (x2 - x1)
#             b = y1 - m * x1
#         else:
#             m = 1e10
#             b = -1e10
#         lines_coeff.append([m, b])
#
#     return lines_coeff
#
# line_coeff_outer = define_lines(line_segments_outer)
# line_coeff_inner = define_lines(line_segments_inner)

# np.savetxt('line_coeff_outer.txt', line_segments_outer, fmt='%d')
# np.savetxt('line_coeff_inner.txt', line_segments_inner, fmt='%d')

# Draw the approximated contours
approx_img = image.copy()
cv2.drawContours(approx_img, [approx_contours_outer, approx_contours_inner], -1, (0, 255, 0), 5)
plt.imshow(cv2.cvtColor(approx_img, cv2.COLOR_BGR2RGB))
plt.show()
