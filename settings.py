import numpy as np
import cv2

### Parameters ###

WIDTH, HEIGHT = 800, 600
TOTAL_GAMETIME = 1000
N_EPISODES = 200
index = 10
N_CARS = 10

ALPHA = 0.005
GAMMA = 0.5
N_ACTIONS = 5
EPSILON = 1.0
EPSILON_END = 0.1
EPSILON_DEC = 0.995
BATCH_SIZE = 128
INPUT_DIMS = 9
REPLACE_TARGET = 25
MEM_SIZE = 25000
LR = 0.003
GOAL_REWARD = 1
LIFE_REWARD = 0
PENALTY = -1
MAX_DISTANCE = 156.5247584249853

## Defition of barriers ##

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
approx_contours_inner = cv2.approxPolyDP(long_contours[-1], epsilon_inner, True)
approx_contours_inner = np.squeeze(approx_contours_inner)

BARRIERS = [approx_contours_outer, approx_contours_inner]

### Definition of checkpoints ###

CHECKPOINTS = np.array([[396, 555, 392, 477],
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

# Definition of starting points and angles
x1 = CHECKPOINTS[:, 0]
y1 = CHECKPOINTS[:, 1]
x2 = CHECKPOINTS[:, 2]
y2 = CHECKPOINTS[:, 3]
x_middle = (x1 + x2) / 2
y_middle = (y1 + y2) / 2

starting_points = []
for i, (x_m, y_m) in enumerate(zip(x_middle, y_middle)):
    index = (i + 1) % (len(x_middle))
    x = (x_m + x_middle[index]) / 2
    y = (y_m + y_middle[index]) / 2
    starting_points.append([x, y])

STARTING_POINTS = np.roll(np.array(starting_points).astype(int), 1, axis=0)

angles = []
for i, (x_m, y_m) in enumerate(zip(x_middle, y_middle)):
    index = (i + 1) % (len(x_middle))
    angle = np.arctan2(y_middle[index] - y_m , x_middle[index] - x_m) * 180 / np.pi
    corrected_angle = (270 - angle % 360) if angle % 360 < 270 else (angle % 360)
    angles.append(corrected_angle)

STARTING_ANGLES = np.roll(np.array(angles).astype(int), 1)

index = 0
INIT_POS = STARTING_POINTS[index]
INIT_ANGLE = STARTING_ANGLES[index]

if index != 0:
    CHECKPOINTS = np.vstack((CHECKPOINTS[index:], CHECKPOINTS[:index]))
elif index == 0:
    CHECKPOINTS = CHECKPOINTS