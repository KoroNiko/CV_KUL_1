import cv2 as cv
import numpy as np

# text fonts
FONT = cv.FONT_HERSHEY_SIMPLEX

# Filenames
VIDEO_FILE_NAME = r'pictures\cv_video.mp4'
OUTPUT_VIDEO_NAME = r'pictures\out.mp4'
TEMPLATE_FILE_NAME = r'pictures\template.png'
WILDCARD_TEMPLATE_1 = r'pictures\fish_temp1.png'
WILDCARD_TEMPLATE_2 = r'pictures\fish_temp2.png'
WILDCARD_TEMPLATE_3 = r'pictures\fish_temp3.png'
WILDCARD_REPLACEMENT_IMAGE = r'pictures\fish_replace.png'

# Shape drawing color
SHAPE_COLOR = (20, 255, 57)

# Gaussian kernel size and increasing step
G_K = 5
G_STEP = 4

# bilateral filter sigma (Increase in value means that farther pixel values are also taken into account)
# neighbourhood increases
B_D = 9
B_D_STEP = 1
B_SIGMA = 20
B_STEP = 50

# bgr bounds
LOWER_BOUND_BGR = np.array([4, 0, 80])
UPPER_BOUND_BGR = np.array([50, 50, 130])

# HSV bounds for apple grab
LOWER_BOUND_HSV = np.array([0, 150, 80])
UPPER_BOUND_HSV = np.array([180, 240, 190])

# Image opening and closing constants
OPEN_SHAPE = cv.MORPH_RECT
OPEN_KERNEL = (5, 5)
CLOSE_SHAPE = cv.MORPH_ELLIPSE
CLOSE_KERNEL = (49, 49)

# Image erosion constants
ERODE_CLOSE_SHAPE = cv.MORPH_ELLIPSE
ERODE_CLOSE_KERNEL = (100, 100)
EROSION_KERNEL = np.ones((5, 5), dtype=np.uint8)
EROSION_ITERATIONS = 2

# Sobel edge detector constants
SOBEL_KERNEL_SIZE = 5
SOBEL_SCALING_FACTOR = 1
SOBEL_X_THRESH = 192
SOBEL_Y_THRESH = 168
SOBEL_X_Y_THRESH = 64
SOBEL_COLORMAP = cv.COLORMAP_JET

# Hough circles constants
# 5 iterations
HOUGH_ROW_DIVISOR = [8, 8, 4, 4, 12]
HOUGH_PARAM = [(100, 30), (200, 40), (180, 30), (126, 35), (255, 20)]
HOUGH_R = [(1,30), (0, 0), (0, 240), (0, 120), (0, 100)]

# Template matching constants
TM_METHOD = cv.TM_CCOEFF_NORMED

# Wildcard part constants
TEMPLATE_THRESH = 0.6
NMS_THRESH = 0.2