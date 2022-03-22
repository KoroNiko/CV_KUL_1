import cv2 as cv
import numpy as np

from constants import * 

def add_subs(frame, msg):
    cv.putText(frame, msg, (10,frame.shape[0]-30), FONT, 1, (0, 255, 0), 2, cv.LINE_AA)
    return frame

def gray2bgr(frame):
    bgr_gray = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    bgr_gray[:,:,0] = bgr_gray[:,:,1] = bgr_gray[:,:,2] = frame
    return bgr_gray


def switch_to_grayscale(frame):
    gray =  cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return gray2bgr(gray)

def gaussian_filter(frame, kernel_size):
    return cv.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def bilateral_filter(frame, d, sigma):
    return cv.bilateralFilter(frame, d, sigmaColor=sigma, sigmaSpace=sigma) 
    
def grab_object_colorspace(frame, low_bounds, high_bounds, colorspace=False):
    if colorspace:
        frame = cv.cvtColor(frame, colorspace)
    mask = cv.inRange(frame, low_bounds, high_bounds)
    # res = cv.bitwise_and(fram, frame, mask=mask)
    return mask

def morphology_operations(frame, method='both'):
    frame = grab_object_colorspace(frame, low_bounds=LOWER_BOUND_HSV, high_bounds=UPPER_BOUND_HSV, \
        colorspace=cv.COLOR_BGR2HSV)
    if method == 'both':
        strel1 = cv.getStructuringElement(OPEN_SHAPE, OPEN_KERNEL)
        strel2 = cv.getStructuringElement(CLOSE_SHAPE, CLOSE_KERNEL)
        opening = cv.morphologyEx(frame, cv.MORPH_OPEN, strel1)
        processed_frame = cv.morphologyEx(opening, cv.MORPH_CLOSE, strel2)
    elif method == 'open':
        strel = cv.getStructuringElement(OPEN_SHAPE, OPEN_KERNEL)
        processed_frame = cv.morphologyEx(frame, cv.MORPH_OPEN, strel)
    elif method == 'erode':
        strel = cv.getStructuringElement(ERODE_CLOSE_SHAPE, ERODE_CLOSE_KERNEL)
        erosion = cv.erode(frame, EROSION_KERNEL, iterations=EROSION_ITERATIONS)
        processed_frame = cv.morphologyEx(erosion, cv.MORPH_CLOSE, strel)
    # get the differences introduced by processing the frame
    mask_xor = frame - processed_frame
    # get frame shape to init a BGR image
    sz = processed_frame.shape
    # init a new BGR image to mark the changes in color
    final_frame = np.zeros((sz[0], sz[1], 3), dtype=np.uint8)
    # B, G channels are same as thresholded image
    final_frame[:, :, 0] = final_frame[:, :, 1] = processed_frame
    # mark the changes in red color
    final_frame[:, :, 2] = cv.bitwise_xor(processed_frame, mask_xor)
    return final_frame


def sobel_edges(frame, edges='both', ksize=SOBEL_KERNEL_SIZE, scale=SOBEL_SCALING_FACTOR):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if edges == 'vertical':
        frame = cv.Sobel(frame, cv.CV_8U, 1, 0, ksize=SOBEL_KERNEL_SIZE)
        frame[frame < SOBEL_X_THRESH] = 0
    if edges == 'horizontal':
        frame = cv.Sobel(frame, cv.CV_8U, 0, 1, ksize=SOBEL_KERNEL_SIZE)
        frame[frame < SOBEL_Y_THRESH] = 0
    if edges == 'both':
        frame = cv.Sobel(frame, cv.CV_8U, 1, 1, ksize=SOBEL_KERNEL_SIZE, scale=scale)
        frame[frame < SOBEL_X_Y_THRESH] = 0
    frame = cv.applyColorMap(frame, SOBEL_COLORMAP)
    return frame

def hough_circles(frame, divisor, param, R):
    # convert to grayscale and apply some blurring to reduce noise and avoid false circle detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows/divisor , \
        param1=param[0], param2=param[1], \
        minRadius=R[0], maxRadius=R[1])
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, SHAPE_COLOR, 3)
    return frame


def template_matching(frame, template, gray_map=False):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    # Apply template matching
    res = cv.matchTemplate(gray, template, TM_METHOD)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if TM_METHOD in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(frame, top_left, bottom_right, SHAPE_COLOR, 2)
    if not gray_map:
        return frame
    res[res < 0] = 0
    new_res = res.copy()
    cv.normalize(new_res, new_res, 0, 255, cv.NORM_MINMAX)
    new_res = new_res.astype(np.uint8)
    scale_size = frame.shape[0:2][::-1]
    new_res = cv.resize(new_res, scale_size, interpolation=cv.INTER_NEAREST)
    return gray2bgr(new_res)

# borrowed from PyImageSearch.com
# non-max suppression to remove overlapping bounding boxes
def nms(boxes, overlapThresh):
    # if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def template_wildcard(frame, template, thresh):
    gray = gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(gray,template,cv.TM_CCOEFF_NORMED)
    loc = np.where( res >= thresh)
    box_list = []
    for pt in zip(*loc[::-1]):
        # pt is (x1, y1)
        # adding w, h gives is (x2, y2)
        box_list.append((pt[0], pt[1], pt[0]+w, pt[1]+h))

    box_array = np.array(box_list)
    boxes = nms(box_array, NMS_THRESH)
    for pt in boxes:
        cv.rectangle(frame, (pt[0], pt[1]), (pt[2], pt[3]), SHAPE_COLOR, 1)
    return frame, boxes

def change_local_colorspace_hsv(frame, boxes):
    if len(boxes) == 0:
        return frame
    for pt in boxes:
        to_convert = frame[pt[1]:pt[3], pt[0]:pt[2], :]
        frame[pt[1]:pt[3], pt[0]:pt[2], :] = cv.cvtColor(to_convert, cv.COLOR_BGR2HSV)
    return frame

def replace_matched_templates(frame, boxes, replacement):
    if len(boxes) == 0:
        return frame
    for pt in boxes:
        scale_size = (pt[2]-pt[0], pt[3]-pt[1])
        resized = cv.resize(replacement, scale_size, interpolation=cv.INTER_NEAREST)
        frame[pt[1]:pt[3], pt[0]:pt[2], :] = resized
    return frame

if __name__ == '__main__':
    pass