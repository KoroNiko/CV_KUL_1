import cv2 as cv

# user defined modules
import processing_functions
from constants import *

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv.CAP_PROP_POS_MSEC)) < upper

if __name__ == '__main__':
    # read template to match in grayscale (apple)
    template = cv.imread(TEMPLATE_FILE_NAME, 0)

    # fish templates (3 pics to improve accuracy)
    wildcard_template_1 = cv.imread(WILDCARD_TEMPLATE_1, 0)
    wildcard_template_2 = cv.imread(WILDCARD_TEMPLATE_2, 0)
    wildcard_template_3 = cv.imread(WILDCARD_TEMPLATE_3, 0)
    wildcard_replace = cv.imread(WILDCARD_REPLACEMENT_IMAGE, cv.IMREAD_COLOR)

    cap = cv.VideoCapture(VIDEO_FILE_NAME)
    fps = int(round(cap.get(cv.CAP_PROP_FPS)))
    print(fps)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # check if we are in the designated video msec range
        if between(cap, 0, 4000):
            if between(cap, 1000, 2000):
                frame = processing_functions.switch_to_grayscale(frame)
                txt = f'Grayscale'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 3000, 4000):
                frame = processing_functions.switch_to_grayscale(frame)
                txt = f'Grayscale'
                frame = processing_functions.add_subs(frame, txt)
        if between(cap, 4000, 8000):
            if between(cap, 4000, 4500):
                frame = processing_functions.gaussian_filter(frame, G_K+G_STEP*0)
                txt = f'Gaussian Blur: {G_K+G_STEP*0}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 4500, 5000):
                frame = processing_functions.gaussian_filter(frame, G_K+G_STEP*1)
                txt = f'Gaussian Blur: {G_K+G_STEP*1}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 5000, 5500):
                frame == processing_functions.gaussian_filter(frame, G_K+G_STEP*2)
                txt = f'Gaussian Blur: {G_K+G_STEP*2}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 5500, 6000):
                frame = processing_functions.gaussian_filter(frame, G_K+G_STEP*3)
                txt = f'Gaussian Blur: {G_K+G_STEP*3}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 6000, 6500):
                frame = processing_functions.gaussian_filter(frame, G_K+G_STEP*4)
                txt = f'Gaussian Blur: {G_K+G_STEP*4}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 6500, 7000):
                frame = processing_functions.gaussian_filter(frame, G_K+G_STEP*5)
                txt = f'Gaussian Blur: {G_K+G_STEP*5}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 7000, 7500):
                frame = processing_functions.gaussian_filter(frame, G_K+G_STEP*6)
                txt = f'Gaussian Blur: {G_K+G_STEP*6}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 7500, 8000):
                frame = processing_functions.gaussian_filter(frame, G_K+G_STEP*7)
                txt = f'Gaussian Blur: {G_K+G_STEP*7}'
                frame = processing_functions.add_subs(frame, txt)
        if between(cap, 8000, 12000):
            if between(cap, 8000, 8500):
                frame = processing_functions.bilateral_filter(frame, B_D+B_D_STEP*0, B_SIGMA + B_STEP*0)
                txt = f'Bilateral filter: {B_D+B_D_STEP*0}, {B_SIGMA + B_STEP*0}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 8500, 9000):
                frame = processing_functions.bilateral_filter(frame, B_D+B_D_STEP*1, B_SIGMA + B_STEP*1)
                txt = f'Bilateral filter: {B_D+B_D_STEP*1}, {B_SIGMA + B_STEP*1}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 9000, 9500):
                frame = processing_functions.bilateral_filter(frame, B_D+B_D_STEP*2, B_SIGMA + B_STEP*2)
                txt = f'Bilateral filter: {B_D+B_D_STEP*2}, {B_SIGMA + B_STEP*2}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 9500, 10000):
                frame = processing_functions.bilateral_filter(frame, B_D+B_D_STEP*3, B_SIGMA + B_STEP*3)
                txt = f'Bilateral filter: {B_D+B_D_STEP*3}, {B_SIGMA + B_STEP*3}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 10000, 10500):
                frame = processing_functions.bilateral_filter(frame, B_D+B_D_STEP*4, B_SIGMA + B_STEP*4)
                txt = f'Bilateral filter: {B_D+B_D_STEP*4}, {B_SIGMA + B_STEP*4}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 10500, 11000):
                frame = processing_functions.bilateral_filter(frame, B_D+B_D_STEP*5, B_SIGMA + B_STEP*5)
                txt = f'Bilateral filter: {B_D+B_D_STEP*5}, {B_SIGMA + B_STEP*5}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 11000, 11500):
                frame = processing_functions.bilateral_filter(frame, B_D+B_D_STEP*6, B_SIGMA + B_STEP*6)
                txt = f'Bilateral filter: {B_D+B_D_STEP*6}, {B_SIGMA + B_STEP*6}'
                frame = processing_functions.add_subs(frame, txt)
            elif between(cap, 11500, 12000):
                frame = processing_functions.bilateral_filter(frame, B_D+B_D_STEP*7, B_SIGMA + B_STEP*7)
                txt = f'Bilateral filter: {B_D+B_D_STEP*7}, {B_SIGMA + B_STEP*7}'
                frame = processing_functions.add_subs(frame, txt)
        if between(cap, 12000, 20000):
            if between(cap, 12000, 14000):
                frame = processing_functions.grab_object_colorspace(frame, low_bounds=LOWER_BOUND_BGR, \
                    high_bounds=UPPER_BOUND_BGR)
                frame = processing_functions.gray2bgr(frame)
                txt = f'Grab object in BGR'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 14000, 16000):
                frame = processing_functions.grab_object_colorspace(frame, low_bounds=LOWER_BOUND_HSV, \
                    high_bounds=UPPER_BOUND_HSV, colorspace=cv.COLOR_BGR2HSV)
                frame = processing_functions.gray2bgr(frame)
                txt = f'Grab object in HSV'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 16000, 17000):
                frame = processing_functions.morphology_operations(frame, method='open')
                txt = f'Open image'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 17000, 18000):
                frame = processing_functions.morphology_operations(frame, method='erode')
                txt = f'Erode image'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 18000, 20000):
                frame = processing_functions.morphology_operations(frame)
                txt = f'Open and then close image'
                frame = processing_functions.add_subs(frame, txt)
            # print(f'12-20 {frame.shape}')
        if between(cap, 20000, 25000):
            if between(cap, 20000, 20500):
                frame = processing_functions.sobel_edges(frame, edges='vertical')
                txt = f'Vertical Sobel edges'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 20500, 21000):
                frame = processing_functions.sobel_edges(frame, edges='horizontal')
                txt = f'Horizontal Sobel edges'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 21000, 22000):
                frame = processing_functions.sobel_edges(frame, edges='both', ksize=SOBEL_KERNEL_SIZE, \
                    scale=SOBEL_SCALING_FACTOR)
                txt = f'Sobel edges: ksize:{SOBEL_KERNEL_SIZE}, scale:{SOBEL_SCALING_FACTOR}'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 22000, 23000):
                frame = processing_functions.sobel_edges(frame, edges='both', ksize=SOBEL_KERNEL_SIZE, \
                    scale=SOBEL_SCALING_FACTOR+2)
                txt = f'Sobel edges: ksize:{SOBEL_KERNEL_SIZE}, scale:{SOBEL_SCALING_FACTOR+2}'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 23000, 24000):
                frame = processing_functions.sobel_edges(frame, edges='both', ksize=SOBEL_KERNEL_SIZE-2, \
                    scale=SOBEL_SCALING_FACTOR)
                txt = f'Sobel edges: ksize:{SOBEL_KERNEL_SIZE-2}, scale:{SOBEL_SCALING_FACTOR}'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 24000, 25000):
                frame = processing_functions.sobel_edges(frame, edges='both', ksize=SOBEL_KERNEL_SIZE-2, \
                    scale=SOBEL_SCALING_FACTOR+2)
                txt = f'Sobel edges: ksize:{SOBEL_KERNEL_SIZE-2}, scale:{SOBEL_SCALING_FACTOR+2}'
                frame = processing_functions.add_subs(frame, txt)
        if between(cap, 25000, 35000):
            if between(cap, 25000, 27000):
                frame = processing_functions.hough_circles(frame, divisor=HOUGH_ROW_DIVISOR[0], \
                    param=HOUGH_PARAM[0], R=HOUGH_R[0])
                txt = f'Hough circles: divisor:{HOUGH_ROW_DIVISOR[0]}, param:{HOUGH_PARAM[0]}, radius:{HOUGH_R[0]}'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 27000, 29000):
                frame = processing_functions.hough_circles(frame, divisor=HOUGH_ROW_DIVISOR[1], \
                param=HOUGH_PARAM[1], R=HOUGH_R[1])
                txt = f'Hough circles: divisor:{HOUGH_ROW_DIVISOR[1]}, param:{HOUGH_PARAM[1]}, radius:{HOUGH_R[1]}'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 29000, 31000):
                frame = processing_functions.hough_circles(frame, divisor=HOUGH_ROW_DIVISOR[2], \
                    param=HOUGH_PARAM[2], R=HOUGH_R[2])
                txt = f'Hough circles: divisor:{HOUGH_ROW_DIVISOR[2]}, param:{HOUGH_PARAM[2]}, radius:{HOUGH_R[2]}'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 31000, 33000):
                frame = processing_functions.hough_circles(frame, divisor=HOUGH_ROW_DIVISOR[3], \
                    param=HOUGH_PARAM[3], R=HOUGH_R[3])
                txt = f'Hough circles: divisor:{HOUGH_ROW_DIVISOR[3]}, param:{HOUGH_PARAM[3]}, radius:{HOUGH_R[3]}'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 33000, 35000):
                frame = processing_functions.hough_circles(frame, divisor=HOUGH_ROW_DIVISOR[4], \
                    param=HOUGH_PARAM[4], R=HOUGH_R[4])
                txt = f'Hough circles: divisor:{HOUGH_ROW_DIVISOR[4]}, param:{HOUGH_PARAM[4]}, radius:{HOUGH_R[4]}'
                frame = processing_functions.add_subs(frame, txt)
        if between(cap, 35000, 40000):
            if between(cap, 35000, 37000):
                frame = processing_functions.template_matching(frame, template)
                txt = f'Find the apple in the picture (bounded here)'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 37000, 40000):
                frame = processing_functions.template_matching(frame, template, gray_map=True)
                print(frame.shape)
                txt = f'probability map for apple location. white dot is apple center'
                frame = processing_functions.add_subs(frame, txt)
        if between(cap, 40000, 60000):
            if between(cap, 42000, 45000):
                frame, boxes = processing_functions.template_wildcard(frame, wildcard_template_1, TEMPLATE_THRESH)
                frame, boxes = processing_functions.template_wildcard(frame, wildcard_template_2, TEMPLATE_THRESH+0.1)
                frame, boxes =  processing_functions.template_wildcard(frame, wildcard_template_3, TEMPLATE_THRESH+0.1)
                txt = f'Try to track the fish using template matching! Initially not very good'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 45000, 53000):
                frame, boxes = processing_functions.template_wildcard(frame, wildcard_template_1, TEMPLATE_THRESH)
                frame = processing_functions.change_local_colorspace_hsv(frame, boxes)
                frame, boxes = processing_functions.template_wildcard(frame, wildcard_template_2, TEMPLATE_THRESH+0.1)
                frame = processing_functions.change_local_colorspace_hsv(frame, boxes)
                frame, boxes =  processing_functions.template_wildcard(frame, wildcard_template_3, TEMPLATE_THRESH+0.1)
                frame = processing_functions.change_local_colorspace_hsv(frame, boxes)
                txt = f'Convert tracked fish to HSV colorspace'
                frame = processing_functions.add_subs(frame, txt)
            if between(cap, 53000, 60000):
                frame, boxes = processing_functions.template_wildcard(frame, wildcard_template_1, TEMPLATE_THRESH)
                frame = processing_functions.replace_matched_templates(frame, boxes, wildcard_replace)
                frame, boxes = processing_functions.template_wildcard(frame, wildcard_template_2, TEMPLATE_THRESH+0.1)
                frame = processing_functions.replace_matched_templates(frame, boxes, wildcard_replace)
                frame, boxes =  processing_functions.template_wildcard(frame, wildcard_template_3, TEMPLATE_THRESH+0.1)
                frame = processing_functions.replace_matched_templates(frame, boxes, wildcard_replace)
                txt = f'replace tracked fish with fish emoji!'
                frame = processing_functions.add_subs(frame, txt)
                

        out.write(frame)
        # play the video
        # cv.imshow('frame', frame)

        # waitkey() argument is delay in msec
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()