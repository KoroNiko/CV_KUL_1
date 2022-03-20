import numpy as np
import cv2 as cv
cap = cv.VideoCapture('higher_res.mp4')

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv.CAP_PROP_POS_MSEC)) < upper

def processing_operations(frame):
    # Convert image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return gray


while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    temp = cap.get(cv.CAP_PROP_POS_MSEC)
    print(temp)

    # check if we are in the designated video msec range
    if between(cap, 5000, 10000):
        frame = processing_operations(frame)


    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


# if __name__ == '__main__':
#     pass