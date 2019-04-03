import numpy as np
import cv2
import imutils
def classifyKey(keyImage):
    keyImage[keyImage > 0] = 255
    # negative
    keyImage = 255 - keyImage
    keyImage = np.uint8(keyImage)
    contours = cv2.findContours(keyImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    if len(contours) > 0:
        # select contnour with biggest area
        contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        # find convex hull
        contour = cv2.convexHull(contour)
        # calculate key height
        top = min(contour[:,0,1])
        bottom = max(contour[:,0,1])
        height = bottom - top
        if height > 80:
            return "Violin key"
        else:
            return "Bass key"
    else:
        return "Cannot classify"
