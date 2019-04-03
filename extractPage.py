import cv2
import imutils
import numpy as np
from os import makedirs
from os.path import exists

def orderPoints(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transformToRectangle(image, pts):
    rect = orderPoints(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def preprocess(img):
    mean = np.mean(img)
    alpha = 1 - mean/510
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    # transform to LAB color scheme
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # split channels
    l, a, b = cv2.split(lab)
    # apply CLAHE to l channel
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # merge channels
    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def extractPage(directory, fileName, saveImages):
    original = cv2.imread(directory + "/" + fileName + ".jpg")
    ratio = original.shape[0] / 500.0
    downsized = imutils.resize(original, height = 500)
    
    gray = preprocess(downsized)
    gray = cv2.cvtColor(downsized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if not exists(directory + "results" + fileName):
        makedirs(directory + "/" + fileName)

    if saveImages:
        cv2.imwrite(directory + "/" + fileName + "/preprocessed.jpg", gray)
    # edged contains downsized blurrd image ready to find contours
    edged = cv2.Canny(gray, gray.min(), gray.max())
    if saveImages:
        cv2.imwrite(directory + "/" + fileName + "/cannyed.jpg", edged)

    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    # select 5 contours with biggest areas
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    # find contour that approximates to a tetragon
    cnts = []
    found = False
    for c in contours:
        c = cv2.convexHull(c)
        cnts.append(c)
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        if len(approx) == 4:
            pageContour = approx
            found = True
            break
    if not found:
        return original, False

    mask = np.zeros(edged.shape, np.uint8)
    cv2.drawContours(mask, [cnts[0]], -1, 255, -1)
    if saveImages:
        cv2.imwrite(directory + "/" + fileName + "/contours.jpg", mask)

    # transform page extracted from picture to a rectangle
    transformed = transformToRectangle(original, pageContour.reshape(4, 2) * ratio)
    (height, width, _) = np.shape(transformed)
    factor =  1492/width
    transformed = cv2.resize(transformed, (int(width*factor), int(height*factor)))
    if saveImages:
        cv2.imwrite(directory + "/" + fileName + "/transformed.jpg", transformed)
    return transformed, True
