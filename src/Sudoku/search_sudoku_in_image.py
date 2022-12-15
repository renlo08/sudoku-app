import numpy as np
import cv2
import os
import imutils
from pathlib import Path
from skimage.segmentation import clear_border

import sys
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


def get_app_dir() -> Path:
    """ return absolute directory path to the application base directory"""
    app_dir = Path.cwd()

    if not hasattr(sys, "frozen"):
        i = 0
        while app_dir.name not in ["src", "Sudoku_papa"] and i < 50:
            app_dir = app_dir.parent
            i += 1
        if i == 50:
            raise Exception("Application dir can´t be determined")

        if app_dir.name == "src":
            app_dir = app_dir.parent

        return app_dir


def grid_capture(framename, file_path):
    # vidéo capture source camera (ici la webcam du portable)
    cap = cv2.VideoCapture(0)

    # return a single frame in variable `frame`
    while(True):
        result,frame = cap.read()
        cv2.imshow(framename, frame) #display the captured image

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(file_path, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): #save on pressing 'y'
            cv2.destroyAllWindows()
            break

    cap.release()


def visualise_image(file_path: Path, show=False):
    img = cv2.imread (str(file_path))
    if show:
        while True:
            cv2.namedWindow('Capture Sudoku', cv2.WINDOW_NORMAL)
            cv2.imshow('Capture Sudoku', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    return img


def resize_image(src, factor=0.7, save=True):
    w, h, d = src.shape
    # new_size = int(min(w, h) * factor)
    # new_dim = (new_size, new_size)
    resized = imutils.resize(src, width=600, height=600)
    # resized = cv2.resize(src, new_dim, interpolation=cv2.INTER_AREA)
    if save:
        app_dir = get_app_dir()
        cv2.imwrite(str(app_dir / 'Images' / 'resized_sudoku.png'), resized)
    return resized


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(src, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(src, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def find_puzzle(file_path, debug=False):

    # read the image
    image = cv2.imread(str(file_path))


    resized_image = resize_image(image)

    # convert in gray
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # filter noice
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # invert color
    thresh = cv2.bitwise_not(thresh)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the Sudoku puzzle so raise an error
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the puzzle
    puzzle = four_point_transform(resized_image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    if debug:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return puzzle, warped

def zoom_on_digit(digit: np.array):
    """ Recenter the image on digit """
    # find the 4 points of the ROI
    x_indexes, y_indexes = np.where(digit == 255)
    x_min, x_max = np.min(x_indexes), np.max(x_indexes)
    y_min, y_max = np.min(y_indexes), np.max(y_indexes)

    # slice the array
    return digit[x_min-1:x_max+1,y_min-1:y_max+1]



def extract_digit(cell, debug=False):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # check to see if we are visualizing the cell thresholding step
    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None

    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None

    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    digit = zoom_on_digit(digit)
    # check to see if we should visualize the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)

    # return the digit to the calling function
    return digit
