import cv2
import numpy as np

from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray

from scipy.ndimage import interpolation as inter
from PIL import Image as im

from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import imutils

class Preprocessor:
    def __init__(self, im_width=1000):
        self.im_width = im_width
    
    def preprocess(self, img):
        """de-skew, de-noise, adjust lighting, change to black and white, etc.

        Args:
            img (ndarray): 3-channel image

        Returns:
            ndarray: image for detecion
            ndarray: image for mixing
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.remove_background(img)
        img = self.binarize_and_denoise(img)
        img = self.deskew(img)
        img = imutils.resize(img, width = self.im_width)
        return img, img.copy()

    def binarize(self, img):
        grey_scale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # adaptive threshold gaussian with denoise
        grey_scale_img = cv2.fastNlMeansDenoising(grey_scale_img, None, 5, 7, 15)
        grey_scale_img = cv2.adaptiveThreshold(grey_scale_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #grey_scale_img contains Binary image
        return cv2.cvtColor(grey_scale_img, cv2.COLOR_GRAY2RGB)
    
    def deskew(self, img):
        # grey_scale_img = rgb2gray(img)
        grey_scale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # convert to binary
        ht, wd = grey_scale_img.shape[0], grey_scale_img.shape[1]
        # pix = np.array(img.convert('1').getdata(), np.uint8)
        # bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
        grey_scale_img = grey_scale_img.reshape((ht, wd)) / 255.0
        def find_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            hist = np.sum(data, axis=1)
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            return hist, score
        delta = 0.5
        limit = 10
        angles = np.arange(-limit, limit+delta, delta)
        scores = []
        for angle in angles:
            hist, score = find_score(grey_scale_img, angle)
            scores.append(score)
        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        print('Best angle: {}'.format(best_angle))
        # correct skew
        data = inter.rotate(grey_scale_img, best_angle, reshape=False, order=0)
        # img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
        # img = im.fromarray((255 * data).astype("uint8"))
        img = cv2.cvtColor((255 * data).astype("uint8"), cv2.COLOR_GRAY2RGB)
        return img
    
    def remove_background(self, img):
        # load the image and compute the ratio of the old height
        # to the new height, clone it, and resize it
        image = imutils.resize(img, width = 3000)
        ratio = image.shape[1] / 500.0
        orig = image.copy()

        image = imutils.resize(image, width = 500)
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour


        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        # loop over the contours
        screenCnt = None
        for c in cnts:
            # approximate the contour
            if cv2.contourArea(c) < 65000:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is not None:
            # apply the four point transform to obtain a top-down
            # view of the original image
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        else:
            warped = orig.copy()
        
        return warped
    
    def binarize_and_denoise(self, img):
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        warped = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        T = threshold_local(warped, 101, offset = 10, method = "gaussian")
        warped = (warped > T).astype("uint8") * 255
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
        return warped

