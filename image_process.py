#this file is for storing the function of image processing from cs2, skimage, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab as pylab
from matplotlib import image
from skimage import io
from skimage import color
from google.colab.patches import cv2_imshow
import cv2
from skimage.transform import hough_line, hough_line_peaks
from skimage.util import random_noise
from skimage.feature import canny
from skimage.filters import sobel
from skimage.draw import polygon

#right_orient_mammogram: orient for pectoral removing
def right_orient_mammogram(image):
    orient_flag = False
    left_nonzero = cv2.countNonZero(image[:, 0:int(image.shape[1]/2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1]/2):])
    if(left_nonzero < right_nonzero):
        image = cv2.flip(image, 1)
        orient_flag = True
    return orient_flag, image

def apply_cvtgray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
def apply_flip(image):
    return cv2.flip(image, 1)

def apply_thresh_otsu(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 

#this function will apply morphlogy operation
#procedure: morph_close --> morph_open -->morph_dilate
def apply_morph(thresh):
    # apply morphology close to remove small regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # apply morphology open to separate breast from other regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # apply morphology dilate to compensate for otsu threshold not getting some areas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29,29))
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel)
    return morph

def get_largest_area_mask(morph):
    #get largest area contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    big_contour_area = cv2.contourArea(big_contour)
    # draw all contours but the largest as white filled on black background as mask
    hh, ww = morph.shape[:2]
    mask = np.zeros((hh,ww), dtype=np.uint8)
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area != big_contour_area:
            cv2.drawContours(mask, [cntr], 0, 255, cv2.FILLED)    
    # invert mask
    mask = 255 - mask
    return mask

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)



def apply_equalizeHist(image):
     return cv2.equalizeHist(image)

def apply_canny_sobel(image):
    return sobel(canny(image, 1))

def apply_cvtgray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_canny(image):
    return canny(image)

#get_hough_lines
#notice: the point1, point2 may out of image size,
# which needs to be considered in remove_pectoral function
def get_hough_lines(canny_image):
    h, theta, d = hough_line(canny_image)
    lines = list()
    #print('\nAll hough lines')
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        #print("Angle: {:.2f}, Dist: {:.2f}".format(np.degrees(angle), dist))
        x1 = 0
        y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
        x2 = canny_image.shape[1] - 1
        y2 = (dist - x2 * np.cos(angle)) / np.sin(angle)
        lines.append({
            'dist': dist,
            'angle': np.degrees(angle),
            'point1': [x1, y1],
            'point2': [x2, y2]
        })
    return lines

#remove pectoral region
def get_pectoral_mask(shortlisted_lines, image):
    shortlisted_lines.sort(key = lambda x: x['dist'])
    pectoral_line = shortlisted_lines[0]
    d = pectoral_line['dist']
    theta = np.radians(pectoral_line['angle'])
    y1 = 0
    x1 = d/np.cos(theta)
    if ( x1>=image.shape[1] ):
      x1 = image.shape[1] - 1
      y1 = (d - x1 * np.cos(theta))/ np.sin(theta)
    x2 = 0
    y2 = d/np.sin(theta)
    #condition 1: it forms a quadrilateral mask
    if( y2 >= image.shape[0]):
        y2 = image.shape[0] - 1
        x2 = (d - y2 * np.sin(theta))/np.cos(theta)
        #return polygon([0, y1, y2], [0, x1, x2])
        return polygon([0, y1, y2, image.shape[0] - 1], [0, x1, x2, 0])
    #condition 2: it forms a triangle mask
    return polygon([0, y1, y2], [0, x1, x2])

#shortlisting lines
def get_shortlist_lines(lines, min_angle = 10, max_angle = 45):
    MIN_ANGLE = min_angle
    MAX_ANGLE = max_angle
    shortlisted_lines = [x for x in lines if
                          (x['angle']>=MIN_ANGLE) &
                          (x['angle']<=MAX_ANGLE)
                        ]
    '''
    print('\nShorlisted lines')
    for i in shortlisted_lines:
        print("Angle: {:.2f}, Dist: {:.2f}".format(i['angle'], i['dist']))
    '''
    return shortlisted_lines

#add_border: to add black border to the picture to denoise
#width: the width of border
def add_border(image, width = 40):
    hh, ww = image.shape[:2]
    # shave 40 pixels all around
    image = image[width:hh-width, width:ww-width]
    # add 40 pixel black border all around
    return cv2.copyMakeBorder(image, width,width,width,width, cv2.BORDER_CONSTANT, value=0)

#opencv_to_scikit: convert opencv_image to scikit image for RGB
def opencv_to_scikit(opencv_image):
  return cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

#scikit_to_opencv: convert scikit image to opencv for RGB
def scikit_to_opencv(scikit_image):
  return cv2.cvtColor(scikit_image, cv2.COLOR_RGB2BGR)

#opencv_to_scikit_gray: convert opencv image to scikit for Gray
def opencv_to_scikit_gray(opencv_image):
  scikit_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2RGB)
  return skimage.color.rgb2gray(scikit_image)

#scikit_to_opencv_gray: convert scikit image to opencv for Gray
def scikit_to_opencv_gray(scikit_image):
  scikit_image = skimage.color.gray2rgb(scikit_image)
  opencv_image = scikit_to_opencv(scikit_image)
  return cv2.cvtColor(scikit_image, cv2.COLOR_RGB2GRAY)