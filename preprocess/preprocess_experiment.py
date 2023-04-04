#@title preprocess_experiment
# this file is for getting the experiment data
from image_process import *


#upload the test_data
'''
%cd /content
from google.colab import files
uploaded = files.upload()
'''


#this part is for remove artifact
image = cv2.imread("/content/test2_MLO.jpg")
gray = apply_cvtgray(image)
thresh = apply_thresh_otsu(gray)
morph = apply_morph(thresh)
mask = get_largest_area_mask(morph)
artifact = apply_mask(image, mask)

# save results
cv2.imwrite('/content/preprocess_test/artifact_thresh.jpg', thresh)
cv2.imwrite('/content/preprocess_test/artifact_morph.jpg', morph)
cv2.imwrite('/content/preprocess_test/artifact_mask.jpg', mask)
cv2.imwrite('/content/preprocess_test/artifact_result.jpg', artifact)


# get the test2_mammogram_equ_gray.jpg
removed_artifact = cv2.imread("/content/preprocess_test/artifact_result.jpg", 0)
equ = apply_equalizeHist(removed_artifact)
cv2.imwrite("/content/preprocess_test/equ.jpg", equ)

#get the canny_sigma1_sobel.jpg
equ = io.imread("/content/preprocess_test/equ.jpg")
orient_flag, equ = right_orient_mammogram(equ)
sobel_canny_equ = apply_canny_sobel(equ)
io.imsave("/content/preprocess_test/sobel_canny_equ.jpg", sobel_canny_equ)

#get the morph_sigma1.jpg
sobel_canny_equ = cv2.imread("/content/preprocess_test/sobel_canny_equ.jpg")
gray = apply_cvtgray(sobel_canny_equ)
morph_sobel_canny_equ = apply_morph(gray)
cv2.imwrite("/content/preprocess_test/morph_sobel_canny_equ.jpg", morph_sobel_canny_equ)


morph_sobel_canny_equ = io.imread("/content/preprocess_test/morph_sobel_canny_equ.jpg")
canny_morph_sobel_canny_equ = apply_canny(morph_sobel_canny_equ)
io.imsave("/content/preprocess_test/canny_morph_sobel_canny_equ.jpg", canny_morph_sobel_canny_equ)



lines = get_hough_lines(canny_morph_sobel_canny_equ)


pectoral = io.imread("/content/preprocess_test/artifact_result.jpg")
if (orient_flag == True):
  pectoral = apply_flip(pectoral)

fig1, ax1 = plt.subplots()
for line in lines:
  ax1.plot((line['point1'][0],line['point2'][0]), (line['point1'][1],line['point2'][1]), '-r')
ax1.imshow(morph_sobel_canny_equ)
fig1.savefig("/content/preprocess_test/lines.jpg")


shortlistLines = get_shortlist_lines(lines)
fig2, ax2 = plt.subplots()
for line in shortlistLines:
  ax2.plot((line['point1'][0],line['point2'][0]), (line['point1'][1],line['point2'][1]), '-r')
ax2.imshow(pectoral)
fig2.savefig("/content/preprocess_test/shortlistLines.jpg")

#Lucky! We get pectoral line
if( len(shortlistLines)!= 0):
  rr, cc = get_pectoral_mask(shortlistLines, pectoral)
  print("cuddly test")
  print("rr, cc", rr, cc)
  print("rr, cc shape", len(rr), len(cc))
  print("canny_morph_sobel_canny_equ shape", canny_morph_sobel_canny_equ.shape)
  plt.imshow(pectoral)
  pectoral[rr, cc] = 0


if( orient_flag == True ):
  pectoral = apply_flip(pectoral)
io.imsave("/content/preprocess_test/pectoral.jpg", pectoral)

border = cv2.imread("/content/preprocess_test/pectoral.jpg")
border = add_border(border, width = 400)
cv2_imshow(border)
cv2.imwrite("/content/preprocess_test/border.jpg", border)


#download the experiment result
from google.colab import files
!zip -r preprocess_test.zip preprocess_test/
files.download('/content/preprocess_test.zip')

#clean the preprocess_test files
'''
!cd /content/preprocess_test
!rm /content/preprocess_test/*.jpg
'''

