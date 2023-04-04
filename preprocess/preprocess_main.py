#@title full preprocess procedure
# this file is for getting the experiment data
from image_process import *
from google.colab.patches import cv2_imshow
import skimage
import os


#remove_artifact: remove the artifact
#input: filename(full path), src_dir, dst_dir
#ouput: None
def remove_artifact(filename, src_dir, dst_dir):
  image = cv2.imread(filename)
  gray = apply_cvtgray(image)
  thresh = apply_thresh_otsu(gray)
  morph = apply_morph(thresh)
  mask = get_largest_area_mask(morph)
  artifact = apply_mask(image, mask)
  cv2.imwrite(os.path.join(dst_dir, 'artifact.jpg'), artifact)

#get_better_canny: this function is to get better edge detection for getting hough lines
# input: removed_artifact: the removed_artifact gray picture
def get_better_canny(removed_artifact, dst_dir):
  equ = apply_equalizeHist(removed_artifact)
  cv2.imwrite(os.path.join(dst_dir, "equ.jpg"), equ)
  equ = io.imread(os.path.join(dst_dir, "equ.jpg"))
  orient_flag, equ = right_orient_mammogram(equ)
  sobel_canny_equ = apply_canny_sobel(equ)
  io.imsave(os.path.join(dst_dir, "sobel_canny_equ.jpg"), sobel_canny_equ)
  sobel_canny_equ = cv2.imread(os.path.join(dst_dir, "sobel_canny_equ.jpg"))
  gray = apply_cvtgray(sobel_canny_equ)
  morph_sobel_canny_equ = apply_morph(gray)
  cv2.imwrite(os.path.join(dst_dir, "morph_sobel_canny_equ.jpg"), morph_sobel_canny_equ)
  morph_sobel_canny_equ = io.imread(os.path.join(dst_dir, "morph_sobel_canny_equ.jpg"))
  canny_morph_sobel_canny_equ = apply_canny(morph_sobel_canny_equ)
  return canny_morph_sobel_canny_equ, orient_flag

#mask_pectoral: use mask to remove pectoral muscle
def mask_pectoral(canny_morph_sobel_canny_equ, orient_flag, dst_dir):
  lines = get_hough_lines(canny_morph_sobel_canny_equ)
  pectoral = io.imread(os.path.join(dst_dir, "artifact.jpg"))
  #condition 1: Terrible! No straight lines is detected!
  if( len(lines)== 0):
    io.imsave(os.path.join(dst_dir, "pectoral.jpg"), pectoral)
    return
  if (orient_flag == True):
    pectoral = apply_flip(pectoral)
  shortlistLines = get_shortlist_lines(lines)
  #condition 2: Lucky! There exist shortlistLines
  if( len(shortlistLines) != 0 ):
    rr, cc = get_pectoral_mask(shortlistLines, pectoral)
    plt.imshow(pectoral)
    pectoral[rr, cc] = 0
  if( orient_flag == True ):
    pectoral = apply_flip(pectoral)
  io.imsave(os.path.join(dst_dir, "pectoral.jpg"), pectoral)

#add_border_denoise: adding black border to denoise
def add_border_denoise(filename, width = 400):
  border = cv2.imread(filename)
  border = add_border(border, width = 400)
  cv2.imwrite(filename, border)

def remove_pectoral(dst_dir):
  removed_artifact = cv2.imread(os.path.join(dst_dir, "artifact.jpg"), 0)
  canny_morph_sobel_canny_equ, orient_flag = get_better_canny(removed_artifact, dst_dir)
  mask_pectoral(canny_morph_sobel_canny_equ, orient_flag, dst_dir)


#preprocess_cc_file: preprocessing the CC image
#input: dst_dir, file_name(full path)
#output: None
def preprocess_cc_file(filename, src_dir, dst_dir):
  remove_artifact(filename, src_dir, dst_dir)
  basename = os.path.basename(filename)
  os.rename(os.path.join(dst_dir, 'artifact.jpg'), os.path.join(dst_dir, basename))
  add_border_denoise(os.path.join(dst_dir, basename))

def preprocess_mlo_file(filename, src_dir, dst_dir):
  remove_artifact(filename, src_dir, dst_dir)
  remove_pectoral(dst_dir)
  basename = os.path.basename(filename)
  os.rename(os.path.join(dst_dir, 'pectoral.jpg'), os.path.join(dst_dir, basename))
  add_border_denoise(os.path.join(dst_dir, basename))

#clean_temp_files: deleting all temp files including equ.jpg, morph_sobel_canny_equ.jpg, sobel_canny_equ.jpg
#input: dst_dir
def clean_temp_files(dst_dir):
  os.remove(os.path.join(dst_dir, "equ.jpg"))
  os.remove(os.path.join(dst_dir, "morph_sobel_canny_equ.jpg"))
  os.remove(os.path.join(dst_dir, "sobel_canny_equ.jpg"))


import glob
import os

show_step = 5

# read all jpg file
root_dir = '/content/drive/MyDrive/graduation_project/dataset/CBIS_DDSM/mass_train'
src_dir = os.path.join(root_dir, 'mass_train')
images_list =  glob.glob(os.path.join(src_dir, '*.jpg'))

# make dst_src
dst_dir = os.path.join(root_dir, 'clean_mass_train')
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

#begin to preprocess
file_cnt = 0
for filename in images_list:
  if( filename[-5]=='O' ):
    preprocess_mlo_file(filename, src_dir, dst_dir)
  if( filename[-5]=='C' ):
    preprocess_cc_file(filename, src_dir, dst_dir)
  file_cnt += 1
  if( file_cnt % show_step == 0):
    print("The amount of files processed is {}/{}".format(file_cnt, len(images_list)))
clean_temp_files(dst_dir)
print('The amount of files processed is ', file_cnt)


'''
somehting to improve:
1, write to argv

problem:
1, opencv_to_scikit, scikit_to_opencv will make image lose precision, which results the inaccuracy.
But many writing and reading increases the preprocessing time.

procedure of single picture:

'''