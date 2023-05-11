# mammogram mass detection
<div> 
This project focuses on detecting the mass for mammogram based on Mask RCNN models using CBIS-DDSM dataset. Considering the small size of dataset, we preprocess it carefully and generate clean data using opencv and scikit. Finally, we use Mask-RCNN pretrained on balloon dataset to train based on detectron2.
</div>

## preprocess 
The preprocess includes removing artifact, removing pectoral and adding black border based on opencv and scikit. Some outcomes are showed below. (The figures of outcome)

<details>
  <summary>methods</summary>
  
  1. cv2.equalizeHist()
  2. skimage.feature.canny()
  3. cv2.morphologyEx()
  4. skimage.filter.sobel()
</details>

<details>
  <summary>procedures</summary>
  1. remove_artifact: image —> gray image —> (cv2.THRESH_OTSU) thresh —>  (cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_DILATE,cv2.morphologyEx) morph —> (get_largest_area)mask —> remove artifact <br />
  2. remove_pectoral: image removed artifact —> orient —> equalHist —> canny detection —> sobel —> morphological operation —> canny edge detection <br />
  3. add_border_denoise
</details>

<details>
  <summary>usage</summary>
  1. adjust your directory like this: <br />
  |--CBIS-DDSM <br />
  |&emsp  |--mass_train <br />
  |&emsp &emsp    |--mass_train <br />
  | &emsp  |--mass_test <br />
  |  &emsp &emsp   |--mass_test<br />
  2. adjust the original directory of CBIS-DDSM and run it
</details>

### preprocess test
Here are some test results and if you want to see more, you can read from the direcotry of /preprocess/preprocess_test_result
original figure            |  after thresholding and morph | after removing artifact
:-------------------------:|:-------------------------:|:-------------------------:
| ![image1](https://github.com/Mia-code112233/mammogram-mass-detection/blob/master/preprocess/preprocess_test_result/preprocess_test2/test2_MLO.jpg) | ![image2](https://github.com/Mia-code112233/mammogram-mass-detection/blob/master/preprocess/preprocess_test_result/preprocess_test2/artifact_morph.jpg) | ![image3](https://github.com/Mia-code112233/mammogram-mass-detection/blob/master/preprocess/preprocess_test_result/preprocess_test2/artifact_result.jpg)

after equalHist,canny,sobel|  after morph                 | after canny
:-------------------------:|:-------------------------:|:-------------------------:
| ![image4](https://github.com/Mia-code112233/mammogram-mass-detection/blob/master/preprocess/preprocess_test_result/preprocess_test2/sobel_canny_equ.jpg) | ![image5](https://github.com/Mia-code112233/mammogram-mass-detection/blob/master/preprocess/preprocess_test_result/preprocess_test2/morph_sobel_canny_equ.jpg) | ![image6](https://github.com/Mia-code112233/mammogram-mass-detection/blob/master/preprocess/preprocess_test_result/preprocess_test2/canny_morph_sobel_canny_equ.jpg)

after hough line detection    |  after selecting 
:-------------------------:|:-------------------------:
| ![image7](https://github.com/Mia-code112233/mammogram-mass-detection/blob/master/preprocess/preprocess_test_result/preprocess_test2/lines.jpg)| ![image8](https://github.com/Mia-code112233/mammogram-mass-detection/blob/master/preprocess/preprocess_test_result/preprocess_test2/shortlistLines.jpg) 
## train

## other
If you like it, please star it.❤️

## references



