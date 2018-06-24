# deep-photo-style-transfer-with-auto-segmentation
Introduction to Multimedia, Final Project  
Still Working.....  
Expect to finish on Jun. 24.  

## Feature
Provide a semantic segmentation tools which is fit to deep photo style transfer. (The method is based on the [original paper](https://arxiv.org/abs/1703.07511) with slightly change)

## Prepare
1. Clone the code and to fulfill the prequisites for both repositories. 
```
git clone https://github.com/LouieYang/deep-photo-styletransfer-tf
git clone https://github.com/hellochick/semantic-segmentation-tensorflow
```
2. Then replace files in semantic-segmentation-tensorflow by the files we provided.

## Usage
1. Run semantic segmentation for style image
```
python3 inference.py --img-path <path to style img> --model "pspnet" --isStyle 1
```
  Then it will generate a segmetation image and print a string with 9 char, which is for style_color.
  
  p.s. meaning of style_color string: first 8 chars: whether color[i] is used or not. last char: color[j], 0<=j<=7 is the most used color in semantic segmentation  
  e.g. 111111003 -> color[0:5] is used, color[6:7] did not appear in style image segmentation. color[3] is the color with largest area in segmentation.  
  
2. Run semantic segmentation for content image
```
python3 inference.py --img-path <path to content img> --model "pspnet" --isStyle 0 --style_color  <style_color string>
```
3. Run deep photo style transfer
```
python deep_photostyle.py --content_image_path <path_to_content_image> --style_image_path <path_to_style_image> --content_seg_path <path_to_content_segmentation> --style_seg_path <path_to_style_segmentation> --style_option 2 --serial <output path>
```
## Alternative way to run
Working....
```
python3 combine.py --content_image_path <path_to_content_image> --style_image_path <path_to_style_image> --serial <output path>
```
Then it will generate a set of images after deep photo style transfer automatically
