import os
import glob
import cv2

# image = '/Volumes/Work/Project/unet_segmentation/Test Image/image.png'
DIR = '/Users/naveen/Studies/Bishops/Rust_detection/data/images'
DES = '/Users/naveen/Studies/Bishops/Rust_detection/data/annotations'

imglist = os.listdir(DIR)
for img_name in imglist:
    img_path = os.path.join(DIR, img_name)
    img_des = os.path.join(DES, img_name)

    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    mask1 = cv2.inRange(img_hsv, (9,95,95), (17,255,255))

    cv2.imwrite(img_des, mask1)
