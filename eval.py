import os
import glob
from model import *
from data import *
import skimage.io as io
import skimage.transform as trans
import cv2
import numpy as np
from skimage import img_as_ubyte

def eval_result(predict_img , true_img):
    h,w = predict_img.shape
    (thresh, im_true_bw) = cv2.threshold(true_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    predict_img = predict_img > 0 
    true_img = true_img > 0
    real_point = np.sum(true_img)
    predict_point = 0
    
    for i in range(h):
        for j in range(w):
            if predict_img[i][j] == 1 and predict_img[i][j] == true_img[i][j]:
                predict_point += 1

    return predict_point/real_point

if __name__ == "__main__":
        
    img_path = '/Users/naveen/Studies/Bishops/Rust_detection/data/images'
    model_dir = '/Users/naveen/Studies/Bishops/Rust_detection/trained_weights/unet.hdf5'
    anno_path = '/Users/naveen/Studies/Bishops/Rust_detection/data/annotations'
    target_size = (256,256)

    model = unet()
    model.load_weights(model_dir)
    print("load success")

    imglist = sorted(glob.glob('{}/*'.format(img_path)))
    precisions = []
    for img in imglist:
        basename = os.path.splitext(os.path.basename(img))[0]
        anno = os.path.join(anno_path, basename + '.png')
        anno_img = cv2.imread(anno, cv2.IMREAD_GRAYSCALE)

        img = cv2.imread(img)
        h, w, c = img.shape
        img = cv2.resize(img, target_size)
        img = img / 255
        img = img[:,:,::-1]
        img = np.reshape(img,(1,)+img.shape)

        img_out = model.predict(img)

        img_result = img_out[0]
        img = img_result[:,:,0]
        cv_image = img_as_ubyte(img)
        cv_image = cv2.resize(cv_image, (w,h), interpolation=cv2.INTER_CUBIC)
        precision = eval_result(cv_image, anno_img)
        precisions.append(precision)

    precisions = np.mean(np.array(precisions))
    print(np.mean(precisions))
    print('Average Acurracy : {}%'.format(round((precisions * 100),3)))
    