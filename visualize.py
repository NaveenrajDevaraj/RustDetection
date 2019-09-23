import os
import glob
from model import *
from data import *
import skimage.io as io
import skimage.transform as trans
import cv2
import numpy as np
from skimage import img_as_ubyte
import colorsys

if __name__ == "__main__":
    

    img_path = '/Users/naveen/Studies/Bishops/Rust_detection/data/images'
    model_dir = '/Users/naveen/Studies/Bishops/Rust_detection/trained_weights/unet.hdf5'
    save_dir = '/Users/naveen/Studies/Bishops/Rust_detection/data/result'
    target_size = (256,256)

    model = unet()
    model.load_weights(model_dir)
    print("load success")

    
    imglist = sorted(glob.glob('{}/*'.format(img_path)))
    for img in imglist:
        basename = os.path.splitext(os.path.basename(img))[0]
        image = cv2.imread(img)
        h,w = image.shape[:2]

        img = cv2.resize(image, target_size)
        img = img / 255
        img = img[:,:,::-1]

        img = np.reshape(img,(1,)+img.shape)

        img_out = model.predict(img)

        img_result = img_out[0]
        img = img_result[:,:,0]
        cv_image = img_as_ubyte(img)
        cv_image = cv2.resize(cv_image, (w,h), interpolation=cv2.INTER_CUBIC)
        cv_image = (cv_image > 0) * 1

        for i in range(h):
            for j in range(w):
                if cv_image[i][j] == 1:
                    image[i][j] = np.array([0,255,0])

        cv2.imwrite('{}/{}_result.png'.format(save_dir, basename),image)
        # io.imsave('{}/{}_result.png'.format(save_dir, basename),img)
