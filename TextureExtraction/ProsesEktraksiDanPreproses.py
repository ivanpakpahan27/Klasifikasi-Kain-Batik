from skimage.feature import greycomatrix
from skimage import feature
import numpy as np
from itertools import chain
from PIL import ImageFilter
from PIL import Image


def Prepocessing(img):
    if (len(img.shape) > 2):
        img = img.astype(float)
        # RGB to grayscale convertion using Luminance
        img = img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.11
    else:
        img = img
    return img

def LBP(image,radius=2, sampling_pixels=8):
    img = Prepocessing(image)
    # converting to uint8 type for 256 graylevels
    img = img.astype(np.uint8)
    # normalize values can also help improving description
    i_min = np.min(img)
    i_max = np.max(img)
    if (i_max - i_min != 0):
        img = (img - i_min) / (i_max - i_min)
    # compute LBP
    lbp = feature.local_binary_pattern(img, sampling_pixels, radius, method="uniform")
    print(lbp)
    # LBP returns a matrix with the codes, so we compute the histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, sampling_pixels + 3), range=(0, sampling_pixels + 2))
    # normalization
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    print(hist)
    # return the histogram of Local Binary Patterns
    return (hist)

def GLCM(image):
    img_feature = []
    properties = ['correlation', 'homogeneity', 'contrast', 'energy']
    # Pertajam citra
    image = image.filter(ImageFilter.SHARPEN)
    # Konversi ke array dan berikan kustomisasi
    glcm = greycomatrix(np.array(image),distances=[1],angles=[0,35,90,135],levels=256, normed=False, symmetric=False)
    #print(glcm[:,:,0,0])
    for i in properties:
        fit = feature.greycoprops(glcm,i)
        fit = fit.tolist()
        img_feature.append(fit)
    # Flatten list
    glcm_feature = list(chain.from_iterable(img_feature))
    glcm_feature = list(chain.from_iterable(glcm_feature))
    return (glcm_feature)



