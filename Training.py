from TextureExtraction.ProsesEktraksiDanPreproses import LBP,GLCM
import glob
from PIL import Image
from itertools import chain
import cv2
import csv
import pandas as pd
import imageio

headerLBP = 'Label,Fitur0,Fitur1,Fitur2,Fitur3,Fitur4,Fitur5,Fitur6,Fitur7,Fitur8,Fitur9'
headerGLCM = 'Label,Path,correlation0,homogeneity0,contrast0,energy0,correlation0,homogeneity0,contrast0,energy0,correlation0,homogeneity0,contrast0,energy0,correlation0,homogeneity0,contrast0,energy0'
# Menulis index dataset pada file csv
output = open("Dataset Index/_index_test_GLCM_.csv", "w")
#data = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#data.writerow(header)
output.write("%s\n" %(headerGLCM))
# looping, untuk membaca semua citra pada direktori dataset
'''Training'''
'''
for Label in glob.glob("Dataset/*"):
    for imagePath in glob.glob(Label+"/*"):
        imageID = imagePath[imagePath.rfind("/") + 1:]
        
        # Fungsi LBP
        # image = imageio.imread(imagePath)
        # print(imagePath)
        # features = LBP(image)

        # Fungsi GLCM
        image = Image.open(imagePath).convert('L')
        features = GLCM(image)

        # Feature
        features = [str(f) for f in features]
        label = Label[8:]
        print(imagePath,"Fitur : ",features)
        output.write("%s,%s\n" % (label, ",".join(features)))
'''
'''Testing'''

for Label in glob.glob("Testing/*"):
    for imagePath in glob.glob(Label + "/*"):
        imageID = imagePath[imagePath.rfind("/") + 1:]
        print(imagePath)
        
        # Fungsi LBP
        # image = imageio.imread(imagePath)
        # features = LBP(image)
        # features = [str(f) for f in features]
        
        # Fungsi GLCM
        image = Image.open(imagePath).convert('L')
        features = GLCM(image)
        features = [str(f) for f in features]
        
        # Feature
        label = Label[8:]
        output.write("%s,%s\n" % (str(label) + "," + str(imagePath), ",".join(features)))

output.close()
