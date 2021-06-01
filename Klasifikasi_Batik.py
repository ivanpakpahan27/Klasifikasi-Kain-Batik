import pandas as pd
import numpy as np
import xlsxwriter
from sklearn.neighbors import KNeighborsClassifier
row = 1
column = 0
workbook = xlsxwriter.Workbook('Dataset Index/KNN/Hasil_KNN(neighbors=1).xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "Target")
worksheet.write(0, 1, "Path")
worksheet.write(0, 2, "Fitur")
worksheet.write(0, 3, "Output")

dataTrain = pd.read_csv('Dataset Index/_index_train_GLCM_.csv',sep=',')
dataTest = pd.read_csv('Dataset Index/_index_test_GLCM_.csv',sep=',')
dataTrain.head()
#from sklearn.preprocessing import LabelEncoder
x = dataTrain.iloc[:,1:].values
y = dataTrain.iloc[:,0].values
xTest = dataTest.iloc[:,2:].values
xTarget = dataTest.iloc[:,0].values
xPath = dataTest.iloc[:,1].values

#le = LabelEncoder() #Untuk categorical ke numeric
#y  = le.fit_transform(y) #Untuk categorical ke numeric

classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(x,y)

y_pred = classifier.predict(xTest)
for i in range(len(y_pred)):
    worksheet.write(row, column, xTarget[i])
    column+=1
    worksheet.write(row, column, xPath[i])
    column += 1
    worksheet.write(row, column, str(xTest[i]))
    column += 1
    worksheet.write(row, column, str(y_pred[i]))
    column = 0
    row += 1
    print(xPath[i],xTest[i],y_pred[i])
workbook.close()




