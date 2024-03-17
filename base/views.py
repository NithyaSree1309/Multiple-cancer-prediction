from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
#from PIL import Image
import numpy as np
#from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
# Create your views here.
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    print(request)
    print (request.POST.dict())
    #print(request.FILES['filePath'])
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    #fs.save(fileObj.name,fileObj)
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    labels = {0: 'Brain_healthy', 1: 'Brain_cancer', 2: 'Kidney_healthy', 3: 'Kidney_cancer', 4: 'Lung_healthy', 5: 'Lung_cancer', 6: 'Oral_healthy',7: 'Oral_cancer'}
    image = cv2.imread(testimage)
    #image = cv2.resize(image, (100, 100))
    image = cv2.resize(image, (100, 100))
    # image = image/255.0
    # print(image1.shape)
    new_image = np.reshape(image, [1, 100, 100, 3])
    # ================== Show Prediction =================================
    model=load_model('model.h5',compile=False)
    prediction = model.predict(new_image)[0]
    # print(prediction1)
    pred =np.argmax(prediction)
    pred = labels[pred]
    context={'filePathName':filePathName,'predictedLabel':pred}
    return render(request,'index.html',context)
