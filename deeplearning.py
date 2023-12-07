import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import easyocr # Example path for Windows


model = tf.keras.models.load_model("C:\\Users\\rajna\\OneDrive\\Desktop\\License-Plate-Detection-Web-App\\static\\models\\object_detection_v2.h5")

def object_detection(path, filename):
    # read image
    image = load_img(path) #PIL Object
    image = np.array(image, dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path, target_size=(224,224))

    # data preprocessing
    image_arr_224 = img_to_array(image1)/255.0 # convert into array and normalize output
   
    # get the shape
    h,w,d = image.shape
    
    # resize
    test_arr = image_arr_224.reshape(1,224,224,3)
    
    # prediction
    coords = model.predict(test_arr)
    
    # denormalize
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    xmin, xmax, ymin, ymax = coords[0]
   
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)
    
    cv2.rectangle(image, pt1, pt2, (0,255,0), 5)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(f"static/predict/{filename}", image_bgr)
    
    return coords

def OCR(path, filename):
    img = np.array(load_img(path))
    coords = object_detection(path, filename)
    xmin, xmax, ymin, ymax = coords[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"static/roi/{filename}", roi_bgr)
    reader = easyocr.Reader(['en',])
    result = reader.readtext(roi)
    text = ' '.join([detection[1] for detection in result])
    return text

