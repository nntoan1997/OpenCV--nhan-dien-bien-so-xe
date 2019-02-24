import cv2
import numpy as np

    

def readimg(add_img):
    img = cv2.imread(add_img,1)
    h = int(img.shape[0]/2)
    w = int(img.shape[1]/2)
    img = cv2.resize(img,(w,h))
    return img

def showimg(img):
    cv2.imshow("img",img)
   
add_img = "bien/Car_ (37).jpg"
    
img = readimg(add_img)
showimg(img)

def preprocess(img):
    img_clone = img.copy()
    img_clone = cv2.cvtColor(img_clone,cv2.COLOR_BGR2GRAY)
    
    S1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,1),(1,0))
    S2 = cv2.getStructuringElement(cv2.MORPH_RECT,(6,1),(3,0))