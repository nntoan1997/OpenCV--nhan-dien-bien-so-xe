import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics

#img_clone anh xam
#img anh goc

#img_plate anh bien so


img = cv2.imread("bien/Car_ (37).jpg",1)
w = int(img.shape[1]/2)
h = int(img.shape[0]/2)
img = cv2.resize(img,(w,h))

#tien xu ly anh
img_clone = img.copy()
img_clone = cv2.cvtColor(img_clone,cv2.COLOR_BGR2GRAY)


S1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,1),(1,0))
S2 = cv2.getStructuringElement(cv2.MORPH_RECT,(6,1),(3,0))


#print(type(S1))
#print(type(S2))

w = img_clone.shape[1]
h = img_clone.shape[0]

img_pyrdown = cv2.pyrDown(img_clone)
mImg = cv2.morphologyEx(img_pyrdown,cv2.MORPH_BLACKHAT,S2)
#cv2.imshow("pre mImg",mImg)
mImg = cv2.normalize(mImg,mImg,0,255,cv2.NORM_MINMAX)

# nhi phan hoa anh mImg

ret,threshold = cv2.threshold(mImg,int (10*cv2.mean(mImg)[0]),255,cv2.THRESH_BINARY)
#cv2.imshow("threshold",threshold)

mini_thresh = threshold.copy()
#cv2.imshow("mini_thresh", mini_thresh)

#loc anh

r = int (mini_thresh.shape[0])
d = int (mini_thresh.shape[1])

dst = np.zeros((r,d))
for i in range(0,r-32,4):
    for j in range(0,d-16,4):

        rect = mini_thresh[i:i+16,j:j+8]
        nonZero1 = cv2.countNonZero(rect)

        rect = mini_thresh[i+16:i+32, j:j + 8]
        nonZero2 = cv2.countNonZero(rect)

        rect = mini_thresh[i:i + 16, j+8:j + 16]
        nonZero3 = cv2.countNonZero(rect)

        rect = mini_thresh[i+16:i + 32, j+8:j + 16]
        nonZero4 = cv2.countNonZero(rect)


        cnt = 0;
        if nonZero1 > 3:
            cnt=cnt+1;
        if nonZero2 > 3:
            cnt=cnt+1;
        if nonZero3 > 3:
            cnt=cnt+1;
        if nonZero4 > 3:
            cnt=cnt+1;

        if cnt > 2:
            rect = mini_thresh[i:i+32,j:j+16]
            dst[i:i+32,j:j+16] = rect

dst_clone = dst.copy()
dst_clone = np.uint8(dst_clone)

S0 = None
dst_clone = cv2.dilate(dst_clone,S0,2)
dst_clone = cv2.erode(dst_clone,S0,2)
dst_clone = cv2.dilate(dst_clone,S1,iterations=9)
dst_clone = cv2.erode(dst_clone,S1,iterations=10)
dst_clone = cv2.dilate(dst_clone,S0)

cv2.imshow("dst_clone",dst_clone)
#cv2.imshow("dst",dst)

#cv2.imshow("pyrDown",img_pyrdown)
#cv2.imshow("mImg",mImg)
cv2.imshow("anh goc", img)

#tim plate va cat
# contour - anh tim duong bao
# grayImg - anh xam
# tmpImg - anh the


#Ve bien
grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
w = img.shape[1]
h = img.shape[0]
tmp = np.zeros(shape=(h,w))
countour = grayImg.copy()
countour = cv2.normalize(countour,countour,0,255,cv2.NORM_MINMAX)
dst_clone = cv2.resize(dst_clone,(w,h))
#cv2.imshow("ds",dst_clone)
_,storePlate,_ =  cv2.findContours(dst_clone,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#print(storePlate)
img_plate = list()
j = 0
for i in range(0,len(storePlate)):
    cnt = storePlate[i]
    x,y,w,h = cv2.boundingRect(cnt)
    s = w*h
#    print(s)
    r = w/h
 #   print(r)
    if s > 5000 and s <20000:
        if r > 3 and r< 5:
            img_plate.append(img[y-3:y + h+3, x-3:x + w+3])
            j = j + 1
            tmp = cv2.rectangle(tmp,(x-5,y-5),(x+w+5,y+h+5),(0,0,255),2)
            img = cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(0,0,255),2)

tapbien = list()

for i in range(0,len(img_plate)):
    count = 0
    img_tmp = img_plate[0]
    img_tmp = cv2.resize(img_tmp,(400,80))
    img_gray = cv2.cvtColor(img_tmp,cv2.COLOR_BGR2GRAY)

    img_binary = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,1)

    _,store,_ = cv2.findContours(img_binary,1,2)
    #print(store)

    for j in range(0,len(store),1):
        cnt = store[j]
        #img_tmp = cv2.drawContours(img_tmp,store,-1,(0,0,0),3)
        #zero = cv2.drawContours(zero, store, -1, (0, 0, 0), 3)

        x,y,w,h = cv2.boundingRect(cnt)
        print(x,y,w,h)
        if w > 20 and w<90 and h >50 and h <90 and h*w >1000:
            count = count + 1
            tapbien.append(img_tmp[y+1:y + h-1,x+1:x + w-1])
            img_tmp = cv2.rectangle(img_tmp, (x, y), (x + w, y + h), (255, 0, 0), 1)

    if count < 5:
        list.clear()


cv2.imshow("img",img_tmp)
cv2.imshow("img_binary0",img_binary)
#cv2.imshow("tmp",tmp)
cv2.imshow("sau xu ly",img)
cv2.imshow("plate",img_plate[0])


img_plate.clear()
for i in range(0,len(tapbien),1):
    tmp = cv2.cvtColor(tapbien[i],cv2.COLOR_BGR2GRAY)
    tmp = cv2.adaptiveThreshold(tmp,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,1)
    img_plate.append(tmp)

cv2.imshow("1",img_plate[0])
cv2.imshow("2",img_plate[1])
cv2.imshow("3",img_plate[2])
cv2.imshow("4",img_plate[3])
cv2.imshow("5",img_plate[4])
cv2.imshow("6",img_plate[5])
cv2.imshow("7",img_plate[6])


#du lieu training
digits = dataset.load_digits()
images_and_labels = list(zip(digits.image, digits.target))


#KNN training






#loc bien
#tim img_bien chinh xac trong list img_plate

#cv2.imshow("a",grayImg)


#cv2.imshow("bien",img_bien)


cv2.waitKey()
cv2.destroyAllWindows()
