import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image_plt(image):
    # for many images 
    #plt.subplot(231),plt.imshow(image,'change'),plt.title('ORIGINAL')
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title('ORIGINAL')
    plt.show()
    #plt.show(block=False)

def show_image_opencv(image):
    cv2.imshow("farshid",image)
    cv2.waitKey(1000)

def save_image_opencv(filename,img):
    if len(filename)==0:
        cv2.imwrite("d:\\farshid.jpg", img)
    else:
        cv2.imwrite([filename], [img])

    

def cartoon_image(image):
    num_down = 5#2 # number of downsampling steps
    num_bilateral = 9#7 # number of bilateral filtering steps
    img_rgb = image
    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    # repeatedly apply small bilateral filter instead of
    # applying one large filter
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    # upsample image to original size
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    #STEP 2 & 3
    #Use median filter to reduce noise
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    #STEP 4
    #Use adaptive thresholding to create an edge mask
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,    cv2.ADAPTIVE_THRESH_MEAN_C,    cv2.THRESH_BINARY,    blockSize=9,    C=2)
    # Step 5
    # Combine color image with edge mask & display picture
    # convert back to color, bit-AND with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    # display
    image=img_cartoon
    cv2.imshow("cartoon", img_cartoon)
    cv2.waitKey(0)
    return image

def face_detection(img):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mainfarshid():
    print("farshid pirahansiah")
    print(cv2.__version__)
    cap=cv2.VideoCapture(0)
    if(cap.isOpened()):
        _,frame=cap.read()     
        rows,cols,channels = frame.shape
        cv2.imshow('farshid original',frame)
        cv2.waitKey(1000)
        #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        #ret, mask = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)
        #ret, mask = cv2.threshold(frame, 127, 255, cv2.THRESH_OTSU)
        th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        #fp.show_image_plt(mask)
        #fp.show_image_opencv(th2)
        #fp.show_image_opencv(th3)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        cap.release()
        cv2.destroyAllWindows()
if __name__== "__main__":
    mainfarshid()
    # frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    # for i in range(rows):
    #     for j in range(cols):
    #         #print(frame.item(i,j))
    #         if frame.item(i,j)>130:
    #             frame.itemset((i,j),255)
    #         else:
    #             frame.itemset((i,j),0)



    # cv2.imshow("farshid",frame)
    # cv2.waitKey(1000)


    # px = frame[100,100]
    # print(px)








    #/////////////////
    # import numpy as np
    # from matplotlib import pyplot as plt

    # BLUE = [255,0,0]

    # img1 = frame 

    # replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
    # reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
    # reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
    # wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
    # constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

    # plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
    # plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
    # plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
    # plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
    # plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
    # plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

    # plt.show()



# cap=cv2.VideoCapture(0)
# if(cap.isOpened()):
#   _,frame=cap.read()     
#   rows,cols,channels = frame.shape
#   cv2.imshow('farshid original',frame)
#   cv2.waitKey(1000)
#   frame=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
#   #fp.show_image_plt(frame)
#   plt.imshow(frame)
#   plt.title('ORIGINAL')
#   plt.show()  


