# Farshid Pirahansiah 22/July/2019
import numpy as np
import os
from matplotlib import pyplot as plt
import opencv_functions as fp
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
import cv2

def mainf():
    print("farshid pirahansiah")
    print(cv2.__version__)
    cap=cv2.VideoCapture(0)
    if(cap.isOpened()):
        _,frame=cap.read()     
        rows,cols,channels = frame.shape
        cv2.imshow('farshid original',frame)
        cv2.waitKey(1000)
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
        fp.show_image_plt(frame)
        #fp.show_image_opencv(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cap.release()
        cv2.destroyAllWindows()
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

if __name__== "__main__":
  mainf()



