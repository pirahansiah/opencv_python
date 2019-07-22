import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image_plt(image):
    plt.subplot(231),plt.imshow(image,'change'),plt.title('ORIGINAL')
    plt.show()
    #plt.show(block=False)

def show_image_opencv(image):
    cv2.imshow("farshid",image)
    cv2.waitKey(1000)
