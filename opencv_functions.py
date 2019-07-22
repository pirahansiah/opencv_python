import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image_plt(image):
    #plt.subplot(231),plt.imshow(image,'change'),plt.title('ORIGINAL')
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title('ORIGINAL')
    plt.show()
    #plt.show(block=False)

def show_image_opencv(image):
    cv2.imshow("farshid",image)
    cv2.waitKey(1000)
