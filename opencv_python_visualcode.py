# Farshid Pirahansiah 22/July/2019
import numpy as np
import os
from matplotlib import pyplot as plt
import opencv_functions as fp
import cv2


def main():
    print("farshid pirahansiah")
    cap=cv2.VideoCapture(0)
    if(cap.isOpened()):
        _,frame=cap.read()     
        rows,cols,channels = frame.shape
        fp.cartoon_image(frame)
        fp.save_image_opencv('',frame)




if __name__== "__main__":
    main()