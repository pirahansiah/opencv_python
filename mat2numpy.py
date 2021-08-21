img=cv2.imread(file,-1)
height, width = img.shape[:2]
an_array1 = np.asarray( img, np.uint8)
an_array2 = np.where(an_array1 < 130, 0, 255)
an_array3=an_array2.astype('uint8')
image=an_array3.reshape(height, width)
cv2.imshow("frame2", image)
k = cv2.waitKey(1000)
cv2.destroyAllWindows()
