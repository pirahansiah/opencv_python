    path = r'c:\\'
    data= glob.glob(path+"/*[0-9].png")
    for file_name in data:
        print(file_name)
        image=cv2.imread(file_name)
        cv2.imshow("image",image)
        cv2.waitKey(100)
