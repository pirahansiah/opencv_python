# dir /s /b /o:n /ad "xcam" > farshid.txt
###################################################################################### import 
import os
import glob
###################################################################################### config
root="C:\\farshid\\"
specific_directories=root+"/**/farshid/**/*.jpg"
path_dir_detection_check=""

###################################################################################### function 
files= glob.glob(specific_directories, recursive=True)
for file in files:     
    b=file.rfind("farshid")
    path_dir_detection=file[0:b-1]    
    if (path_dir_detection != path_dir_detection_check):        
        dirname = os.path.dirname(file)
        print("******************************************************* Next Directories ************************")
        print(dirname)    
        path_dir_detection_check=path_dir_detection       
    print(file) 
