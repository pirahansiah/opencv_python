# store all image list in the list
    data= glob.glob(src+"/*.png")
    total_data_len = len(data)
    
# progress bar that updates at every 10%
progress = ((progressBar) * 100) / total_data_len
if progress >= lastProg:
    print ("Progress: "+str(progress) + "%")
    lastProg +=10
progressBar += 1
