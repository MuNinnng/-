import os
import cv2
import numpy as np

path='C:/Users/DELL/Desktop/project/'
filelist = os.listdir(path)

fps=20
size=(640,480)

video=cv2.VideoWriter("C:/Users/DELL/Desktop/project/VideoTest.avi",cv2.VideoWriter_fourcc('I','4','2','0'),fps,size)

for item in filelist:
    if item.endswith('.jpg'):
        item=path+item
        img=cv2.imread(item)
        video.write(img)

video.release()
cv2.destroyAllWindows()

##ffmpeg -f images2 -i C:/Users/Dell/Desktop/project/pic%d.jpg -vcodec libx264 test.h264