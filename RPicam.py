import cv2
import zmq
import base64
import numpy as np

context = zmq.Context()
footage_socket = context.socket(zmq.PAIR)
footage_socket.bind('tcp://*:5555')
i=1
while(True):
    i=i+1
    try:
        frame = footage_socket.recv_string()
        img = base64.b64decode(frame)
        npimg = np.frombuffer(img,dtype = np.uint8)
        source = cv2.imdecode(npimg,1)
        img_name="C:/Users/DELL/Desktop/project/pic"+str(i)+".jpg"
        cv2.imwrite(img_name,source)
        cv2.imshow("Stream",source)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyWindows()
        break



