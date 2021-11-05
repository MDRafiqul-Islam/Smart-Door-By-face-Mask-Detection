from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from imutils.video import VideoStream
import imutils
import controller
def detect_and_predict_mask(frame,faceNet,maskNet):
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
    faceNet.setInput(blob)
    detections=faceNet.forward()
    faces=[]
    locs=[]
    preds=[]
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>0.5:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            faces.append(face)
            locs.append((startX,startY,endX,endY))
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNet.predict(faces,batch_size=12)
        return (locs,preds)

prototxtPath=os.path.sep.join([r'F:\FAce Mask Detector','deploy.prototxt'])
weightsPath=os.path.sep.join([r'F:\FAce Mask Detector','res10_300x300_ssd_iter_140000.caffemodel'])
faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)
maskNet=load_model(r'F:\FAce Mask Detector\mobilenet_v2.model')
vs=VideoStream(src="http://192.168.0.2:4747/video").start()
while True:
    frame=vs.read()
    frame=imutils.resize(frame,width=400)
    (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)
    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY)=box
        (mask,withoutMask)=pred
        if mask>withoutMask:
            label='Mask'
            #controller.doorAutomate(0)
        else:
            label='No Mask' 
            #controller.doorAutomate(1)
        color=(0,255,0) if label=='Mask' else (0,0,255)
        cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
    cv2.imshow("Mask Detector",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
        
cv2.destroyAllWindows()
vs.stop()