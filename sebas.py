"""
@author:IngeniousWorks/ferso8
"""
import cv2
import numpy as np
import dlib
import sys
import argparse
import imutils
import serial
from imutils import face_utils

#Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shapepredictor_68_facelandmarks.dat")

#Cam to test
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('./dt/Maud.mp4')
kernel = np.ones((5,5),np.uint8)
ym=5
xm=5

#Cx serial Port 
port = serial.Serial("COM8", baudrate=9600)              #Arduino Windows 
#port = serial.Serial("/dev/ttyACM0", baudrate=9600)      #Arduino Linux     
#port = serial.Serial("/dev/ttyTHS1", baudrate=9600)      #Jetson                                    
text = ""

#Threshold calibration 
tcr=[0,0]
rtc=10
dr=0

#Right & Left eye center cords
rcx=0
rcy=0
lcx=0
lcy=0

#Counters for serial Cx
r=0
l=0
u=0
d=0
c=0

#input_dir = './samples/'
#out_dir = './result/'

def extract_roi(img, shape, i, j):
    # extract the ROI of the face region as a separate image
    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    roi = img[y-ym:y+h+ym,x-xm:x+w+xm]
    roi = imutils.resize(roi, width=300, inter=cv2.INTER_CUBIC)
    return roi

while True:

    try:
        # Capture frame-by-frame
        ret, img = cap.read()

        # Resize the image for video files.
        #img = imutils.resize(img, width=580)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face
        rects = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y) -coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (ri, rj) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
            (li, lj) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']

            #windows sizes
            w=extract_roi(img,shape,ri,rj)
            rows, cols, _ = w.shape
            #print(w.shape)
            lbx= int ((cols/9)*4)
            rbx= int ((cols/9)*5)
            uby= int ((rows/9)*0)
            #dby= int ((rows/9)*(11/2))

            #Right EYE
            re=extract_roi(img,shape,ri,rj)
            grayr = cv2.cvtColor(re,cv2.COLOR_BGR2GRAY)
            mr = cv2.medianBlur(grayr,5)
            eror = cv2.erode(mr,kernel,cv2.BORDER_REFLECT)
            retr,threshr = cv2.threshold(eror,rtc,255,cv2.THRESH_BINARY_INV)
            contoursr, _ = cv2.findContours(threshr,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            #contours, _ = cv2.findContours(threshr,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(re,contoursr,-1,(0,255,0))
            contoursr = sorted(contoursr,key=lambda x: cv2.contourArea(x),reverse=True) 

            if contoursr:
                
                if dr==0:
                    tcr[dr]=rtc
                    dr=dr+1
                    tcr[dr]=tcr[0]+10
                    dr=dr+1  

                for cnt in contoursr:
                    rtc=tcr[1]
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    #cv2.circle(le,(int (colsl/2),int(rowsl/2)),5,(155,0,150),-1)       # center
                    cv2.circle(re,(x+int(w/2),y+int(h/2)),int (h/3),(5,0,250),2)        # cv2 circles parameters (img,center(x,y),ratio,color,thickness)
                    #print('right eye',x+int(w/2),y+int(h/2))
                    rcx=(x+(w/2))
                    rcy=(y+(h/2))
                    cv2.line(re,(x+int(w/2),0),(x+int(w/2),rows),(250,0,10),1)          # cv2 lines parameters (origin(x,y),end (x,y),color,thickness)
                    cv2.line(re,(0,y+int(h/2)),(cols,y+int(h/2)),(250,0,10),1)
                    cv2.line(re,(lbx,0),(lbx,rows),(5,250,5),1)
                    cv2.line(re,(rbx,0),(rbx,rows),(5,250,5),1)
                    cv2.line(re,(0,uby),(cols,uby),(5,250,5),1)
                    break
            
            else:
                rtc=rtc+5


            #Right
            if  rcx < lbx:
                if rcy >= uby:
                    cv2.putText(img,'Right',(30,30),0,1,(255,0,0),2)        #Text Parameters (img,"Txt",cords(x,y),Font,Scale,Color,Thickness)     
                    r=r+1
                    if (r==2):
                        port.write(b"R\n")
                        print('Right')
                        r=0
                        l=0
                        u=0
                        d=0
                        c=0

            #Center    
            if lbx <= rcx <= rbx and uby <= rcy:
                cv2.putText(img,'Center',(30,30),2,1,(255,0,0),2)
                c=c+1
                if (c==2):
                    port.write(b"C\n")
                    print('Center')
                    r=0
                    l=0
                    u=0
                    d=0
                    c=0

            #Left
            if  rcx > rbx:
                if rcy >= uby:
                    cv2.putText(img,'Left',(30,30),0,1,(255,0,0),2)
                    l=l+1
                    if (l==2):
                        port.write(b"L\n")
                        print('Left')
                        r=0
                        l=0
                        u=0
                        d=0
                        c=0

            #Up
            if rcy <= uby:
                cv2.putText(img,'Up',(30,30),2,1,(255,0,0),2)
                u=+1
                if (u==2):
                    port.write(b"U\n")
                    print('Up')
                    r=0
                    l=0
                    u=0
                    d=0
                    c=0

            print ('Eye Cords ==> ',rcx,rcy)

            cv2.imshow("right eye", re)
            cv2.imshow("right", threshr)
            cv2.imshow("Image", img)

    except: 
        print("Error")
        rtc=10
        dr=0
    
    if cv2.waitKey(1) & 0xFF == ord('r'):
        rtc=10
        dr=0

    if cv2.waitKey(1) & 0xFF == 27:     # esc key to exit
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()