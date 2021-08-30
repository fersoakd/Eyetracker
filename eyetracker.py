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
predictor = dlib.shape_predictor('./shapepredictor_68_facelandmarks.dat')

#Video or Cam to test
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('./dt/roc.mp4')
kernel = np.ones((5,5),np.uint8)
ym=5
xm=5

#Threshold calibration 
tcr=[0,0]
tcl=[0,0]
rtc=10
ltc=10
dr=0
dl=0

#Right & Left eye center cords
rcx=0
rcy=0
lcx=0
lcy=0

#Cx serial Port 
#port = serial.Serial("COM8", baudrate=9600)              #Arduino Windows 
#port = serial.Serial("/dev/ttyACM0", baudrate=9600)      #Arduino Linux     
#port = serial.Serial("/dev/ttyTHS1", baudrate=9600)      #Jetson                                  
#text = ""
#port.write(b"D\n")

#input_dir = './samples/'
#out_dir = './result/'

def extract_roi(img, shape, i, j):
    # extract the ROI of the face region as a separate image
    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    roi = img[y-ym:y+h+ym,x-xm:x+w+xm]
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    return roi

while True:

    try:
        # Capture frame-by-frame
        ret, img = cap.read()

        # Resize the image for video files.
        img = imutils.resize(img, width=550)

        # Convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

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
            lbx= int ((cols/9)*(3.75))       #Left bound
            rbx= int ((cols/9)*(5.25))       #Right bound        
            uby= int ((rows/9)*(2))          #Up bound
            #dby= int ((rows/9)*(11/2))      #Down bound

            #Right EYE
            re=extract_roi(img,shape,ri,rj)
            gr = cv2.cvtColor(re,cv2.COLOR_BGR2GRAY)
            mr = cv2.medianBlur(gr,15)
            br = cv2.GaussianBlur(mr,(15,15),0)
            #er = cv2.erode(mr,kernel,cv2.BORDER_REFLECT)
            retr,threshr = cv2.threshold(br,rtc,255,cv2.THRESH_BINARY_INV)
            contoursr, _ = cv2.findContours(threshr,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            #contours, _ = cv2.findContours(threshr, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(re, contoursr, -1, (0, 255, 0))
            contoursr = sorted(contoursr,key=lambda x:cv2.contourArea(x),reverse=True) 

            if contoursr:
                
                if dr==0:
                    tcr[dr]=rtc
                    dr=dr+1
                    tcr[dr]=tcr[0]+5
                    dr=dr+1  

                for cnt in contoursr:
                    rtc=tcr[1]
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    #cv2.circle(le,(int (colsl/2),int(rowsl/2)),5,(155,0,150),-1)    # center
                    cv2.circle(re,(x+int(w/2),y+int(h/2)),int (h/3),(5,0,250),2)    # cv2 circles parameters (img,center(x,y),ratio,color,thickness)
                    #print('right eye',x + int(w/2),y+int(h/2))
                    rcx=(x+(w/2))
                    rcy=(y+(h/2))
                    cv2.line(re,(x+int(w/2),0),(x+int(w/2),rows),(250,0,10),1)       # cv2 lines parameters (origin(x,y),end (x,y),color,thickness)
                    cv2.line(re,(0,y+int(h/2)),(cols,y+int(h/2)),(2500,0,10),1)
                    cv2.line(re,(lbx,0),(lbx,rows),(0,250,0),1)
                    cv2.line(re,(rbx,0),(rbx,rows),(0,250,0),1)
                    cv2.line(re,(0,uby),(cols,uby),(0,250,0),1)
                    break
            
            else:
                rtc=rtc+5

            #Left EYE
            le=extract_roi(img,shape,li,lj)
            gl = cv2.cvtColor(le,cv2.COLOR_BGR2GRAY)
            ml = cv2.medianBlur(gl,15)
            bl = cv2.GaussianBlur(ml,(15,15),0)
            #el = cv2.erode(ml,kernel,cv2.BORDER_REFLECT)
            retl,threshl = cv2.threshold(bl,ltc,255,cv2.THRESH_BINARY_INV)
            contoursl, _ = cv2.findContours(threshl,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            contoursl = sorted(contoursl,key=lambda x: cv2.contourArea(x),reverse=True)
            
            if contoursl:
            
                if dl==0:
                    tcl[dl]=ltc
                    dl=dl+1
                    tcl[dl]=tcl[0]+5
                    dl=dl+1
                    ltc=tcl[1]

                for cnt in contoursl:
                    ltc=tcl[1]
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    cv2.circle(le,(x+int(w/2),y+int(h/2)),int (h/3),(5,0,250),2)  
                    lcx=(x+(w/2))
                    lcy=(y+(h/2))
                    cv2.line(le,(x+int(w/2),0),(x+int(w/2),rows),(250,0,10),1)
                    cv2.line(le,(0,y+int(h/2)),(cols,y+int(h/2)),(250,0,10),1)
                    cv2.line(le,(lbx,0),(lbx,rows),(5,250,5),1)
                    cv2.line(le,(rbx,0),(rbx,rows),(5,250,5),1)
                    cv2.line(le,(0,uby),(cols,uby),(5,250,5),1)
                    break

            else:
                ltc=ltc+5

            #Right
            if  rcx < lbx or lcx < lbx:
                if rcy >= uby and lcy >= uby:
                    cv2.putText(img,'Right',(30,30),0,1,(255,0,0),2)        #Text Parameters (img,"Txt",cords(x,y),Font,Scale,Color,Thickness)
                    #print ('Right')   

            #Center    
            if lbx <= rcx <= rbx and lbx <= lcx <= rbx and uby <= rcy and uby <= lcy:
                cv2.putText(img,'Center',(30,30),2,1,(255,0,0),2)
                #print ('Center')

            #Left
            if  rcx > rbx or lcx > rbx :
                if rcy >= uby and lcy >= uby:
                    cv2.putText(img,'Left',(30,30),0,1,(255,0,0),2)
                    #print ('Left')

            #Up
            if rcy <= uby and lcy <= uby:
                cv2.putText(img,'Up',(30,30),2,1,(255,0,0),2)
                #print ('Up')

            print ('Right Eye ==> ',rcx,rcy)
            print ('Left Eye  ==> ',lcx,lcy)

            cv2.imshow("right eye",re)
            cv2.imshow("right eye blur",br)
            cv2.imshow("right threshold",threshr)
            cv2.imshow("left eye",le)
            cv2.imshow("left eye blur",bl)
            cv2.imshow("left threshold",threshl)
            cv2.imshow("Image",img)

    except: 
        print("Error")
        rtc=10
        ltc=10
        dl=0
        dr=0
    
    if cv2.waitKey(1) & 0xFF == ord('r'):
        rtc=10
        ltc=10
        dl=0
        dr=0
        
    if cv2.waitKey(1) & 0xFF == ord('q') :      # for 'esc' key to exit cv2.waitKey(1) & 0xFF == 27
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
