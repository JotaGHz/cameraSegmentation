import cv2
import numpy as np
import datetime
#inicialização da captura da webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

x, y, w, h = 640, 360, 100, 50
track_window1 = (x, y, w, h)
track_window2 = (x, y, w, h)

#dimensões do monitor para o controle do ponteiro do mouse
width = 1920
height = 1080

cap_width = np.size(frame, 1)
cap_height = np.size(frame, 0)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
m_flag = 0
while True:
    ret, frame = cap.read()
    period = datetime.datetime.now()
    if ret == True:
        
        frame = cv2.flip(frame, 1)
         
        cv2.imshow("mouseHelper", frame)
        key = cv2.waitKey(30) & 0xff
    
    else:
        break
    if key == 27:
        break
    
    
cap.release()
cv2.destroyAllWindows()
