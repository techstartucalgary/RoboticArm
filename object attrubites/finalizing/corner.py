from math import sqrt
import numpy as np 
import cv2 

    
camera = cv2.VideoCapture(0)

while True:
    ret,frame = camera.read()
    all_corners = []
    y_value = []
    x_value = []
    area = 400
    centre_x = 20
    centre_y = 40
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.flip(frame_gray,1)

    #eculidane distance (min) between each corner is the last parameter
    #second parameter is the min quaility 
    #first parameter maxium number of corners to return 
    corners = cv2.goodFeaturesToTrack(frame_gray, 100, 0.01, 50)
    corners = np.int0(corners)
    
    #function for the top camera
    for z in corners:
        x,y = z.ravel()
        all_corners.append([x,y]) 
        y_value.append(y)
        x_value.append(x)
        cv2.circle(frame_gray, (x,y), 3, 255, -1)
    #take the area of the contour
    
    sides = sqrt(area)
    half_dist = sides/2
    for n in y_value:
        if(((centre_y-half_dist-10) >= n) and (n >= (centre_y+half_dist+10))):
            index = y_value.index(n)
            del y_value[index]
            del x_value[index]
    for m in x_value:
         if(((centre_x-half_dist-10) >= m) and (m >= (centre_x+half_dist+10))):
            index = x_value.index(n)
            del y_value[index]
            del x_value[index]     
    
    doupe_y = list(set(y_value))
    doupe_y.sort()
    print(doupe_y)
    doupe_x = list(set(x_value))
    doupe_x.sort()
    print(doupe_x)
    
    
    dif_x = max(doupe_x) - min(doupe_x)
    dif_y = max(doupe_y) - min(doupe_y) 
    print("diff in x ", dif_x)
    print("diff in y ", dif_y)
    #function for the side camera 
    for o in corners: 
        max_height = y_value[0]
        x,y = z.ravel()
        for a in y_value:
            if(max_height < a):
                max_height = a
    
    print("the max height is ", max_height)
    
    cv2.imshow('Camera', frame_gray)
    
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
