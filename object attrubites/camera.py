import numpy as np 
import cv2 
#this does not contain the click, maybe work on each dimension...?
import xml.etree.cElementTree as xml
import numpy as np 
import cv2
import imutils 
font = cv2.FONT_ITALIC

input_camera = cv2.VideoCapture(0)
overall = []   
master = None

#creating the objects for the files
tree = xml.parse('/Users/nesmoh/Downloads/gym-0.19.0 2/gym/envs/robotics/assets/fetch/push.xml')
root = tree.getroot()
tree2 = xml.parse('/Users/nesmoh/Downloads/gym-0.19.0 2/gym/envs/robotics/assets/fetch/pick_and_place.xml')
root2 = tree2.getroot()
tree3 = xml.parse('/Users/nesmoh/Downloads/gym-0.19.0 2/gym/envs/robotics/assets/fetch/slide.xml')
root3 = tree3.getroot()
object0 = root.find("./worldbody/body[@name='object0']")
object1 = root2.find("./worldbody/body[@name='object0']")
object2 = root3.find("./worldbody/body[@name='object0']")

 

while True:
    ret,frame = input_camera.read()
    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame2 = cv2.GaussianBlur(frame1,(15,15),0)
        # initialize master
    if master is None:
        master = frame2
        continue
    frame3 = cv2.absdiff(master,frame2) 
    frame4 = cv2.threshold(frame3,15,255,cv2.THRESH_BINARY)[1]
    kernel = np.ones((2,2),np.uint8)
    frame5 = cv2.erode(frame4,kernel,iterations=4)
    frame5 = cv2.dilate(frame5,kernel,iterations=8)
    contours, nada = cv2.findContours(frame5.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    frame6 = frame.copy()
        # target contours
    targets = []
        # loop over the contours
    for c in contours:
            
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 500:
                    continue
            #overall.clear() 
            # contour data
            overall.clear() # Move here
            M = cv2.moments(c)#;print( M )
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(c)
            rx = x+int(w/2)
            ry = y+int(h/2)
            ca = cv2.contourArea(c)
            # plot contours
            cv2.drawContours(frame6,[c],0,(0,0,255),2)
            cv2.rectangle(frame6,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame6,(cx,cy),2,(0,0,255),2)
            cv2.circle(frame6,(rx,ry),2,(0,255,0),2)
            targets.append((cx,cy,ca))
            overall.append(cx)
            overall.append(cy)



    mx = 0
    my = 0
    if targets:
            area = 0
            for x,y,a in targets:
                if a > area:
                    mx = x
                    my = y
                    area = a
    print("this is the statement", overall)    
        # plot target
    tr = 50
    frame7 = frame.copy()
    if targets:
            cv2.circle(frame7,(mx,my),tr,(0,0,255,0),2)
            cv2.line(frame7,(mx-tr,my),(mx+tr,my),(0,0,255,0),2)
            cv2.line(frame7,(mx,my-tr),(mx,my+tr),(0,0,255,0),2)
    
    if(len(overall) != 0): 
        final_str = "still at" + 'x: ' + str(overall[0]) + ' y:' + str(overall[1]) 
        cv2.putText(frame7, final_str, (20,1000), font,1,(255,255,255)) 
        total_str = str(overall[0]) + str(overall[1])  + "0" #the z axis is a zero for the final value
        object0.set('pos', total_str)
        object1.set('pos', total_str)
        object2.set('pos', total_str)
        tree.write('push.xml')
        tree2.write('pick_and_place.xml')
        tree3.write('slide.xml')
        
    master = frame2
        # display
    point1 = np.array((400,500))
    point2 = np.array((mx,my))
    distance = np.linalg.norm(point1 - point2)
    distance_str = 'the distance between: ' + str(distance)

    frame7 = cv2.circle(frame7, (400,500), 10, (255,255,255), 2)
    frame7 = cv2.putText(frame7, '(400,500)', (410, 510), font, 1, (255,255,255))
    if((mx != 0) and (my != 0)): 
            frame7 = cv2.line(frame7, (400,500), (mx,my), (0,255,0),2)
            frame7 = cv2.putText(frame7, distance_str, (0, 1000), font, 1, (255,255,255))
    cv2.imshow("Frame7: Target",frame7)

    if cv2.waitKey(1) == ord('q'):
         break