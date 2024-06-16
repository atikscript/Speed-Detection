import cv2
import numpy as np
import vehicles
import time

cnt_up=0
cnt_down=0

carCascade = cv2.CascadeClassifier('vech.xml')
cap=cv2.VideoCapture("carsVid.mp4")

#Get width and height of video

w=cap.get(3)
h=cap.get(4)
frameArea=h*w
areaTH=frameArea/400

#Lines
line_up=int(2*(h/5))
line_down=int(3*(h/5))

up_limit=int(1*(h/5))
down_limit=int(4*(h/5))

print("Red line y:",str(line_down))
print("Blue line y:",str(line_up))
line_down_color=(255,0,0)
line_up_color=(255,0,255)
pt1 =  [0, line_down]
pt2 =  [w, line_down]
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit]
pt8 =  [w, down_limit]
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

#Background Subtractor
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)

#Kernals
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
# kernalCl = np.ones((11,11),np.uint)
kernalCl = np.ones((11,11),np.uint8)


font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1


while(cap.isOpened()):
    ret,frame=cap.read()
    for i in cars:
        i.age_one()
    fgmask=fgbg.apply(frame)
    fgmask2=fgbg.apply(frame)

    if ret==True:

        #Binarization
        ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2=cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        #OPening i.e First Erode the dilate
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)
        mask2=cv2.morphologyEx(imBin2,cv2.MORPH_CLOSE,kernalOp)

        #Closing i.e First Dilate then Erode
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalCl)
        mask2=cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernalCl)


        #Find Contours
        #_, countours0,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        countours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            area=cv2.contourArea(cnt)
            print(area)
            if area>areaTH:
                ####Tracking######
                m=cv2.moments(cnt)
                cx=int(m['m10']/m['m00'])
                cy=int(m['m01']/m['m00'])
                x,y,w,h=cv2.boundingRect(cnt)

                new=True
                if cy in range(up_limit,down_limit):
                    for i in cars:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                            new = False
                            i.updateCoords(cx, cy)

                            if i.going_UP(line_down,line_up)==True:
                                cnt_up+=1
                                print("ID:",i.getId(),'crossed going up at', time.strftime("%c"))
                            elif i.going_DOWN(line_down,line_up)==True:
                                cnt_down+=1
                                print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                            break
                        if i.getState()=='1':
                            if i.getDir()=='down'and i.getY()>down_limit:
                                i.setDone()
                            elif i.getDir()=='up'and i.getY()<up_limit:
                                i.setDone()
                        if i.timedOut():
                            index=cars.index(i)
                            cars.pop(index)
                            del i

                    if new==True: #If nothing is detected,create new
                        p=vehicles.Car(pid,cx,cy,max_p_age)
                        cars.append(p)
                        pid+=1

                cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        for i in cars:
            cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 1.0, i.getRGB(), 1, cv2.LINE_AA)





        str_up='UP: '+str(cnt_up)
        str_down='DOWN: '+str(cnt_down)
        frame=cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
        frame=cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
        frame=cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame=cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Frame',frame)

        if cv2.waitKey(1)&0xff==ord('q'):
            break

    else:
        break

####################
import cv2
import dlib
import time
import math

carCascade = cv2.CascadeClassifier('vech.xml')
# video = cv2.VideoCapture('video3.mp4')

WIDTH = 1280
HEIGHT = 720


def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter('outNew.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = cap.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print("Removing carID " + str(carID) + ' from list of trackers. ')
            print("Removing carID " + str(carID) + ' previous location. ')
            print("Removing carID " + str(carID) + ' current location. ')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                            x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    print(' Creating new tracker' + str(currentCarID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x1, y2, w2, h2])

                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(x1 + w1 / 2), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100), 2)

        cv2.imshow('result', resultImage)

        out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    trackMultipleObjects()
#############
cap.release()
cv2.destroyAllWindows()









