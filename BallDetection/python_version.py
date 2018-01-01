import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

yellow_min = np.array([21,101,68])
yellow_max = np.array([105,211,218])

cv2.namedWindow('adjust')

#create trackbars
cv2.createTrackbar('H-','adjust',yellow_min[0],360,nothing)
cv2.createTrackbar('S-','adjust',yellow_min[1],255,nothing)
cv2.createTrackbar('V-','adjust',yellow_min[2],255,nothing)
cv2.createTrackbar('H+','adjust',yellow_max[0],360,nothing)
cv2.createTrackbar('S+','adjust',yellow_max[1],255,nothing)
cv2.createTrackbar('V+','adjust',yellow_max[2],255,nothing)
cv2.createTrackbar('param1','adjust',20,100,nothing)
cv2.createTrackbar('param2','adjust',16,100,nothing)
cv2.createTrackbar('erosion','adjust',2,10,nothing)
cv2.createTrackbar('dilation','adjust',9,10,nothing)

while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h1 = cv2.getTrackbarPos('H-','adjust')
    s1 = cv2.getTrackbarPos('S-','adjust')
    v1 = cv2.getTrackbarPos('V-','adjust')
    h2 = cv2.getTrackbarPos('H+','adjust')
    s2 = cv2.getTrackbarPos('S+','adjust')
    v2 = cv2.getTrackbarPos('V+','adjust')
    p1 = cv2.getTrackbarPos('param1','adjust')
    p2 = cv2.getTrackbarPos('param2','adjust')
    erosion = cv2.getTrackbarPos('erosion','adjust')
    dilation = cv2.getTrackbarPos('dilation','adjust')

    yellow_min = np.array([h1,s1,v1])
    yellow_max = np.array([h2,s2,v2])
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv,11)
    yellow_mask = cv2.inRange(hsv,yellow_min,yellow_max)
    if erosion > 0:
        yellow_mask = cv2.erode(yellow_mask,None,iterations=erosion)
    if dilation > 0:
        yellow_mask = cv2.dilate(yellow_mask, None, iterations=dilation)

    yellow_image = cv2.bitwise_and(hsv,hsv,mask = yellow_mask)

    #yellow_gray = gray
    yellow_gray = yellow_image[:,:,2]
    #yellow_gray = cv2.GaussianBlur(yellow_gray,(5,5),0)
    #yellow_gray = cv2.medianBlur(yellow_gray,5)
    
    cimg = cv2.cvtColor(yellow_gray,cv2.COLOR_GRAY2BGR)
    
    #get circles with Hough circle transform
    circles = cv2.HoughCircles(yellow_mask,cv2.HOUGH_GRADIENT,dp=1,minDist=20,
                                param1=p1,param2=p2,minRadius=0,maxRadius=0)
    
    #Draw circles on image
    if not circles is None:
        print('num circles found:',len(circles))
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    else:
        print('num circles found:',0)
    #edges = cv2.Canny(yellow_image, 100, 200) 
    """
    cnts = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        #c = max(cnts, key=cv2.contourArea)
        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
    """
    cv2.imshow('hsv',hsv)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',yellow_mask)
    cv2.imshow('yellow',yellow_image)
    cv2.imshow('cimg',cimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
