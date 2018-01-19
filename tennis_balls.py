import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np

def nothing(x):
    pass
yellow_min = np.array([21,101,68])
yellow_max = np.array([100,211,218])
cv2.namedWindow('adjust')
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
#last try:20,16,175,37,255,249,20,18,-,-

def find_ball(im):
    """
    cv2.namedWindow('adjust')
    cv2.createTrackbar('H-','adjust',yellow_min[0],360,nothing)
    cv2.createTrackbar('S-','adjust',yellow_min[1],255,nothing)
    cv2.createTrackbar('V-','adjust',yellow_min[2],255,nothing)
    cv2.createTrackbar('H+','adjust',yellow_max[0],360,nothing)
    cv2.createTrackbar('S+','adjust',yellow_max[1],255,nothing)
    cv2.createTrackbar('V+','adjust',yellow_max[2],255,nothing)
    cv2.createTrackbar('param1','adjust',20,100,nothing)
    cv2.createTrackbar('param2','adjust',16,100,nothing)
    cv2.createTrackbar('erosion','adjust',2,10,nothing)
    cv2.createTrackbar('dilation','adjust',9,10,nothing)"""
    im2 = im.copy()
    while(True):
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
        #yellow_min = np.array([21,101,68])
        #yellow_max = np.array([100,211,218])
        yellow_min = np.array([h1,s1,v1])
        yellow_max = np.array([h2,s2,v2])
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv,5)
        yellow_mask = cv2.inRange(hsv,yellow_min,yellow_max)
        if erosion > 0:
            yellow_mask = cv2.erode(yellow_mask,None,iterations=erosion)
        if dilation > 0:
            yellow_mask = cv2.dilate(yellow_mask, None, iterations=dilation)
        #yellow_mask = cv2.erode(yellow_mask,None,iterations=2)
        #yellow_mask = cv2.dilate(yellow_mask, None, iterations=9)
        yellow_image = cv2.bitwise_and(hsv,hsv,mask = yellow_mask)
        yellow_gray = yellow_image[:,:,2]
        #circles = cv2.HoughCircles(yellow_gray,cv2.HOUGH_GRADIENT,dp=1,minDist=20,
                                    #param1=20,param2=25,minRadius=0,maxRadius=0)
        circles = cv2.HoughCircles(yellow_mask,cv2.HOUGH_GRADIENT,dp=1,minDist=20,
                                    param1=p1,param2=p2,minRadius=0,maxRadius=0)
       
        im2 = im.copy()
        if not circles is None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(im2,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(im2,(i[0],i[1]),2,(0,0,255),3)
        cv2.imshow('yellow image',yellow_image)
        cv2.imshow('hsv',hsv)
        cv2.imshow('mask',yellow_mask)
        cv2.imshow('found',im2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    return circles

ball_ann = 'C:/Users/djorna.Pokedex/Documents/ZED/TennisBallAnnotations/'
ball_im = 'C:/Users/djorna.Pokedex/Documents/ZED/TennisBallImages/'

for fname in os.listdir(ball_ann):  
    imfile = ball_im + fname.split('.')[0] + '.JPEG'
    if not os.path.exists(imfile):
        continue

    tree = ET.parse(ball_ann + fname)
    root = tree.getroot() 

    im = cv2.imread(imfile) 
    #show circles from algorithm
    circles = find_ball(im)
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        print(xmin,xmax,ymin,ymax)
        cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),3)
        #draw center
        cv2.circle(im,((xmin+xmax)//2,(ymin+ymax)//2),2,(255,0,0),3)

    
    """im2 = im.copy()
    if not circles is None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(im2,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(im2,(i[0],i[1]),2,(0,0,255),3)
    else:
        print('ball not found')
    cv2.imshow('found',im2)
    
    cv2.imshow(fname,im)    
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    cv2.destroyAllWindows()
    """
cv2.destroyAllWindows()
