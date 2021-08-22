import gym
import pix_main_arena
import time
import pybullet as p
import pybullet_data
import cv2
import os
import numpy as np
import cv2.aruco as aruco
from collections import defaultdict, deque
import math
import cv2.aruco as aruco
import cv2



def find_centroid(img):
# Constant parameters used in Aruco methods
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)


    # Create grid board object we're using in our stream
    board = aruco.GridBoard_create(
            markersX=2,
            markersY=2,
            markerLength=0.09,
            markerSeparation=0.01,
            dictionary=ARUCO_DICT)

    # Create vectors we'll be using for rotations and translations for postures
    rvecs, tvecs = None, None

        #Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    # Make sure all 5 markers were detected before printing them out
    if ids is not None:
                # Print corners and ids to the console
        for i, corner in zip(ids, corners):
            #print('ID: {}; Corners: {}'.format(i, corner))
            cx=(corner[0][0][0]+corner[0][2][0])/2
            cy=(corner[0][0][1]+corner[0][2][1])/2
            #print((cx,cy))
            x=(corner[0][0][0]+corner[0][1][0])/2
            y=(corner[0][0][1]+corner[0][1][1])/2
            centroid=[cx,cy]
            front=[x,y]
            #print("centroid :",centroid)
            #print("front :",front)
            vector=[front[0]-centroid[0], front[1]-centroid[1]]

        return [centroid,vector]
    
def runBot(nodes_cord,prev_vector, prev_centroid):        
    vector_path=[]
    angles=[]
    for i in range (len(nodes_cord)-1):
        vector_path.append([nodes_cord[i+1][0]-nodes_cord[i][0],nodes_cord[i+1][1]-nodes_cord[i][1]])
    i=0
    min_dist=1000000000
    count1=1
    count2=1
    box=[[32, 260, 30, 30, 6, 0], [298, 69, 30, 31, 1, 7]]
    pink_nodes=[(1, 7), (6, 0)]
    while True:
        print("node",i)
        img=env.camera_feed()
        img=cv2.cvtColor(cv2.resize(img,(512,512)), cv2.COLOR_BGR2GRAY)
        cv=find_centroid(img)
        if (cv is None):
            vector=prev_vector
            centroid= prev_centroid
        else:
            vector=cv[1]
            centroid=cv[0]
        print(centroid)
        if(centroid is not None):
            prev_centroid=centroid
        #move right if 180 angle and left if 90 angle
        dist=math.dist(centroid,nodes_cord[i])
        min_dist=min(min_dist,dist)
        print("dist",dist)
        if(i<len(nodes_cord)-1):
            angle=np.degrees(np.math.atan2(np.linalg.det([vector_path[i],vector]),np.dot(vector_path[i],vector)))
            print("angle",angle)
        if(dist<=17):
            idx=i
            frame=100
            while(frame):
                p.stepSimulation()
                env.move_husky(0, 0, 0, 0)
                frame=frame-1
            i=i+1
            if(i==14 and count1==1):
                time.sleep(5)
                env.remove_cover_plate(1,7)
                img_p=env.camera_feed()
                img_p=cv2.cvtColor(cv2.resize(img_p,(512,512)), cv2.COLOR_BGR2GRAY)
                img_p=img_p[box[1][1]:box[1][1]+box[1][3],box[1][0]:box[1][0]+box[1][2]]
                cv2.imshow("img_shape",img_p)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                count1=0

            if(i==40 and count2 ==1):
                time.sleep(5)
                env.remove_cover_plate(6,0)
                img_p=env.camera_feed()
                img_p=cv2.cvtColor(cv2.resize(img_p,(512,512)), cv2.COLOR_BGR2GRAY)
                img_p=img_p[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
                cv2.imshow("img_shape",img_p)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                count2=0
            if(i==len(nodes_cord)-1 and angle<=5 and angle>=-5):
                frame=300
                while(frame):
                    print("ending")
                    p.stepSimulation()
                    env.move_husky(14,14,14,14)
                    frame=frame-1
                break
            if(angle>=60 and angle<=180):
                while(angle>=15 or (i==29 and angle>=1) or (i==41 and angle>=1)):
                    print(i)
                    print("turning right")
                    p.stepSimulation()
                    env.move_husky(-5,5 , -5, 5)
                    img=env.camera_feed()
                    img=cv2.cvtColor(cv2.resize(img,(512,512)), cv2.COLOR_BGR2GRAY)
                    cv=find_centroid(img)
                    if (cv is None):
                        vector=prev_vector
                        centroid= prev_centroid
                    else:
                        vector=cv[1]
                        centroid=cv[0]
                        prev_centroid=cv[0]
                        prev_vector=cv[1]
                    #move right if 180 angle and left if 90 angle
                    angle=np.degrees(np.math.atan2(np.linalg.det([vector_path[idx],vector]),np.dot(vector_path[idx],vector)))
                    print(angle)
                frame=100
                while(frame):
                    p.stepSimulation()
                    env.move_husky(0, 0, 0, 0)
                    frame=frame-1
            if(angle<=-60 and angle>=-180):
                while(angle<=-15 or (i==29 and angle>=1) or (i==40 and angle>=9)):
                    print(i)
                    print("turning left")
                    p.stepSimulation()
                    env.move_husky(5,-5 ,5, -5)
                    img=env.camera_feed()
                    img=cv2.cvtColor(cv2.resize(img,(512,512)), cv2.COLOR_BGR2GRAY)
                    cv=find_centroid(img)
                    if (cv is None):
                        vector=prev_vector
                        centroid= prev_centroid
                    else:
                        vector=cv[1]
                        centroid=cv[0]
                        prev_centroid=cv[0]
                        prev_vector=cv[1]
                    print(centroid)
                    #move right if 180 angle and left if 90 angle
                    angle=np.degrees(np.math.atan2(np.linalg.det([vector_path[idx],vector]),np.dot(vector_path[idx],vector)))
                    print(angle)
                    frame=100
                while(frame):
                    p.stepSimulation()
                    env.move_husky(0, 0, 0, 0)
                    frame=frame-1
        elif(dist>17):
            p.stepSimulation()
            env.move_husky(10, 10, 10, 10)
            if(angle<=-1 and angle>=-50):
                while(angle<=-1):
                    print(i)
                    print("turning left")
                    p.stepSimulation()
                    env.move_husky(5,-5 ,5, -5)
                    img=env.camera_feed()
                    img=cv2.cvtColor(cv2.resize(img,(512,512)), cv2.COLOR_BGR2GRAY)
                    cv=find_centroid(img)
                    if (cv is None):
                        vector=prev_vector
                        centroid= prev_centroid
                    else:
                        vector=cv[1]
                        centroid=cv[0]
                        prev_centroid=cv[0]
                        prev_vector=cv[1]
                    print(centroid)
                    #move right if 180 angle and left if 90 angle
                    angle=np.degrees(np.math.atan2(np.linalg.det([vector_path[i],vector]),np.dot(vector_path[i],vector)))
                    print(angle)
            if(angle>=1 and angle<=50):
                while(angle>=1):
                    print(i)
                    print("turning right")
                    p.stepSimulation()
                    env.move_husky(-5,5 , -5, 5)
                    img=env.camera_feed()
                    img=cv2.cvtColor(cv2.resize(img,(512,512)), cv2.COLOR_BGR2GRAY)
                    cv=find_centroid(img)
                    if (cv is None):
                        vector=prev_vector
                        centroid= prev_centroid
                    else:
                        vector=cv[1]
                        centroid=cv[0]
                        prev_centroid=cv[0]
                        prev_vector=cv[1]
                    #move right if 180 angle and left if 90 angle
                    angle=np.degrees(np.math.atan2(np.linalg.det([vector_path[i],vector]),np.dot(vector_path[i],vector)))
                    print(angle)
    return prev_vector



start_vector=[-1,0]

if __name__=="__main__":
    parent_path = os.path.dirname(os.getcwd())
    os.chdir(parent_path)
    env = gym.make("pix_main_arena-v0")
    time.sleep(2)
    img=env.camera_feed()
    img=cv2.cvtColor(cv2.resize(img,(512,512)), cv2.COLOR_BGR2GRAY)
    time.sleep(1)
    env.respawn_car()
    cv2.imshow('feed', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #[144, 143, 142, 130, 118, 106, 94, 82, 70, 58, 57, 56, 44, 32, 20.0]
    #nodes_cord0=[[466, 466], [427, 464], [393, 464], [393, 426], [393, 388], [393, 350], [393, 312], [393, 274], [393, 249], [393, 200], [351, 200], [312, 200], [312, 164], [312, 122], [312, 84], [312, 30], [275, 47], [222, 47], [236, 84], [236, 122], [236, 176], [290, 160], [275, 198], [275, 236], [275, 255], [275, 312], [275, 350], [275, 405], [312, 395], [275, 388], [275, 350], [275, 312], [236, 312], [199, 312], [147, 312], [160, 274], [160, 236], [122, 236], [84, 236], [46, 236], [46, 274],[46, 236], [84, 236], [122, 236], [122, 198], [160, 198], [157, 161], [199, 160], [236, 160], [275, 160], [275, 198], [312, 198], [351, 198], [351, 236], [351, 274], [389, 274], [427, 274]]
    nodes_cord0=[[466, 466], [427, 464], [393, 464], [388, 416], [388, 378], [388, 340], [388, 302], [388, 264], [388, 249], [388, 193], [342, 198], [304, 196], [312, 153], [312, 116], [312, 76], [312, 30], [265, 45], [220, 44], [236, 90], [236, 133], [236, 170], [285, 160], [275, 209], [275, 246], [275, 265], [275, 322], [275, 360], [275, 412], [324, 390], [267, 389], [275, 345], [275, 303], [227, 313], [185, 313], [147, 313], [161, 264], [160, 224], [112, 236], [76, 236], [26, 236], [45, 285],[47, 226], [95, 236], [132, 236], [122, 188], [171, 198], [162, 152], [208, 161], [245, 161], [285, 161], [275, 208], [320, 198], [361, 198], [351, 246], [351, 285], [398, 276], [437, 274]]
    prev_centroid=nodes_cord0[0]
    prev_vector=[-1,0]
    prev_vector=runBot(nodes_cord0,prev_vector,prev_centroid)
    #print(prev_vector)
    #[20.0, 8, 7, 6, 18, 30, 42, 43, 55, 67, 79, 91, 103, 115, 116.0]


    #nodes_cord1=[[312, 84], [312, 30], [275, 47], [222, 47], [236, 84], [236, 122], [236, 176], [290, 160], [275, 198], [275, 236], [275, 255], [275, 312], [275, 350], [275, 395], [312, 395]]
    #prev_centroid=nodes_cord1[0]
    #prev_vector=runBot(nodes_cord1,prev_vector,prev_centroid)
    
    #[116.0, 115, 103, 91, 90, 89, 88, 76, 64, 63, 62, 61, 73.0]
    #nodes_cord2=[[312, 388], [275, 388], [275, 350], [275, 312], [236, 312], [199, 312], [160, 312], [160, 274], [160, 236], [122, 236], [84, 236], [46, 236], [46, 274]]
    #prev_centroid=nodes_cord2[0]
    #prev_vector=runBot(nodes_cord2,prev_vector,prev_centroid)
    
    #[73.0, 61, 62, 63, 51, 52, 40, 41, 42, 43, 55, 56, 57, 69, 81, 82, 83.0]
    #nodes_cord3=[[46, 274], [46, 236], [84, 236], [122, 236], [122, 198], [160, 198], [157, 161], [199, 160], [236, 160], [275, 160], [275, 198], [312, 198], [351, 198], [351, 236], [351, 274], [389, 274], [427, 274]]
    #prev_centroid=nodes_cord3[0]
    #prev_vector=runBot(nodes_cord3,prev_vector,prev_centroid)



        


