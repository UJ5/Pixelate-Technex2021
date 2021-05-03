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


if __name__=="__main__":
    parent_path = os.path.dirname(os.getcwd())
    os.chdir(parent_path)
    env = gym.make("pix_main_arena-v0")
    time.sleep(2)
    #while True:
    #    p.stepSimulation()
    #    env.move_husky(0.2, 0.2, 0.2, 0.2)
    env.remove_car()
    img=env.camera_feed()
    img=cv2.resize(img,(512,512))
    img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    time.sleep(1)
    env.respawn_car()
    cv2.imshow('feed', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #time.sleep(100)
    cd=np.zeros((12,12),dtype='i,i')
    
    

    #red_masking
    mask_red=np.copy(img1)
    row,columns=mask_red.shape
    for i in range(0,row):
        for j in range(0,columns):
            if ( mask_red[i][j]==43):
                mask_red[i][j]=255
            else: 
                mask_red[i][j]=0

    red_edges = cv2.Canny(mask_red,10,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated_red = cv2.dilate(red_edges, kernel)
    red_contours,hierarchy = cv2.findContours(dilated_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #draw=cv2.drawContours(img, red_contours, -1,(0,0,255),3)
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if(15<(perimeter*perimeter)/area<18 and area>800):
        #    shape="sqr"
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print(cx,cy)
            cd[(cy-33)//37][(cx-32)//37]=(cx,cy)
        #else:
        #    shape="null" 
        #print(shape)
        #print("peri/area:",(perimeter*perimeter)/area)
        #print("area:",area)
        #print(cx,cy)
        #print("\n")

    cv2.imshow('mask_red', red_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cd)
    
    
    
    
    #green_masking
    mask_green=np.copy(img1)
    row,columns=mask_green.shape
    for i in range(0,row):
        for j in range(0,columns):
            if (mask_green[i][j]==133):
                mask_green[i][j]=255
            else: 
                mask_green[i][j]=0
    green_edges = cv2.Canny(mask_green,100,100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated_green = cv2.dilate(green_edges, kernel)
    green_contours,hierarchy = cv2.findContours(dilated_green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in green_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        if(15<(perimeter*perimeter)/area<18 and area>800):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            shape="sqr"
        else:
            shape="null"
        print(area) 
        print(cx,cy)
        cd[(cy-33)//37][(cx-33)//37]=(cx,cy)
        print((perimeter*perimeter)/area)
    cv2.imshow('mask_green', dilated_green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cd)
    
    
    
    
    #### yellow_masking
    mask_yellow=np.copy(img1)
    row,columns=mask_yellow.shape
    for i in range(0,row):
        for j in range(0,columns):
            if (mask_yellow[i][j]==201):
                mask_yellow[i][j]=255
            else: 
                mask_yellow[i][j]=0
    yellow_edges = cv2.Canny(mask_yellow,100,100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated_yellow = cv2.dilate(yellow_edges, kernel)
    yellow_contours,hierarchy = cv2.findContours(dilated_yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in yellow_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        if(15<(perimeter*perimeter)/area<18 and area>800):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            shape="sqr"
        #print(shape) 
        print(cx,cy)
        cd[(cy-33)//37][(cx-33)//37]=(cx,cy)
        print((perimeter*perimeter)/area)
    cv2.imshow('mask_yellow', dilated_yellow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cd)



    #white_masking
    mask_white=np.copy(img1)
    row,columns=mask_white.shape
    for i in range(0,row):
        for j in range(0,columns):
            if (mask_white[i][j]==227):
                mask_white[i][j]=255
            else: 
                mask_white[i][j]=0
    white_edges = cv2.Canny(mask_white,10,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated_white = cv2.dilate(white_edges, kernel)
    white_contours,hierarchy = cv2.findContours(dilated_white,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in white_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        if(15<(perimeter*perimeter)/area<18 and area>800):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cd[(cy-33)//37][(cx-33)//37]=(cx,cy)
            shape="sqr"
        #print(shape) 
        #print(cx,cy)
        print((perimeter*perimeter)/area)
    cv2.imshow('mask_white', dilated_white)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cd)
    
    
    #pink_masking
    mask_pink=np.copy(img1)
    row,columns=mask_pink.shape
    for i in range(0,row):
        for j in range(0,columns):
            if (mask_pink[i][j]==154):
                mask_pink[i][j]=255
            else: 
                mask_pink[i][j]=0
    pink_edges = cv2.Canny(mask_pink,10,10)
    kernel_pink = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated_pink = cv2.dilate(pink_edges, kernel)
    pink_contours,hierarchy = cv2.findContours(dilated_pink,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in pink_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        if(15<(perimeter*perimeter)/area<18 and area>800):
            shape="sqr"
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        #print(shape) 
        print(cx,cy)
        cd[(cy-33)//37][(cx-33)//37]=(cx,cy)
        print((perimeter*perimeter)/area)
    cv2.imshow('mask_pink', dilated_pink)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cd)


    #sky_masking
    mask_sky=np.copy(img1)
    row,columns=mask_sky.shape
    for i in range(0,row):
        for j in range(0,columns):
            if (mask_sky[i][j]==159):
                mask_sky[i][j]=255
            else: 
                mask_sky[i][j]=0
    sky_edges = cv2.Canny(mask_sky,10,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated_sky = cv2.dilate(sky_edges, kernel)
    sky_contours,hierarchy = cv2.findContours(dilated_sky,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sky_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        if(15<(perimeter*perimeter)/area<18 and area>800):
            shape="sqr"
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        #print(shape) 
        print(cx,cy)
        cd[(cy-33)//37][(cx-33)//37]=(cx,cy)
        print((perimeter*perimeter)/area)
    cv2.imshow('mask_sky', dilated_sky)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(len(sky_contours))
    
    
    
    
    mask_s=np.copy(img1)
    row,columns=mask_s.shape
    for i in range(0,row):
        for j in range(0,columns):
            if (mask_s[i][j]==53):
                mask_s[i][j]=255
            else: 
                mask_s[i][j]=0
    s_edges = cv2.Canny(mask_s,10,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated_s = cv2.dilate(s_edges, kernel)
    s_contours,hierarchy = cv2.findContours(dilated_s,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in s_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        if(12<(perimeter*perimeter)/area<16 and area>800):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print(cx,cy)
            cd[(cy-33)//37][(cx-33)//37]=(cx,cy)
        print(area)
        print((perimeter*perimeter)/area)
    cv2.imshow('mask_s',dilated_s)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cd)



    #designating each node in arena
    count=0
    nodes=np.zeros((12,12))
    for i in range(12):
        for j in range(12):
            count=count+1
            nodes[i][j]=count

    print(nodes,"\n")
    print(cd)




    #dijkstra_algorithm_to_find_out_minimum_path

    class Graph(object):
        def __init__(self):
            self.nodes = set()
            self.edges = defaultdict(list)
            self.distances = {}

        def add_node(self, value):
            self.nodes.add(value)

        def add_edge(self, from_node, to_node, distance):
            self.edges[from_node].append(to_node)
            self.edges[to_node].append(from_node)
            self.distances[(from_node, to_node)] = distance


    def dijkstra(graph, initial):
        visited = {initial: 0}
        path = {}

        nodes = set(graph.nodes)

        while nodes:
            min_node = None
            for node in nodes:
                if node in visited:
                    if min_node is None:
                        min_node = node
                    elif visited[node] < visited[min_node]:
                        min_node = node
            if min_node is None:
                break

            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in graph.edges[min_node]:
                try:
                    weight = current_weight + graph.distances[(min_node, edge)]
                except:
                    continue
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node

        return visited, path


    def shortest_path(graph, origin, destination):
        visited, paths = dijkstra(graph, origin)
        full_path = deque()
        _destination = paths[destination]

        while _destination != origin:
            full_path.appendleft(_destination)
            _destination = paths[_destination]

        full_path.appendleft(origin)
        full_path.append(destination)

        return visited[destination], list(full_path)

    
    
    exl_nodes=[15,40,44,70,97]
    g=Graph()
    for i in range(144):
        g.add_node(int(i+1))
        
    def weights(x,y):
        if(img1[cd[x][y][1],cd[x][y][0]]==154):
            return(1)
        if(img1[cd[x][y][1],cd[x][y][0]]==25):
            return(1)
        if(img1[cd[x][y][1],cd[x][y][0]]==201):
            return(3)
        if(img1[cd[x][y][1],cd[x][y][0]]==133):
            return(2)
        if(img1[cd[x][y][1],cd[x][y][0]]==227):
            return(1)
        if(img1[cd[x][y][1],cd[x][y][0]]==43):
            return(4)
        if(img1[cd[x][y][1],cd[x][y][0]]==0):
            return(0)
        if(img1[cd[x][y][1],cd[x][y][0]]==53):
            return(1)

    for i in range(12):
        for j in range(12):
            curr_node=nodes[i][j]
            if(img1[cd[i][j][1]][cd[i][j][0]]!=0):
                if(i-1>=0 and img1[cd[i-1][j][1]][cd[i-1][j][0]]!=0):
                    if curr_node not in exl_nodes and nodes[i-1][j] not in exl_nodes:
                        dist=weights(i-1,j)
                        g.add_edge(int(curr_node),int(nodes[i-1][j]),dist)
                if(j-1>=0 and img1[cd[i][j-1][1]][cd[i][j-1][0]]!=0):
                    if curr_node not in exl_nodes and nodes[i][j-1] not in exl_nodes:
                        dist=weights(i,j-1)
                        g.add_edge(int(curr_node),int(nodes[i][j-1]),dist)
                if(i+1<12 and img1[cd[i+1][j][1]][cd[i+1][j][0]]!=0):
                    if curr_node not in exl_nodes and nodes[i+1][j] not in exl_nodes:
                        dist=weights(i+1,j)
                        g.add_edge(int(curr_node),int(nodes[i+1][j]),dist)
                if(j+1<12 and img1[cd[i][j+1][1]][cd[i][j+1][0]]!=0):
                    if curr_node not in exl_nodes and nodes[i][j+1] not in exl_nodes:
                        dist=weights(i,j+1)
                        g.add_edge(int(curr_node),int(nodes[i][j+1]),dist)

    g.add_edge(int(15),int(14),2)
    g.add_edge(int(14),int(15),2)
    g.add_edge(int(16),int(15),2)
    g.add_edge(int(28),int(40),3)
    g.add_edge(int(39),int(40),3)
    g.add_edge(int(52),int(40),3)
    g.add_edge(int(40),int(41),1)
    g.add_edge(int(44),int(32),3)
    g.add_edge(int(43),int(44),2)
    g.add_edge(int(56),int(44),2)
    g.add_edge(int(70),int(58),1)
    g.add_edge(int(69),int(70),1)
    g.add_edge(int(82),int(70),1)
    g.add_edge(int(97),int(85),4)
    g.add_edge(int(98),int(97),3)
    g.add_edge(int(144),int(132),4)
    g.add_edge(int(144),int(143),2)
    g.add_edge(int(115),int(116),0)
    g.add_edge(int(116),int(117),4)
    g.add_edge(int(82),int(83),0)
    g.add_edge(int(83),int(84),4)
    print(nodes,"\n")
    print(g.distances)
    
    
    #p,f = dijkstra(g,144)
    #print(f)
    #print("\n")
    #printing_paths and centeroids of nodes involved
    path_nodes=shortest_path(g, 144, 20)[1]
    print(path_nodes)
    #g.add_edge(int(76),int(64),2)
    #g.add_edge(int(76),int(88),3)
    #g.add_edge(int(88),int(76),0)
    #g.add_edge(int(64),int(76),0)
    
    path_nodes1=shortest_path(g, 20, 116)[1]
    print(path_nodes1)
    path_nodes2=shortest_path(g, 116, 73)[1]
    print(path_nodes2)
    
    #g.add_edge(int(32),int(20),2)
    #g.add_edge(int(32),int(44),2)
    #g.add_edge(int(20),int(32),0)
    #g.add_edge(int(44),int(32),0)
    
    path_nodes3=shortest_path(g, 73, 83)[1]
    print(path_nodes3)
    
    #path_nodes2.pop(0)
    #print(path_nodes2)
    
    
    co_ord=[]
    for i in path_nodes:
        for j in range(12):
            for k in range(12):
                if(i==nodes[j][k]):
                    co_ord.append(np.array([cd[j][k][0],cd[j][k][1]]))
    
    for i in path_nodes1:
        for j in range(12):
            for k in range(12):
                if(i==nodes[j][k]):
                    co_ord.append(np.array([cd[j][k][0],cd[j][k][1]]))
    
    for i in path_nodes2:
        for j in range(12):
            for k in range(12):
                if(i==nodes[j][k]):
                    co_ord.append(np.array([cd[j][k][0],cd[j][k][1]]))
    
    for i in path_nodes3:
        for j in range(12):
            for k in range(12):
                if(i==nodes[j][k]):
                    co_ord.append(np.array([cd[j][k][0],cd[j][k][1]]))
                    
    print(len(co_ord))
    print(co_ord)
    print("\n")
    
                    
    
    u1=np.array([0,-1])
    u2=np.array([0,1])
    u3=np.array([1,0])
    u4=np.array([-1,0])
    
    
    
    for i in range(len(co_ord)-1):
        print(co_ord[i],"\n")   
        dist=100000000000000
        para=20
        #corners=[[466, 495],
        #         [466, 465],
        #         [496, 466],
        #         [495, 495]]
        #p1=np.array(list(corners[1]))
        #p2=((corners[0][0]+corners[1][0])//2,(corners[0][1]+corners[1][1])//2)
        #p1=((corners[1][0]+corners[3][0])//2,(corners[1][1]+corners[3][1])//2)
        #v0 = np.array(p2) - np.array(p1)
        #p3=np.array(list(co_ord[i]))
        #p4=np.array(list(co_ord[i+1]))
        #v1 = np.array(p4) - np.array(p3)
        #angle = np.math.atan2(np.linalg.det([v1,v0]),np.dot(v1,v0))
        #print("angle between:",v0,v1)
        #print (np.degrees(angle))
        x=0
        #angle1=-90
        u=0
        while (dist>=para):
            img2=env.camera_feed()
            img2=cv2.resize(img2,(512,512))
            gray=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY )
            #cv2.imshow('detect', gray)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            ARUCO_PARAMETERS = aruco.DetectorParameters_create()
            ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            corners=np.squeeze(corners)
            ids=np.squeeze(ids) 
            if(ids.any()==None):
                print("Trapped: running blindly")
                if(x<100):
                    p.stepSimulation()
                    env.move_husky(6, 6, 6, 6)
                    x=x+1
                elif(x>=100):
                    if(np.math.atan2(np.linalg.det([v1,u1]),np.dot(v1,u1))<=10 and np.math.atan2(np.linalg.det([v1,u1]),np.dot(v1,u1))>=-10):
                        p.stepSimulation()
                        env.move_husky(-2, 2, -2, 2)
                    elif(np.math.atan2(np.linalg.det([v1,u2]),np.dot(v1,u2))<=10 and np.math.atan2(np.linalg.det([v1,u2]),np.dot(v1,u2))>=-10):
                        p.stepSimulation()
                        env.move_husky(2, -2, 2, -2)
                    elif(np.math.atan2(np.linalg.det([v1,u3]),np.dot(v1,u3))<=10 and np.math.atan2(np.linalg.det([v1,u3]),np.dot(v1,u3))>=-10):
                        p.stepSimulation()
                        env.move_husky(-2, 2, -2, 2)
                    elif(np.math.atan2(np.linalg.det([v1,u4]),np.dot(v1,u4))<=10 and np.math.atan2(np.linalg.det([v1,u4]),np.dot(v1,u4))>=-10):
                        p.stepSimulation()
                        env.move_husky(2, -2, 2, -2)
                    
            else:   
                print(ids)
                print(corners)
                #x=x+1
                #if(i==10 or i==17 or i==9):

                #while(ids==None):
                #    p.stepSimulation()
                #    env.move_husky(1, 1, 1, 1)
                #if(i!=10 and i!=13):
                cent=(corners[0]+corners[1])//2
                #elif(i==10 and i==13):
                #    cent1=(corners[0]+corners[1])//2
                #    cent2=(corners[0]+corners[2])//2
                #    cent=(cent1+cent2)//2
                p1=np.array(list((corners[0]+corners[1])//2))
                p2=np.array(list((corners[1]+corners[3])//2))
                v0 = np.array(p1) - np.array(p2)
                if(i+3<=(len(co_ord)-1)):
                    p3=np.array(list(co_ord[i]))
                    p4=np.array(list(co_ord[i+1]))
                    v1 = np.array(p4) - np.array(p3)
                    angle1 = np.math.atan2(np.linalg.det([v1,v0]),np.dot(v1,v0))
                    p5=np.array(list(co_ord[i+1]))
                    p6=np.array(list(co_ord[i+2]))
                    v2 = np.array(p6) - np.array(p5)
                    angle2 = np.math.atan2(np.linalg.det([v2,v0]),np.dot(v2,v0))
                    p7=np.array(list(co_ord[i+2]))
                    p8=np.array(list(co_ord[i+3]))
                    v3 = np.array(p8) - np.array(p7)
                    angle3 = np.math.atan2(np.linalg.det([v3,v0]),np.dot(v3,v0))
                
                if(i==11 and dist<=27 and dist>=22 and u==0):
                #if(path_nodes[i]== path_nodes[len(path_nodes)-2] and u==0):
                    p.stepSimulation()
                    time.sleep(3)
                    env.remove_cover_plate(3,3)
                    u=1
                #if(i==29 and dist==30):
                #if(path_nodes[i]== path_nodes[len(path_nodes)-2] and u==0):
                #    p.stepSimulation()
                #    time.sleep(3)
                #    env.remove_cover_plate(3,3) 
                if(i==31 and dist<=25 and dist>=22 and u==1):
                    p.stepSimulation()
                    time.sleep(3)
                    env.remove_cover_plate(8,0)
                    u=2
                p3=np.array(list(co_ord[i]))
                p4=np.array(list(co_ord[i+1]))
                v1 = np.array(p4) - np.array(p3)
                angle = np.math.atan2(np.linalg.det([v1,v0]),np.dot(v1,v0))
                print("angle between:",v0,v1)
                angle1=np.degrees(angle)
                print (np.degrees(angle))
                dist= math.sqrt( (cent[0]-p4[0])**2 + (cent[1]-p4[1])**2 )
                print("dist",dist)
                print("i=",i)
                #u=0
                #if(dist<=15 and dist>=12):
                #    p.stepSimulation()
                #    env.move_husky(-10000, -10000,-10000,10000)
                    #break
                #if(dist>15):    
                if(np.degrees(angle1)<=2.5 and -2.5<=np.degrees(angle1) and np.degrees(angle2)<=2.5 and -2.5<=np.degrees(angle2) and np.degrees(angle3)<=2.5 and -2.5<=np.degrees(angle3)):
                    print("forward high speed")
                    p.stepSimulation()
                    env.move_husky(20, 20, 20, 20) 
                if(np.degrees(angle)<=1 and -1<=np.degrees(angle)):
                    print("forward_slowly")
                    p.stepSimulation()
                    env.move_husky(10, 10, 10, 10)
                if(np.degrees(angle)>1 and np.degrees(angle)<=90):
                    print("Turning left")
                    p.stepSimulation()
                    env.move_husky(-80, 80, -80, 80)
                if(np.degrees(angle)<=180 and 90<np.degrees(angle)):
                    p.stepSimulation()
                    env.move_husky(-5,5,-5,5)
                if(np.degrees(angle)>=-179 and np.degrees(angle)<-1):
                    print("Turning right")
                    p.stepSimulation()
                    env.move_husky(80, -80, 80, -80)
            
     
    
    
