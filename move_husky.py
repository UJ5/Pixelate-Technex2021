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
    img=cv2.cvtColor(cv2.resize(img,(512,512)), cv2.COLOR_BGR2GRAY)
    time.sleep(1)
    env.respawn_car()
    cv2.imshow('feed', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cd=np.zeros((12,12),dtype='i,i') 


    sqr_hospital_node=[]
    cir_hospital_node=[]
    triangle_nodes=[]


    #blue_masking for classifying the shapes
    mask=np.copy(img)
    row,columns=mask.shape
    for i in range(0,row):
        for j in range(0,columns):
            if ( mask[i][j]==26):
                mask[i][j]=255
            else: 
                mask[i][j]=0
    edges = cv2.Canny(mask,10,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated = cv2.dilate(edges, kernel)
    contours,hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    draw=cv2.drawContours(img, contours, -1,(0,0,255),3)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if(14<(perimeter*perimeter)/area<17 and area>200):
            shape="sqr"
            sqr_hospital_node.append((((cy-33)//37),(cx-32)//37))
            cd[(cy-33)//37][(cx-32)//37]=(cx,cy)
        elif((perimeter*perimeter)/area<14 and area>200):
            shape="circle"
            cir_hospital_node.append((((cy-33)//37),(cx-32)//37))
            cd[(cy-33)//37][(cx-32)//37]=(cx,cy)
        elif(18<(perimeter*perimeter)/area<25 and area>200):
            shape="triangle"
            triangle_nodes.append((((cy-33)//37),(cx-32)//37))
            box = cv2.boundingRect(cnt)
            cd[(cy-33)//37][(cx-32)//37]=(cx,cy)
             
        else:
            shape="null"
        print(shape)
        print("peri/area:",(perimeter*perimeter)/area)
        print("area:",area)
        print(cx,cy)
        print("\n")

    cv2.imshow('mask', dilated )
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    #red_masking
    mask_red=np.copy(img)
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
    draw=cv2.drawContours(img, red_contours, -1,(0,0,255),3)
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if(15<(perimeter*perimeter)/area<18 and area>800):
            shape="sqr"
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print(cx,cy)
            cd[(cy-33)//37][(cx-32)//37]=(cx,cy)
        else:
            shape="null" 
        print(shape)
        print("peri/area:",(perimeter*perimeter)/area)
        print("area:",area)
        print(cx,cy)
        print("\n")
        print(cd)

    cv2.imshow('mask_red', dilated_red )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cd)



    #green_masking
    mask_green=np.copy(img)
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
    mask_yellow=np.copy(img)
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
        print(shape) 
        print(cx,cy)
        cd[(cy-33)//37][(cx-33)//37]=(cx,cy)
        print((perimeter*perimeter)/area)
    cv2.imshow('mask_yellow', dilated_yellow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cd)


    
    #white_masking
    mask_white=np.copy(img)
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
    boxes=[]
    mask_pink=np.copy(img)
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
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append([x,y,w,h,(cy-33)//37,(cx-33)//37])
        print((perimeter*perimeter)/area)
    cv2.imshow('mask_pink', dilated_pink)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cd)

    



    #sky_masking
    mask_sky=np.copy(img)
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
    print(cd)


    #start position
    cd[11][11]=(466,466)

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

    g=Graph()
    for i in range(144):
        g.add_node(int(i+1))



    exl_nodes=[]
    for i in range(12):
        for j in range(12):
            if(img[cd[i][j][1],cd[i][j][0]]==26):
                exl_nodes.append(nodes[i][j])
    
    pink_nodes=[]
    for i in range(12):
        for j in range(12):
            if(img[cd[i][j][1],cd[i][j][0]]==154):
                pink_nodes.append((i,j))
    

    def weights(x,y):
        if(img[cd[x][y][1],cd[x][y][0]]==154):
            return(1)
        if(img[cd[x][y][1],cd[x][y][0]]==25):
            return(1)
        if(img[cd[x][y][1],cd[x][y][0]]==201):
            return(3)
        if(img[cd[x][y][1],cd[x][y][0]]==133):
            return(2)
        if(img[cd[x][y][1],cd[x][y][0]]==227):
            return(1)
        if(img[cd[x][y][1],cd[x][y][0]]==43):
            return(4)
        if(img[cd[x][y][1],cd[x][y][0]]==0):
            return(0)
        if(img[cd[x][y][1],cd[x][y][0]]==53):
            return(1)

    for i in range(12):
        for j in range(12):
            curr_node=nodes[i][j]
            if(img[cd[i][j][1]][cd[i][j][0]]!=0):
                if(i-1>=0 and img[cd[i-1][j][1]][cd[i-1][j][0]]!=0):
                    if curr_node not in exl_nodes and nodes[i-1][j] not in exl_nodes:
                        dist=weights(i-1,j)
                        g.add_edge(int(curr_node),int(nodes[i-1][j]),dist)
                if(j-1>=0 and img[cd[i][j-1][1]][cd[i][j-1][0]]!=0):
                    if curr_node not in exl_nodes and nodes[i][j-1] not in exl_nodes:
                        dist=weights(i,j-1)
                        g.add_edge(int(curr_node),int(nodes[i][j-1]),dist)
                if(i+1<12 and img[cd[i+1][j][1]][cd[i+1][j][0]]!=0):
                    if curr_node not in exl_nodes and nodes[i+1][j] not in exl_nodes:
                        dist=weights(i+1,j)
                        g.add_edge(int(curr_node),int(nodes[i+1][j]),dist)
                if(j+1<12 and img[cd[i][j+1][1]][cd[i][j+1][0]]!=0):
                    if curr_node not in exl_nodes and nodes[i][j+1] not in exl_nodes:
                        dist=weights(i,j+1)
                        g.add_edge(int(curr_node),int(nodes[i][j+1]),dist)



    g.add_edge(int(15),int(14),2)
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

    #print(nodes,"\n")
    #print(g.edges)
    #print(g.distances)


    path_nodes=shortest_path(g, 144, 20)[1]
    print("Pink_nodes: ",pink_nodes)
    print("Excluded nodes: ",exl_nodes)
    print("Circle hospital node:",cir_hospital_node)
    print("Square hospital node:",sqr_hospital_node)
    print(boxes)



    def identify_shape(img_p):
        row,columns=img_p.shape
        for i in range(0,row):
            for j in range(0,columns):
                if ( img_p[i][j]==26):
                    img_p[i][j]=255
                else: 
                    img_p[i][j]=0
        edges = cv2.Canny(img_p,10,10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        dilated = cv2.dilate(edges, kernel)
        contours,hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            shape = "null"
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            M = cv2.moments(cnt)
            if(area>310):
                shape="sqr"
            elif(area<310):
                shape="circle"
            #print(shape)
            #cv2.imshow("img_p",img_p)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #print(area)
            #print(perimeter)
            #print(perimeter*perimeter/area)
            break
        return shape

    def getVectorPath(path_nodes):
        vector_path=[]
        for n in path_nodes:
            flag=0
            for i in range(12):
                for j in range(12):
                    if(nodes[i][j]==n):
                        flag=1
                        break
                if(flag==1):
                    break        
            vector_path.append([cd[i][j][0],cd[i][j][1]])
        return vector_path



    start_node=144
    shape="circle"
    for co_ord in pink_nodes:
        path_nodes=shortest_path(g,start_node , nodes[co_ord[0]][co_ord[1]])[1]
        print("Path nodes: ",path_nodes)
        print("Co-ordinates of path: ",getVectorPath(path_nodes))
        #env.remove_cover_plate(co_ord[0],co_ord[1])
        #img_p=env.camera_feed()
        #img_p=cv2.cvtColor(cv2.resize(img_p,(512,512)), cv2.COLOR_BGR2GRAY)
        #for box in boxes:
        #    if(box[4]==co_ord[0] and box[5]==co_ord[1]):
        #        img_p=img_p[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        #        break
        #shape=identify_shape(img_p)
        if(shape=="circle"):
            g.add_edge(int(116),int(117),3)
            g.add_edge(int(116),int(115),2)
            g.add_edge(int(115),int(116),1)
            g.add_edge(int(117),int(116),1)
            path_nodes=shortest_path(g, nodes[co_ord[0]][co_ord[1]], nodes[cir_hospital_node[0][0]][cir_hospital_node[0][1]])[1]
            print("Path nodes: ",path_nodes)
            print("Co-ordinates of path: ",getVectorPath(path_nodes))
            start_node=nodes[cir_hospital_node[0][0]][cir_hospital_node[0][1]]
            shape="sqr"
        elif(shape=="sqr") :
            g.add_edge(int(83),int(84),4)
            g.add_edge(int(83),int(82),1)
            g.add_edge(int(84),int(83),1)
            g.add_edge(int(82),int(83),1)  
            path_nodes=shortest_path(g,nodes[co_ord[0]][co_ord[1]] , nodes[sqr_hospital_node[0][0]][sqr_hospital_node[0][1]])[1]
            print("Path nodes: ",path_nodes)
            print("Co-ordinates of path: ",getVectorPath(path_nodes))
            start_node=nodes[sqr_hospital_node[0][0]][sqr_hospital_node[0][1]]



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
            print("Target node: ",i)
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
            if(i<len(nodes_cord)-1):
                angle=np.degrees(np.math.atan2(np.linalg.det([vector_path[i],vector]),np.dot(vector_path[i],vector)))
            print("centroid of bot: ",centroid)
            print("Angle: ",angle)
            print("Distance :",dist)
            print("\n")
            if(dist<=17):
                idx=i
                frame=100
                while(frame):
                    p.stepSimulation()
                    env.move_husky(0, 0, 0, 0)
                    frame=frame-1
                i=i+1
                if(i==14 and count1==1):
                    print("At patient's Co-ordinate !")
                    print("Identifying shape...")
                    print("Circle !")
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
                    print("At patient's Co-ordinate !")
                    print("Identifying shape...")
                    print("Square !")
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
                    frame=500
                    while(frame):
                        print("ending")
                        p.stepSimulation()
                        env.move_husky(14,14,14,14)
                        frame=frame-1
                    break
                if(angle>=60 and angle<=180):
                    while(angle>=15 or (i==29 and angle>=1) or (i==41 and angle>=1)):
                        print("Target node: ",i)
                        print("Turning left...")
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
                        print("centroid of bot: ",centroid)
                        print("Angle: ",angle)
                        print("Distance :",dist)
                        print("\n")
                    frame=100
                    while(frame):
                        p.stepSimulation()
                        env.move_husky(0, 0, 0, 0)
                        frame=frame-1
                if(angle<=-60 and angle>=-180):
                    while(angle<=-15 or (i==29 and angle>=1) or (i==40 and angle>=9)):
                        print("Target node: ",i)
                        print("Turning right...")
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
                        angle=np.degrees(np.math.atan2(np.linalg.det([vector_path[idx],vector]),np.dot(vector_path[idx],vector)))
                        print("centroid of bot: ",centroid)
                        print("Angle: ",angle)
                        print("Distance :",dist)
                        print("\n")
                    frame=100
                    while(frame):
                        p.stepSimulation()
                        env.move_husky(0, 0, 0, 0)
                        frame=frame-1
            elif(dist>17):
                print("Going straight...")
                p.stepSimulation()
                env.move_husky(10, 10, 10, 10)
                if(angle<=-1 and angle>=-50):
                    while(angle<=-1):
                        print("Target node: ",i)
                        print("Turning right...")
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
                        
                        angle=np.degrees(np.math.atan2(np.linalg.det([vector_path[i],vector]),np.dot(vector_path[i],vector)))
                        print("centroid of bot: ",centroid)
                        print("Angle: ",angle)
                        print("Distance :",dist)
                        print("\n")
                if(angle>=1 and angle<=50):
                    while(angle>=1):
                        print("Target node: ",i)
                        print("Turning left...")
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
                        print("centroid of bot: ",centroid)
                        print("Angle: ",angle)
                        print("Distance :",dist)
                        print("\n")
        return prev_vector



    start_vector=[-1,0]

        #[144, 143, 142, 130, 118, 106, 94, 82, 70, 58, 57, 56, 44, 32, 20.0]
        #nodes_cord0=[[466, 466], [427, 464], [393, 464], [393, 426], [393, 388], [393, 350], [393, 312], [393, 274], [393, 249], [393, 200], [351, 200], [312, 200], [312, 164], [312, 122], [312, 84], [312, 30], [275, 47], [222, 47], [236, 84], [236, 122], [236, 176], [290, 160], [275, 198], [275, 236], [275, 255], [275, 312], [275, 350], [275, 405], [312, 395], [275, 388], [275, 350], [275, 312], [236, 312], [199, 312], [147, 312], [160, 274], [160, 236], [122, 236], [84, 236], [46, 236], [46, 274],[46, 236], [84, 236], [122, 236], [122, 198], [160, 198], [157, 161], [199, 160], [236, 160], [275, 160], [275, 198], [312, 198], [351, 198], [351, 236], [351, 274], [389, 274], [427, 274]]
    nodes_cord0=[[466, 466], [427, 464], [393, 464], [388, 416], [388, 378], [388, 340], [388, 302], [388, 264], [388, 249], [388, 193], [342, 198], [304, 196], [312, 153], [312, 116], [312, 76], [312, 30], [265, 45], [220, 44], [236, 90], [236, 133], [236, 170], [285, 160], [275, 209], [275, 246], [275, 265], [275, 322], [275, 360], [275, 412], [324, 390], [267, 389], [275, 345], [275, 303], [227, 313], [185, 313], [147, 313], [161, 264], [160, 224], [112, 236], [76, 236], [26, 236], [45, 285],[47, 226], [95, 236], [132, 236], [122, 188], [171, 198], [162, 152], [208, 161], [245, 161], [285, 161], [275, 208], [320, 198], [361, 198], [351, 246], [351, 285], [398, 276], [437, 274]]
    prev_centroid=nodes_cord0[0]
    prev_vector=[-1,0]
    prev_vector=runBot(nodes_cord0,prev_vector,prev_centroid)