# Technex'2021
# Pixelate-A computer project:

Pixelate is an image procesing and robotics event of Technex (annual techno-management fest of IIT-BHU Varanasi).
This repository is about my work on pixelate projcet. 

#### Language: Pyhton
#### Modules used: OpenCV, Numpy, math, gym, pybullet  

**Problem Statement**: [Detailed Problem Statement](https://github.com/ujjawalmodanwal/Pixelate-Technex2021/blob/main/Pixelate21_PS.pdf).

Before moving, forward, I recommend to go throught the PS to understand what I did :) 


## My approach to complete the task: ## 

I divided this task in two parts: 
1) Preprocessing part
2) Running the bot car on arena

<h2>Preprocessing part</h2>

- Here, is the colored image of scanned arena provided to me:

<img src=https://github.com/ujjawalmodanwal/Pixelate-Technex2021/blob/main/arena(2).png>

- I did image processing using the OpenCV module for detecting the different colors, shapes present on the arena. 

- Different colors are detected using the pixel values for particular color. I created mask for each color (rest colors converted to black). I found the canny edges, contours of shapes (differen rectangles of same color) and then used moment menthod on that contour points to find the centroid of each square node. Contour is nothing but a set of continious points for each shape.Below is the image:


     ![shape_mask](https://github.com/ujjawalmodanwal/Pixelate-Technex2021/blob/main/images/shape_mask.png)       ![white_mask](https://github.com/ujjawalmodanwal/Pixelate-Technex2021/blob/main/images/white_mask.png)
- I classified the shapes on the basis of following formula: **(perimeter* perimeter)/area** . If its values <= 14, then its a circle, if 14< and <17 then square, else it will be a triangle. In all the cases area bounded by each shape is >200.

- After I created a matrix of 12 * 12 matrix. Each cell of this matrix has centroid coordinates of nodes. Below is how this matrix looked like ((0,0) represents the restricted node):


     ![centroid matrix](https://github.com/ujjawalmodanwal/Pixelate-Technex2021/blob/main/images/Coordinates_matrix.png)
- Now based on differen colors and weights as given in the problem statement, I created a directed graph using the adjacency list in python. I connected the node values (1,2,3...143,144). While creating the graph, I excluded those nodes whose centroid coordinate was giving the pixel values of black color.
         
- After creating the directional graph, I used the **Dijkstra algorithm** for finding the path that has shortest cost from destination to source. Input for Dijkstra algorithm was adjacency list, containing connected nodes via their color weights. Output is list of nodes which has shortest path.    

<h2>Running the bot-car</h2>

After finding the node path, its time to roll the botcar. But how ? Here is what I did:

-There is an **aruco marker** provided to us on the top of the car. It moves with the car. Aruco markers can be used to identify the location of an object in an image. Its is a rectangular marker. It provides us the coordinates of its four corners in terms of pixel locations. Using these coordinates, we can find the centroid of marker and so of the bot car.

- The distance between centroid of aruco marker and nodes coordinates is what, I used to move the car. I moved the car straight until the distance between these to points becomes less than a certain threshold value. 

- In case, where I had to rotate the car, I used the angle between vectors. I created an abstract vector on top of the car. It is always directed towards the forward moment of car. Another vector I created, between current node at which car is present and the next node where it has to go. I used nodes coordinates to create this vector.

- Then I was tracking the angle between botcar's vector and the path vector. 
  - If -1<= angle <=1, moved the car straight.  
  - If angle>=1, turn the car in anticloackwise direction until it comes in the range of -1 to 1.
  - If angle =<-1 turn the car in clockwise direction until it comes in the range of -1 to 1.

- While maintaing the above strategy, I moved the car. 

<h3>To see the above explained process in live run:</h3>
Check out the youtube video here: 



[<img src="https://github.com/ujjawalmodanwal/Pixelate-Technex2021/blob/main/images/thumbnail.png" width="600" height="300">](https://youtu.be/3k_BcKAblLI)
