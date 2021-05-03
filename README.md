# Pixelate-Technex'2021

Pixelate is an image procesing event under Technex (annual techno-management fest of IIT-BHU Varanasi).
This repository is about my work on pixelate projcet. I have uploaded the problem statement of this event, you must check it. 

**Problem Statement**: Pixelate of Technex 2021, expexted me to write a python code using the OpenCV library to perform the color segemntation,identification and
shape detection on a colored grid arena. Continious live feed of arena is provided by a overhead cam(virtually). The detailed porblem statement can be hound here:


**My approach to complete the task:**

- I used openCV library to detect the different colors, shapes. 
- Then, centroid of each shapes(nodes) are found using moments method in openCV.
- Based on a cost of colors detected, a directional weighted graph is made for further implementation of Dijkstra algorithm.  
- Basically, centroid of the nodes are used to connect the nodes in Dijkstra algorithm.
- Shortest path with less cost is found using Dijkstra algorithm.
- Bot is moved forward on calculated shortest path. Live position of bot-car is identified by aruco marker on its top.

**Check out the youtube video here:**
![Watch the video](https://i9.ytimg.com/vi/kUSyqiDj9bw/mq2.jpg?sqp=CIyGwIQG&rs=AOn4CLDehS5xtrFBPBfm4CxWPMQLNwaLtA)](https://youtu.be/kUSyqiDj9bw)
