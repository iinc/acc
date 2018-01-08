# Adaptive Cruise Control

The goal of this project is to implement an [adaptive cruise control](https://en.wikipedia.org/wiki/Autonomous_cruise_control_system) system without relying on expensive radar sensors. The biggest challenge is accurately determining the distance from a vehicle in lane in front of the car. 

## Computer Vision  
My first attempt at determining the distance from cars relies on computer vision with [OpenCV](https://opencv.org/). I will walk you through the algorithm.

Initial image. The green lines show the area we are focusing on.

![](images/dAQGWua.jpg?raw=true)

Convert the image to gray scale, blur it using a Gaussian filter to reduce noise, and apply Canny edge detection.

![](images/wmHKDcW.jpg?raw=true)

Transform the image to only contain the area we want to focus on.

![](images/N255LUZ.png?raw=true)

Attempt to detect the empty potion of the lane. It starts at the bottom, center of the image and works up. The edges of the lane influences that starting point in the next row.

![](images/3SdwGub.png?raw=true)

The empty lane is transformed back and drawn on the initial image.

![](images/zPmd2uY.jpg?raw=true)

This works well when the road surface is a single color, the road markings are clear, and there are no shadows on the road.
You can see this technique applied to a video here: [https://www.youtube.com/watch?v=20JxqClLxec](https://www.youtube.com/watch?v=20JxqClLxec)
There are plenty of improvements that could be done, but I decided to try out a different approach. I may come back to this in the future. 

### Convolution Neural Networks
My second attempt was to use a neural network as it would be able to handle the noise on the road more easily. 

Initial image.

![](images/fZQtLj4.jpg?raw=true)

Crop and transform the image to the portion we care about.

![](images/j3rWyEn.jpg?raw=true)

Scale the image down so that it is quicker for the neural network to process.

![](images/bTqLnGK.jpg?raw=true)

I did this to thousands of images and manually classified them based on the distance to the car in front. 

![](images/HQqT7Y8.jpg?raw=true)
![](images/DwsvqP3.jpg?raw=true)

After for training for 10 hours on 100,000 images, here is the result.

[https://youtu.be/8dSeKicEVf0](https://youtu.be/8dSeKicEVf0) 

[https://youtu.be/ECbU_EvyUqM](https://youtu.be/ECbU_EvyUqM)

![](images/Dws3qP3.jpg?raw=true)

The green lines represent the different classes. The thickness of the line increases as the network is more confident that the image belongs in that class. The network was not trained on these videos. It is important to note that the network is not confused by shadows on the road like the computer vision approach was. 

By knowing the approximate distance from the car in front, the vehicles speed can be automatically adjusted to maintain a safe following distance.

