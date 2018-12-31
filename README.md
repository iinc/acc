# Adaptive Cruise Control Prototype

The goal of this project was to create an [adaptive cruise control](https://en.wikipedia.org/wiki/Autonomous_cruise_control_system) prototype without relying on expensive hardware components (such as radar or lidar). The biggest challenge was figuring out how to accurately determine the distance to the vehicle ahead of my car. To do this, I relied on computer vision and machine learning techniques to analyze images from a camera mounted at the top of the windshield.

### Computer Vision
My first attempt at determining the distance from cars relies on computer vision with the [OpenCV](https://opencv.org/) library. I will walk you through the algorithm.

Here is the initial image. The green lines highlight the area we are focusing on.

![](images/dAQGWua.jpg?raw=true)

First, we convert the image to gray scale `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`, blur it using a Gaussian filter to reduce noise `cv2.GaussianBlur(grayscale_image, (3, 3), 0)`, and apply edge detection `cv2.Canny(blurred_image, low_threshold, high_threshold)`. This leaves us with a quiet image where the lane is easily distinguishable.

![](images/wmHKDcW.jpg?raw=true)

Next, we can warp/transform the image to only contain the area we want to focus on. The green lines in the first image shown outline the source area. The destination area is a rectangle with dimensions equal to the maximum width/height of the source area. 

`cv2.warpPerspective(image, cv2.getPerspectiveTransform(src_area, dst_area), dst_dimensions, flags=cv2.INTER_LINEAR & cv2.WARP_FILL_OUTLIERS)`


![](images/N255LUZ.png?raw=true)

Then we can try to detect the empty portion of the lane. We start out at the bottom center of the image and work our way up. The green dots represent the starting position in each row. The horizontal white lines extend out from the dot until they contact an edge. The edge intersections of the current row influce the starting position of the next row. This allows a curved road to be followed. The white lines continue to the top of the image or until they are blocked by an edge.  

![](images/3SdwGub.png?raw=true)

In the image above, the green polygon surrounding the while lines represents the empty portion of the lane ahead. We can transform this back and draw it on the initial image.

![](images/zPmd2uY.jpg?raw=true)

With an outline of the empty portion of the lane, the distance to the car ahead can be approximated. However, this only technique works when the road surface is a single color, the road markings are clear, and there are no shadows on the road.
You can see this technique applied to a video here: [https://www.youtube.com/watch?v=20JxqClLxec](https://www.youtube.com/watch?v=20JxqClLxec) The issues with this algorithm are apparent in the video. There are plenty of improvements that could be done, but I wanted to try out a different approach.

[computer vision code](cv.py)

### Convolution Neural Networks
My second attempt to estimate the distance was to use a neural network as it would be able to handle the noise on the road more easily. Convolution neural networks are great for working with images.

Here I will walk through processing images so they can be used as input to the network.

![](images/fZQtLj4.jpg?raw=true)

First, we will crop and transform the image to the portion we care about. 

![](images/j3rWyEn.jpg?raw=true)

Next, we shrink the image down so that it is quicker for a neural network to process. 

![](images/bTqLnGK.jpg?raw=true)

This tiny image is the input to the network. I manually classified thousands these images into eight different categories based on the distance to the car ahead. The output of the network is the probability of an image belonging to each one of these categories.

Below are two videos showcasing the network in action.

[https://youtu.be/8dSeKicEVf0](https://youtu.be/8dSeKicEVf0) 

[https://youtu.be/ECbU_EvyUqM](https://youtu.be/ECbU_EvyUqM)

The green lines represent the different categories. The thickness of the line increases as the network is more confident that the image belongs in that category. The network was not trained on these videos. It is important to note that the network is not confused by shadows on the road like the computer vision approach was. 

![](images/Dws3qP3.jpg?raw=true)

Based on output of the network, we can estimate the relative distance from the car ahead. With this information, we can make a decision on how to adjust the car's speed.

[CNN training code](cnn.py)

### Hardware

There are two ways I can electronically control my car’s accelerator. 

The first is the more basic approach of using the existing cruise control system and changing the speed by pressing the accelerate and coast buttons. The second way is to directly use the speed control servo that the existing cruise control uses to actuate the throttle. (If you are interested on how it works, there is a great video [here](https://www.youtube.com/watch?v=nZhwYZYvhNA).) To control the servo, I would need a device with general-purpose input/output that supports pulse width modulation. The Raspberry Pi I am using does not support that, so I chose to go with the more basic approach of simply pressing the cruise control buttons. Additionally, pressing the buttons and letting the car handle the throttle is much safer than trying to reverse engineer the servo and use it directly.

The cruise control buttons are extremely simple. There are only two wires. When you press a button, it completes the circuit with a specific resistance. The computer knows what button you pressed because each button has a unique resistance. 

In the images below, you can see the cruise control buttons and the connector to the car's computer. I unplugged the buttons from the connector and plugged in my wires.

![](images/1.jpg?raw=true) 

![](images/2.jpg?raw=true)

![](images/3.jpg?raw=true)


Below is the circuit used to "press" the cruise control buttons. The Raspberry Pi controls the relays that complete the circuit to the car’s computer with a specific resistance.

![](images/6.png?raw=true)

![](images/5.png?raw=true)

![](images/4.png?raw=true)


The Raspberry Pi captures images using a camera. With a small neural network, the distance can be estimated a few times per second. Right now, the adaptive cruise control is fairly naive. When the vehicle in lane ahead is far away, it increases the cruise speed, and when the vehicle is too close, it decreases the cruise speed. This could be greatly improved to take the speed at which the distance is changing into account. 

[adaptive cruise control code](app.py)

Testing the system was done with the Raspberry Pi disconnected from the car to ensure safety. For example: you can simply replace the relays with LEDs and manually press the cruise control buttons when the LEDs light up. This allows testing the system without actually giving it direct control of the car.

### Future Improvements

Use a [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) to more accurately estimate the distance from cars ahead. Right now, I use an average of previous measurements to attempt to handle noise.

Ideally, I would want to be to control the throttle directly instead of relying on the car's existing cruise control. To do this safely I would need to be able to monitor the throttle position, engine rpm, and speed. All of these measurements are available directly from the car's on-board diagnostics port. With these measurements and direct control over the car's throttle, I would be able to more effectively control the car's speed.
