# **Finding Lane Lines on the Road** 

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

To complete the project, two files will be submitted: a file containing project code and a file containing a brief write up explaining your solution. We have included template files to be used both for the [code](https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb) and the [writeup](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).The code file is called P1.ipynb and the writeup template is writeup_template.md 

To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)


Project 1 : Finding Lane Lines on the Road
---
Lane markings are a great information in order to steer an autonous vehicle. In order to be able tu used them, it's necessary to detect them in datas given by sensor. In our case, we will use images provided by a camera mounted on vehicle. 
This is placed behind the windshield of the car.
The goal of this project is to define a simple pipeline based on computer vision. At this stage, no machine learning was used. 
As mention in the lesson, I used the [tool](https://github.com/maunesh/opencv-gui-helper-tool) developed by another student in order to simplify filter parameters  finding.  

Setup
---
## If you have already installed the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) you should be good to go!   If not, you should install the starter kit to get started on this project. ##

**Step 1:** Set up the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) if you haven't already.

**Step 2:** Open the code in a Jupyter Notebook

You will complete the project code in a Jupyter notebook.  If you are unfamiliar with Jupyter Notebooks, check out [Udacity's free course on Anaconda and Jupyter Notebooks](https://classroom.udacity.com/courses/ud1111) to get started.

Jupyter is an Ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, use terminal to navigate to your project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 carnd-term1 environment as described in the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) installation instructions!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook.  Follow the instructions in the notebook to complete the project.  


Pipeline
---
**1. Loading image and color space**

As we will try to detect yellow lines as well white lines, I choose to convert image to grayscale. Thus our pipeline can be applied in both cases. 

<img src="https://github.com/Dynaa/lanelines/blob/master/test_images_output_old/gray.jpg" width="480" alt="Grayscaled Image" />

**2. Gaussian Blur**

In order to remove noise, a gaussian blur is applied on the grayscaled image

<img src="https://github.com/Dynaa/lanelines/blob/master/test_images_output_old/gaussian.jpg" width="480" alt="Gaussian Image" />

**3. Canny Edge detection**

A Canny Edge Detector is then applied. 

<img src="https://github.com/Dynaa/lanelines/blob/master/test_images_output_old/canny.jpg" width="480" alt="Canny Edge detection" />

**4. Region of interest**

The image obtained after Canny Edge detection contains edges that are not relevant for our use case. In order to eliminate them, a region of interest is defined. A mask is applied using vertices defining the region to conserve. 

<img src="https://github.com/Dynaa/lanelines/blob/master/test_images_output_old/roi.jpg" width="480" alt="Region of interest" />

**5. Hough transform**

Finally, we apply the OpenCV function HoughLinesP function to find lines in the image. The outuput of this function consist in an array containing the endpoints of all line segments detected by the transform operation. 

<img src="https://github.com/Dynaa/lanelines/blob/master/test_images_output_old/hough.jpg" width="480" alt="Region of interest" />

**6. Merge with orignal image**

To display the result in the original image, I fused the output of Hough transform result with the original image. 

<img src="https://github.com/Dynaa/lanelines/blob/master/test_images_output_old/solidYellowCurve.jpg" width="480" alt="Region of interest" />

**7. drawlines() improvement**

As seen on the previous image, the output of Hough transform don't provide continuous line. In order to endl this issue, I choose to apply a linear regression in order to find the line model that fit the best to the detected line. 
Endpoint are find by searching extrema such as minimal and maximal pixel coordinates, linked to endpoints detection and image size. 

    def draw_lines_improved(img, lines, color=[255, 0, 0], thickness=2): 
        """
        As advised, slope will be computed in order to determinate if a segment is part of the left
        line or part of the right lane
        """
        slope_left = []
        slope_right = []
        line_left_x = []
        line_left_y = []
        line_right_x = []
        line_right_y = []
        for line in lines: 
            for x1,y1,x2,y2 in line: 
                line_slope = slope(x1,y1,x2,y2)
                # Testing slope value in order to know if it's a right or left line
                if ((line_slope<-0.5) and (line_slope>-0.8)): 
                    # Left case
                    slope_left.append(line_slope)
                    line_left_x.append(x1)
                    line_left_x.append(x2)
                    line_left_y.append(y1)
                    line_left_y.append(y2)
                else :
                    if ((line_slope>0.5)and (line_slope<0.8)): 
                        # Right case
                        slope_right.append(line_slope)
                        line_right_x.append(x1)
                        line_right_x.append(x2)
                        line_right_y.append(y1)
                        line_right_y.append(y2)

        if(len(line_left_x)>0):         
            line_left_x = np.array(line_left_x)
            line_left_y = np.array(line_left_y)
            x_max_left = np.amax(line_left_x)
            y_max_left = np.amax(line_left_y)
            x_min_left = np.amin(line_left_x)
            y_min_left = np.amin(line_left_y)
            x_left = np.array(line_left_x)
            y_left = np.array(line_left_y)
            model_left = np.polyfit(x_left,y_left, 1)
            # Giving the left model, determine points to be used for lane drawing
            x_value_left = int((img.shape[0]-model_left[1])/model_left[0])
            cv2.line(img, (x_max_left, y_min_left), (x_value_left, img.shape[0]), color=[255, 0, 0], thickness=12)

        if(len(line_right_x)>0):         
            line_right_x = np.array(line_right_x)
            line_right_y = np.array(line_right_y)
            x_max_right = np.amax(line_right_x)
            y_max_right = np.amax(line_right_y)
            x_min_right = np.amin(line_right_x)
            y_min_right = np.amin(line_right_y)
            x_right = np.array(line_right_x)
            y_right = np.array(line_right_y)
            model_right = np.polyfit(x_right,y_right, 1)
            # Giving the left model, determine points to be used for lane drawing
            x_value_right = int((img.shape[0]-model_right[1])/model_right[0])
            cv2.line(img, (x_min_right, y_min_right), (x_value_right, img.shape[0]), color=[255, 0, 0], thickness=12)

In order to determine if outputs of Hough transform belong to left or right line slope values are computed. Line with positive slope are classified as being on the left line, while those with negative value are linked to right line. 

<img src="https://github.com/Dynaa/lanelines/blob/master/test_images_output/solidYellowCurve.jpg" width="480" alt="Region of interest" />

Test implementation pipeline on videos 
---
The whole pipepline was tested on 2 videos provided. 

**1. Video1 :**

On the first video a continuous whitle line is present at right and a dashed one at left. The result obtain is not so bad on this situation. 

[![Alt text](https://github.com/Dynaa/lanelines/blob/master/test_images_output/youtube_white_lane.png)](https://youtu.be/oIgQ1tX5zNU)

**2. Video2 :**

On the second video a continuous yellow line is present at left and a white dashed one at right. The result obtain is not so bad on this situation. 

[![Alt text](https://github.com/Dynaa/lanelines/blob/master/test_images_output/youtube_yellow_lane.png)](https://www.youtube.com/watch?v=awFEhbWAkmk)

**3. Challenge video:**

A final video was provided, this one was really challenging for some points. I have to adapt the ROI size to use due some limitations of my pipeline, for exemple impact of curvature of the road. 

[![Alt text](https://github.com/Dynaa/lanelines/blob/master/test_images_output/youtube_challenge.png)](https://youtu.be/onINe6HNWKM)


These differents tests allows me to discover the limitations of my pipeline, and give some ideas on the improvements needed to have a more robust and evicient pipeline. 

Remarks and improvements
---
1. Dash line : we observe that dashed line introduce some "lost" of detection, it could be interesting to introduct some tracking step in order to hace "memories" and don't lose line detection. 

2. Curvature : due to curvature line detection based on linear regression seems to be limited. The line can be advantageously fit on more complexe model such as polynomial. 
