Reflection

1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

R= My pipeline consisted of 8 main steps.  First I created a filter that transformed the image from rgb to hsv. 
Then I created two masks with one detecting the color white, and one detecting the color yellow.  
Those two masks were combined into a single image so that both yellow and white lines could be visible.
The second step was converting the image from HSV with applied yellow/white line masks into grayscale in prep for Canny edge line detection.  The third step was applying a gaussian blur to the image to further refine the image for Canny Edge detection.
The fourth step was applying Canny Edge detection in order to see borders in the image more clearly.  The fifth step I created a polygon mask that would eliminate any information that was outside of the polygons borders so that we could just
focus on the left and right lanes.The sixth step I created Hough Lines for the the detected white/yellow lane lines and adjusted the 
associated variables to get reasonable outputs.  The seventh step was where I took the lower x,y and upper x,y extracted from the Hough 
Lines and send them into a logical if/else to determine whether the sloped for each line was a negative or positive.
If it was a positive, that defined the slope of the left lane line.  If negative, then the slope of the right lane.
I then input each independent variable in empty arrays and then took the average of each independent variable.  After, I inserted these averages into OpenCV's line function in order to create a smooth Hough Line.  This was how I modified the Line function.  The last step was combining the finalized filtered/overlayed video on top of the original video.

2. Identify potential shortcomings with your current pipeline

I believe a potential shortcoming for my pipeline is better detecting colors under shadows.  I still had a difficult time with this; however, even in the challenge video, I managed to get my pipeline to work as I was able to draw just enough information from the lane lines to continue projecting my Hough Line.  I believe that HSV derived masks are a stronger choice than rgb masks due to their ability to better detect colors in shadow.

3. Suggest possible improvements to your pipeline

I think potentially (I did not experiment with this) that YCbCr could have potentially led to better detecting colors under shadow as a possible improvement.  

The second improvement I thought of would be to potentially develop some sort of PID algorithm to make HoughLines maintain position over lane lines.  I have never used a PID on CV and am not sure if it would even be viable, but thinking of using averages to make the Hough Line smooth caused the thought to occur to me.
