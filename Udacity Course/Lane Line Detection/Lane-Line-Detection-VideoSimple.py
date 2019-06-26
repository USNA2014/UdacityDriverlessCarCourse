#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

#Capture Video

cap = cv2.VideoCapture('CarND-LaneLines-P1-master\\test_videos\\solidYellowLeft.mp4') #solidWhiteRight.mp4, solidYellowLeft.mp4, challenge.mp4 

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

#initialize arrays that will be used to collect x values of HoughLines

lower_x = []
upper_x = []
lower_y = []
upper_y = []
lower_x2 = []
upper_x2 = []
lower_y2 = []
upper_y2 = []
 
while(cap.isOpened()):

    # Capture frame-by-frame for "image"
    ret, image = cap.read()

    if np.shape(image) == (): #In Case a NoneType is returned
        print('Empty Frame')
    else:
      

      color_myriad= np.copy(image)
      color_select= np.copy(image) #alternate method

      # Convert to HSV
      hsv = cv2.cvtColor(color_myriad, cv2.COLOR_BGR2HSV)

      # Mask of White
      mask1 = cv2.inRange(color_myriad, (220, 220, 220), (255, 255,255))

      # Mask of Yellow
      mask2 = cv2.inRange(color_myriad, (0,120,100), (100, 255, 255))

      

      # Final Mask and Yellow/White Mask
      mask = cv2.bitwise_or(mask1, mask2)
      target = cv2.bitwise_and(color_myriad,color_myriad, mask=mask)


      # Define our color criteria lower end (other method if HSV not desired)
##      red_threshold = 20 #180 20
##      green_threshold = 190 #170 190
##      blue_threshold = 200 #80 200
##      rgb_threshold = [red_threshold, green_threshold, blue_threshold]
   

      # Mask pixels below the threshold for white
##      color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
##                          (image[:,:,1] < rgb_threshold[1]) | \
##                          (image[:,:,2] < rgb_threshold[2])
    
      

      # Mask color selection
##      color_select[color_thresholds] = [0,0,0]

      gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY) #grayscale conversion

      # Define a kernel size for Gaussian smoothing / blurring
      kernel_size = 5
      blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

      # Define parameters for Canny and run it

      low_threshold = 50
      high_threshold = 150
      edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

      # Create a masked edges image using cv2.fillPoly()

      mask = np.zeros_like(edges)   
      ignore_mask_color = 255



      # This time we are defining a four sided polygon to mask

      imshape = image.shape #470 490 320
      vertices = np.array([[(0,imshape[0]),(470, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
      cv2.fillPoly(mask, vertices, ignore_mask_color)
      masked_edges = cv2.bitwise_and(edges, mask)

       

      # Define the Hough transform parameters
      # Make a blank the same size as our image to draw on
      rho = 2
      theta = np.pi/180
      threshold = 15
      min_line_length = 100
      max_line_gap = 200
      line_image = np.copy(image)*0 #creating a blank to draw lines on

      # Run Hough on edge detected image

      lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)

      # Iterate over the output "lines" and draw lines on the blank
      
      for line in lines:
          for x1,y1,x2,y2 in line:

            if ((y2-y1)/(x2-x1))<0: #if slope is negative, send x/y's here for left lane line

              lower_x.append(x1-50)
              upper_x.append(x2+80)#70
              lower_x_avg= int(np.average(lower_x))
              upper_x_avg= int(np.average(upper_x))
              lower_y.append(y1+40)
              upper_y.append(y2-50)#60
              lower_y_avg= int(np.average(lower_y))
              upper_y_avg= int(np.average(upper_y))
            
              cv2.line(line_image,(lower_x_avg,lower_y_avg),(upper_x_avg,upper_y_avg),(0,0,255),15)
              
            else: #if slope is positive, send x/y's here for right lane line

              #store x/y values then take the average of them, add x/y offset as needed
              lower_x2.append(x1-80) 
              upper_x2.append(x2+90)
              lower_x_avg2= int(np.average(lower_x2))#values must be integers
              upper_x_avg2= int(np.average(upper_x2))
              lower_y2.append(y1-40)
              upper_y2.append(y2+60)
              lower_y_avg2= int(np.average(lower_y2))
              upper_y_avg2= int(np.average(upper_y2))
              
              cv2.line(line_image,(lower_x_avg2,lower_y_avg2),(upper_x_avg2,upper_y_avg2),(0,0,255),15)
              

      # Create a "color" binary image to combine with line image

      color_edges = np.dstack((edges, edges, edges))

      # Draw the lines on the edge image
      lines_edges = cv2.addWeighted(target, 0.8, line_image, 1, 0)

      image_lines= cv2.addWeighted(line_image, 1, image, 1, 0)
    

      # Display the resulting frame
      cv2.imshow('Frame',image_lines)
    if cv2.waitKey(17) & 0xFF == ord('q'):
        break


