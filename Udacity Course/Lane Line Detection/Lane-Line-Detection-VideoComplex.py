#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

#Capture Video

cap = cv2.VideoCapture('CarND-LaneLines-P1-master\\test_videos\\challenge.mp4') #0 Allows feed through webcam

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):

    # Capture frame-by-frame for "image"
    ret, image = cap.read()

    if np.shape(image) == ():
        print('Empty Frame')
    else:
      
      color_myriad= np.copy(image)
      #color_select= np.copy(image) #alternate method

      # Convert to HSV
      hsv = cv2.cvtColor(color_myriad, cv2.COLOR_BGR2HSV)

      # Mask of White
      mask1 = cv2.inRange(color_myriad, (220, 220, 220), (255, 255,255))

      # Mask of Yellow
      mask2 = cv2.inRange(color_myriad, (0,120,100), (100, 255, 255))

      

      # Final Mask and Yellow/White Mask
      mask = cv2.bitwise_or(mask1, mask2)
      target = cv2.bitwise_and(color_myriad,color_myriad, mask=mask)

      # Define color criteria as alternate method vice HSV
      #red_threshold = 100 #180 20
      #green_threshold = 80 #170 190
      #blue_threshold = 80 #80 200
      #rgb_threshold = [red_threshold, green_threshold, blue_threshold]

      # Mask pixels below the threshold for white
      #color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                       #   (image[:,:,1] < rgb_threshold[1]) | \
                       #   (image[:,:,2] < rgb_threshold[2])

      # Mask color selection
      #color_select[color_thresholds] = [0,0,0]

      #Convert RGB to Gray
      gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY) #grayscale conversion

      # Kernel size for Gaussian smoothing / blurring
      
      kernel_size = 5
      blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

      # Parameters for Canny

      low_threshold = 50
      high_threshold = 150
      edges = cv2.Canny(gray, low_threshold, high_threshold)

      # Create masked edges image using cv2.fillPoly()

      mask = np.zeros_like(edges)   
      ignore_mask_color = 255



      # Define a four sided polygon to mask

      imshape = image.shape
      vertices = np.array([[(150,660),(600, 440), (720, 440), (1100,660)]], dtype=np.int32)
      cv2.fillPoly(mask, vertices, ignore_mask_color)
      masked_edges = cv2.bitwise_and(edges, mask)

       

      # Define the Hough transform parameters
      # Make a blank the same size as image to draw on
      rho = 2
      theta = 1*np.pi/180
      threshold = 50
      min_line_length = 100
      max_line_gap = 250
      line_image = np.copy(image)*0 #creating a blank to draw lines on

      # Run Hough on edge detected image

      lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)

      # Iterate over the output "lines" and draw lines on the blank

      for line in lines:
          for x1,y1,x2,y2 in line:
              cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),15)

      # Create a "color" binary image to combine with line image

      color_edges = np.dstack((edges, edges, edges))

      # Draw the lines on the edge image
      lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

      image_lines= cv2.addWeighted(line_image, 1, image, 1, 0)


      # Display the resulting frame
      cv2.imshow('Frame',image_lines)
    if cv2.waitKey(17) & 0xFF == ord('q'):
        break


#cv2.inRange() for color selection
#cv2.fillPoly() for regions selection
#cv2.line() to draw lines on an image given endpoints
#cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
#cv2.bitwise_and() to apply a mask to an image
