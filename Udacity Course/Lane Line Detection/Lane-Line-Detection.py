#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline




#reading in an image
image = mpimg.imread('CarND-LaneLines-P1-master\\test_images\\solidYellowCurve.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)

## convert to hsv
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

## mask of white (36,0,0) ~ (70, 255,255) 8 20
mask1 = cv2.inRange(hsv, (5, 0, 180), (30, 50,255))

## mask of yellow (15,0,0) ~ (36, 255, 255) 45,76,76; 65,255,255
mask2 = cv2.inRange(hsv, (90,30,100), (110, 180, 255))

#mkk= cv2.cvtColor(mask2, cv2.COLOR_HSV2BGR)

## final mask and masked
mask = cv2.bitwise_or(mask1, mask2)
target = cv2.bitwise_and(image,image, mask=mask)


gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY) #grayscale conversion

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
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

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 10
max_line_gap = 120
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image

color_edges = np.dstack((edges, edges, edges))

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)



# Grab the x and y size and make a copy of the image and region

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)

#line_image = np.copy(image)

# Define our color selection criteria

red_threshold = 180
green_threshold = 170
blue_threshold = 80
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold

thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]


# Display the image

plt.imshow(lines_edges)
#plt.imshow(edges, cmap='Greys_r')
#plt.imshow(line_image)
#plt.imshow(image)
#plt.imshow(gray, cmap='gray')
#plt.imshow(color_select)
#plt.imshow(image)
plt.show()






#cv2.inRange() for color selection
#cv2.fillPoly() for regions selection
#cv2.line() to draw lines on an image given endpoints
#cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
#cv2.bitwise_and() to apply a mask to an image
