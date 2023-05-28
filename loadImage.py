from PIL import Image
import numpy as np
from numpy import asarray
import cv2
import pandas as pd

def getFirstPx(image):
    small_x = len(image)
    small_y = len(image[0])
    for line_nr, line in enumerate(image):
        for px_nr, px_color in enumerate(line):
            if px_color != 0:
                if line_nr < small_x: small_x = line_nr
                if px_nr < small_y: small_y = px_nr
    return small_x, small_y

def getLastPx(image):
    big_x = 0
    big_y = 0
    for line_nr, line in reversed(list(enumerate(image))):
        for px_nr, px_color in reversed(list(enumerate(line))):
            if px_color != 0:
                if line_nr > big_x: big_x = line_nr
                if px_nr > big_y: big_y = px_nr
    return big_x, big_y

im = cv2.imread('clock.jpg', 0)
gray_negative = abs(255-im)
print(gray_negative)

# np.savetxt("gray.txt", gray_negative)
firstPoint = getFirstPx(gray_negative)
print("line: "+str(firstPoint[0])+", pixel: "+str(firstPoint[1]))
lastPoint = getLastPx(gray_negative)
print("line: "+str(lastPoint[0])+", pixel: "+str(lastPoint[1]))

scaled = cv2.resize(gray_negative[firstPoint[0]:lastPoint[0], firstPoint[1]:lastPoint[1]], (64, 64))

cv2.imwrite('cut.png', scaled)    





# gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# print(gray_im.shape)

# # Reshape the 4D array to 2D
# flattened_array = data.reshape((-1, data.shape[-1]))
# # Save the flattened array to a file in text format (.txt or .csv)
# np.savetxt("drawing.txt", flattened_array)

