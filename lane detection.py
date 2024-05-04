#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install opencv-python')


# In[1]:


import numpy as np
import cv2


# In[2]:


def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    cv2.fillPoly(mask, [np.array(region_of_interest_vertices, np.int32)], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    line = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
 
    line_image = np.zeros_like(image)
    if line is not None:
        for line in line:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    c_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    return c_image


# In[4]:



reading = cv2.VideoCapture("C:\\Users\\RR Fashion Point\\Downloads\\VID_20240412141729.mp4")

while reading.isOpened():
    ret, frame = reading.read()
    if not ret:
        break

    processed_frame = detect_lanes(frame)
    cv2.imshow('Lane Detection', processed_frame)
   


 #quite by pressing key "k"
    if cv2.waitKey(1) & 0xFF == ord('k'):
        break
reading.release()
cv2.destroyAllWindows()
    


# In[1]:





# In[ ]:




