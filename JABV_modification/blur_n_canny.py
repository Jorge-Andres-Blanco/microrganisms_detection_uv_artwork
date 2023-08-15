"""
Last modified Aug 14 2023

@author: Jorge Blanco
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as sf
from functions import *


""""The purpose of this program is to create a transformation for the images for later use of the matchTemplate function of OpenCV"""



#THIS IS THE TRANSFORMATION:

"""

WHY THIS TRANSFORMATION?

FIRST: A gaussianblur is applied. This is made to reduce the noise in the picture.
Because changes in color can be later interpred as edges by the canny function, a preliminary gaussian blur is needed.

The (5,5) parameter shows the size of the gaussian kernel. The bigger this is, the blurrier the final image

SECOND: The second step is to define the edges by the canny function.
The apertureSize parameter behaves as a parameter for sensitivity.

THIRD: Another gaussian blur is applied to make the borders wider, the idea behind this was that the borders are more likely to fit those of the template if both are wider. 


"""

def blur_canny_transform(img, is_template = False):

  if is_template:
    #TEMPLATE TRANSFORMATION
    temp = img

    blur = cv2.GaussianBlur(temp, (9,9), cv2.BORDER_DEFAULT) #First applies a high blur
    canny = cv2.Canny(blur, 200, 200, apertureSize=5) #Draws the edges of the blurred template
    blur_canny = cv2.GaussianBlur(canny, (9,9), cv2.BORDER_DEFAULT) # Applies anoerther high blur to the edges in order to make them softer
    
    """canny  = cv2.Canny(blur_canny, 125, 175, apertureSize=5)
    temp_mod = cv2.GaussianBlur(canny, (9,9), cv2.BORDER_DEFAULT)""" #The process can be repeated, however, this can be unnecessary and creates multiple lines from the canny

    temp_mod = blur_canny
    return temp_mod
  
  else:
    #IMAGE TRANSFORMATION
    I = img

    blur = cv2.GaussianBlur(I, (5,5), cv2.BORDER_DEFAULT) #First applies a mild blur
    canny = cv2.Canny(blur, 200, 200, apertureSize=5)#Draws the edges of the blurred template
    blur_canny = cv2.GaussianBlur(canny, (5,5), cv2.BORDER_DEFAULT) #First applies a mild blur

    """canny  = cv2.Canny(blur_canny, 125, 175, apertureSize=5)
    I_mod = cv2.GaussianBlur(canny, (5,5), cv2.BORDER_DEFAULT)"""
    I_mod = blur_canny
    return I_mod
  

  


#THIS IS THE MAIN PROGRAM


#IMAGE
I=read_img('/home/oem/Desktop/Asistencia_IIArte/code_mod/MII.jpg', False) #All paths should be changed accordingly. 
color_I = cv2.imread('/home/oem/Desktop/Asistencia_IIArte/code_mod/MII.jpg')
I_mod = blur_canny_transform(I)#TRANSFORMATION

#TEMPLATE
temp=read_img('/home/oem/Desktop/Asistencia_IIArte/code_mod/templates/MusasII-penicillium-49A3-49B3.png', False)
temp_mod = blur_canny_transform(temp, is_template=True)#TRANSFORMATION


#CORRELATION MATRICES GRAY IMAGE

c, c_rot = temp_rotation_correlation(temp, I)

#CORRELATION MATRICES BLUR-CANNY IMAGE

c_mod, c_rot_mod = temp_rotation_correlation(temp_mod, I_mod)


(height_t, width_t) = temp.shape[:2]

if height_t == width_t:

  c_tot = np.maximum(c,c_rot)
  c_tot_mod = np.maximum(c_mod,c_rot_mod)

  I_mod = box(color_I,temp_mod,c_tot_mod,0.5,20,False)
  I = box(color_I,temp,c_tot,0.75,20,False)

else:

  I_mod = box(color_I,temp_mod,c_mod,0.58,20,False)
  I_mod_rot = box(I_mod,temp_mod,c_rot_mod,0.58,20,False)
  I_rot = box(I_mod_rot,temp,c_rot,0.69,20,False) 
  I = box(I_rot,temp,c,0.7,20,False)  


I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)

I.astype(np.float32)

plt.imshow(I)
plt.show()
