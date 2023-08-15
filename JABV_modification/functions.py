"""
Last modified Aug 14 2023

@author: Jorge Blanco
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as sf

#Read images and turn into grayscale
def read_img(name, fl_32 = False): #Function read images
  
  img1 = cv2.imread(name) #cv2 reads as uint8, plt as float32

  if fl_32:
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0 #turn images to grayscale and convert dtype in float 32

  else:
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #turn images to grayscale
  
  return img1





def rotate(img, angle, center = True):
  
  (height, width) = img.shape[:2]

  if center:
    rot_point = (width//2,height//2)

  
  if angle in [90, 270, -90]:

    rot_mat = cv2.getRotationMatrix2D(rot_point, angle, 1.0)
    dimensions = (height, width)# Rotates the shape by 90 degrees

    return cv2.warpAffine(img, rot_mat, dimensions)

  else:
    #If template the rotations aren't 90 or 270, the original shape of the image will be maintained"

    rot_mat = cv2.getRotationMatrix2D(rot_point, angle, 1.0)
    dimensions = (width, height)

    return cv2.warpAffine(img, rot_mat, dimensions)






def blur_canny_transform(img, is_template = False):

  if is_template:
    #TEMPLATE TRANSFORMATION
    temp = img

    blur = cv2.GaussianBlur(temp, (9,9), cv2.BORDER_DEFAULT) #First applies a high blur
    canny = cv2.Canny(blur, 200, 200, apertureSize=5) #Draws the edges of the blurred template
    blur_canny = cv2.GaussianBlur(canny, (9,9), cv2.BORDER_DEFAULT) # Applies anoerther high blur to the edges in order to make them softer
    
    """canny  = cv2.Canny(blur_canny, 125, 175, apertureSize=5)
    temp_mod = cv2.GaussianBlur(canny, (9,9), cv2.BORDER_DEFAULT)""" #The process can be repeated, however, this can be unnecessary and create multiple lines from the canny

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
  


    


#Correlation Coefficient matrix with Opencv2 function
def correlation_map(I, R): #I: image, R: template
    
    C = cv2.matchTemplate(I, R,5, mask=None) #5 = TM_CCOEFF_NORMED, mask=None
    
    return C





def temp_rotation_correlation(temp, large_img):

  #ROTATIONS OF TEMPLATE

  temp90 = rotate(temp, 90)
  temp180 = rotate(temp, 180)
  temp270 = rotate(temp, 270)

  
  #Correlation_matrices
  c0 = correlation_map(large_img, temp)
  c90 = correlation_map(large_img, temp90)
  c180 = correlation_map(large_img, temp180)
  c270 = correlation_map(large_img, temp270)


  #Total correlation matrix

  c = np.maximum(c0,c180) #ORIGINAL SHAPE

  c_rot = np.maximum(c90,c270) #TRANSPOSED SHAPE

  return c, c_rot






#Locate position of points according to threshold
def loc(c,t): #matrix c of corr_coeff, t=threshold(select a threshold by comparing the number of outputs. The selection of threshold can be improved by automatizing it)

  l = sf.peak_local_max(c,min_distance=100 ,threshold_abs=t, exclude_border=False) #change min_distance as required(size of matrix): 100 pixels

  return l #returns list with the location of the maximums





#Draw box on the location of the maximums, box of the size of the template
def box (color_I,R,c,t,f,score):#I: original image(grayscale),R(template), c(coefficient matrix),t:threshold, f:fontScale, score:boolean(show value inside the box)

  l=loc(c,t)

  w, h = R.shape[::-1]

  #I = np.array(cv2.imread(img))
  #I = I.astype(np.float32)

  for i in range(len(l)):
    x = l[i][1]
    y = l[i][0]

    color_I = cv2.rectangle(color_I, (x,y), (x+w,y+h), color=(0,255,0),thickness=10)


    print(f'{(x,y)}, {(x+w,y+h)},  {c[y][x]}')
    
    if score == True:
      color_I = cv2.putText(color_I, str(c[x][y]), (y,x+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=f, color=(255,0,0))
    

  return color_I





#MAIN


#IMAGE
I=read_img('/home/oem/Desktop/Asistencia_IIArte/code_mod/MII.jpg', False)
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
