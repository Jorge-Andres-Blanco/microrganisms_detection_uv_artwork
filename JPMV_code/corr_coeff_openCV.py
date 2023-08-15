# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:11:46 2022

@author: JeanM
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage.feature as sf

#Read images and turn into grayscale
def read_img(name): #Function read images
  img1 = cv.imread(name) #cv reads as uint8, plt as float32
  img1 = np.array(img1)
  img1 = img1.astype(np.float32) #do not use uint8, only float32
  img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) #turn images to grayscale
  
  return img1

#Correlation Coefficient matrix with OpenCV function
def correlation_map(I, R): #I: image, R: template
    
    C = cv.matchTemplate(I, R,5, mask=None) #5 = TM_CCOEFF_NORMED, mask=None
    
    return C

#Locate position of points according to threshold
def loc(c,t): #matrix c of corr_coeff, t=threshold(select a threshold by comparing the number of outputs. The selection of threshold can be improved by automatizing it)

  l = sf.peak_local_max(c,min_distance=100 ,threshold_abs=t, exclude_border=False) #change min_distance as required(size of matrix): 100 pixels

  return l #returns list with the location of the maximums

#Draw box on the location of the maximums, box of the size of the template
def box (I,R,c,t,f,score):#I: original image(grayscale),R(template), c(coefficient matrix),t:threshold, f:fontScale, score:boolean(show value inside the box)

  l=loc(c,t)

  m = R.shape[0]
  n = R.shape[1]

  #I = np.array(cv.imread(img))
  #I = I.astype(np.float32)

  for i in range(len(l)):
    x = l[i][0]
    y = l[i][1]

    I = cv.rectangle(I, (y,x), (y+n,x+m), color=(0,255,0),thickness=1)
    
    if score == True:
      I = cv.putText(I, str(c[x][y]), (y,x+10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=f, color=(255,0,0))
    

  return I
