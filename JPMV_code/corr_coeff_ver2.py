
"""
Jean Mena
8/8/22
Pattern detection: correlation coefficient template matching method
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage.feature as sf
import pandas as pd


###########################Functions###############################

def read_img(name): #Function read images
  img1 = cv.imread(name) #cv reads as uint8, plt as float32
  img2 = np.array(img1)
  img2 = img2.astype(np.float32) #do not use uint8, only float32
  img3 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) #turn images to grayscale
  
  return img3

def mean_sr_k(R): #mean and s_r of template, only needed once, only computed once (see equation from Burger 2016)
  
  K = R.size

  r_mean = round(np.mean(R),2)
  s_r = round(np.sqrt(K*np.var(R)),2) 

  return r_mean, s_r, K

#correlation coefficient matrix
def c_matrix(I,R): #image, template(reference) 
  M = I.shape[0] #num rows of I
  N = I.shape[1] #num col of I 
  m = R.shape[0] #num rows of R
  n = R.shape[1] #num col of R

  num_rows = M-m+1 #matrix c, search area(subimage of I)
  num_col  = N-n+1 

  C  = np.zeros((num_rows,num_col), dtype=np.float32)

  return C  #matrix of zeros

def cff_matrix(I,lista): #image, list of templates

  #I=read_img(img)
  #R=read_img(ref)

  c_lista=[]

  for R in lista:
    
    r_mean, s_r, K = mean_sr_k(R)
    #print(r_mean, s_r, K)
    c = c_matrix(I,R)
    #print(c)
    numerador   = 0
    denominador = 0
    coeff = 0


    with np.nditer(c, flags = ['multi_index'], op_flags=['readwrite']) as it: #move in c
      while not it.finished:
        #print('new box C'+str(it.multi_index))
        gh = 0
        gp = 0
        fr = 0
        with np.nditer(R, flags = ['multi_index'], op_flags=['readwrite']) as ir: #move in reference
          while not ir.finished:
            #print('new box R--------'+ str(ir.multi_index))
            b_I=0
            b_r=0
            b_I = I[it.multi_index[0]+ir.multi_index[0]][it.multi_index[1]+ir.multi_index[1]]
            #print('b_I: '+str(b_I))
            b_r = R[ir.multi_index[0]][ir.multi_index[1]]
            #print('b_r: '+ str(b_r))
            gh  = gh + b_I
            #print('gh'+str(gh))
            gp  = gp + b_I*b_I
            #print('gp: '+str(gp))
            fr  = fr + b_I*b_r
            #print('fr: '+str(fr))
            #print('finish box R')

            ir.iternext()

        numerador   = fr - (gh*r_mean)
        denominador = np.sqrt(gp - ((gh**2)/K))*s_r   
        coeff = numerador/(1+denominador)
        c[it.multi_index[0]][it.multi_index[1]] =  round(coeff,2)
        #print(g2 - (gh**2/K))

        it.iternext()

    c_lista.append(c)
    #print("new template")
  return c_lista #returns list of matrices

def loc(c,t): #matrix c of coeff, t=threshold

  l = sf.peak_local_max(c,min_distance=100 ,threshold_abs=t, exclude_border=False) #change min_distance as required

  return l #returns lis(matrix)t with the location of the maximums located


def box (img,R,c,t,f,score):#imagen original(no grayscale),R(template), c(coefficient matrix),t:threshold, f:fontScale

  l=loc(c,t)

  m = R.shape[0]
  n = R.shape[1]

  I = np.array(cv.imread(img))
  I = I.astype(np.float32)

  for i in range(len(l)):
    x = l[i][0]
    y = l[i][1]

    I = cv.rectangle(I, (y,x), (y+n,x+m), color=(0,255,0),thickness=1)
    if score == True:
      I = cv.putText(I, str(c[x][y]), (y,x+10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=f, color=(255,0,0))
    
    else:
      continue


  return I

##########################################################




