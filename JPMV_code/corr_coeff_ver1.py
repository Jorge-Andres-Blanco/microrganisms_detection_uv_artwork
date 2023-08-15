#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 23:02:23 2022

@author: jean
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#matrix = [[2,8,3,5,4],[0,1,2,5,3],[7,4,9,6,4],[4,9,1,2,0]]
#plt.imshow(matrix)
#plt.colorbar()

#ref = [[1,2],[4,9]]
#plt.imshow(ref)
#plt.colorbar()

I = np.asarray(matrix)
R = np.asarray(ref)

M = I.shape[0] #fila Image
N = I.shape[1] #columnas Image
m = R.shape[0] #fila reference
n = R.shape[1] #columnas reference
K = R.size     #m*n

 #calculate R mean and S_r
sum  = 0
sum2 = 0
KK = R.shape

for i in range(KK[0]):
  for j in range(KK[1]):
      sum  = sum  + R[i,j]
      sum2 = sum2 + R[i,j]* R[i,j]

r_mean = sum / K 
s_r    = np.sqrt(sum2 - (K * r_mean**2))

 #create correlation map

C  = np.zeros((M-m+1,N-n+1), dtype=np.float32)
CC = C.shape

 #place R at (r,s): coordinates of subimage I (piece of I evaluated)

for r in range(CC[0]):
  for s in range (CC[1]):

 #compute correlation coefficient for (r,s)

     sumI  = 0
     sumI2 = 0
     sumIR = 0
     for i in range(KK[0]):
         for j in range(KK[1]):
             a_I = I[r+i,s+j]
             a_R = R[i,j]
             sumI  = sumI  + a_I #mean of subimage I for each (r,s)
             sumI2 = sumI2 + a_I**2 
             sumIR = sumIR + a_I*a_R
     
     C[r,s] = (sumIR - sumI*r_mean)/ (1 + (np.sqrt(sumI2 - (sumI**2/K)) * s_r))

print(C)

plt.imshow(C, cmap='gray')
plt.colorbar()
