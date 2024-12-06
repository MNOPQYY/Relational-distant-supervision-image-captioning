#coding=utf-8
import pickle
import numpy as np
import scipy.io as sio
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

  
def mainpro(input_dir):
   for i in range(1,1971):
     datapath = input_dir + '/vid' +str(i) + '.avi.txt'
     if (os.path.exists(datapath)==0):
       print(i)
       
if __name__ == '__main__':
    mainpro('/media/mcislab/sdb1/home/mcislab/qiyayun/msvd_detected')