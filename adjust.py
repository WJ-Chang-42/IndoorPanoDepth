import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
def linear_adjust(depth,alpha):
    mapping = np.ones_like(depth)
    bar = depth - alpha
    mapping[bar>0] = 1 + 0.4*bar[bar>0]
    return mapping[...,None]



###Config###
data_dir = '***/Matterport3D/'
save_dir = '***/adjust/'
distance=1.5
#####################
scenes = ['0_0b217f59904d4bdf85d35da2cab963471', '1_0b724f78b3c04feeb3e744945517073d1', '0_a2577698031844e7a5982c8ee0fecdeb1', '0_9f2deaf4cf954d7aa43ce5dc70e7abbe1', '0_7812e14df5e746388ff6cfe8b043950a1', '4_0b724f78b3c04feeb3e744945517073d1',"2_0b217f59904d4bdf85d35da2cab963471" ,"1_7812e14df5e746388ff6cfe8b043950a1","47_a2577698031844e7a5982c8ee0fecdeb1","45_a2577698031844e7a5982c8ee0fecdeb1"]
for scene in scenes:
    positions = ['Right','Up', 'Left_Down']
    for position in positions:
        gt_depth = np.array(cv2.imread(data_dir+'%s_depth_0_%s_0.0.exr'%(scene,position),cv2.IMREAD_UNCHANGED)[:,:,0])
        img = np.array(cv2.imread(data_dir+'%s_color_0_%s_0.0.png'%(scene,position), cv2.IMREAD_ANYCOLOR))/255
        mapping = linear_adjust(gt_depth,distance)
        cv2.imwrite(save_dir+'%s_color_0_%s_0.0.png'%(scene,position),(img*mapping).clip(0,1)*255)
    os.system('cp %s%s_depth* %s'%(data_dir,scene,save_dir))