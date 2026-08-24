import pandas as pd
import os
import numpy as np
import datetime
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import time

import mypkg.data_files.parameter_set as parameter_set
from mypkg.heat_analysis.common_img import GetPtsFromImg

# ======================================================================
# ======================================================================

class Img_TorchTrack():
    def __init__(s, showsubsteps=False):
        s._imdebug = showsubsteps
        s._common_img = GetPtsFromImg()
    
    def _calibrate_torch(s, torch_im):

        if isinstance(torch_im, str): 
            # Reading in image
            img = cv2.imread(
                str(torch_im),
                cv2.IMREAD_GRAYSCALE
            )
        else: 
            img = torch_im

        # ==================
        # TORCH MASK 
        # ==================
        print("->>Click on as many points around the torch as you can.")
        torch_pts = s._common_img.clickfind_pts(img)

        filled_img = img.copy()
        filled_img = cv2.fillPoly(filled_img, np.int32([torch_pts]), (255,0,0))
        
        if s._imdebug:
            cv2.imshow("Filled Polygon", filled_img)
            cv2.imshow("Original", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

        x = input("->>Press 1 to click on approx center pt of torch, or 2 to type in the coordinate relative to your plate.")
        # val='N'
        # while val=='N':
        if x == '1': 
            center_pt = s._common_img.clickfind_pts(img)
            center_pt = center_pt[0]
        if x=='2':
            x = input('Type your x position (will be rounded to ~0.000 mm).')
            y = input('Type your y position (will be rounded to ~0.000 mm).')
            # Find u,v of node x,y
            H = np.load(os.path.join(os.getcwd(),r'Thesis_Work/Obj 3 - Online Control/OBJ3_PACKAGE/src/mypkg/data_files/h_matrix.npy'))
            p_hom = [float(x), float(y), 1] # (float(parameter_set.DICT['Substrate Length'].split(' ')[0])*25.4) - 
            p1_hom = H@p_hom
            p1 = [x / p1_hom[2] for x in p1_hom[:-1]]
            # print("P:{}, H@P:{}".format(pt, p1))
            center_pt = [np.round(p1[0],3), np.round(p1[1],3)]
            

        filled_img = cv2.circle(filled_img, (int(round(center_pt[0],0)),int(round(center_pt[1],0))), radius=5, color=(255, 255, 255), thickness=-1)
        if s._imdebug: 
            cv2.imshow("Labelled Centerpt", filled_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                
            # print('Press Y to accept, N to redo.')
            # val = input('Input:\t\t')

        torch_mask_ptsx = np.array(torch_pts)[:,0] - center_pt[0]
        torch_mask_ptsy = np.array(torch_pts)[:,1] - center_pt[1]
        arr = np.zeros(len(torch_mask_ptsx))
        arr[:2] = np.array([center_pt[0], center_pt[0]])
        torch_mask_pts=[torch_mask_ptsx, torch_mask_ptsy]
        np.save(
            os.path.join(os.getcwd(),r'Thesis_Work/Obj 3 - Online Control/OBJ3_PACKAGE/src/mypkg/data_files/torch_mask.npy'),
            torch_mask_pts
        )

    def _calibrate_plate(s, plate_im):
        val='N'
        
        while val=='N':
            # ==================
            # BUILD PLATE MASK
            # ==================
            if isinstance(plate_im, str): 
                # Reading in image
                img = cv2.imread(
                    str(plate_im),
                    cv2.IMREAD_GRAYSCALE
                )
            else: 
                img = plate_im

            print("->>Click on corners starting from [0,0], CW.")
            crop_pts = s._common_img.clickfind_pts(img)

            newimg = img
            for i in range(0,len(crop_pts)-1): 
                cv2.line(newimg,(crop_pts[i][0],crop_pts[i][1]),(crop_pts[i+1][0],crop_pts[i+1][1]),(255,0,0),5)
            cv2.line(newimg,(crop_pts[len(crop_pts)-1][0],crop_pts[len(crop_pts)-1][1]),(crop_pts[0][0],crop_pts[0][1]),(255,0,0),5)

            print('Press Y to accept, N to redo.')
            cv2.namedWindow("Resizable Window", cv2.WINDOW_NORMAL)
            cv2.imshow("Resizable Window", newimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            val = input('Input:\t\t')

        np.save(
            os.path.join(os.getcwd(),r'Thesis_Work\Obj 3 - Online Control\OBJ3_PACKAGE\src\mypkg\data_files\buildplate_pts.npy'),
            crop_pts
        )

        # crop_pts = np.load(os.path.join(os.getcwd(),'Thesis_Work\Obj 3 - Online Control\OBJ3_PACKAGE\src\mypkg\data_files','buildplate_pts.npy'))

        # ==================
        # FIND H MATRIX
        # ==================

        dst = np.array(crop_pts) 
        src = np.array([
            [0,0], #1
            [0,float(parameter_set.DICT['Substrate Length'].split(' ')[0])*25.4], #2
            [float(parameter_set.DICT['Substrate Width'].split(' ')[0])*25.4, float(parameter_set.DICT['Substrate Length'].split(' ')[0])*25.4], #3
            [float(parameter_set.DICT['Substrate Width'].split(' ')[0])*25.4,0], #4
        ])

        x, y, u, v = src[:,0], src[:,1], dst[:,0], dst[:,1]
        A = np.zeros((9,9))
        j = 0
        for i in range(4):
            A[j,:] = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*u[i], y[i]*u[i], u[i]])
            A[j+1,:] = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*v[i], y[i]*v[i], v[i]])
            j += 2
        A[8, 8] = 1   # assuming h_9 = 1
        b = [0]*8 + [1]

        H = np.reshape(np.linalg.solve(A, b), (3,3))
        print(H)

        np.save(
            os.path.join(os.getcwd(),r'Thesis_Work\Obj 3 - Online Control\OBJ3_PACKAGE\src\mypkg\data_files\h_matrix.npy'),
            H
        )

        ps, p1s = [], [] 

        for i in range(4):
            p, q = src[i], dst[i]
            p_hom = [p[0], p[1],1] 
            p1_hom = H@p_hom
            p1 = [x / p1_hom[2] for x in p1_hom[:-1]]
            ps.append(p)
            p1s.append(p1)
            print("P:{}, P':{}, H@P:{}".format(p, q, p1))

        # plot the original and transformed points
        # img = mpimg.imread(str(plate_im))
        txt_list =['1','2','3','4']
        ps, p1s = np.array(ps), np.array(p1s)
        plt.figure(figsize=(10,3))
        plt.subplot(121),   plt.scatter(ps[:,0], ps[:,1], color='g', s=50), plt.title('P', size=20), plt.grid(), plt.axis('equal')
        for i, txt in enumerate(txt_list): 
            plt.annotate(txt, (ps[i,0], ps[i,1]))
        plt.subplot(122),   plt.imshow(img, origin='lower'),    plt.scatter(p1s[:,0], p1s[:,1], color='r', s=50), plt.title(r'P$^{\prime}$', size=20), plt.grid()
        for i, txt in enumerate(txt_list): 
            plt.annotate(txt,(p1s[i,0], p1s[i,1]))
        plt.show()

    # ==========================
    # ==== UNUSED FUNCTIONS ====
    # ==========================
    def move_mask(s, mask, pt, img=-1):
        x=mask[0] + pt[0]
        y=mask[1] + pt[1]

        if s._imdebug: 
            filled_img=img.copy()
            filled_img = cv2.line(filled_img, (int(mask[2][0]),int(mask[2][1])),(pt[0],pt[1]), (255,0,0), thickness=2)
            filled_img = cv2.fillPoly(filled_img, np.int32([np.array(list(zip(x,y)))]), (255,0,0))
            filled_img = cv2.circle(filled_img, (pt[0],pt[1]), radius=5, color=(0, 255, 255), thickness=-1)
            cv2.imshow("New Loc", filled_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return np.array(list(zip(x,y)))
    
    def get_torch_mask_pts(s, img, torch_pts):
        filled_img = img.copy()
        filled_img = cv2.fillPoly(filled_img, np.int32([torch_pts]), (255,0,0))
        torch_mask = np.float64(filled_img)
        torch_mask[torch_mask < 255] = 0
        torch_mask = cv2.morphologyEx(torch_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        torch_mask[torch_mask == 255] = 1
        torch_mask_pts = np.where(torch_mask==1)
        return torch_mask_pts
    
    def move_n_snap(s, pos, name):
        # Move robot to position
        x=input(f"Move robot to pos == {pos} ==, press enter to continue.")

        # Get snapshot
        img = s._FLIR_Snapshot(name)
        
        print("->>Click on the centerpoint of the torch, as closely as possible.")
        return (s._common_img.clickfind_pts(img))

    def find_torch(s, img):
        # PIL
        #img.show()

        #OPEN CV 
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # cv2.imshow('img',img)

        # TEMPERATURE
        img[img>100]=0
        img[img<30]=0
        plt.imshow(img, cmap='inferno', aspect='auto')
        plt.colorbar(format='%.2f').set_label('°C')
        plt.show()
        pass
    # ==========================
    # ==========================
    # ==========================
if __name__ == '__main__':
    print("Initializing...")
    main = Img_TorchTrack(True) #imdebug

    data_path = r'C:\Users\kkelly97\Documents\Kate Files\FRANUC_DATA\2026-04-21_GT-Layer1_IR_Values'

    # main._calibrate_torch(torch_im=r'C:\Users\kkelly97\Documents\GitHub\Thesis_Work\Obj 3 - Online Control\Test Tracking Images\00003849_25.55_1154.75.tiff')
    main._calibrate_plate(plate_im=os.path.join(os.getcwd(),r'Thesis_Work/Obj 3 - Online Control/OBJ3_RESULTS/Debug Files','IR_IMG.tiff'))
    print("==========\nDONE!\n==========")