import pandas as pd
import os
import numpy as np
import cv2 
import skimage
import matplotlib.pyplot as plt

class CameraCalculation():
    def __init__(s,emissivity):
        calib_vals = [  19457.3671875, # ------------ R
                        1517.0, # ------------------- B
                        1.5, # ---------------------- F
                        0.7319999933242798, # ------- X
                        1.2390000136974777e-08, # --- A1
                        1.1094999585736787e-08, # --- A2
                        0.0031799999997019768, # ---- B1
                        0.0031802000012248755, # ---- B2
                        9.749829292297363, # -------- J1
                        5877, # --------------------- J0
                        293.15, # ------------------- TRefl
                        293.15, # ------------------- TAtm
                        20.0, # --------------------- TAtmC
                        0.55, # --------------------- Humidity
                        2, # ------------------------ Dist
                        1, # ------------------------ EOR
                        273.15, # ------------------- EOT
                    ]
        
        # Unpack calibration values
        R, B, F, X, A1, A2, B1, B2, J1, J0, TRefl, TAtm, TAtmC, Humidity, Dist, EOR, EOT = calib_vals

        # Calculations from FLIR
        H2O = Humidity * np.exp(1.5587 + 0.06939 * TAtmC - 0.00027816 * TAtmC * TAtmC + 0.00000068455 * TAtmC * TAtmC * TAtmC)
        Tau = X * np.exp(-np.sqrt(Dist) * (A1 + B1 * np.sqrt(H2O))) + (1 - X) * np.exp(-np.sqrt(Dist) * (A2 + B2 * np.sqrt(H2O)))
        r1 = ((1 - emissivity) / emissivity) * (R / (np.exp(B / TRefl) - F))
        r2 = ((1 - Tau) / (emissivity * Tau)) * (R / (np.exp(B / TAtm) - F))
        r3 = ((1 - EOR) / (emissivity * Tau * EOR)) * (R / (np.exp(B / EOT) - F))
        K2 = r1 + r2 + r3

        s._ir_calib_list=[J0, J1, B, R, emissivity, Tau, K2, F]    

    def calculate(s, data):
        J0, J1, B, R, emissivity, Tau, K2, F = s._ir_calib_list
        image_Radiance = (data - J0) / J1
        temperature = (B / np.log(R / ((image_Radiance / emissivity / Tau) - K2) + F)) - 273.15
        return temperature
    
    def reverse_calc(s, temp):
        #d=J1*T*e*(K2 - R/(F - exp(B/(C + 5463/20))) + J0/(J1*T*e))
        J0, J1, B, R, emissivity, Tau, K2, F = s._ir_calib_list
        K2 = K2.item()
        Tau = Tau.item()
        data = J1*Tau*emissivity*(K2-R/(F-np.exp(B/(temp+5463/20)))+J0/(J1*Tau*emissivity))
        return data
    
    def get_emiss(s, temp, true): # Takes in as celcius 
        J0, J1, B, R, emissivity, Tau, K2, F = s._ir_calib_list
        image_Radiance = ((temp+273.15) - J0) / J1
        X = K2 + R / (np.exp(B / (true+273.15)) - F)
        emissivity = image_Radiance / (Tau * X)
        return emissivity

class EmissivityHandler(): 
    def __init__(s, imdebug=False):
        s.imdebug = imdebug

    def solve_for_emiss(s, data, old_emiss, true): 
        old_temp = CameraCalculation(old_emiss)
        IR_data = old_temp.reverse_calc(data)
        emiss= old_temp.get_emiss(IR_data, true)
        return emiss

    def dual_emiss_maskfilter(s, data, mask, old_emiss, emiss, yellow_emiss):
        old_temp = CameraCalculation(old_emiss)
        # real_temp = CameraCalculation(emiss)
        oxide_temp = CameraCalculation(yellow_emiss)

        IR_data = old_temp.reverse_calc(data)

        # correctIR=np.zeros((data.shape[0],data.shape[1]))
        correctIR = data
        # Aluminum
        # correctIR[mask==1] = real_temp.calculate(IR_data[mask==1])
        # Soot
        correctIR[mask!=1] = oxide_temp.calculate(IR_data[mask!=1])

        if s.imdebug: 
            plt.figure(figsize=(10,3))

            plt.subplot(221)
            plt.imshow(IR_data, cmap='inferno', aspect='auto', vmin=25)
            plt.colorbar(format='%.2f').set_label('°C')
            plt.title('Original Img', size=20)
            plt.grid()

            plt.subplot(222)
            plt.imshow(correctIR * mask, cmap='inferno', aspect='auto', vmin=25)
            plt.colorbar(format='%.2f').set_label('°C')
            plt.title('Bead Temps', size=20)
            plt.grid()

            plt.subplot(223)
            plt.imshow(correctIR * (~(mask.astype(bool))).astype(int), cmap='inferno', aspect='auto', vmin=25)
            plt.colorbar(format='%.2f').set_label('°C')
            plt.title('Soot Temps', size=20)
            plt.grid()

            plt.subplot(224)
            plt.imshow(correctIR, cmap='inferno', aspect='auto', vmin=25)
            plt.colorbar(format='%.2f').set_label('°C')
            plt.title('Result', size=20)
            plt.grid()

            plt.show()
    
        return correctIR

    def dual_emiss_colorfilter(s, data, old_emiss, emiss, yellow_emiss):
        old_temp = CameraCalculation(old_emiss)
        real_temp = CameraCalculation(emiss)
        oxide_temp = CameraCalculation(yellow_emiss)

        IR_data = old_temp.reverse_calc(data)
        IR_color_data = (255 * (IR_data - np.min(IR_data)) /
                        (np.max(IR_data) - np.min(IR_data))
                        ).astype(np.uint8)
        new_image = cv2.applyColorMap(IR_color_data, cv2.COLORMAP_HOT)

        # alpha = 2.5 #1.0 - 3.0
        # beta = 0 #0 - 100
        # new_image = np.zeros(IR_color_data.shape, IR_color_data.dtype)
        # for y in range(IR_color_data.shape[0]):
        #     for x in range(IR_color_data.shape[1]):
        #         for c in range(IR_color_data.shape[2]):
        #             new_image[y,x,c] = np.clip(alpha*IR_color_data[y,x,c] + beta, 0, 255)

        hsv = cv2.cvtColor(new_image,cv2.COLOR_BGR2HSV)
        yellow_lower = np.array([0, 30, 130]) #([20, 100, 100])
        yellow_upper = np.array([255, 255, 255]) #([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_output = cv2.bitwise_and(new_image, new_image, mask=mask_yellow)

        yellow_img = np.mean(yellow_output, axis=2)

        correctIR=np.zeros((yellow_img.shape[0],yellow_img.shape[1]))
        # Aluminum
        correctIR[yellow_img==0] = real_temp.calculate(IR_data[yellow_img==0])
        # Soot
        correctIR[yellow_img!=0] = oxide_temp.calculate(IR_data[yellow_img!=0])

        if s.imdebug: 
            cv2.imshow("Brightened Data", new_image)
            cv2.imshow("Color Data", IR_color_data)
            # cv2.imshow("Mask Yellow", mask_yellow)
            cv2.imshow("Yellow Output", yellow_output)
            # cv2.imshow("Correct IR", correctIR)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
    
        return correctIR

    def single_emiss(s, data, old_emiss, emiss):
        old_temp = CameraCalculation(old_emiss)
        real_temp = CameraCalculation(emiss)

        IR_data = old_temp.reverse_calc(data)
        
        correctIR = real_temp.calculate(IR_data)
    
        return correctIR
    
    def chart_al_emiss(s,temp):
        if temp <= 60:
            return 0.4
        elif temp <= 80:
            return 0.255
        elif temp <= 100:
            return 0.25
        elif temp <=140:
            return 0.205
        else:
            return 0.2
    
    def chart_ox_emiss(s,temp):
        if temp <= 60:
            return 0.4
        elif temp <= 80:
            return 0.255
        elif temp <= 90:
            return 0.25
        elif temp <=120:
            return 0.205
        else:
            return 0.2

if __name__ == '__main__':
    print("Initializing...")
    emissivity_handler = EmissivityHandler(True)
    exp_name, root_folder = emissivity_handler.get_exp()
    print("Experiments retrieved. Running...")
    if len(exp_name)==1:
        emissivity_handler.image_process(exp_name[0], root_folder)
    else:
        for i in range(0, len(exp_name)):
            emissivity_handler.image_process(exp_name[i], root_folder)
    print("==========\nDONE!\n==========")