import pandas as pd
import os
import time
import PySpin
import numpy as np
import datetime
import cv2 
from queue import Queue #object type is still used even if this says otherwise
import threading


# Add parent directory to path
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import mypkg.data_files.parameter_set as parameter_set

"""
    IRHandler Class is created. Ensure correct installation 
    of camera checking feed through SpinView SDK, and that 
    FLIR camera is on the same subgroup as the ethernet adapter. 

    Collected data: 
        SAVES images.
"""

# Gige ex. refers to gige_example_A400_A700.py

class Flag: 
    def __init__(self,val=True):
        self.val = val
    
    def set(self, val):
        self.val = val

    def __bool__(self):
        return self.val
    
class IRHandler:

    def __init__(self):
        print("IR CAMERA :: Initializing IR Camera...")

        self.init_fail = False

        self.outpath = os.path.join(parameter_set.DICT['Root Folder'],str(parameter_set.DICT['Experiment Name']+"_IR_Values"))
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        else:
            print(f"\nPath ({parameter_set.DICT['Experiment Name']}_IR_Values) already exists!\n")
            self.outpath = os.path.join(parameter_set.DICT['Root Folder'],str(parameter_set.DICT['Experiment Name']+"-1_IR_Values"))
            if not os.path.isdir(self.outpath):
                os.mkdir(self.outpath)
                print(f"Using path ({str(parameter_set.DICT['Experiment Name'])}-1_IR_Values) instead.")
            else:
                found_files = 1
                for file in os.listdir(parameter_set.DICT['Root Folder']):
                    if parameter_set.DICT['Experiment Name'] in file:
                        found_files=found_files+1
                self.outpath = os.path.join(parameter_set.DICT['Root Folder'],str(parameter_set.DICT['Experiment Name']+"-"+str(found_files)+"_IR_Values"))
                os.mkdir(self.outpath)
                print(f"Using path ({str(parameter_set.DICT['Experiment Name'])}-{str(found_files)}_IR_Values) instead.")


        # ===== Gige ex. {def main:} =====

        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()

        # There should be one camera detected. 
        # See Gige ex. if adding more cameras and change the below code.
        if num_cameras == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()

            print('IR CAMERA :: No cameras detected. Aborting init...')
            self._init_fail = True
            return
        elif num_cameras > 1: 
            print("IR CAMERA :: Multiple cameras detected! Aborting init...")
            self._init_fail = True
            return
        
        self.cam = self.cam_list[0]
        
        try: 
            # ===== Gige ex. {def run_single_camera:} =====
            # Initialize camera
            self.cam.Init()

            # Retrieve GenICam nodemap
            s.nodemap = s.cam.GetNodeMap()
            sNodemap = s.cam.GetTLStreamNodeMap()
            
            # ===== Gige ex. {def acquire_and_display_images:} =====

            # Change bufferhandling mode to NewestOnly
            node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
            if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsReadable(node_bufferhandling_mode):
                print('IR CAMERA :: Unable to set stream buffer handling mode.. Aborting...')
                s._init_fail = True
                return
            
            node_IRFormat = PySpin.CEnumerationPtr(s.nodemap.GetNode('IRFormat'))
            node_temp_radiometric = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('Radiometric'))
            node_radiometric = node_temp_radiometric.GetValue()
            node_IRFormat.SetIntValue(node_radiometric)

            # Retrieve entry node from enumeration node
            node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
            if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
                print('IR CAMERA :: Unable to set stream buffer handling mode.. Aborting...')
                s._init_fail = True
                return
            
            # Retrieve integer value from entry node
            node_newestonly_mode = node_newestonly.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

            node_acquisition_mode = PySpin.CEnumerationPtr(s.nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('IR CAMERA :: Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                s._init_fail = True
                return

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('IR CAMERA :: Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                s._init_fail = True
                return

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            # Retrieve Calibration details
            CalibrationQueryR_node = PySpin.CFloatPtr(s.nodemap.GetNode('R'))
            R = CalibrationQueryR_node.GetValue()

            CalibrationQueryB_node = PySpin.CFloatPtr(s.nodemap.GetNode('B'))
            B = CalibrationQueryB_node.GetValue()

            CalibrationQueryF_node = PySpin.CFloatPtr(s.nodemap.GetNode('F'))
            F = CalibrationQueryF_node.GetValue()

            CalibrationQueryX_node = PySpin.CFloatPtr(s.nodemap.GetNode('X'))
            X = CalibrationQueryX_node.GetValue()

            CalibrationQueryA1_node = PySpin.CFloatPtr(s.nodemap.GetNode('alpha1'))
            A1 = CalibrationQueryA1_node.GetValue()

            CalibrationQueryA2_node = PySpin.CFloatPtr(s.nodemap.GetNode('alpha2'))
            A2 = CalibrationQueryA2_node.GetValue()

            CalibrationQueryB1_node = PySpin.CFloatPtr(s.nodemap.GetNode('beta1'))
            B1 = CalibrationQueryB1_node.GetValue()

            CalibrationQueryB2_node = PySpin.CFloatPtr(s.nodemap.GetNode('beta2'))
            B2 = CalibrationQueryB2_node.GetValue()

            CalibrationQueryJ1_node = PySpin.CFloatPtr(s.nodemap.GetNode('J1'))    # Gain
            J1 = CalibrationQueryJ1_node.GetValue()

            CalibrationQueryJ0_node = PySpin.CIntegerPtr(s.nodemap.GetNode('J0'))   # Offset
            J0 = CalibrationQueryJ0_node.GetValue()

            Emiss = float(parameter_set.DICT['Material Emissivity'])

            TRefl = 293.15
            TAtm = 293.15
            TAtmC = TAtm - 273.15
            Humidity = 0.55

            Dist = 2
            ExtOpticsTransmission = 1
            ExtOpticsTemp = 311.4 #Kelvin
            
            calibration_vals = [R, B, F, X, A1, A2, B1, B2, J1, J0, 
                                TRefl, TAtm, TAtmC, Humidity, 
                                Dist, ExtOpticsTransmission, ExtOpticsTemp]
            
            print(f"IR CAMERA :: Calibration Vals:\n{calibration_vals}")

            H2O = Humidity * np.exp(1.5587 + 0.06939 * TAtmC - 0.00027816 * TAtmC * TAtmC + 0.00000068455 * TAtmC * TAtmC * TAtmC)

            Tau = X * np.exp(-np.sqrt(Dist) * (A1 + B1 * np.sqrt(H2O))) + (1 - X) * np.exp(-np.sqrt(Dist) * (A2 + B2 * np.sqrt(H2O)))

            # Pseudo radiance of the reflected environment
            r1 = ((1 - Emiss) / Emiss) * (R / (np.exp(B / TRefl) - F))

            # Pseudo radiance of the atmosphere
            r2 = ((1 - Tau) / (Emiss * Tau)) * (R / (np.exp(B / TAtm) - F))

            # Pseudo radiance of the external optics
            r3 = ((1 - ExtOpticsTransmission) / (Emiss * Tau * ExtOpticsTransmission)) * (R / (np.exp(B / ExtOpticsTemp) - F))

            K2 = r1 + r2 + r3

            self.radiometric_calibration = {'J0' : J0,
                            'J1' : J1,
                            'B' : B,
                            'R' : R,
                            'Emissivity' : Emiss,
                            'Tau' : Tau,
                            'K2' : K2,
                            'F' : F}
            
        except PySpin.SpinnakerException as ex:
            print('IR CAMERA :: Error: %s' % ex)
            self._init_fail = True
            return
        
    def run_camera(self, msgqueue, runflag, savequeue, imgqueue=0, robo_flag=0): 
        # If setup was not performed properly, do not run this code. 
      
        if self._init_fail:
            print("IR CAMERA :: Initialization of camera failed.")
            return

        #  Begin acquiring images
        self.cam.BeginAcquisition() 

        # Retrieve and display images
        print("IR CAMERA :: Reading Camera Data!")
        self.count = 0
        # s.start = datetime.datetime.now()
        while runflag.val:
            try:
                #  Retrieve next received image
                self.image_result = self.cam.GetNextImage()
                if self.image_result.IsIncomplete():
                    pass

                else:
                    # Getting the image data as a numpy array
                    image_data = s.image_result.GetNDArray() # 480 x 640 res
                    #print(image_data.shape) # Uncomment to get resolution of camera

                    # Find maximum temperature: 
                    image_Radiance = (image_data - s._ir_val_list["J0"]) / s._ir_val_list["J1"]
                        
                    image_Temp = (s._ir_val_list["B"] / 
                                  np.log(s._ir_val_list["R"] / 
                                         ((image_Radiance / s._ir_val_list["Emissivity"] / s._ir_val_list["Tau"]) - s._ir_val_list["K2"]) 
                                         + s._ir_val_list["F"])
                                  ) - 273.15
                    # C=(B/(log((R/((d/e/T)-K2))+F)))-273.15/

                    # image_Temp=np.transpose(image_Temp)
                    # image_Temp=np.flip(image_Temp)
                    # image_Temp=np.flip(image_Temp,axis=1)
                    
                    # Save image: 
                    Tmax = np.max(image_Temp)
                    Tmin = np.min(image_Temp)
                    image_16b = (65535 * (image_Temp - Tmin) / (Tmax - Tmin)).astype(np.uint16)

                    savequeue.put([str((os.path.join(s.outpath ,f'{s.count:08}_{Tmin:0.2f}_{Tmax:0.2f}.tiff'))), image_16b])
                    imgqueue.put(image_Temp)

                    # Send max temp back to main script:
                    msgqueue.put([['Machine','FroniusA700'],
                                ['Timestamp',pd.to_datetime(datetime.datetime.now())],
                                ['Count', s.count],
                                ["Max Temp", Tmax]])
                    
                    if robo_flag!=0:
                        if Tmax>=parameter_set.DICT['Interpass Temperature']:
                            robo_flag.set(True)
                        else:
                            robo_flag.set(False)
                        print(f'IR THREAD :: Dwell is {robo_flag.val}.')
                    
                    s.count = s.count+1

                # Pause to make sure images release. 
                # If line is removed, thread will not shut down successfully.
                #time.sleep(0.05) 
                try: 
                    s.image_result.Release()
                except:
                    s._init_fail = True
                    return

            except PySpin.SpinnakerException as ex:
                print('IR CAMERA :: Error: %s' % ex)
                s._init_fail = True
                return
        
    def handle_close(s):
        # s.end = datetime.datetime.now()
        # print(f"\n-\nProcessed {s.count} frames in {(s.end-s.start).total_seconds()}.\nfps: {s.count/(s.end-s.start).total_seconds()}\n-\n")

        print ("IR CAMERA :: Closing IR Camera...")
        s._record_onoff = False

        try: 
            s.image_result.Release()
        except:
            pass
        
        cv2.destroyAllWindows()

        # ===== Gige ex. {acquire_and_display_images} =====
        s.cam.EndAcquisition()

        # ===== Gige ex. {run_single_camera} =====
        s.cam.DeInit()

        # ===== Gige ex. {main} =====
        del s.cam
        s.cam_list.Clear()
        s.system.ReleaseInstance()
    
    def debug(s):
        s._runflag_debug=Flag(True)
        s._debugging=True

    def debug_close(s):
        s._runflag_debug.set(False)

class MiniHandler:
    def __init__(s):
        s._running = Flag(True)
        s._threads = [
            s._ir_thread,
            s._img_thread,
        ]
        s.msg_queue = Queue(maxsize=0)
        s.img_queue = Queue(maxsize=0)

    def run(s):
        s._thread_objs=[]
        for thread in s._threads:
            t = threading.Thread(target=thread)
            t.start()
            s._thread_objs.append(t)
    
    def kill_threads(s):
        print("IR CAMERA :: Entered kill sequence.")
        s._running.set(False)
        for thread in s._thread_objs:
            thread.join()
        time.sleep(1)
        print("IR CAMERA :: Done! :)")

    def _ir_thread(s):
        ir_handler = IRHandler()
        ir_handler.run_camera(s.msg_queue, s.img_queue, s._running)
        while s._running:
            time.sleep(0.100)
        ir_handler.handle_close()
        print("IR CAMERA :: Exit _ir_thread")

    def _img_thread(s):
        while s._running or (not s.img_queue.empty()):
            msg = s.img_queue.get()
            cv2.imwrite(msg[0], msg[1])
        print("IR CAMERA :: Exit _img_thread")

if __name__ == '__main__':
    app = MiniHandler()
    app.run()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt: 
        print("IR CAMERA :: Killing threads...")
        final=app.kill_threads()