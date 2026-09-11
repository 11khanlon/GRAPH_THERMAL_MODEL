import datetime
import numpy as np
import PySpin

from radiometric_model import RadiometricModel
from thermal_frame import ThermalFrame


class A700Camera:

    def __init__(self, environment, emissivity):

        self.environment = environment
        self.emissivity = emissivity

        self.system = None
        self.camera_list = None
        self.camera = None

        self.nodemap = None
        self.stream_nodemap = None

        self.camera_constants = None
        self.radiometric_model = None

        self.is_initialized = False
        self.is_acquiring = False


    # -------- Initialization ----------

    def initialize(self):

        print("A700 :: Initializing camera...")
        self.system = PySpin.System.GetInstance()
        self.camera_list = self.system.GetCameras()
        number_of_cameras = self.camera_list.GetSize()

        if number_of_cameras == 0:
            self.release_system()
            raise RuntimeError("No FLIR cameras detected.")

        if number_of_cameras > 1:
            self.release_system()

            raise RuntimeError("Multiple FLIR cameras detected. " 
                               "Camera selection is required.")


        self.camera = self.camera_list[0]
        self.camera.Init()
        self.nodemap = self.camera.GetNodeMap()
        self.stream_nodemap = self.camera.GetTLStreamNodeMap()
        

        self.configure_ir_format()
        self.configure_buffer()
        self.configure_acquisition()
        self.camera_constants = self.read_radiometric_constants()

        self.radiometric_model = (
            RadiometricModel(
                camera_constants = self.camera_constants,
                environment = self.environment,
                emissivity = self.emissivity)
                )

        self.is_initialized = True
        print("A700 :: Camera initialized.")


    #-------- Camera configuration ------------
    def configure_ir_format(self):

        node = PySpin.CEnumerationPtr(self.nodemap.GetNode("IRFormat"))
        entry = node.GetEntryByName("Radiometric")
        node.SetIntValue(entry.GetValue())

    def configure_buffer(self):

        node = PySpin.CEnumerationPtr(self.stream_nodemap.GetNode("StreamBufferHandlingMode"))
        entry = node.GetEntryByName("NewestOnly")
        node.SetIntValue(entry.GetValue())

    def configure_acquisition(self):

        node = PySpin.CEnumerationPtr(self.nodemap.GetNode("AcquisitionMode"))
        entry = node.GetEntryByName("Continuous")
        node.SetIntValue(entry.GetValue())


    #------- Radiometric constants ----------
    def read_radiometric_constants(self):

        return {"R": self.get_float_node("R"),
                "B": self.get_float_node("B"),
                "F": self.get_float_node("F"),
                "X": self.get_float_node("X"),
                "A1": self.get_float_node("alpha1"),
                "A2": self.get_float_node("alpha2"), 
                "B1": self.get_float_node("beta1"),
                "B2": self.get_float_node("beta2"),
                "J1": self.get_float_node("J1"),
                "J0": self.get_integer_node("J0")
                }


    def get_float_node(self, node_name):

        node = PySpin.CFloatPtr(self.nodemap.GetNode(node_name))

        if (not PySpin.IsAvailable(node) or not PySpin.IsReadable(node)):

            raise RuntimeError(f"Unable to read " f"{node_name}")
        
        return node.GetValue()


    def get_integer_node(self, node_name):

        node = PySpin.CIntegerPtr(self.nodemap.GetNode(node_name))

        if (not PySpin.IsAvailable(node) or not PySpin.IsReadable(node)):

            raise RuntimeError(f"Unable to read " f"{node_name}")

        return node.GetValue()


    # ---------- Acquisition -------------

    def start(self):

        if not self.is_initialized:

            raise RuntimeError("Camera must be initialized before acquisition")

        self.camera.BeginAcquisition()

        self.is_acquiring = True


    def acquire_frame(self):

        if not self.is_acquiring:

            raise RuntimeError("Camera acquisition has not started")

        image_result = self.camera.GetNextImage()

        try:
            if image_result.IsIncomplete():
                return None

            raw_image = image_result.GetNDArray().copy()
            temperature = self.radiometric_model.raw_to_temperature(raw_image)
            
            frame = ThermalFrame(frame_id = image_result.GetFrameID(),
                                 camera_timestamp=None,
                                 host_timestamp = datetime.datetime.now(),
                                 raw=raw_image,
                                 temperature = temperature.astype(np.float32)
                                 )

            return frame


        finally:

            image_result.Release()


    # Shutdown

    def close(self):

        print("A700 :: Closing camera...")

        if (self.camera is not None and self.is_acquiring):

            self.camera.EndAcquisition()

        if self.camera is not None:

            self.camera.DeInit()

        self.release_system()
        self.is_initialized = False
        self.is_acquiring = False


    def release_system(self):

        if self.camera_list is not None:

            self.camera_list.Clear()


        if self.system is not None:

            self.system.ReleaseInstance()