class ThermalFrameWriter:

    def __init__(
        self,
        output_folder
    ):

        self.output_folder = Path(
            output_folder
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )


    def save_frame(
        self,
        frame
    ):

        frame_name = (
            f"{frame.frame_id:08d}"
        )

        np.save(
            self.output_folder
            / f"{frame_name}_raw.npy",

            frame.raw
        )

        np.save(
            self.output_folder
            / f"{frame_name}_temperature.npy",

            frame.temperature.astype(
                np.float32
            )
        )