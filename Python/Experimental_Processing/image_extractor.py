import cv2 
import numpy as np
from skimage.draw import line

class ImagePointExtractor:

    def __init__(self):
        self.points = [] 
        self.display_image = None
        self.original_image = None
        self.window_name = None

    #-----------------------------
    def clickevent(self, event, x, y, flags, param):

        #OpenCV calls image coordinates x,y 
        #change to u, v 
        u = x 
        v = y 

        #Left click adds points 
        if event == cv2.EVENT_LBUTTONDOWN:

            self.points.append([u, v])
            print(f"Point {len(self.points)}: " f"u={u}, v={v}")
            self.redraw_points()


        #Right click removes points 
        elif event == cv2.EVENT_RBUTTONDOWN:

            if len(self.points) > 0:

                removed = self.points.pop()
                print("Removed point:", removed)
                self.redraw_points()

    #-----------------------------------
    def redraw_points(self):
    
            self.display_image = (
                self.original_image.copy()
            )
    
            for number, (u, v) in enumerate(self.points, start = 1):
    
                cv2.circle(
                    self.display_image,
                    (int(u), int(v)),
                    radius = 5,
                    color = (0,0,255),
                    thickness = -1)
    
                cv2.putText(
                    self.display_image,
                    str(number),
                    (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, 255, 2)
    
            cv2.imshow(self.window_name, self.display_image)           

    #-----------------------------------
    def select_points(self, 
                      img, 
                      window_name = "Select Image Points",
                      max_points = None):

        self.points = []
        self.original_image = (img.copy())
        self.display_image = (img.copy())
        self.window_name = window_name
        

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.display_image)
        cv2.setMouseCallback(self.window_name, self.clickevent)

        print("\nLEFT CLICK  : add point")
        print("\nRIGHT CLICK : remove last point")
        print("\nPRESS KEY   : finish selection")

        while True:

            key = cv2.waitKey(20) & 0xFF

            if max_points is not None:
                if len(self.points) >= max_points:
                    break

            if key == 13:
                break

            if key == 27:
                self.points = []
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("\nLEFT CLICK  : add point")
        print("RIGHT CLICK : remove last point")
        print("ENTER  : finish selection")
        print("ESC : cancel")

        return np.asarray(self.points, dtype = int)


    #-----------------------------
    def select_independent_lines(self, img):

        points = self.select_points(img, window_name = "Select Independent Lines")

        if len(points) % 2 != 0: 
            raise ValueError("Independent lines require even number of points")

        line_mask = np.zeros(img.shape[:2], dtype = np.uint8)
        line_pixels = []

        for i in range(0, len(points), 2):

            u0, v0 = points[i]
            u1, v1 = points[i + 1]

            # skimage uses row, column = v, u
            rows, cols = line(v0, u0, v1, u1)
            line_mask[rows, cols] = 1
            pixels = np.column_stack((cols, rows))
            line_pixels.append(pixels)


        if line_pixels:

            line_pixels = np.vstack(line_pixels)

        else:

            line_pixels = np.empty(
                (0, 2), dtype=int)


        return (line_mask, line_pixels)


    # --------------------------------------------------------
    # Continuous polyline
    
    def select_continuous_line(
        self,
        img):

        points = self.select_points(
            img,
            window_name="Select Continuous Line")

        if len(points) < 2:

            raise ValueError(
                "At least two points are required "
                "to create a continuous line.")

        line_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        line_pixels = []

        for i in range(len(points) - 1):

            u0, v0 = points[i]
            u1, v1 = points[i + 1]

            rows, cols = line(v0, u0, v1, u1)

            line_mask[rows, cols] = 1

            pixels = np.column_stack((cols, rows))

            line_pixels.append(pixels)

        line_pixels = np.vstack(line_pixels)


        return (line_mask, line_pixels)


    # --------------------------------------------------------
    # Bounding-box crop

    def crop_image(
        self,
        img,
        corners):

        corners = np.asarray(corners, dtype=int)

        u_min = np.min(corners[:, 0])
        u_max = np.max(corners[:, 0])
        v_min = np.min(corners[:, 1])
        v_max = np.max(corners[:, 1])

        return img[v_min:v_max, u_min:u_max]


if __name__ == "__main__":

    # Create artificial test image
    image = np.zeros((500, 800), dtype=np.uint8)

    # Draw test rectangle
    cv2.rectangle(
        image,
        (150, 100),
        (650, 400),
        color=150,
        thickness=3)

    extractor = ImagePointExtractor()
    points = extractor.select_points(image)
    print("\nSelected camera coordinates:")
    print(points)

    line_mask, line_pixels = (
        extractor.select_independent_lines(image)
    )

    cv2.imshow(
        "Independent Line Mask",
        line_mask * 255
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    line_mask, line_pixels = (
        extractor.select_continuous_line(
            image
        )
    )

    cv2.imshow("Continuous Line Mask", line_mask * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()