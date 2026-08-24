import cv2 
import numpy as np
import skimage

class GetPtsFromImg:
    def __init__(s):
        pass

    def clickfind_pts(s, img):
        global xpts, ypts
        xpts=[]
        ypts=[]

        cv2.namedWindow("Resizable Window", cv2.WINDOW_NORMAL)
        cv2.imshow("Resizable Window", img)
        cv2.setMouseCallback("Resizable Window", clickevent)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        fin_pts=[]
        for i in range(len(xpts)):
            fin_pts.append([xpts[i],ypts[i]])
        
        return fin_pts
    
    def clickfind_indp_lines(s, img):
        global xpts, ypts
        xpts=[]
        ypts=[]

        cv2.imshow('image', img)
        cv2.setMouseCallback('image', clickevent)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for i in range(len(xpts)):
            if i%2==1:
                start=[xpts[i-1],ypts[i-1]]
                end=[xpts[i],ypts[i]]
                discrete_line = list(zip(*skimage.draw.line(*start, *end)))
                discrete_line=np.asarray(discrete_line)
                if i==1:
                    line_loc = discrete_line
                else:
                    line_loc=np.concatenate((line_loc,discrete_line),axis=0)

        lines = np.zeros((img.shape[0],img.shape[1]))
        for pt in range(len(line_loc)):
            #line_loc[pt]=abs(line_loc[pt]-[IR_color_data.shape[0],IR_color_data.shape[1]])
            # xpt = abs(line_loc[pt][0]-IR_color_data.shape[0])
            # ypt = abs(line_loc[pt][1]-IR_color_data.shape[1])
            xpt = line_loc[pt][1]
            ypt = line_loc[pt][0]
            lines[xpt,ypt]=1

        lines=lines.astype(np.uint8)

        return lines, line_loc
    
    def clickfind_cont_lines(s, img):
        global xpts, ypts
        xpts=[]
        ypts=[]

        cv2.imshow('image', img)
        cv2.setMouseCallback('image', clickevent)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for i in range(len(xpts)-1):
            start=[xpts[i],ypts[i]]
            end=[xpts[i+1],ypts[i+1]]
            discrete_line = list(zip(*skimage.draw.line(*start, *end)))
            discrete_line=np.asarray(discrete_line)
            if i==0:
                line_loc = discrete_line
            else:
                line_loc=np.concatenate((line_loc,discrete_line),axis=0)

        lines = np.zeros((img.shape[0],img.shape[1]))
        for pt in range(len(line_loc)):
            #line_loc[pt]=abs(line_loc[pt]-[IR_color_data.shape[0],IR_color_data.shape[1]])
            # xpt = abs(line_loc[pt][0]-IR_color_data.shape[0])
            # ypt = abs(line_loc[pt][1]-IR_color_data.shape[1])
            xpt = line_loc[pt][1]
            ypt = line_loc[pt][0]
            lines[xpt,ypt]=1

        lines=lines.astype(np.uint8)

        return lines, line_loc
    
    def crop_img(s, img,corners):
        xpts=[]; ypts=[]
        for pt in corners:
            xpts.append(pt[0])
            ypts.append(pt[1])
        xpts.sort()
        ypts.sort()

        return img[ypts[0]:ypts[3],xpts[0]:xpts[3]]

def clickevent(event,x, y, flags, param):
    global xpts, ypts
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"X: {x}\tY: {y}")
        xpts.append(x)
        ypts.append(y)