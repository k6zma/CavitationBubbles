from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from math import hypot, pi
import numpy as np
import cv2
import requests

def select_image():
    global panelA, panelB
    
    path = filedialog.askopenfilename()
	
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(path) > 0:
        image_original = cv2.imread(path)
        image_edited = cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)
        height = image_edited.shape[0]
        width = image_edited.shape[1]
        ret, thresh = cv2.threshold(image_edited, 40, 255, 0, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        a = 1
        b = 0

        for i, contour in enumerate(contours):
            if a < len(contour) and [0, 0] not in contour and [0, height] not in contour and [width, height] not in contour and [width, 0] not in contour:
                a = len(contour)
                b = i
                bigger_contour = contour
            else:
                pass

        try:
            M = cv2.moments(bigger_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centre_coordinstes = (cX, cY)
            cv2.putText(image_edited, ".", (cX, cY), font, 0.5, (255, 255, 255), 2)
        except Exception as e:
            pass

        distance = []

        for point in bigger_contour:
            new_distance = hypot(point[0][0] - cX, point[0][1] - cY)
            distance.append(new_distance)
        distance = np.array(distance)
        dist_res = distance
        distance = min(distance)
        radius = distance
        color = (255, 0, 0)
        image_edited = cv2.cvtColor(image_edited, cv2.COLOR_GRAY2RGB)
        new_img = cv2.circle(image_edited, centre_coordinstes,
                            int(distance), color, thickness=4)
        scale_bar = 200
        coeff = (scale_bar * 4) / width
        area = sum(dist_res) + pi * radius ** 2
        res_area = coeff * area
        res_radius = coeff * radius

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 9 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(image=image_edited, contours=bigger_contour, contourIdx=-1, color=(0, 255, 0), thickness=4,
                            lineType=cv2.LINE_AA)
        cv2.putText(image_edited, "Area: " + str(res_area), (50, 50), font, 1,
                (255, 255, 255), 2)
        cv2.putText(image_edited, "Radius: " + str(res_radius), (50, 80), font, 1,
            (255, 255, 255), 2)
        
        file = {'file': open(path, 'rb')}
        prediction = requests.post(
            url='http://k6zma.ru/upload_file',
            files=file
            ).json()

        cv2.putText(image_edited, "Concentration: " + str(prediction['class_name']), (50, 110), font, 1,
            (255, 255, 255), 2)
        
        image_original = Image.fromarray(image_original)
        image_edited = Image.fromarray(image_edited)

        image_original = ImageTk.PhotoImage(image_original)
        image_edited = ImageTk.PhotoImage(image_edited)


        if panelA is None or panelB is None:
            panelA = Label(image=image_original)
            panelA.image = image_original
            panelA.pack(side="left", padx=10, pady=10)
            panelB = Label(image=image_edited)
            panelB.image = image_edited
            panelB.pack(side="right", padx=10, pady=10)
        else:
            panelA.configure(image=image_original)
            panelB.configure(image=image_edited)
            panelA.image = image_original
            panelB.image = image_edited

root = Tk()
panelA = None
panelB = None
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
root.mainloop()