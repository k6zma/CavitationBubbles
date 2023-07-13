import tkinter as tk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import requests
import io

def select_image():
    global panelA, panelB
    
    path = filedialog.askopenfilename()
    file = {'file': open(path, 'rb')}
    image_original = cv2.imread(path)
    resposnse = requests.post('http://k6zma.ru/upload_image_for_generating_image_normal', files=file)
    image_edited = Image.open(io.BytesIO(resposnse.content))
    
    image_original = Image.fromarray(image_original)
    
    image_original = image_original.resize((600,600))
    image_edited = image_edited.resize((600,600))

    image_original = ImageTk.PhotoImage(image_original)
    image_edited = ImageTk.PhotoImage(image_edited)

    if panelA is None or panelB is None:
        panelA = tk.Label(image=image_original)
        panelA.image = image_original
        panelA.pack(side='left', padx=10, pady=10)
        panelB = tk.Label(image=image_edited)
        panelB.image = image_edited
        panelB.pack(side='right', padx=10, pady=10)
    else:
        panelA.configure(image=image_original)
        panelB.configure(image=image_edited)
        panelA.image = image_original
        panelB.image = image_edited

root = tk.Tk()

panelA = None
panelB = None

btn = tk.Button(root, text='Select an image', command=select_image)
btn.pack(side='bottom', fill='both', expand='yes', padx='10', pady='10')

root.mainloop()
