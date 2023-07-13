import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn
from io import BytesIO
from PIL import Image
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
from math import hypot, pi
import numpy as np
import base64

class Predict(BaseModel):
    class_name: str

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('model_torch_vgg19.pth', map_location=device)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

augmentation = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)).convert('RGB')
    return image

def decoding(temp_prediction: int):
    alcohol_levels = {
        0: '0% alcohol',
        1: '12.5% alcohol',
        2: '25% alcohol',
        3: '50% alcohol',
        4: '5% alcohol',
        5: '75% alcohol',
        6: '96% alcohol'
    }
    final_predict = alcohol_levels.get(temp_prediction)
    return final_predict

def prediction(file):
    img = file
    img_preproc = augmentation(img)
    batch_img = torch.unsqueeze(img_preproc, 0)
    model.eval()
    out = model(batch_img)
    out = out.tolist()[0]
    max_value = max(out)
    temp_prediction = out.index(max_value)
    final_prediction = decoding(temp_prediction)
    print(final_prediction)
    return final_prediction

def genereaion_image(path):
    font = cv2.FONT_HERSHEY_SIMPLEX
    nparr = np.frombuffer(path, np.uint8)
    image_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_original_2 = read_imagefile(path)
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
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        centre_coordinstes = (cX, cY)
        cv2.putText(image_edited, '.', (cX, cY), font, 0.5, (255, 255, 255), 2)
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
    cv2.putText(image_edited, 'Area: ' + str(res_area), (50, 50), font, 1,
            (255, 255, 255), 2)
    cv2.putText(image_edited, 'Radius: ' + str(res_radius), (50, 80), font, 1,
        (255, 255, 255), 2)
    
    prediction_for_image = prediction(image_original_2)

    cv2.putText(image_edited, 'Concentration: ' + str(prediction_for_image), (50, 110), font, 1,
        (255, 255, 255), 2)
    
    is_success, buffer = cv2.imencode('.jpg', image_edited)
    byte_im = buffer.tobytes()

    return byte_im


@app.post('/upload_image_for_concentration')
async def predict(file: UploadFile = File(...)):
    file_read = await file.read()
    img = read_imagefile(file_read)
    final_predict = prediction(img)

    return Predict(class_name=final_predict)

@app.post('/upload_image_for_generating_image_normal')
async def image_concentration_normal(file: UploadFile = File(...)):
    file_read = await file.read()
    byte_im = genereaion_image(file_read)

    return StreamingResponse(BytesIO(byte_im), media_type="image/jpeg")

@app.post('/upload_image_for_generating_image_base64')
async def image_concentration_base64(file: UploadFile = File(...)):
    file_read = await file.read()
    byte_im = genereaion_image(file_read)
    base64_encoded_result = base64.b64encode(byte_im).decode('utf-8')

    return {'image': base64_encoded_result}


if __name__ == "__main__":
    uvicorn.run(app)
