import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Predict(BaseModel):
    class_name: str


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('data/model/model_torch/model_torch_vgg19.pth', map_location=device)

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


@app.post('/upload_file')
async def predict(file: UploadFile = File(...)):
    file_read = await file.read()
    img = read_imagefile(file_read)
    img_preproc = augmentation(img)
    batch_img = torch.unsqueeze(img_preproc, 0)
    model.eval()
    out = model(batch_img)
    out = out.tolist()[0]
    max_value = max(out)
    temp_prediction = out.index(max_value)

    if temp_prediction == 0:
        final_predict = '0% alcohol'
    elif temp_prediction == 1:
        final_predict = '12,5% alcohol'
    elif temp_prediction == 2:
        final_predict = '25% alcohol'
    elif temp_prediction == 3:
        final_predict = '50% alcohol'
    elif temp_prediction == 4:
        final_predict = '5% alcohol'
    elif temp_prediction == 5:
        final_predict = '75% alcohol'
    elif temp_prediction == 6:
        final_predict = '96% alcohol'

    return Predict(class_name=final_predict)


if __name__ == "__main__":
    uvicorn.run(app)
