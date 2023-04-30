import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
from schemas import Predict
from starlette.middleware.cors import CORSMiddleware

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('data/model/model_torch_vgg19.pth', map_location=device)

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
    out = out.tolist()
    return Predict(
        water_0=round(out[0][0] * 100, 3),
        alcohol_5=round(out[0][1] * 100, 3),
        alcohol_12_5=round(out[0][2] * 100, 3),
        alcohol_25=round(out[0][3] * 100, 3),
        alcohol_50=round(out[0][4] * 100, 3),
        alcohol_75=round(out[0][5] * 100, 3),
        alcohol_96=round(out[0][6] * 100, 3),
    )


if __name__ == "__main__":
    uvicorn.run(app)
