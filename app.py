from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
import os

app = FastAPI()

device = "cpu"  # Render uses CPU

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model_path = os.path.join(os.path.dirname(__file__), "model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))model.to(device)
model.eval()

class_names = ['damage', 'no_damage']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/")
def home():
    return {"message": "Damage Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs, dim=1).item()

    return {"prediction": class_names[pred]}
