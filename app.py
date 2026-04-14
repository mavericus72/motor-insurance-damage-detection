from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
import os
import requests

app = FastAPI()

# ------------------ DOWNLOAD MODEL ------------------
MODEL_URL = "https://github.com/mavericus72/motor-insurance-damage-detection/raw/refs/heads/main/model1.pth"
MODEL_PATH = "model1.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded.")
    else:
        print("Model already exists.")

download_model()


# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

model1 = models.resnet18(pretrained=False) # Instead of pretrained=False use weights=False 
model1.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model1.fc.in_features, 2)
)
model1.load_state_dict(torch.load("model1.pth", map_location=device, weights_only=False))
model1.to(device)
model1.eval()

# Class names
class_names = ['damage', 'no_damage']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], # Normalize the input data since we use the pre-trained model
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "Motor Insurance Damage Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model1(image)
        pred = torch.argmax(outputs, dim=1).item()

    return {"prediction": class_names[pred]}
