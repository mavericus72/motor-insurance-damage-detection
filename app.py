from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
import os

app = FastAPI()

# Downlaod Model
MODEL_URL = "https://raw.githubusercontent.com/mavericus72/motor-insurance-damage-detection/main/model.pth"
MODEL_PATH = "model.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        try:
            response = requests.get(MODEL_URL, stream=True)

            if response.status_code != 200:
                raise Exception("Failed to download model")

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)

            print("Model downloaded successfully!")

        except Exception as e:
            print("ERROR downloading model:", e)
            raise e  # this will show error in logs
    else:
        print("Model already exists.")

download_model()


# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Class names
class_names = ['damage', 'no_damage']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/")
def home():
    return {"message": "Motor Insurance Damage Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs, dim=1).item()

    return {"prediction": class_names[pred]}
