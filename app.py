from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn

app = FastAPI()

# Downlaod Model
MODEL_URL = "https://drive.google.com/uc?id=1Mbl_M8uvY_Riy-TPALQqA4k1XRy_WQrY"
MODEL_PATH = "model.pth"

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
