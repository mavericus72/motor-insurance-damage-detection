## Fast API code for model1 (EfficientNetB0)

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
import os
import requests
import torch.nn.functional as F

app = FastAPI()

# ------------------ DOWNLOAD MODEL ------------------
MODEL_URL = "https://github.com/mavericus72/motor-insurance-damage-detection/raw/refs/heads/main/efficientnet_model.pth"
MODEL_PATH = "efficientnet_model.pth"

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

model = models.efficientnet_b0(pretrained=False)

model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 2)
)

model.load_state_dict(torch.load("efficientnet_model.pth", map_location=device))
model.to(device)
model.eval()

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
        outputs = model(image)
        
        # Get prediction
        pred = torch.argmax(outputs, dim=1).item()
        
        # Confidence check
        probs = F.softmax(outputs, dim=1)
        confidence = probs[0][pred].item()

    # Decision logic check
    if confidence < 0.75:
        decision = "manual_review"
    elif class_names[pred] == "damage":
        decision = "approve_claim_flow"
    else:
        decision = "no_damage"

    return {
        "prediction": class_names[pred],
        "confidence": round(confidence, 3),
        "decision": decision
    }

