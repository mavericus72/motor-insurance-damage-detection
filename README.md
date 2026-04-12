# 🚗 Motor Insurance Damage Detection (AI)

An end-to-end **Deep Learning project** that detects whether a vehicle is **damaged or not** from an image.
Built using **PyTorch + ResNet18 + FastAPI**, and deployed as a **live API**.

---

## 📌 Problem Statement

In motor insurance, manual inspection of vehicle damage is:

* Time-consuming
* Expensive
* Prone to human error

👉 This project automates **damage detection** using AI.

---

## 🚀 Features

* Upload car image → get prediction
* Binary classification:

  * **Damage**
  * **No Damage**
* REST API using FastAPI
* Deployable on cloud platforms

---

## 🧠 Model Details

* Architecture: ResNet18 (Transfer Learning)
* Framework: PyTorch
* Training Strategy:

  * Freeze → Train final layer
  * Fine-tune entire model
* Image Size: 224 × 224
* Output: 2 classes

---

## 📊 Performance

* Validation Accuracy: **~91%**
* Optimized using:

  * Data augmentation
  * Fine-tuning
  * Adam optimizer

---

## 🏗️ Project Structure

```
motor-insurance-damage-detection/
│
├── app.py
├── model.pth
├── requirements.txt
├── Procfile
│
├── data/
├── notebooks/
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/motor-insurance-damage-detection.git
cd motor-insurance-damage-detection

pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🌐 API Usage

### Endpoint:

```
POST /predict
```

### Input:

* Upload image file

### Output:

```json
{
  "prediction": "damage"
}
```

---

## 🚀 Deployment

Deployed using:

* Render (Cloud Platform)
* FastAPI (Backend API)

---

## 🧰 Tech Stack

* Python
* PyTorch
* Torchvision
* FastAPI
* Uvicorn

---

## 📸 Example Use Cases

* Insurance claim automation
* Fleet damage monitoring
* Vehicle inspection systems

---

## 🔥 Future Improvements

* Multi-class damage classification
* Severity detection
* Bounding box (object detection)

---

## 👨‍💻 Author

**Aniket Patil**

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!
