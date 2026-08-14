from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io

app = FastAPI(title="MNIST CNN PyTorch API")

# -----------------------------
# CNN model definition
# IMPORTANT: This architecture must match the model used during training.
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Load trained model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()

# Assumes mnist_cnn_pytorch.pth contains model.state_dict().
checkpoint = torch.load(
    "mnist_cnn_pytorch.pth",
    map_location=device,
    weights_only=True
)

# Supports either a plain state_dict or a checkpoint containing "model_state_dict".
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# -----------------------------
# Home page
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MNIST CNN Digit Classifier</title>
    </head>
    <body>
        <h1>MNIST CNN Digit Classifier</h1>
        <p>Upload an image of a handwritten digit (0-9).</p>

        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """


# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        return {"error": "Invalid image file."}

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_digit].item()

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence * 100, 2)
    }
