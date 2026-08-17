import io
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Path definitions
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "pest_classifier.pt"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference image preprocessing (matching val_test_transform from training)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_pest_model():
    """
    Loads MobileNetV2 architecture, restores trained weights from pest_classifier.pt,
    loads class names from class_names.json, and sets the model to evaluation mode.
    """
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Model or class names file not found in {MODELS_DIR}. "
            "Please ensure pest_classifier.pt and class_names.json exist."
        )

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    num_classes = len(class_names)

    # Recreate MobileNetV2 architecture matching train_pest_model.py
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    return model, class_names

# Module-level single load
model, class_names = load_pest_model()

def predict_pest(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes, applies exact training preprocessing,
    runs model inference, and returns predicted class, confidence, and all class probabilities.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid or corrupt image format: {str(e)}")

    # Preprocess image
    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    conf, pred_idx = torch.max(probabilities, dim=0)

    predicted_class = class_names[pred_idx.item()]
    confidence = float(conf.item())

    all_class_probabilities = {
        name: float(prob.item()) for name, prob in zip(class_names, probabilities)
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_class_probabilities": {
            k: round(v, 4) for k, v in all_class_probabilities.items()
        }
    }
