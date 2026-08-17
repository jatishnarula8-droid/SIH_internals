import os
import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

def main():
    start_time = time.time()
    
    # Define paths relative to script root
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_DIR = PROCESSED_DIR / "train"
    VAL_DIR = PROCESSED_DIR / "val"
    TEST_DIR = PROCESSED_DIR / "test"

    if not (TRAIN_DIR.exists() and VAL_DIR.exists() and TEST_DIR.exists()):
        print(f"Error: Processed dataset directory not found at {PROCESSED_DIR}")
        sys.exit(1)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA GPU not available. Training on CPU will be slower.")
    else:
        print(f"Using GPU device: {torch.cuda.get_device_name(0)}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=val_test_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    batch_size = 32
    # Windows DataLoader compatibility
    num_workers = 2 if os.name != 'nt' else 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load MobileNetV2 with pretrained weights
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    # Freeze all feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model = model.to(device)

    # Loss function and optimizer (optimizing classifier parameters only)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    epochs = 8
    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        print(f"Epoch [{epoch}/{epochs}] - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    # Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    test_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

    final_test_loss = test_loss / total_test
    final_test_acc = (correct_test / total_test) * 100.0
    print(f"Final Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_acc:.2f}%")

    # Save model weights and class names
    model_save_path = MODELS_DIR / "pest_classifier.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    class_names_path = MODELS_DIR / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to: {class_names_path}")

    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\nTotal training time: {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()
