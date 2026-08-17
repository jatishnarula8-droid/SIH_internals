import os
import sys
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define the paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PLANTVILLAGE_DIR = RAW_DIR / "plantvillage"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Define the target classes we want to keep (source_folder_name: target_folder_name)
TARGET_CLASSES = {
    "Tomato_healthy": "Tomato___healthy",
    "Tomato_Early_blight": "Tomato___Early_blight",
    "Tomato_Late_blight": "Tomato___Late_blight",
    "Potato___healthy": "Potato___healthy",
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
}

def main():
    if not PLANTVILLAGE_DIR.exists():
        print(f"Error: PlantVillage dataset not found in {PLANTVILLAGE_DIR}")
        print("\nPlease follow these manual steps:")
        print("1. Go to Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("2. Download the dataset and extract the 'PlantVillage' folder into your raw data directory.")
        print(f"   (Ensure the folder path is exactly: {PLANTVILLAGE_DIR})")
        print("3. Run this script again.")
        sys.exit(1)

    print("Found PlantVillage dataset. Starting processing...")
    
    # Clear out existing processed directory before reprocessing
    if PROCESSED_DIR.exists():
        print(f"Clearing existing processed directory: {PROCESSED_DIR}")
        shutil.rmtree(PROCESSED_DIR)

    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    # Create output directories
    for split in splits.keys():
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)
        for class_name in set(TARGET_CLASSES.values()):
            (PROCESSED_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
            
    # Track metrics for summary
    summary = {split: {cls: 0 for cls in set(TARGET_CLASSES.values())} for split in splits.keys()}

    # Check inner structure, sometimes there is an extra nested 'PlantVillage' folder
    search_dir = PLANTVILLAGE_DIR
    if (PLANTVILLAGE_DIR / 'PlantVillage').exists():
        search_dir = PLANTVILLAGE_DIR / 'PlantVillage'

    # Filter, split, and copy
    for class_folder in search_dir.iterdir():
        if not class_folder.is_dir():
            continue
            
        class_name = class_folder.name
        if class_name not in TARGET_CLASSES:
            continue
            
        target_class_name = TARGET_CLASSES[class_name]
        print(f"\nProcessing class: {class_name} -> {target_class_name}")
        
        # PlantVillage usually has .JPG or .jpg extensions
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.JPG"))
        if not images:
            print(f"  Warning: No images found in {class_folder}")
            continue
            
        # Split: 80% train, 20% temp
        train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
        # Split temp: 50% val, 50% test (10% overall each)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        split_dict = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs
        }
        
        for split_name, img_list in split_dict.items():
            dest_dir = PROCESSED_DIR / split_name / target_class_name
            for img_path in tqdm(img_list, desc=f"  -> {split_name}", leave=False):
                dest_path = dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)
            summary[split_name][target_class_name] = len(img_list)
            
    print("\n" + "="*40)
    print("DATASET PREPARATION SUMMARY")
    print("="*40)
    for split_name, class_counts in summary.items():
        print(f"\n[{split_name.upper()}] split:")
        total_split = 0
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count} images")
            total_split += count
        print(f"  Total {split_name} images: {total_split}")

    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
