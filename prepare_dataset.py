import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def create_dataset_structure():
    """Create the basic folder structure for the dataset"""
    base_dir = 'dataset'
    subdirs = ['train/male', 'train/female', 'test/male', 'test/female']
    
    # Create directories
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    return base_dir

def prepare_dataset(base_dir, image_size=(64, 64)):
    """Prepare training and testing datasets"""
    # Lists to store images and labels
    images = []
    labels = []
    
    # Process male images
    male_dir = os.path.join(base_dir, 'train/male')
    for img_name in os.listdir(male_dir):
        img_path = os.path.join(male_dir, img_name)
        try:
            # Read and resize image
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            # Normalize pixel values
            img = img / 255.0
            # Add to lists
            images.append(img)
            labels.append(1)  # 1 for male
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Process female images
    female_dir = os.path.join(base_dir, 'train/female')
    for img_name in os.listdir(female_dir):
        img_path = os.path.join(female_dir, img_name)
        try:
            # Read and resize image
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            # Normalize pixel values
            img = img / 255.0
            # Add to lists
            images.append(img)
            labels.append(0)  # 0 for female
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, y_train, y_val

def main():
    print("Dataset Preparation Tool")
    print("----------------------")
    
    # Create dataset structure
    base_dir = create_dataset_structure()
    
    print("\nDataset directory structure created!")
    print("\nPlease add your images to the following directories:")
    print(f"- {base_dir}/train/male: for male training images")
    print(f"- {base_dir}/train/female: for female training images")
    print(f"- {base_dir}/test/male: for male testing images")
    print(f"- {base_dir}/test/female: for female testing images")
    
    input("\nPress Enter when you've added your images...")
    
    # Prepare the dataset
    try:
        X_train, X_val, y_train, y_val = prepare_dataset(base_dir)
        print("\nDataset prepared successfully!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
    except Exception as e:
        print(f"\nError preparing dataset: {str(e)}")

if __name__ == "__main__":
    main()