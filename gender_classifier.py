import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

class GenderClassifier:
    def __init__(self):
        self.model = self._build_model()
        self.image_size = (64, 64)  # Standard size for our input images
        
    def _build_model(self):
        """Build a simple CNN model for gender classification"""
        model = models.Sequential([
            # First Convolutional Layer
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Layer
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Layer
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten the output for dense layers
            layers.Flatten(),
            
            # Dense layers for classification
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),  # Prevent overfitting
            layers.Dense(1, activation='sigmoid')  # Output layer (1 for binary classification)
        ])
        
        # Compile the model
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        # Read and resize image
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_size)
        
        # Normalize pixel values
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def train(self, train_images, train_labels, validation_split=0.2, epochs=10):
        """Train the model with the provided dataset"""
        history = self.model.fit(
            train_images,
            train_labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32
        )
        return history
    
    def predict_gender(self, image_path):
        """Predict gender for a single image"""
        # Preprocess the image
        processed_img = self.preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(processed_img)[0][0]
        
        # Convert prediction to label (assuming 0 = female, 1 = male)
        gender = "male" if prediction > 0.5 else "female"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return gender, confidence * 100

def main():
    print("Gender Classification Demo")
    print("-------------------------")
    
    # Initialize the classifier
    classifier = GenderClassifier()
    
    # Here you would normally train the model with a dataset
    # For demo purposes, we'll just use the model to make predictions
    # You'll need to add your own training data and train the model first
    
    while True:
        # Get image path from user
        print("\nEnter the path to an image file (or 'q' to quit):")
        image_path = input().strip()
        
        if image_path.lower() == 'q':
            break
            
        if not os.path.exists(image_path):
            print("Error: File not found!")
            continue
            
        try:
            # Make prediction
            gender, confidence = classifier.predict_gender(image_path)
            print(f"\nPrediction: {gender.capitalize()} (Confidence: {confidence:.2f}%)")
        except Exception as e:
            print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()