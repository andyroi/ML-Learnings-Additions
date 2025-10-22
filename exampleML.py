import numpy as np
from sklearn import datasets, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

class BasicML:
    def __init__(self):
        """Initialize the ML model and preprocessing tools"""
        # Initialize the neural network with a simple architecture
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
            max_iter=1000,                 # Maximum iterations for training
            activation='relu',             # ReLU activation function
            solver='adam',                 # Adam optimizer
            random_state=42                # For reproducibility
        )
        # Initialize the scaler for data preprocessing
        self.scaler = StandardScaler()
        
    def load_sample_data(self):
        """Load a sample dataset (iris dataset) for demonstration"""
        # Load the classic iris dataset (a simple dataset with 3 types of flowers)
        iris = datasets.load_iris()
        self.X = iris.data        # Features (measurements of the flowers)
        self.y = iris.target      # Labels (type of flower)
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
    def preprocess_data(self):
        """Scale the data to have zero mean and unit variance"""
        # Fit the scaler on training data and transform both training and test data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train(self):
        """Train the model and print progress"""
        print("Training the model...")
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Calculate and print training accuracy
        train_accuracy = self.model.score(self.X_train_scaled, self.y_train)
        print(f"Training accuracy: {train_accuracy:.2%}")
        
    def evaluate(self):
        """Evaluate the model on test data"""
        test_accuracy = self.model.score(self.X_test_scaled, self.y_test)
        print(f"Test accuracy: {test_accuracy:.2%}")
        
        # Make predictions on test data
        predictions = self.model.predict(self.X_test_scaled)
        
        # Print some example predictions
        print("\nExample predictions:")
        for i in range(5):
            true_label = self.target_names[self.y_test[i]]
            predicted_label = self.target_names[predictions[i]]
            print(f"True: {true_label}, Predicted: {predicted_label}")
            
    def visualize_data(self):
        """Create a simple visualization of the data"""
        # We'll plot the first two features
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='viridis')
        plt.xlabel(self.feature_names[0])
        plt.ylabel(self.feature_names[1])
        plt.title('Iris Dataset - First Two Features')
        plt.colorbar(scatter)
        plt.show()
        
    def predict_single_sample(self, measurements):
        """Make a prediction for a single sample"""
        # Scale the input data
        scaled_sample = self.scaler.transform([measurements])
        # Make prediction
        prediction = self.model.predict(scaled_sample)
        # Get probability estimates
        probabilities = self.model.predict_proba(scaled_sample)
        
        return {
            'class': self.target_names[prediction[0]],
            'probabilities': {
                name: prob for name, prob in zip(self.target_names, probabilities[0])
            }
        }

def main():
    # Create an instance of our ML class
    ml = BasicML()
    
    # Load and prepare the data
    print("Loading data...")
    ml.load_sample_data()
    
    # Visualize the raw data
    print("\nShowing data visualization...")
    ml.visualize_data()
    
    # Preprocess the data
    print("\nPreprocessing data...")
    ml.preprocess_data()
    
    # Train the model
    print("\nTraining model...")
    ml.train()
    
    # Evaluate the model
    print("\nEvaluating model...")
    ml.evaluate()
    
    # Interactive prediction
    print("\nNow you can try predicting with your own measurements!")
    print("The iris dataset uses these features:", ml.feature_names)
    
    while True:
        try:
            print("\nEnter 4 measurements (separated by spaces), or 'q' to quit:")
            user_input = input().strip()
            
            if user_input.lower() == 'q':
                break
                
            # Convert input string to list of floats
            measurements = [float(x) for x in user_input.split()]
            if len(measurements) != 4:
                print("Please enter exactly 4 measurements!")
                continue
                
            # Make prediction
            result = ml.predict_single_sample(measurements)
            
            # Print results
            print(f"\nPredicted class: {result['class']}")
            print("\nProbabilities:")
            for flower, prob in result['probabilities'].items():
                print(f"{flower}: {prob:.2%}")
                
        except ValueError:
            print("Invalid input! Please enter numbers separated by spaces.")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
