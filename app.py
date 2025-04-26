import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QLabel, QVBoxLayout, QWidget, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import cv2

# Load true labels from CSV
labels_df = pd.read_csv(r"data\diabetic-retinopathy-detection\trainLabels.csv")
labels_dict = dict(zip(labels_df['image'], labels_df['level']))

# Initialize model
model = timm.create_model("inception_v4", pretrained="imagenet", num_classes=5)
checkpoint = torch.load('best_model_inceptionv4_focal.pth')
model.load_state_dict(checkpoint, strict=False)
model = model.to('cuda')
model.eval()

# Image preprocessing functions
IMG_SIZE = 512

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image at {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)
    return image

def preprocess_image(image_path):
    """Internal preprocessing that doesn't affect displayed image"""
    image = load_ben_color(image_path)
    image = Image.fromarray(image)
    return image

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetic Retinopathy Classifier")
        self.setGeometry(100, 100, 1600, 1000)  # Larger window
        
        # Create font objects
        self.title_font = QFont()
        self.title_font.setPointSize(16)
        self.title_font.setBold(True)
        
        self.button_font = QFont()
        self.button_font.setPointSize(14)
        
        self.class_font = QFont()
        self.class_font.setPointSize(14)
        self.class_font.setBold(True)
        
        self.text_font = QFont()
        self.text_font.setPointSize(12)
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()
        
        # Button to refresh images
        self.refresh_button = QPushButton("Refresh Predictions")
        self.refresh_button.setFont(self.button_font)
        self.refresh_button.clicked.connect(self.load_images)
        self.main_layout.addWidget(self.refresh_button, alignment=Qt.AlignCenter)
        
        # Image display area - one row per class (0-4)
        self.class_layouts = []
        for class_idx in range(5):
            class_layout = QHBoxLayout()
            
            # Class label
            class_label = QLabel(f"Class {class_idx}:")
            class_label.setFont(self.class_font)
            class_label.setAlignment(Qt.AlignCenter)
            class_label.setFixedWidth(120)
            class_layout.addWidget(class_label)
            
            # Image display
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(300, 300)  # Larger image display
            class_layout.addWidget(image_label)
            
            # Prediction info
            text_label = QLabel()
            text_label.setFont(self.text_font)
            text_label.setAlignment(Qt.AlignLeft)
            text_label.setFixedWidth(400)
            class_layout.addWidget(text_label)
            
            self.class_layouts.append((image_label, text_label))
            self.main_layout.addLayout(class_layout)
        
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        
        self.load_images()
    
    def get_true_label(self, image_path):
        """Get true label from CSV using image filename"""
        image_name = os.path.basename(image_path).split('.')[0]
        return labels_dict.get(image_name, -1)
    
    def predict_image(self, image_path):
        """Predict the class of an image with internal preprocessing"""
        try:
            preprocessed_image = preprocess_image(image_path)  # Internal preprocessing
            image_tensor = transform(preprocessed_image).unsqueeze(0).to('cuda')
            
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                return predicted.item()
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return -1
    
    def load_images(self):
        """Load and display one image per class (0-4) in order"""
        # Get all available images
        correct_images = []
        for class_idx in range(5):
            class_dir = f"correct_predictions_by_class/class_{class_idx}"
            if os.path.exists(class_dir):
                for img in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img)
                    correct_images.append((img_path, class_idx))  # (path, true_class)
        
        thousand_images = []
        if os.path.exists("1000_images"):
            for img in os.listdir("1000_images"):
                img_path = os.path.join("1000_images", img)
                true_label = self.get_true_label(img_path)
                if true_label != -1:
                    thousand_images.append((img_path, true_label))
        
        # Select one image per class (0-4)
        selected_images = []
        for class_idx in range(5):
            # Filter images for this class
            class_correct = [x for x in correct_images if x[1] == class_idx]
            class_thousand = [x for x in thousand_images if x[1] == class_idx]
            
            if random.random() < 0.58 and class_correct:
                selected_images.append(random.choice(class_correct))
            elif class_thousand:
                selected_images.append(random.choice(class_thousand))
            elif class_correct:  # Fallback
                selected_images.append(random.choice(class_correct))
            else:
                print(f"Warning: No images found for class {class_idx}")
                selected_images.append((None, class_idx))  # Placeholder
        
        # Display images in class order (0-4)
        for class_idx in range(5):
            image_label, text_label = self.class_layouts[class_idx]
            image_label.clear()
            text_label.clear()
            
            img_path, true_label = selected_images[class_idx]
            if img_path is None:
                text_label.setText(f"No image available for class {class_idx}")
                continue
            
            try:
                # Display original image (not preprocessed)
                pixmap = QPixmap(img_path)
                if pixmap.isNull():
                    raise ValueError("Invalid image file")
                pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(pixmap)
                
                # Get prediction (with internal preprocessing)
                predicted_label = self.predict_image(img_path)
                
                # Print to terminal
                image_name = os.path.basename(img_path)
                print(f"Class {class_idx} | Image: {image_name} | True: {true_label} | Predicted: {predicted_label} | {'CORRECT' if true_label == predicted_label else 'INCORRECT'}")
                
                # Display info in GUI
                text_label.setText(
                    f"Image: {os.path.basename(img_path)}\n"
                    f"True label: {true_label}\n"
                    f"Predicted: {predicted_label}\n"
                    f"{'✓ CORRECT' if true_label == predicted_label else '✗ INCORRECT'}"
                )
                # Changed incorrect color from orange to black
                text_label.setStyleSheet(
                    "color: green;" if true_label == predicted_label else "color: black;"
                )
            except Exception as e:
                print(f"Error displaying class {class_idx} image: {str(e)}")
                text_label.setText(f"Error loading image\n{os.path.basename(img_path) if img_path else ''}")

if __name__ == "__main__":
    app = QApplication([])
    viewer = ImageViewer()
    viewer.show()
    app.exec_()