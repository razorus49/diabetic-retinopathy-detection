import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import timm
import time

base_image_dir = r"C:\Users\Illuminatus\projects\diabetic_retinopathy\data\diabetic-retinopathy-detection"
retina_df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels.csv'))
retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])

# Define the image directory correctly
image_dir = os.path.join(base_image_dir, "train", "train")

# Correctly construct the paths to the images
retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(image_dir, f'{x}.jpeg'))

# Check if images exist
retina_df['exists'] = retina_df['path'].map(os.path.exists)
# print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

# Add a column for left or right eye
retina_df['eye'] = retina_df['image'].map(lambda x: 1 if x.split('_')[-1] == 'left' else 0)

# Categorize the levels
retina_df['level_cat'] = retina_df['level'].map(lambda x: np.eye(1 + retina_df['level'].max())[x])

# Drop rows with missing data and filter rows with existing images
retina_df.dropna(inplace=True)
retina_df = retina_df[retina_df['exists']]

# idris_base_dir = r"C:\Users\Illuminatus\projects\diabetic_retinopathy\data\Idris\Idris"
# idris_image_dir = os.path.join(idris_base_dir, "images", "test")
# idris_label_path = os.path.join(idris_base_dir, "labels", "testing_labels.csv")
# retina_df = pd.read_csv(idris_label_path)
# retina_df['path'] = retina_df['Image name'].map(lambda x: os.path.join(idris_image_dir, f'{x}.jpg'))
# image_paths = retina_df['path'].values
# retina_df['exists'] = retina_df['path'].map(os.path.exists)
# # print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

# retina_df = retina_df[retina_df['exists']]

# if retina_df.empty:
#     raise ValueError("no images found")

# # Extract the labels (Retinopathy grade)
# retina_df['level'] = retina_df['Retinopathy grade']



# Get the list of image paths and labels from retina_df
image_paths = retina_df['path'].values
labels = retina_df['level'].values

# # Split the data into training+validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    image_paths, 
    labels, 
    test_size=0.3,  # 20% for testing, adjust as needed
    random_state=42  # Seed for reproducibility
)

# Further split the training+validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, 
    y_train_val, 
    test_size=0.3,  # 20% of the original data, making it 25% of the remaining 80%
    random_state=42  # Seed for reproducibility
)

#Reduce the training set size to 2000 
# selected_indices = np.random.choice(X_train.shape[0], 20000, replace=False)
# X_train = X_train[selected_indices]
# y_train = y_train[selected_indices]

# selected_indices = np.random.choice(X_val.shape[0], 2000, replace=False)
# X_val = X_val[selected_indices]
# y_val = y_val[selected_indices]



# print("Training set size:", X_train.shape[0], y_train.shape[0])
# print("Validation set size:", X_val.shape[0], y_val.shape[0])
# print("Test set size:", X_test.shape[0], y_test.shape[0])

IMG_SIZE = 512
def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
        
def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

class RetinopathyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = load_ben_color(img_path)
        image = Image.fromarray(image)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label



# Define transformations for validation/test data (without augmentation)
val_test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

batch_size= 16
# train_loader_len = int(len(X_train)/batch_size)
# train_dataset = RetinopathyDataset(X_train, y_train, transform=train_transform)
# val_dataset = RetinopathyDataset(X_val, y_val, transform=val_test_transform)
test_dataset = RetinopathyDataset(X_test, y_test, transform=val_test_transform)

# from dataloader_wrapper import DataloaderWrapper

if __name__ == '__main__':
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    from collections import Counter

    # Check class distribution
    test_dist = Counter(labels)
    print("Testing distribution:", test_dist)

    model =timm.create_model("inception_v4", pretrained="imagenet", num_classes=5)


    checkpoint = torch.load('best_model_inceptionv4_focal.pth')
    model.load_state_dict(checkpoint, strict=False)
    model = model.to('cuda')
    print("model init")


    class FocalLoss(nn.Module):
        
        def __init__(self, weight=None, 
                    gamma=2., reduction='mean'):
            nn.Module.__init__(self)
            self.weight = weight
            self.gamma = gamma
            self.reduction = reduction
            
        def forward(self, input_tensor, target_tensor):
            if self.weight is not None:
                self.weight = self.weight.to(input_tensor.device)
            log_prob = F.log_softmax(input_tensor, dim=-1)
            prob = torch.exp(log_prob)
            return F.nll_loss(
                ((1 - prob) ** self.gamma) * log_prob, 
                target_tensor, 
                weight=self.weight,
                reduction = self.reduction
            )
        
    weights = [0.0179, 0.245, 0.098, 0.335, 0.305]

    best_acc = 0.0
    since = time.time()

    model.eval()
    all_preds = []
    all_labels = []
    count_predict = {0:0, 1:0, 2:0, 3:0, 4:0}
    count_truth = {0:0, 1:0, 2:0, 3:0, 4:0}
    correct_predict = {0:0, 1:0, 2:0, 3:0, 4:0}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to('cuda'), data[1].to('cuda')
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            # Collect predictions and labels for Kappa
            all_preds.extend(predictions.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            
            # Existing counts for class-wise accuracy
            for label, prediction in zip(labels, predictions):
                count_truth[label.item()] += 1
                count_predict[prediction.item()] += 1
                if label == prediction:
                    correct_predict[label.item()] += 1

    # Calculate and print Quadratic Weighted Kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    print(f'\nQuadratic Weighted Kappa: {kappa:.4f}')

    # Existing class-wise accuracy calculations
    print("\nClass-wise Accuracy:")
    for classname, correct_count in correct_predict.items():
        accuracy = 100 * float(correct_count) / count_predict[classname] if count_predict[classname] != 0 else 0
        print(f'Class {classname}: {accuracy:.1f}%')
    print("predictions", count_predict)
    print("truths", count_truth)