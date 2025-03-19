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
import albumentations as A
import cv2
import timm
import time

#2015 dataset
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
# idris_image_dir = os.path.join(idris_base_dir, "images", "train")
# idris_label_path = os.path.join(idris_base_dir, "labels", "training_labels.csv")
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

# Split the data into training+validation and test sets
X_train, X_val, y_train, y_val = train_test_split(
    image_paths, 
    labels, 
    test_size=0.15,  # 20% for testing, adjust as needed
    random_state=42  # Seed for reproducibility
)

# Further split the training+validation set into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(
#     image_paths, 
#     labels, 
#     test_size=0.15,  # 20% of the original data, making it 25% of the remaining 80%
#     random_state=42  # Seed for reproducibility
# )


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

# Apply transformations to resize and normalize the images

train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomAffine(
        degrees=(-180, 180),
        scale=(0.8889, 1.0),
        shear=(-36, 36)),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(contrast=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((512, 512)),           # Resize to 224x224
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(                    # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Example usage
train_dataset = RetinopathyDataset(X_train, y_train, transform=train_transform)
val_dataset = RetinopathyDataset(X_val, y_val, transform=val_test_transform)
# test_dataset = RetinopathyDataset(X_test, y_test, transform=val_test_transform)

batch_size = 16
if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    from collections import Counter

    # Check class distribution
    train_dist = Counter(y_train)
    # test_dist = Counter(y_test)
    val_dist = Counter(y_val)
    # print("Training distribution:", train_dist)
    # print("Valid distribution:", val_dist)
    # print("Testing distribution:", val_dist)


    model =timm.create_model("inception_v4", pretrained="imagenet", num_classes=5)


    # checkpoint = torch.load('inceptionv4focal_8-3.pth')
    # model.load_state_dict(checkpoint, strict=False)
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
    # weights = [0.0120, 0.0581, 0.1266, 0.3582, 0.4451]
    # weights = [0.0213, 0.202, 0.0635, 0.3802, 0.333]    
    # weights = [0.0211, 0.2, 0.0889, 0.375, 0.315]
    # weights = [0.0185, 0.24, 0.0965, 0.34, 0.305]
    weights = [0.0179, 0.245, 0.098, 0.335, 0.305]

    num_epochs = 15
    scaler = torch.amp.GradScaler('cuda')
    criterion = FocalLoss(weight=torch.tensor(weights)).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch = len(train_loader), epochs = num_epochs)


    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # PATH = './inceptionv4_2015_focal_scheduled.pth'
    best_model_params_path = './inceptionv4focal_10-3.pth'
    best_acc = 0.0
    since = time.time()

    print("Training start")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()


                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                scheduler.step()
                print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    print('Saved new best model')


    # Training complete
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    print("train acc history", train_acc_history)
    print("val acc history",val_acc_history)
    print("train loss history", train_loss_history)
    print("val loss histroy", val_loss_history)