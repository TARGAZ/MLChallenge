#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Challenge
# 
# Étudiants :
# - MONNIER Raphaël
# - PLANCHENAULT Allan
# - POINGT Tanguy
# - TAROT Bastien
# 
# Technique de Machine Learning utilisée pour construire le modèle : Se baser sur les images pour construire un modèle CNN

# ## Installation des libraries

# In[226]:


#get_ipython().run_line_magic('pip', 'install numpy matplotlib pandas scikit-learn keras tensorflow torch torchvision')


# In[227]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ## Code réutilisable

# ### Variables globales

# In[228]:


emotion_to_number = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "anger": 3,
    "surprise": 4,
    "disgust": 5,
    "fear": 6
}

number_to_emotion = {
    0 : "neutral",
    1 : "happy",
    2 : "sad",
    3 : "anger",
    4 : "surprise",
    5 : "disgust",
    6 : "fear"
}

BATCH_SIZE = 8
LEARNING_RATE = 0.00065
EPOCHS = 600


# ### Classes

# In[229]:


# Define the EmotionDataset class based on the Dataset class from PyTorch
class EmotionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, usecols=[0, 1])
        self.landmarks = pd.read_csv(annotations_file, usecols=lambda col: col not in ['id', 'labels'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        # Extract landmarks
        landmarks = self.landmarks.iloc[idx, :].values
        x = landmarks[:68]
        y = landmarks[68:]

        # Bounding box
        x_min, x_max = x.min() - 60, x.max() + 60
        y_min, y_max = y.min() - 120, y.max() + 60

        # Crop the image
        image = image.crop((x_min, y_min, x_max, y_max))

        label = self.img_labels.iloc[idx, 1]
        label = emotion_to_number[label]

        if self.transform:
            image = self.transform(image)

        return image, label


# ## Préparation des données

# In[230]:


# Define a temporary transform without normalization
temp_transform = transforms.Compose([
    transforms.Resize((1000, 750)),
    transforms.ToTensor()
])

# Load the dataset with the temporary transform
img_dir = "./project_data/training_set/"
csv_file = "./project_data/training_set.csv"

temp_dataset = EmotionDataset(annotations_file=csv_file, img_dir=img_dir, transform=temp_transform)
temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize variables to calculate mean and std
mean = 0.0
std = 0.0
nb_samples = 0

# Iterate through the dataset to calculate mean and std
for data in temp_loader:
    images, _ = data
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')

# Define the final transform with normalization
transform = transforms.Compose([
    #transforms.RandomRotation(degrees=5),
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])

# Load the final dataset with the normalization transform
dataset = EmotionDataset(annotations_file=csv_file, img_dir=img_dir, transform=transform)

# Split the dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to denormalize and convert tensor to PIL image
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Visualize some images after transformations
def show_images(loader, mean, std, num_images=5):
    data_iter = iter(loader)
    images, _ = next(data_iter)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img = denormalize(images[i], mean, std)
        img = transforms.ToPILImage()(img)
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()

# Show some images from the training loader
show_images(train_loader, mean.tolist(), std.tolist(), num_images=5)


# In[231]:


class CNNEmotionClassifier(nn.Module):
    def __init__(self):
        super(CNNEmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.60)
        self.dropout2 = nn.Dropout(p=0.70)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


model = CNNEmotionClassifier()


# In[232]:


# Model, criterion, optimizer
model = CNNEmotionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Training loop with scheduler
train_acc = []
val_acc = []
for epoch in range(EPOCHS):
    # Train phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val

    # Reduce learning rate if validation loss plateaus
    scheduler.step(val_loss)

    print(f'Epoch [{epoch+1}/{EPOCHS}], Training Loss: {running_loss/len(train_loader):.4f}, '
          f'Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
    
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)

# Make a plot of the training and validation accuracies
def plot_accuracies(train_acc, val_acc):
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

plot_accuracies(train_acc, val_acc)


# In[233]:


# Test phase
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f'Accuracy of the model on the test set: {test_accuracy:.2f}%')


# In[234]:


model_save_path = "./cnn_model_tanguy6.pth"
torch.save(model.state_dict(), model_save_path)


# ## Test du modèle (dans le cadre du rendu intermédiaire du challenge)

# ### Chargement des données de test

# In[235]:


test_csv_path = "./testing_data/testing_data.csv"
test_image_path = "./testing_data/testing_img/"

# Load the torch model
#model = torch.load("cnn_model.pth")


# ### Classe personnalisée pour le dataset de test comme on a pas les labels

# In[236]:


# Create a TestDataset espacially for the testing data (no labels)
class TestDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_ids = pd.read_csv(annotations_file, usecols=[0])
        self.landmarks = pd.read_csv(annotations_file, usecols=lambda col: col not in ['id'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_ids.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        # Extract landmarks
        landmarks = self.landmarks.iloc[idx, :].values
        x = landmarks[:68]
        y = landmarks[68:]

        # Bounding box
        x_min, x_max = x.min() - 60, x.max() + 60
        y_min, y_max = y.min() - 120, y.max() + 60

        # Crop the image
        image = image.crop((x_min, y_min, x_max, y_max))

        if self.transform:
            image = self.transform(image)

        return image


# ### Préparation des données de tests (normalisation, etc.)

# In[237]:


temp_dataset = TestDataset(annotations_file=test_csv_path, img_dir=test_image_path, transform=temp_transform)
temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize variables to calculate mean and std
mean = 0.0
std = 0.0
nb_samples = 0

# Iterate through the dataset to calculate mean and std
for data in temp_loader:
    images = data
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')

# Define the final transform with normalization
transform = transforms.Compose([
    #transforms.RandomRotation(degrees=5),
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])

# Load the final dataset with the normalization transform
dataset = TestDataset(annotations_file=test_csv_path, img_dir=test_image_path, transform=transform)
test_loader_ = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to denormalize and convert tensor to PIL image
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Visualize some images after transformations
def show_images(loader, mean, std, num_images=5):
    data_iter = iter(loader)
    images = next(data_iter)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img = denormalize(images[i], mean, std)
        img = transforms.ToPILImage()(img)
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()

# Show some images from the test loader
show_images(test_loader_, mean.tolist(), std.tolist(), num_images=5)


# ### Prédiction sur les données de test

# In[238]:


# Test phase
model = CNNEmotionClassifier()
model.load_state_dict(torch.load("cnn_model_tanguy6.pth"))
model.eval()

results = []

with torch.no_grad():
    for images in test_loader_:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for pred in predicted:
            results.append(number_to_emotion[pred.item()])
df = pd.DataFrame(results, columns=['labels'])
df.to_csv('results.csv', index=False)


# In[239]:


# Compare res.csv and results.csv on column labels
res = pd.read_csv('./testing_data/tests_pseudo_labels.csv')
results = pd.read_csv('results.csv')

# Measure accuracy
accuracy = np.sum(res['labels'] == results['labels']) / len(res['labels'])
print(f"Accuracy: {accuracy*100:.2f}%")

