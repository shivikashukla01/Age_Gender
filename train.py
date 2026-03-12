import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


from dataset import UTKFaceDataset
from model import AgeGenderModel


dataset_path = "dataset/UTKFace"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


dataset = UTKFaceDataset(dataset_path, transform)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset,[train_size,val_size])


train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32)


model = AgeGenderModel().to(device)


age_loss_fn = nn.MSELoss()
gender_loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)


epochs = 5


for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, ages, genders in train_loader:

        images = images.to(device)
        ages = ages.float().to(device)
        genders = genders.to(device)

        pred_age, pred_gender = model(images)

        pred_age = pred_age.squeeze()

        loss_age = age_loss_fn(pred_age, ages)
        loss_gender = gender_loss_fn(pred_gender, genders)

        loss = loss_age + loss_gender

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(train_loader)}")


torch.save(model.state_dict(),"models/age_gender_model.pth")

print("Model Saved!")