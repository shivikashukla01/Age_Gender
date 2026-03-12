import os
from PIL import Image
from torch.utils.data import Dataset


class UTKFaceDataset(Dataset):

    def __init__(self, dataset_path, transform=None):

        self.dataset_path = dataset_path
        self.transform = transform
        self.data = []

        for file in os.listdir(dataset_path):

            try:
                age = int(file.split("_")[0])
                gender = int(file.split("_")[1])

                self.data.append((file, age, gender))

            except:
                continue


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        file, age, gender = self.data[idx]

        img_path = os.path.join(self.dataset_path, file)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, age, gender