import torch.nn as nn
import torchvision.models as models


class AgeGenderModel(nn.Module):

    def __init__(self):

        super(AgeGenderModel, self).__init__()

        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        num_features = base_model.fc.in_features

        base_model.fc = nn.Identity()

        self.base = base_model

        self.age_head = nn.Linear(num_features, 1)
        self.gender_head = nn.Linear(num_features, 2)


    def forward(self, x):

        features = self.base(x)

        age = self.age_head(features)
        gender = self.gender_head(features)

        return age, gender