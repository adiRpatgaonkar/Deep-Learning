import torch.nn as nn


def models(model_id):
    if model_id == 'cnn0':
        return CNN_0()
    if model_id == 'cnn1':
        return CNN_1()
    if model_id == 'cnn2':
        return CNN_2()
    if model_id == 'cnn3':
        return CNN_3()
    if model_id == 'cnn4':
        return CNN_4()

# CNN Model (1 conv layers, 1 fc layers)
# model_id -> cnn1
class CNN_0(nn.Module):
    def __init__(self):
        super(CNN_0, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(14*14*16, 10)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# CNN Model (2 conv layers, 3 fc layers)
# model_id -> cnn1
class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(5*5*32, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# CNN Model (3 conv layers, 3 fc layers)
# model_id -> cnn2
class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# CNN Model (2 conv layers, 3 fc layers)
# with dropouts i.e. cnn1 with dropouts
# model_id -> cnn3
class CNN_3(nn.Module):
    def __init__(self):
        super(CNN_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(5*5*32, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 10),
            nn.Dropout())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# CNN Model (6 conv layers, 1 fc layer, dropouts in Conv layers)
#TODO This model has a visualization feature. Messy code. Can / should be changed.
# Credits: https://appliedmachinelearning.wordpress.com/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
# Abhijeet Kumar
# model_id -> cnn4
class CNN_4(nn.Module):
    def __init__(self):
        super(CNN_4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3))
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4))
        self.fc = nn.Linear(2048, 10)

    def forward(self, x, vis=None):
        if vis is True:
            self.out_buffer = []
        out = self.conv1(x)
        if vis is True:
            self.out_buffer.append(out)
        #print(out.size())
        out = self.conv2(out)
        if vis is True:
            self.out_buffer.append(out)
        #print(out.size())
        out = self.conv3(out)
        if vis is True:
            self.out_buffer.append(out)
        #print(out.size())
        out = self.conv4(out)
        if vis is True:
            self.out_buffer.append(out)
        #print(out.size())
        out = self.conv5(out)
        if vis is True:
            self.out_buffer.append(out)
        #print(out.size())
        out = self.conv6(out)
        if vis is True:
            self.out_buffer.append(out)
        #print(out.size())
        out = out.view(out.size(0), -1)
        #print(out.size())
        out = self.fc(out)
        if vis is True:
            self.out_buffer.append(out)
        #print(out.size())
        return out

