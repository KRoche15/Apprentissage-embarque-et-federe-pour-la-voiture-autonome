import torch 
import torch.nn as nn


class Model_classif_1(nn.Module):
    def __init__(self):
        super(Model_classif_1, self).__init__()
        
        self.deepwise_conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding='same', groups=3).cuda()
        self.pointwise_conv2 = torch.nn.Conv2d(3, 64, kernel_size=1, padding='same').cuda()
        self.bn1 = torch.nn.BatchNorm2d(3).cuda()
        self.bn2 = torch.nn.BatchNorm2d(64).cuda()

        self.deepwise_conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same', groups=64).cuda()
        self.pointwise_conv4 = torch.nn.Conv2d(64, 128, kernel_size=1, padding='same').cuda()
        self.bn3 = torch.nn.BatchNorm2d(64).cuda()
        self.bn4 = torch.nn.BatchNorm2d(128).cuda()

        self.deepwise_conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=128).cuda()
        self.pointwise_conv6 = torch.nn.Conv2d(128, 256, kernel_size=1, padding='same').cuda()
        self.bn5 = torch.nn.BatchNorm2d(128).cuda()
        self.bn6 = torch.nn.BatchNorm2d(256).cuda()

        self.deepwise_conv7 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256).cuda()
        self.pointwise_conv8 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same').cuda()
        self.bn7 = torch.nn.BatchNorm2d(256).cuda()
        self.bn8 = torch.nn.BatchNorm2d(256).cuda()

        self.deepwise_conv9 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256).cuda()
        self.pointwise_conv10 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same').cuda()
        self.bn9 = torch.nn.BatchNorm2d(256).cuda()
        self.bn10 = torch.nn.BatchNorm2d(256).cuda()

        self.dense1 = torch.nn.Linear(23296, 512, bias=True).cuda()
        self.dense2 = torch.nn.Linear(512, 128, bias=True).cuda()
        self.dense3 = torch.nn.Linear(128, 9, bias=True).cuda()

        self.maxpooling = torch.nn.MaxPool2d(2, stride=2).cuda()
        self.maxpooling_half = torch.nn.MaxPool2d((1,2), stride=(1,2)).cuda()
        self.relu = torch.nn.ReLU().cuda()

        self.dropout = torch.nn.Dropout(0.1)

    '''
    this function is made to compute prediction using the given batch
    args:
        x: torch tensor representing one batch of data
    
    return:
        x: torch tensor which contains a batch of prediction
    '''
    def forward(self, x):

        x = self.deepwise_conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pointwise_conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = self.deepwise_conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.pointwise_conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = self.deepwise_conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.pointwise_conv6(x)
        x = self.relu(x)
        x = self.bn6(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = self.deepwise_conv7(x)
        x = self.relu(x)
        x = self.bn7(x)
        x = self.pointwise_conv8(x)
        x = self.relu(x)
        x = self.bn8(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = self.deepwise_conv9(x)
        x = self.relu(x)
        x = self.bn9(x)
        x = self.pointwise_conv10(x)
        x = self.relu(x)
        x = self.bn10(x)
        x = self.maxpooling_half(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x


def get_mobilenet_classif_1():
    model = Model_classif_1()
    if torch.cuda.is_available():
        model.cuda()

    return model