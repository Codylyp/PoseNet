import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle


def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init(key+"/1x1", nn.Conv2d(in_channels, n1x1, kernel_size=1), weights),
            nn.ReLU()
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            init(key+"/3x3_reduce", nn.Conv2d(in_channels=in_channels, out_channels=n3x3red, kernel_size=1), weights),
            nn.ReLU(),
            init(key+"/3x3", nn.Conv2d(in_channels=n3x3red, out_channels=n3x3, kernel_size=3, padding=1), weights),
            nn.ReLU()
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            init(key+"/5x5_reduce", nn.Conv2d(in_channels=in_channels, out_channels=n5x5red, kernel_size=1), weights),
            nn.ReLU(),
            init(key+"/5x5", nn.Conv2d(in_channels=n5x5red, out_channels=n5x5, kernel_size=5, padding=2), weights),
            nn.ReLU()
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            init(key+"/pool_proj", nn.Conv2d(in_channels=in_channels, out_channels=pool_planes, kernel_size=1), weights),
            nn.ReLU()
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        feature1 = self.b1(x)
        feature2 = self.b2(x)
        feature3 = self.b3(x)
        feature4 = self.b4(x)

        out = torch.concat([feature1, feature2, feature3, feature4], dim=1)
        return out


class LossHeader(nn.Module):
    def __init__(self, key, weights=None):
        super(LossHeader, self).__init__()

        self.key = key;
        # TODO: Define loss headers
        if key == "loss1":
          self._loss = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.ReLU(),
            init(key+"/conv", nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1), weights),
            nn.ReLU(),
            nn.Flatten(),
            init(key+"/fc", nn.Linear(in_features=2048, out_features=1024), weights),
            nn.Dropout(p=0.7)
          )

        if key == "loss2":
          self._loss = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.ReLU(),
            init(key+"/conv", nn.Conv2d(in_channels=528, out_channels=128, kernel_size=1), weights),
            nn.ReLU(),
            nn.Flatten(),
            init(key+"/fc", nn.Linear(in_features=2048, out_features=1024), weights),
            nn.Dropout(p=0.7)
          )

        if key == "loss3":
          self._loss = nn.Sequential(
            nn.AvgPool2d(kernel_size=7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.Dropout(p=0.4)
          )

        self.out_xyz = nn.Linear(in_features=1024, out_features=3)
        self.out_wpqr = nn.Linear(in_features=1024, out_features=4)
        self.out_3_xyz = nn.Linear(in_features=2048, out_features=3)
        self.out_3_wpqr = nn.Linear(in_features=2048, out_features=4)

    def forward(self, x):
        # TODO: Feed data through loss headers
        loss = self._loss(x)
        if self.key == "loss3":
          xyz = self.out_3_xyz(loss)
          wpqr = self.out_3_wpqr(loss)
          
        else:
          xyz = self.out_xyz(loss)
          wpqr = self.out_wpqr(loss)

        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),##############
            nn.LocalResponseNorm(size=5),######################
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1), weights),
            nn.ReLU(),
            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5),#################
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #print("1f"),
            nn.ReLU()#################
        )
        
        # Example for InceptionBlock initialization
        #self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights) # TA given
        self.first_group = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32, "inception_3a", weights),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64, "inception_3b", weights),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(480, 192, 96, 208, 16, 48, 64, "inception_4a", weights),
        )
        self._loss1 = LossHeader("loss1", weights)

        self.second_group = nn.Sequential(
            InceptionBlock(512, 160, 112, 224, 24, 64, 64, "inception_4b", weights),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64, "inception_4c", weights),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64, "inception_4d", weights)
        )
        self._loss2 = LossHeader("loss2", weights)

        self.third_group = nn.Sequential(
            InceptionBlock(528, 256, 160, 320, 32, 128, 128, "inception_4e", weights),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(832, 256, 160, 320, 32, 128, 128, "inception_5a", weights),
            InceptionBlock(832, 384, 192, 384, 48, 128, 128, "inception_5b", weights)
        )
        self._loss3 = LossHeader("loss3")


        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        x1 = self.pre_layers(x)
        x1 = self.first_group(x1)
        loss1_xyz = self._loss1(x1)[0]
        loss1_wpqr = self._loss1(x1)[1]
        
        x2 = self.second_group(x1)
        loss2_xyz = self._loss2(x2)[0]
        loss2_wpqr = self._loss2(x2)[1]
        
        x3 = self.third_group(x2)
        loss3_xyz = self._loss3(x3)[0]
        loss3_wpqr = self._loss3(x3)[1]

        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr


    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        #poseGT[:,0:3] = poseGT[:,0:3]*100
        mse = nn.MSELoss()
        loss_1 = mse(p1_xyz, poseGT[:,0:3]) + self.w1_wpqr*mse(p1_wpqr, poseGT[:,3:7]/mse(poseGT[:,3:7], torch.zeros([20,4]).to("cuda")))
        loss_2 = mse(p2_xyz, poseGT[:,0:3]) + self.w2_wpqr*mse(p2_wpqr, poseGT[:,3:7]/mse(poseGT[:,3:7], torch.zeros([20,4]).to("cuda")))
        loss_3 = mse(p3_xyz, poseGT[:,0:3]) + self.w3_wpqr*mse(p3_wpqr, poseGT[:,3:7]/mse(poseGT[:,3:7], torch.zeros([20,4]).to("cuda")))
        #loss_1 = torch.norm((p1_xyz - poseGT[:,0:3]), p=2) + self.w1_wpqr*torch.norm((p1_wpqr - poseGT[:,3:7]/torch.norm(poseGT[:,3:7], p=1)), p=2)
        #loss_2 = torch.norm((p2_xyz - poseGT[:,0:3]), p=2) + self.w2_wpqr*torch.norm((p2_wpqr - poseGT[:,3:7]/torch.norm(poseGT[:,3:7], p=1)), p=2)
        #loss_3 = torch.norm((p3_xyz - poseGT[:,0:3]), p=2) + self.w3_wpqr*torch.norm((p3_wpqr - poseGT[:,3:7]/torch.norm(poseGT[:,3:7], p=1)), p=2)
        loss = self.w1_xyz*loss_1 + self.w2_xyz*loss_2 + self.w3_xyz*loss_3
        
        return loss
