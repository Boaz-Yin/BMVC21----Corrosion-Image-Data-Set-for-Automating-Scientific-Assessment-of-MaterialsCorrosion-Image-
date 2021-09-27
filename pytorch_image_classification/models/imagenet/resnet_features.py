import torch
import torch.nn as nn
import torch.nn.functional as F

#from Sklearn_PyTorch import TorchRandomForestClassifier


class Res_code(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):

        f1 = self.network._forward_conv(x)
        f1 = f1.view(f1.size(0), -1)

        return f1



class Res_mlp(nn.Module):
    def __init__(self, network, mode, act):
        super().__init__()
        self.network = network
        self.mode = mode
        self.act = act
        # 512 = self.network.feature_size

        # print('---size---', 512)

        if self.mode == 0 and self.act =='Relu':

            self.mlp = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 5)
            )
            
    def forward(self, x):

        f1 = self.network._forward_conv(x)


        f1 = f1.view(f1.size(0), -1)

        # print('--s---',f1.size())

        f2 = self.mlp(f1)

        return f2



def classifier_RF(target, feature, name, eva):

    if eva == 'train':
        my_model = TorchRandomForestClassifier(nb_trees=100, nb_samples=3, max_depth=5, bootstrap=True)

        my_model.fit(feature, target)
    else:
        my_result = my_model.predict(feature)


def classifier_RF(target, feature, name, eva):
    if eva == 'train':
        my_model = TorchRandomForestClassifier(nb_trees=100, nb_samples=3, max_depth=5, bootstrap=True)

        my_model.fit(feature, target)
    else:
        my_result = my_model.predict(feature)


from torch import nn

def create_combined_model(model_fe):
  # Step 1. Isolate the feature extractor.
  model_fe_features = nn.Sequential(
    # model_fe.quant,  # Quantize the input
    model_fe.conv1,
    model_fe.bn1,
    model_fe.relu,
    model_fe.maxpool,
    model_fe.layer1,
    model_fe.layer2,
    model_fe.layer3,
    model_fe.layer4,
    model_fe.avgpool,
    # model_fe.dequant,  # Dequantize the output
  )

  # Step 2. Create a new "head"
  new_head = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 2),
  )

  # Step 3. Combine, and don't forget the quant stubs.
  new_model = nn.Sequential(
    model_fe_features,
    nn.Flatten(1),
    new_head,
  )
  return new_model
