# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SARBinarySegmentationModel(nn.Module):
    def __init__(self, in_channels, n_classes, device, encoder_name="mobilenet_v2", encoder_weights=None, activation="softmax"):
        super(SARBinarySegmentationModel, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            activation=activation
        )
        self.device = device
        self.model = self.model.to(self.device)  
    
    def forward(self, x):   
        return self.model(x.to(self.device))  

def load_model(model_path, device, in_channels, n_classes):
    model = SARBinarySegmentationModel(in_channels, n_classes, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    new_state_dict = {k.replace('model.model', 'model'): v for k, v in checkpoint['state_dict'].items() if k != "aug.rotate._param_generator.degrees"}
    model.load_state_dict(new_state_dict)
    if device.type == 'cuda': 
        model = model.half()
    else:
        model = model.float()
    model = model.to(device)
    model.eval()
    return model