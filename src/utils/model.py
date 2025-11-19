# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class SARBinarySegmentationModel(nn.Module):
    def __init__(
        self, in_channels, n_classes, device, encoder_name="mobilenet_v2", encoder_weights=None, activation="softmax"
    ):
        super(SARBinarySegmentationModel, self).__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            activation=activation,
        )
        self.device = device
        self.model = self.model.to(self.device)

    def forward(self, x):
        return self.model(x.to(self.device))


def load_model(model_path, device, in_channels, n_classes):
    model = SARBinarySegmentationModel(in_channels, n_classes, device)

    # Load the file based on extension
    if model_path.lower().endswith(('.pth', '.pt')):
        # Clean .pth/.pt file with just state_dict tensors (platform-agnostic)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

    elif model_path.lower().endswith('.ckpt'):
        # Full checkpoint file - Unix/Mac only (contains PosixPath objects)
        if platform.system() == "Windows":
            raise RuntimeError(
                "Cannot load .ckpt files on Windows due to PosixPath compatibility issues. "
                "Please use the .pth version of the model instead."
            )
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]

    else:
        raise ValueError(
            f"Unsupported model file format: {model_path}. Expected .pth, .pt, or .ckpt"
        )

    # Adjust state_dict keys for compatibility
    new_state_dict = {
        k.replace("model.model", "model"): v
        for k, v in state_dict.items()
        if k != "aug.rotate._param_generator.degrees"
    }
    model.load_state_dict(new_state_dict)

    # Set model precision
    if device.type == "cuda":
        model = model.half()
    else:
        model = model.float()

    model = model.to(device)
    model.eval()

    return model