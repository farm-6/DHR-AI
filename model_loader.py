# model_loader.py
import torch
from efficientnet_pytorch import EfficientNet


def load_model(model_path, num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    in_features = model._fc.in_features
    model._fc = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.eval()

    return model
