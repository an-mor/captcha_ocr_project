import torch
from torch import nn


class CaptchaOCR(nn.Module):
    def __init__(self, num_classes, captcha_length, hidden_size=128):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            nn.MaxPool2d(2),
            self._conv_block(32, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 192),
        )
        self.sequence_model = nn.LSTM(
            input_size=192,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.position_pool = nn.AdaptiveAvgPool1d(captcha_length)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def extract_sequence(self, images):
        features = self.features(images)
        return features.mean(dim=2).transpose(1, 2)

    def forward(self, images):
        features = self.extract_sequence(images)
        sequence, _ = self.sequence_model(features)
        positions = self.position_pool(sequence.transpose(1, 2)).transpose(1, 2)
        return self.classifier(positions)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
