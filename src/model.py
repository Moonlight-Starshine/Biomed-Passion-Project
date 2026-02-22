"""
Model architectures for disease classification.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class DiseaseClassifier(nn.Module):
    """ResNet50-based disease classifier with custom head."""
    
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.3):
        """
        Args:
            num_classes: Number of disease classes
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout rate in classifier head
        """
        super(DiseaseClassifier, self).__init__()
        
        # Backbone: ResNet50
        backbone = models.resnet50(pretrained=pretrained)
        
        # Remove classification head
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classifier head
        feature_dim = 2048  # ResNet50 output dimension
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LightweightClassifier(nn.Module):
    """MobileNetV2-based lightweight classifier for edge deployment."""
    
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.2):
        """
        Args:
            num_classes: Number of disease classes
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout rate in classifier head
        """
        super(LightweightClassifier, self).__init__()
        
        # Backbone: MobileNetV2
        backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Remove classification head
        self.features = backbone.features
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classifier head
        feature_dim = 1280  # MobileNetV2 output dimension
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(num_classes, architecture="resnet50", pretrained=True):
    """
    Get a model architecture.
    
    Args:
        num_classes: Number of disease classes
        architecture: 'resnet50' or 'mobilenet_v2'
        pretrained: Use ImageNet pretrained weights
    
    Returns:
        model: PyTorch model
    """
    if architecture == "resnet50":
        model = DiseaseClassifier(num_classes, pretrained=pretrained)
    elif architecture == "mobilenet_v2":
        model = LightweightClassifier(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model
