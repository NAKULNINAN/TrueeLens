import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class XceptionNet(nn.Module):
    """
    XceptionNet model for deepfake detection
    Based on the Xception architecture with modifications for binary classification
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionNet, self).__init__()
        
        # Load pretrained Xception model
        # Note: PyTorch doesn't have Xception directly, so we'll use a custom implementation
        # or adapt from EfficientNet which has similar properties
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Replace the classifier for binary classification
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class CustomXception(nn.Module):
    """
    Custom Xception-like architecture for deepfake detection
    """
    def __init__(self, num_classes=2):
        super(CustomXception, self).__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Separable convolution blocks
        self.sep_conv1 = self._make_separable_conv(64, 128, 2)
        self.sep_conv2 = self._make_separable_conv(128, 256, 2)
        self.sep_conv3 = self._make_separable_conv(256, 728, 2)
        
        # Middle flow (8 blocks of separable convolutions)
        self.middle_blocks = nn.ModuleList([
            self._make_separable_conv(728, 728, 1) for _ in range(8)
        ])
        
        # Exit flow
        self.sep_conv4 = self._make_separable_conv(728, 1024, 2)
        
        self.sep_conv5 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1, groups=1024, bias=False),
            nn.Conv2d(1024, 1536, 1, bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True)
        )
        
        self.sep_conv6 = nn.Sequential(
            nn.Conv2d(1536, 1536, 3, padding=1, groups=1536, bias=False),
            nn.Conv2d(1536, 2048, 1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
        
    def _make_separable_conv(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, 
                     groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Entry flow
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        
        # Middle flow
        for block in self.middle_blocks:
            residual = x
            x = block(x)
            x += residual
        
        # Exit flow
        x = self.sep_conv4(x)
        x = self.sep_conv5(x)
        x = self.sep_conv6(x)
        
        # Global average pooling and classification
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def load_xception_model(version='latest', num_classes=2, use_custom=True, load_pretrained=False):
    """
    Load XceptionNet model for deepfake detection
    
    Args:
        version: Model version (not used in this implementation)
        num_classes: Number of output classes
        use_custom: Whether to use custom or standard architecture
        load_pretrained: Whether to attempt loading pretrained weights (disabled by default)
    """
    if use_custom:
        model = CustomXception(num_classes=num_classes)
        model_name = 'CustomXception'
    else:
        model = XceptionNet(num_classes=num_classes)
        model_name = 'XceptionNet'
    
    if load_pretrained:
        # Only attempt to load pretrained weights if explicitly requested
        # and you have actual model files available
        print(f"Warning: Pretrained weights not available for {model_name}")
        print("To use pretrained weights, please provide model files and update this function")
    else:
        print(f"✅ {model_name} loaded with randomly initialized weights")
        print("   Note: For production use, train the model or provide pretrained weights")
    
    model.eval()
    return model
