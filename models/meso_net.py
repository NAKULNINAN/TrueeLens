import torch
import torch.nn as nn
import torch.nn.functional as F

class Meso4(nn.Module):
    """
    MesoNet-4 model for deepfake detection
    """
    def __init__(self, num_classes=1):
        super(Meso4, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(4, 4)
        
        # Fully connected layers
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 7 * 7, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.maxpool4(self.relu4(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class MesoInception4(nn.Module):
    """
    MesoInception-4 model for deepfake detection
    Uses Inception modules for better feature extraction
    """
    def __init__(self, num_classes=1):
        super(MesoInception4, self).__init__()
        
        # Inception modules
        self.inception1 = InceptionModule(3, 1, 4, 4, 2)
        self.inception2 = InceptionModule(8, 2, 4, 4, 2)
        
        # Convolutional layers
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(4, 4)
        
        # Fully connected layers
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 7 * 7, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # Inception modules
        x = self.inception1(x)
        x = self.inception2(x)
        
        # Convolutional layers
        x = self.maxpool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.maxpool4(self.relu4(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class InceptionModule(nn.Module):
    """
    Inception module for MesoInception
    """
    def __init__(self, in_channels, a, b, c, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, a, 1),
            nn.BatchNorm2d(a),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c, 5, padding=2),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        
        # Max pool + 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
        # Final pooling and batch norm
        self.final_pool = nn.MaxPool2d(2, 2)
        self.final_bn = nn.BatchNorm2d(a + b + c + pool_proj)
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate all branches
        outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
        outputs = self.final_pool(outputs)
        outputs = self.final_bn(outputs)
        
        return outputs

def load_meso_model(model_type='meso4', version='latest', num_classes=1, load_pretrained=False):
    """
    Load MesoNet model for deepfake detection
    
    Args:
        model_type: 'meso4' or 'mesoinception4'
        version: Model version (not used in this implementation)
        num_classes: Number of output classes
        load_pretrained: Whether to attempt loading pretrained weights (disabled by default)
    """
    if model_type.lower() == 'meso4':
        model = Meso4(num_classes=num_classes)
        model_name = 'Meso4'
    elif model_type.lower() == 'mesoinception4':
        model = MesoInception4(num_classes=num_classes)
        model_name = 'MesoInception4'
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'meso4' or 'mesoinception4'")
    
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
