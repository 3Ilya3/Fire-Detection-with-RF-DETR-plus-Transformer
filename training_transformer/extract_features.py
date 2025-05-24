import torch
import torchvision.models as models


class FeatureExtractor:
    def __init__(self, model_name='efficientnet_b0', weights='DEFAULT', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = models.__dict__[model_name](weights=weights).to(self.device)
        self.model.classifier = torch.nn.Identity()  # Without classifier
        self.model.eval()

    def extract_features_batch(self, sequences):
        sequences = sequences.to(self.device)
        batch_size, seq_len, channels, height, width = sequences.shape
        images = sequences.view(-1, channels, height, width)
        with torch.no_grad():
            features = self.model(images)  # [batch_size * seq_len, feature_dim]
        features = features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, feature_dim]
        return features