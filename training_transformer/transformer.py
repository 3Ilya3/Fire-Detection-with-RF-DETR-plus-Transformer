import torch
import torch.nn as nn
import math
import os
from sklearn.metrics import f1_score, precision_score, recall_score

from .extract_features import FeatureExtractor

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # Register as buffer for saving

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return x

class FireDetectionTransformer(nn.Module):
    def __init__(self, d_model=1280, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, lr=1e-4, model_filename=None, device=None):
        super(FireDetectionTransformer, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Feature extractor initialization
        self.feature_extractor = FeatureExtractor(device=self.device)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder layer
        # Includes: Multi-Head Self-Attention, FFN (256→1024→256 with GELU),
        # residual connections, Layer Normalization, Dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,  
            activation='gelu',  
            batch_first=True
        )
        # Encoder with 6 layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer for classification
        self.classifier = nn.Linear(d_model, 1)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        self.start_epoch = 0

        # Load checkpoint if model_filename is specified
        if model_filename is not None:
            print("Loading model...")
            if os.path.exists(model_filename):
                self.start_epoch = self.load_checkpoint(model_filename)
            else:
                raise FileNotFoundError(f"Model file '{model_filename}' not found")

        # Move model to device
        self.to(self.device)

    def forward(self, seq):
        # seq: [batch_size, 30, 1280]
        batch_size = seq.size(0)

        # Expand CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, 1280]

        # Concatenate CLS token with sequence
        full_seq = torch.cat([cls_tokens, seq], dim=1)  # [batch_size, 31, 1280]

        # Add positional encoding
        full_seq = self.pos_encoding(full_seq.permute(1, 0, 2)).permute(1, 0, 2)  # [batch_size, 31, 1280]

        # Process with encoder
        output = self.transformer_encoder(full_seq)  # [batch_size, 31, 1280]

        # Extract CLS token
        cls_output = output[:, 0, :]  # [batch_size, 1280]

        # Prediction probability
        pred = torch.sigmoid(self.classifier(cls_output))  # [batch_size, 1]

        return pred

    # Function to save checkpoint
    def save_checkpoint(self, epoch, loss, filepath):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved in '{filepath}' (epoch {epoch + 1}, loss: {loss:.4f})")
    
    # Function to load checkpoint
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Explicitly move optimizer state to self.device
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
    
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from '{filepath}' (epoch {start_epoch}, loss: {loss:.4f})")
        return start_epoch

    # Training loop
    def train_mod(self, dataloader, test_loader, num_epochs, save_mod=10, output_dir=None):

        if output_dir is None:
            output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        start_epoch = self.start_epoch
        num_epochs = start_epoch + num_epochs

        for epoch in range(start_epoch, num_epochs):
            self.train()
            total_loss = 0
            for seq, label in dataloader:

                
                seq = seq.to(self.device)
                label = label.to(self.device).float()

                # Extract features
                seq = self.feature_extractor.extract_features_batch(seq)  # [batch_size, 30, 1280]

                # Forward pass
                pred = self(seq)  # [batch_size, 1]
                loss = self.criterion(pred.squeeze(), label)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Save checkpoint at epochs divisible by save_mod
            if (epoch + 1) % save_mod == 0:
                self.eval()
                correct = 0
                total = 0
                test_loss = 0
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for idx, (seq, labels) in enumerate(test_loader):
                        seq = seq.to(self.device)
                        labels = labels.to(self.device).float()
                        pred = self.evaluate_mod(seq).squeeze()
                        test_loss += self.criterion(pred, labels).item()
                        predicted = (pred > 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                accuracy = 100 * correct / total
                avg_test_loss = test_loss / len(test_loader)
                f1 = f1_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds)
                recall = recall_score(all_labels, all_preds)
                print(f"Test accuracy: {accuracy:.2f}%, Test loss: {avg_test_loss:.4f}, "
                      f"F1 score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

                model_filename = os.path.join(output_dir, f"model_{epoch + 1}.pth")
                self.save_checkpoint(epoch, avg_loss, model_filename)

        model_filename = os.path.join(output_dir, f"model_{epoch + 1}.pth")
        self.save_checkpoint(epoch, avg_loss, model_filename)

    # Inference
    def evaluate_mod(self, input_seq):
        self.eval()
        with torch.no_grad():
            input_seq = input_seq.to(self.device)  
            input_seq = self.feature_extractor.extract_features_batch(input_seq)  # [batch_size, 30, 1280]
            pred = self(input_seq)
        return pred