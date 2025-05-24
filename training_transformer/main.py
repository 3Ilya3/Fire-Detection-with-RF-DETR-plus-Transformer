import torch

from .dataloader import get_dataloaders
from .transformer import FireDetectionTransformer


if __name__ == "__main__":
    
    working_dir = "dataset"
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(working_dir, batch_size=batch_size, num_workers=4)

    transformer = FireDetectionTransformer(device=device)
    
    print("Training Transformer...")
    transformer.train_mod(train_loader, test_loader, num_epochs=10, save_mod=5, output_dir="output2")
    