from model import LSTMModel
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from argparse import ArgumentParser

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import lightning as L

def main():
    transform = transforms.ToTensor()
    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
    
    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(train_set, num_workers=15)
    valid_loader = DataLoader(valid_set, num_workers=15)
    test_loader = DataLoader(test_set, num_workers=15)
    
    # automatically save the best model
    checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints", 
            filename="lstm-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1, 
            mode="min", 
        )   
    
    # load model from checkpoint and evaluate
    if(args.eval != ""):
        trainer = L.Trainer(devices=1, num_nodes=1)
        LSTM = LSTMModel.load_from_checkpoint(args.eval)
        trainer.test(model=LSTM, dataloaders=test_loader)
    
    # train model
    else:
        LSTM = LSTMModel(input_dim=28, hidden_dim=100, layer_dim=1, output_dim=10, seq_dim=28)
        
        trainer = L.Trainer(
            max_epochs=20, 
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
            default_root_dir="checkpoints")
        
        trainer.fit(model=LSTM, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        trainer.test(model=LSTM, dataloaders=test_loader)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval", type=str, default="", help="Path to model to evaluate")
    args = parser.parse_args()
    main()