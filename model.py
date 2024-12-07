"""
Inspired by: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#step-3-create-model-class
"""
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import lightning as L

import torch.nn.functional as F

class LSTMModel(L.LightningModule):
    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim,seq_dim):
        super().__init__()
        self.save_hyperparameters()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Number of steps to unroll
        self.seq_dim = seq_dim
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.preds = []
        self.labels = []

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), self.seq_dim, -1)
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), self.seq_dim, -1)
        y_pred = self.forward(x)
        val_loss = F.cross_entropy(y_pred, y)
        self.log("val_loss", val_loss, sync_dist=True)
        self.preds.append(torch.argmax(y_pred, dim=1).cpu())
        self.labels.append(y.cpu())
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), self.seq_dim, -1)
        y_pred = self.forward(x)
        test_loss = F.cross_entropy(y_pred, y)
        self.log("test_loss", test_loss, sync_dist=True)
        self.preds.append(torch.argmax(y_pred, dim=1).cpu())
        self.labels.append(y.cpu())
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)

        # Calculate F1 score
        f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
        self.log("Final Test F1 Score", f1, prog_bar=True, sync_dist=True)

        # Clear the stored predictions and labels for the next epoch
        self.preds.clear()
        self.labels.clear()
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), self.seq_dim, -1)
        pred = self.forward(x)
        return pred