import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
class DrugDiscoveryModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_channels=64, heads=3, lr=0.001):
        super().__init__()
        self.save_hyperparameters() 
        self.conv1 = GATv2Conv(input_dim, hidden_channels, heads=heads, concat=False)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False)       
        self.lin = Linear(hidden_channels, 1)
        self.lr = lr
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)        
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)      
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(out, batch.y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(out, batch.y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }