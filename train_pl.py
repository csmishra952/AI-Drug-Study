import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from utils import scaffold_split, smiles_to_graph
from model_pl import DrugDiscoveryModel
import wandb
def main():
    wandb.init(project="ai-drug-discovery", name="GATv2-Scaffold-Split")
    print("🚀 Starting MLOps Training Pipeline...")
    dataset = MoleculeNet(root='.', name='ESOL')  
    print("Featurizing molecules...")
    data_list = []
    for data in dataset:
        rich_data = smiles_to_graph(data.smiles)
        if rich_data:
            rich_data.y = data.y
            rich_data.smiles = data.smiles
            data_list.append(rich_data)
    print("Performing Scaffold Split...")
    train_idx, val_idx, test_idx = scaffold_split(data_list)
    train_dataset = [data_list[i] for i in train_idx]
    val_dataset = [data_list[i] for i in val_idx]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )  
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )
    num_features = data_list[0].x.shape[1]
    model = DrugDiscoveryModel(input_dim=num_features, hidden_channels=64, lr=0.001)
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        logger=WandbLogger(),
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    wandb.finish()
if __name__ == '__main__':
    main()