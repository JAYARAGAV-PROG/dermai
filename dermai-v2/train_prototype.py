import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import time
from engine import load_engine

class PerfectSyntheticDataset(Dataset):
    """
    A 'Perfect' dataset where the labels are strictly correlated 
    with the metadata and image patterns for architectural validation.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # Metadata: [Age, Gender, Anatomy, FamilyHistory]
        self.metadata = torch.rand(num_samples, 4)
        # Image: [3, 224, 224] - We'll create patterns (e.g., bright spots) correlated with labels
        self.images = torch.randn(num_samples, 3, 224, 224) * 0.1
        
        self.labels_risk = []
        self.labels_mutation = []
        
        for i in range(num_samples):
            md = self.metadata[i]
            # Rule 1: High age + family history = High Risk
            if md[0] > 0.5 and md[3] > 0.5:
                risk = 1.0
                # Add a visual signal (red channel boost) to the image
                self.images[i, 0, :, :] += 0.5 
            else:
                risk = 0.0
                
            # Rule 2: Gender 'Male' (1.0) + Anatomy 'Head/Neck' (0.0) = BRAF Mutation
            if md[1] > 0.5 and md[2] < 0.25:
                mut = 1.0
                # Add a visual signal (green channel boost)
                self.images[i, 1, :, :] += 0.5
            else:
                mut = 0.0
                
            self.labels_risk.append(risk)
            self.labels_mutation.append(mut)
            
        self.labels_risk = torch.tensor(self.labels_risk).unsqueeze(1)
        self.labels_mutation = torch.tensor(self.labels_mutation).unsqueeze(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.metadata[idx], self.labels_risk[idx], self.labels_mutation[idx]

def train_engine():
    print("=====================================================")
    print("  DermAI V2 Engine: Training Architectural Prototype ")
    print("=====================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_engine() # Already on device from load_engine
    model.train()
    
    dataset = PerfectSyntheticDataset(num_samples=200)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Validation set
    val_dataset = PerfectSyntheticDataset(num_samples=50)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    criterion = nn.BCELoss()
    # Fine-tune classification heads + cross-attention; keep ViT frozen for speed
    optimizer = optim.Adam([
        {'params': model.cross_attn.parameters()},
        {'params': model.risk_head.parameters()},
        {'params': model.mutation_head.parameters()}
    ], lr=1e-3)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, mds, risk_labels, mut_labels in train_loader:
            imgs, mds = imgs.to(device), mds.to(device)
            risk_labels, mut_labels = risk_labels.to(device), mut_labels.to(device)
            
            optimizer.zero_grad()
            risk_pred, mut_pred = model(imgs, mds)
            
            loss_risk = criterion(risk_pred, risk_labels)
            loss_mut = criterion(mut_pred, mut_labels)
            loss = loss_risk + loss_mut
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        all_risk_preds, all_risk_labels = [], []
        all_mut_preds, all_mut_labels = [], []
        
        with torch.no_grad():
            for imgs, mds, risk_labels, mut_labels in val_loader:
                imgs, mds = imgs.to(device), mds.to(device)
                risk_pred, mut_pred = model(imgs, mds)
                
                all_risk_preds.extend((risk_pred > 0.5).cpu().numpy())
                all_risk_labels.extend(risk_labels.numpy())
                all_mut_preds.extend((mut_pred > 0.5).cpu().numpy())
                all_mut_labels.extend(mut_labels.numpy())
        
        acc_risk = accuracy_score(all_risk_labels, all_risk_preds)
        acc_mut = accuracy_score(all_mut_labels, all_mut_preds)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} | Risk Acc: {acc_risk:.4f} | Mutation Acc: {acc_mut:.4f}")
        
        if acc_risk >= 0.99 and acc_mut >= 0.99:
            print("\n[SUCCESS] Target accuracy of 99%+ reached!")
            break

    # Save the trained prototype weights
    torch.save(model.state_dict(), "engine_prototype_99.pth")
    print("\nModel weights saved to 'engine_prototype_99.pth'")

if __name__ == "__main__":
    train_engine()
