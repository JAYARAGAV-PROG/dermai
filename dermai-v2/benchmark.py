import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from engine import load_engine
import time
from train_prototype import PerfectSyntheticDataset
from torch.utils.data import DataLoader

def run_benchmark():
    print("=====================================================")
    print("  DermAI V2 Engine: Full Pipeline Benchmark  ")
    print("=====================================================\n")
    
    # Suppress huggingface warnings for a clean output
    import logging
    import transformers
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    model, preprocess, device = load_engine()
    model.eval()
    
    num_samples = 100
    print(f"Generating {num_samples} validation samples...")
    dataset = PerfectSyntheticDataset(num_samples=num_samples)
    val_loader = DataLoader(dataset, batch_size=1)
    
    print("Running inference sequentially...")
    start_time = time.time()
    
    all_risk_preds, all_risk_labels = [], []
    all_mut_preds, all_mut_labels = [], []
    all_risk_probs, all_mut_probs = [], []
    
    with torch.no_grad():
        for imgs, mds, risk_labels, mut_labels in val_loader:
            imgs, mds = imgs.to(device), mds.to(device)
            risk_prob, mut_prob = model(imgs, mds)
            
            all_risk_probs.extend(risk_prob.cpu().numpy())
            all_mut_probs.extend(mut_prob.cpu().numpy())
            
            all_risk_preds.extend((risk_prob > 0.5).cpu().numpy())
            all_risk_labels.extend(risk_labels.numpy())
            all_mut_preds.extend((mut_prob > 0.5).cpu().numpy())
            all_mut_labels.extend(mut_labels.numpy())
            
    end_time = time.time()
    inference_time = (end_time - start_time) / num_samples
    
    try:
        acc_mut = accuracy_score(all_mut_labels, all_mut_preds)
        auc_mut = roc_auc_score(all_mut_labels, all_mut_probs)
        prec_mut = precision_score(all_mut_labels, all_mut_preds, zero_division=0)
        rec_mut = recall_score(all_mut_labels, all_mut_preds, zero_division=0)
    except ValueError:
        auc_mut = 0.0
    
    try:
        acc_risk = accuracy_score(all_risk_labels, all_risk_preds)
        auc_risk = roc_auc_score(all_risk_labels, all_risk_probs)
    except ValueError:
        auc_risk = 0.0

    print("\n=====================================================")
    print("                 BENCHMARK RESULTS                   ")
    print("=====================================================")
    print(f"Device:           {device}")
    print(f"Inference Speed:  {inference_time*1000:.2f} ms / sample")
    print("-----------------------------------------------------")
    print("[Task 1: BRAF V600E Mutation Prediction]")
    print(f"  Accuracy:  {acc_mut:.4f}")
    print(f"  AUC-ROC:   {auc_mut:.4f}")
    print(f"  Precision: {prec_mut:.4f}")
    print(f"  Recall:    {rec_mut:.4f}")
    print("-----------------------------------------------------")
    print("[Task 2: Metastatic Risk Prediction]")
    print(f"  Accuracy:  {acc_risk:.4f}")
    print(f"  AUC-ROC:   {auc_risk:.4f}")
    print("=====================================================\n")

if __name__ == "__main__":
    run_benchmark()
