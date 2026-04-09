import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from engine import load_engine
import time

def generate_synthetic_data(num_samples=100):
    # Images: [B, 3, 224, 224] representing the ViT expected input
    pixel_values = torch.randn(num_samples, 3, 224, 224)
    # Metadata: [B, 4] representing [Age, Gender, Anatomy, FamilyHistory]
    metadata = torch.rand(num_samples, 4)
    
    # Random binary labels for Risk and Mutation Ground Truths
    labels_risk = torch.randint(0, 2, (num_samples, 1)).float()
    labels_mutation = torch.randint(0, 2, (num_samples, 1)).float()
    
    return pixel_values, metadata, labels_risk, labels_mutation

def run_benchmark():
    print("=====================================================")
    print("  Initializing DermAI V2 Engine Benchmark Pipeline  ")
    print("=====================================================\n")
    print("NOTE: The ViT multi-task heads and Cross-Attention layers are")
    print("currently randomly initialized (untrained) as part of the")
    print("framework prototype. Therefore, accuracy/AUC will reflect")
    print("random guessing (~0.50). This benchmark measures pipeline")
    print("functionality and inference speed.\n")
    
    # Suppress huggingface warnings for a clean output
    import logging
    import transformers
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    model, preprocess, device = load_engine()
    model.eval()
    
    num_samples = 100
    print(f"Generating {num_samples} synthetic multi-modal patient samples...")
    pixel_values, metadata, labels_risk, labels_mutation = generate_synthetic_data(num_samples)
    pixel_values = pixel_values.to(device)
    metadata = metadata.to(device)
    
    print("Running inference sequentially to prevent OOM...")
    start_time = time.time()
    
    predictions_risk = []
    predictions_mutation = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Process one sample at a time
            pv = pixel_values[i:i+1]
            md = metadata[i:i+1]
            risk_score, mutation_prob = model(pv, md)
            predictions_risk.append(risk_score.item())
            predictions_mutation.append(mutation_prob.item())
            
    predictions_risk = np.array(predictions_risk)
    predictions_mutation = np.array(predictions_mutation)
            
    end_time = time.time()
    inference_time = (end_time - start_time) / num_samples
    
    # Extract Ground Truth vs Predictions
    y_true_mut = labels_mutation.numpy()
    y_pred_prob_mut = predictions_mutation
    y_pred_class_mut = (y_pred_prob_mut > 0.5).astype(float)
    
    try:
        acc_mut = accuracy_score(y_true_mut, y_pred_class_mut)
        auc_mut = roc_auc_score(y_true_mut, y_pred_prob_mut)
        prec_mut = precision_score(y_true_mut, y_pred_class_mut, zero_division=0)
        rec_mut = recall_score(y_true_mut, y_pred_class_mut, zero_division=0)
    except ValueError:
        auc_mut = 0.0 # If synthetic data happens to be all one class
    
    y_true_risk = labels_risk.numpy()
    y_pred_prob_risk = predictions_risk
    y_pred_class_risk = (y_pred_prob_risk > 0.5).astype(float)
    
    try:
        acc_risk = accuracy_score(y_true_risk, y_pred_class_risk)
        auc_risk = roc_auc_score(y_true_risk, y_pred_prob_risk)
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
    print("Pipeline check complete. To achieve clinical accuracy,")
    print("fine-tune the `DermAIEngine` on a labeled multimodal dataset.")

if __name__ == "__main__":
    run_benchmark()
