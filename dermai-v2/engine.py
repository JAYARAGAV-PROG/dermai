import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from PIL import Image
from torchvision import transforms

class MetadataCrossAttention(nn.Module):
    """
    Simplified Cross-Attention that uses metadata as the 'Query' 
    to filter image features ('Key/Value').
    """
    def __init__(self, embed_dim, metadata_dim):
        super().__init__()
        self.metadata_proj = nn.Linear(metadata_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, metadata):
        # x: [B, N, E] (image patches)
        # metadata: [B, M] (clinical features)
        
        m_proj = self.metadata_proj(metadata).unsqueeze(1) # [B, 1, E]
        
        q = self.query_proj(m_proj)
        k = self.key_proj(x)
        v = self.value_proj(x)
        
        # Metadata queries the image patches
        attn_out, _ = self.attn(q, k, v)
        # Add & Norm (simplified)
        out = self.norm(m_proj + attn_out)
        return out.squeeze(1) # [B, E]

class DermAIEngine(nn.Module):
    def __init__(self, metadata_dim=4):
        super().__init__()
        # Load pre-trained ViT
        self.config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # Cross-Attention layer
        self.cross_attn = MetadataCrossAttention(self.config.hidden_size, metadata_dim)
        
        # Multi-task heads
        self.risk_head = nn.Linear(self.config.hidden_size, 1) # Metastatic Risk Score
        self.mutation_head = nn.Linear(self.config.hidden_size, 1) # BRAF V600E Probability

    def forward(self, pixel_values, metadata):
        # 1. Image Embeddings from ViT backbone
        vit_outputs = self.vit(pixel_values=pixel_values)
        patch_embeddings = vit_outputs.last_hidden_state # [B, 197, 768]
        
        # 2. Metadata Cross-Attention (metadata filters the image patches)
        fused_features = self.cross_attn(patch_embeddings, metadata) # [B, 768]
        
        # 3. Multi-task output
        risk_score = torch.sigmoid(self.risk_head(fused_features))
        mutation_prob = torch.sigmoid(self.mutation_head(fused_features))
        
        return risk_score, mutation_prob

def load_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DermAIEngine(metadata_dim=4).to(device)
    
    # NEW: Load trained weights if they exist
    import os
    base_dir = os.path.dirname(__file__)
    weights_path = os.path.join(base_dir, "engine_prototype_99.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[OK] High-Accuracy Engine loaded: {weights_path}")
    
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    return model, preprocess, device

def predict(model, preprocess, device, image, metadata_list):
    # metadata_list: [age_norm, gender_encoded, anatomy_encoded, family_hist_encoded]
    
    pixel_values = preprocess(image).unsqueeze(0).to(device)
    metadata_tensor = torch.tensor([metadata_list], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        risk_score, mutation_prob = model(pixel_values, metadata_tensor)
        
    return risk_score.item(), mutation_prob.item()
