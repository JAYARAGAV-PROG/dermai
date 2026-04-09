# DermAI V2: Multi-Modal Virtual Biopsy & 3D Digital Twin

DermAI V2 is a fully local, lightweight web application built with Python and Streamlit. It represents a paradigm shift from traditional image-only classification by introducing a **Multi-Modal Vision Transformer (ViT)** and a hardened **Statistical Machine Learning** treatment recommender, visualized through an interactive **3D Digital Twin**.

---

## 🧬 Architecture Pipeline

The pipeline is divided into four distinct stages that execute sequentially when a user inputs patient data:

### Stage 1: Multi-Modal Ingestion
The application accepts both a standard digital photo of the skin lesion and critical clinical metadata (Age, Gender, Anatomical Location, and Family History). This multi-modal approach mirrors real-world clinical diagnosis where visual context is augmented by patient history.

### Stage 2: The Virtual Biopsy Engine (Multi-Task ViT)
Instead of utilizing standard Convolutional Neural Networks (like EfficientNet), V2 leverages a powerful **Vision Transformer (ViT-Base)**.
- **Cross-Attention Fusion:** The network employs a custom PyTorch Cross-Attention layer. It uses the patient's tabular metadata as a mathematical "Query" to filter and prioritize the visual patch embeddings ("Keys/Values") produced by the ViT. This dynamically tells the model *where* to look in the image based on the patient's background.
- **Multi-Task Output:** The unified representation is passed through two distinct multi-task heads to simultaneously predict:
  1. The overarching **Metastatic Risk Score**.
  2. The specific probability of a **BRAF V600E Genomic Mutation** (acting as an in-silico genetic profiling tool).

### Stage 3: Data-Driven Treatment Recommender
To eliminate the risk of Large Language Model (LLM) hallucinations in a medical context, the treatment engine strictly utilizes a hardened logic system grounded in real historical clinical trial data.
- If a BRAF V600E mutation is flagged, the application queries clinical statistics from the **COMBI-d / COMBI-v Phase III Clinical Trials**.
- It calculates expected clinical outcomes for Targeted Therapies (Dabrafenib + Trametinib), such as a 68% Objective Response Rate and 71% 5-Year Overall Survival (in patients achieving a complete response).

### Stage 4: 3D Digital Twin & RAG Concierge
The software surfaces results using advanced interactive visualization tools:
- **3D Digital Twin:** Utilizing an embedded WebGL/Three.js framework, the app renders a 3D morphological tumor model. A treatment timeline slider allows users to visually observe the tumor physically shrink in 3D space, mapped linearly to the 68% volume reduction statistic calculated in Stage 3.
- **RAG Concierge:** A lightweight Retrieval-Augmented Generation (RAG) pipeline operates locally using `FAISS` and `sentence-transformers`. It dynamically surfaces exact, verbatim textbook citations from **NCCN Oncology Guidelines** to corroborate the statistical treatment plan, ensuring transparency and trust.

---

## 🚀 Running the Application Locally

**Prerequisites:**
- Python 3.10+
- Installed dependencies.

**Installation:**
```bash
cd dermai-v2
pip install -r requirements.txt
```

**Launch the Streamlit Dashboard:**
```bash
streamlit run app.py
```
The application will be hosted locally at `http://localhost:8501`.

---

## 📊 Benchmarking & Performance

You can evaluate the inference speed and pipeline architecture of the multi-modal engine using the included benchmark suite:

```bash
python benchmark.py
```

### Interpretation of Results
*Note: The current V2 framework utilizes a prototype `DermAIEngine` where the Custom Cross-Attention block and Multi-Task predictive heads are **randomly initialized (untrained)** to demonstrate the pipeline architecture without requiring hours of training on a massive dataset.*

Therefore, the benchmark runs on synthetic patient data and yields metrics that reflect random guessing:
- **Accuracy / AUC-ROC:** ~0.40 - 0.60
- **Inference Speed:** ~790 ms per sample on CPU.

**To achieve clinical accuracy:** The `DermAIEngine` PyTorch module within `engine.py` must be fine-tuned on a robust multi-modal dataset (such as combining the ISIC Archive with matched clinical metadata and genomic profiles).