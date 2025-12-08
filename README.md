# 🛫 Singapore Airlines Customer Review NLP Analysis
#### **Semi-Supervised Sentiment Classification · Topic Modeling · BERTopic · MLOps**
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![BERTopic](https://img.shields.io/badge/BERTopic-Topic_Modeling-purple.svg)](https://bertopic.com//)
[![WandB](https://img.shields.io/badge/WandB-Experiment%20Tracking-orange)](https://wandb.ai/)

## ✨ Summary
This project builds a domain-specific sentiment analysis system for Singapore Airlines using 10K+ customer reviews (2018-2024).Using semi-supervised learning, the pipeline combines zero-shot pseudo-labeling with a confidence-based automated refinement step to construct a high-quality, domain-aware dataset with three sentiment classes: Positive, Negative, and Mixed. Transformer models (DistilBERT and RoBERTa) were fine-tuned using Focal Loss to address class imbalance and tracked with Weights & Biases for full reproducibility and experiment comparison. Sentiment outputs were integrated with BERTopic to identify operational themes, compute a custom Impact Index, and analyze pre- and post-pandemic sentiment trends, revealing measurable shifts in customer experience across key service areas.

> Demonstrating an end-to-end applied NLP workflow from data preparation to actionable business insights.

- ---

## 🎯 Business Impact Delivered
| Challenge                                | Technical Solution                                 | Business Outcome                                                               |
| ---------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------ |
| Star ratings unreliable (25% mismatch)   | Semi-supervised labeling + transformer fine-tuning | **Accurate sentiment signals** for CX and loyalty teams                        |
| Mixed/ambiguous reviews hard to classify | Focal Loss + manual validation                     | **Identified service inconsistencies** affecting loyalty (71% mixed class recall)                    |
| Root causes of sentiment unclear         | BERTopic + Impact Index                            | **Clear prioritization** of focus areas (refunds, cabin comfort, crew service) |
| Need for reproducible experiments        | WandB tracking integrated                          | **Experiment transparency** & model reproducibility                            |
|"What's driving sentiment changes over time?"| 	BERTopic + Temporal Analysis (2018-2024) |	Prioritized improvement areas with pre-post pandemic insights|
- ---

## 🔧 Project Highlights
Project delivers a full end-to-end customer voice analytics pipeline using modern NLP techniques. Key components include:
- **Semi-Supervised Learning Pipeline:** Generated 10k+ "Silver Standard" labels using Zero-Shot Classification (MPNet) with a strict 0.85 confidence threshold to bypass expensive manual annotation.
> Critical insight: Low zero-shot confidence ≈ mixed sentiment (praise + complaints), not neutral
- **Imbalance Correction:** Implemented Focal Loss with dynamic class weighting to stabilize training on a 70/20/10 distribution, boosting Mixed-Sentiment *F1 to 71%*.
- **MLOps Maturity:** Fine-tuned Transformer models (DistilBERT/RoBERTa) with Weights & Biases (WandB) for rigorous experiment tracking and reproducibility.
- **Strategic Quantification:** Developed the Impact Index (Topic Frequency $\times$ Net Sentiment), translating raw text into a ranked list of operational priorities.
- **Temporal Insight:** Mapped the "Pandemic Pivot," revealing a distinct shift from Experiential topics (Flight Service) to Logistical topics (Process/Refunds).


## 📊 Key Outcomes & Performance
Actionable Insights Delivered
- **High-impact:** Crew professionalism drives 68% of positive sentiment (Recommendation: Maintain investment). Impact Index: +1,023 (maintain investment)
- **Quick win:** Clearer delay communication could reduce negative sentiment by 15%.
- **Strategic:** In-flight service consistency identified as the key differentiator vs. competitors. Impact Index: +1,177 (primary loyalty driver)
- **Critical Alert:** Refunds & Support identified as the primary bleed point with an **Impact Index of -612**.

Real Customer Patterns Validated
Topic analysis revealed patterns consistent with aviation industry:

* Premium carriers receive mostly positive reviews.
* Negatives cluster around delays, seat comfort, communication gaps.
* Mixed reviews combine great staff with service inconsistencies (meals, timing).



## Top Business Insights (Impact Index)
**Impact Index = Topic Frequency (%) × Net Sentiment Score**  

| Topic | Net Sentiment | Impact Index | Temporal Trend |
|--------|--------------|--------------|----------------|
| **In-Flight Experience & Crew** | +78.9 | **+1,545** | Strong, stable positive driver |
| **Brand Reputation (SQ)** | +76.6 | **+1,053** | Slight decline post-COVID but still high |
| **Service & Recommendations** | +87.9 | **+758** | Increasingly positive after 2022 |
| **Australian Routes** | +64.3 | **+678** | Recovered well post-pandemic |
| **Changi Airport Experience** | +76.7 | +125 | Steady satisfaction driver |
| **Crew Commendations** | +95.3 | +116 | Clear service strengths |
| **Baggage Handling Issues** | -77.3 | -142 | Long-standing operational pain point |
| **In-Flight Service Incidents** | -78.1 | -373 | Noticeable during recovery period |
| **Booking/Refund & Customer Service** | -91.5 | **-510** | Peaked during COVID; top negative driver |


#### Topic Impact Bar Chart
<img width="854" height="470" alt="image" src="https://github.com/user-attachments/assets/93c6c2d1-2c0e-4f13-8852-941fc595253c" />

Positive values indicate strong satisfaction drivers; negative values highlight operational pain points.

---

## Model Performance

| Model                   | Accuracy | Macro-F1 | Training Time | Best For             |
|-------------------------|----------|----------|----------------|-----------------------|
| VADER (baseline)        | 68.2%    | 51.4%    | –              | Quick prototyping     |
| TextBlob                | 71.5%    | 55.8%    | –              | General sentiment     |
| DistilBERT + Focal Loss | 90.1%    | 82.0%    | 10 min           | Production deployment |
| RoBERTa + Focal Loss    | 91.3%    | 83.2%    | 25 min           | Maximum accuracy      |

---

## Per-Class Performance 
### DistilBERT

| Sentiment | Precision | Recall | F1-Score | Support | Insight |
|-----------|-----------|--------|----------|----------|---------|
| Positive | 97% | 91% | 94% | 1214 | Confident detection of praise |
| Negative | 95% | 89% | 92% | 482 | Strong complaint detection |
| Mixed | 60% | 79% | 68% | 304 | Recovers mixed reviews created via zero-shot low-confidence filtering |

- **Accuracy** : **89%** 
- **Macro Avg**: **85%** 
- **Weighted Avg** : **89%**

#### WandB hyperparameter sweeps
<img width="630" height="300" alt="image" src="https://github.com/user-attachments/assets/78013abd-03c2-4b54-985f-c889f8f92a3e" />

### RoBERTa
| Sentiment | Precision | Recall | F1-Score | Support | Insight |
|-----------|-----------|--------|----------|---------|---------|
| Positive | 96% | 94% | 95% | 1214 | Best-in-class stability |
| Negative | 93% | 91% | 92% | 482 | Robust across all complaint types |
| Mixed | 66% | 76% | 71% | 304 | Improved recognition of dual-polarity reviews |


- **Accuracy** :  **90%** 
- **Macro Avg** : **86%** 
- **Weighted Avg** : **91%** 

#### WandB hyperparameter sweeps
<img width="630" height="300" alt="image" src="https://github.com/user-attachments/assets/dc60bd15-c823-4372-86f7-c57b8b0608e5" />


**Note:** *The "Mixed" class was created from low-confidence zero-shot predictions where reviews expressed both positive and negative aspects.*

- ---

## 🧠 Modeling Approach

### **1. Semi-Supervised Pseudo-Labeling (Automated + Confidence-Based)**

- Rating data proved unreliable, so initial sentiment labels were generated using MPNet zero-shot classification.
- A 0.85 confidence threshold ensured high-precision assignments for Positive/Negative.
- No human annotation loop was used. Instead:
  - Low-confidence cases (<0.85) were automatically flagged as ambiguous.
  - These samples consistently contained both positive and negative expressions, supporting the Mixed Sentiment Hypothesis.

#### The Mixed Sentiment Hypothesis

During early analysis, we observed a consistent pattern:
  - High zero-shot confidence (> 0.85) → Clearly positive or negative reviews
  - Low confidence (0.50–0.85) → Reviews mixing praise and complaints
  > Example: "The crew was amazing but the 5-hour delay was unacceptable."

A small random audit confirmed that low-confidence samples were genuinely mixed rather than neutral.
This justified formalizing Mixed Sentiment as a third supervised class for fine-tuning.

 #### **Produced clean, domain-specific dataset without exhaustive manual labeling**

### **2. Fine-Tuning with Focal Loss**
- 3-class classification: positive, negative, mixed
- Custom Focal Loss with class-balanced α values (not external weighting)
- Addressed severe imbalance (70% / 20% / 10%) by focusing on hard examples
- WandB tracking for loss curves, metrics, and hyperparameter comparisons

### **3. Topic Modeling + Impact Index & Temporal Analysis**

BERTopic applied on transformer embeddings to extract themes:
 - Each topic was enriched with:
   - Topic frequency
   - Average predicted sentiment
   - A simple Impact Index to rank customer-experience drivers

     ```Impact Index = Topic Frequency (%) × Net Sentiment Score```
- Temporal and cohort segmentation analysis
  -  Pre & Post Covid Pandemic shifts

### **Why This Approach?**
- Semi-supervised: Scalable labeling for domain-specific applications
- Focal Loss: Superior for imbalanced datasets compared to weighted CE
- WandB: Reproducible experiments, easy comparison of DistilBERT vs. RoBERTa
- BERTopic + Impact Index & Temporal: Moves beyond accuracy to actionable business intelligence



- ---
## 📂 Repository Structure

```text
.
├── data/
│   ├── raw_reviews.csv              # Original dataset
│   └── labeled_dataset_0.85.csv     # "Silver Standard" dataset (with Mixed labels)
├── notebooks/
│   ├── 02_preprocessing.ipynb       # Zero-Shot Labeling & Hypothesis Testing
│   ├── 03_distilbert_training.ipynb # Fine-tuning with Focal Loss & WandB
│   ├── 04_roberta_training.ipynb    # RoBERTa Model Benchmarking
│   └── 05_bertopic_analysis.ipynb   # Temporal Analysis & Impact Index
├── models/
│   └── Model_Link.txt               # Link to HuggingFace Models
└── README.mdg
```
> Note: The DistilBERT and RoBERTa models are pushed to HuggingFace.

### 🛠️ Tech Stack
- ML Frameworks: PyTorch, Transformers, Scikit-learn
- NLP Models: DistilBERT, RoBERTa, MPNet (zero-shot & embeddings)
- MLOps: Weights & Biases (experiment tracking with parallel coordinates)
- Analysis: BERTopic, UMAP, HDBSCAN, pandas, matplotlib
- Temporal Analysis: Time-series decomposition, cohort analysis
- Visualization: Parallel coordinates, confusion matrices

- ---
## 🚀 Quick Start
Installation
```bash
pip install torch transformers datasets bertopic umap-learn wandb
pip install scikit-learn pandas matplotlib seaborn
```
Python version: 3.10
GPU used: Google Colab T4
Inference Example
```python

from transformers import pipeline

# Load fine-tuned model
classifier = pipeline(
    "sentiment-analysis",
    model="Yoshaaa7/distilbert-SGairline-sentiment-private"
)

# Predict
result = classifier("Crew was excellent but the delay was frustrating")

# Output: {'label': 'MIXED', 'score': 0.87}
```

## Future notes:
- Unlike a traditional human-in-the-loop validation pipeline, this project relied entirely on automated threshold-based rules rather than manual annotation. Incorporating HITL in future iterations could further improve Mixed-class separability.
