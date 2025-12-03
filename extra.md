# 🛫 Singapore Airlines Customer Review NLP Analysis
#### **Semi-Supervised Sentiment Classification · Topic Modeling · BERTopic · MLOps**

## ✨ Summary
This project builds a domain-specific sentiment analysis system for Singapore Airlines using 10K+ customer reviews (2018-2024). Using semi-supervised learning, we combined zero-shot pseudo-labeling with manual validation to create a high-quality training set, then fine-tuned transformer models (DistilBERT and RoBERTa) with Focal Loss to handle severe class imbalance. The pipeline integrates sentiment predictions with BERTopic to surface operational themes driving customer satisfaction over time, quantified through a custom Impact Index and temporal tracking, revealing distinct pre-pandemic and post-recovery customer experience shifts, quantified through net sentiment.

Complete with WandB experiment tracking and modular design, this demonstrates an end-to-end applied NLP workflow from data preparation to actionable business insights

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

Premium carriers receive mostly positive reviews
Negatives cluster around delays, seat comfort, communication gaps
Mixed reviews combine great staff with service inconsistencies (meals, timing)

## Model Performance

| Model                   | Accuracy | Macro-F1 | Training Time | Best For             |
|-------------------------|----------|----------|----------------|-----------------------|
| VADER (baseline)        | 68.2%    | 51.4%    | –              | Quick prototyping     |
| TextBlob                | 71.5%    | 55.8%    | –              | General sentiment     |
| DistilBERT + Focal Loss | 90.1%    | 82.0%    | 2.1h           | Production deployment |
| RoBERTa + Focal Loss    | 91.3%    | 83.2%    | 4.5h           | Maximum accuracy      |

---

## Per-Class Performance (DistilBERT)

| Sentiment | Precision | Recall | F1-Score | Support | Key Insight                               |
|----------|-----------|--------|---------|---------|--------------------------------------------|
| Positive | 94.0%     | 94.7%  | 94.4%   | 1,500   | Strong baseline performance                |
| Negative | 84.8%     | 85.3%  | 85.1%   | 600     | Reliable detection of complaints           |
| Mixed    | 70.2%     | 66.7%  | 68.4%   | 300     | Critical: identified service inconsistencies |

**Note:** *The "Mixed" class was created from low-confidence zero-shot predictions where reviews expressed both positive and negative aspects.*

---

## Top Business Insights (Impact Index)

| Topic                | Net Sentiment | Impact Index | Temporal Trend              |
|----------------------|---------------|--------------|-----------------------------|
| In-Flight Experience | +79           | +1,177       | Stable positive driver      |
| Crew Professionalism | +88           | +1,023       | Improved post-pandemic      |
| Refund Processing    | –91           | –612         | 15× spike during pandemic   |
| Baggage Handling     | –65           | –287         | Worsened during recovery    |

**Impact Index = Topic Frequency (%) × Net Sentiment Score**  
Positive values indicate strong satisfaction drivers; negative values highlight operational pain points.

DistilBerta:
<img width="464" height="235" alt="image" src="https://github.com/user-attachments/assets/78013abd-03c2-4b54-985f-c889f8f92a3e" />

Roberta:
<img width="662" height="299" alt="image" src="https://github.com/user-attachments/assets/dc60bd15-c823-4372-86f7-c57b8b0608e5" />


- ---

## 🧠 Modeling Approach

**1. Semi-Supervised Pseudo-Labeling**
- After determining rating as unreliable projcet used MPNet zero-shot classifier for initial labeling
- Applied 0.85 confidence threshold for high-precision labels
- Manual validation of ambiguous cases introduced "mixed sentiment" category
  - The Mixed Sentiment Hypothesis
    When analyzing zero-shot confidence scores, we observed a pattern:
    - High confidence (≥0.85): Clear positive/negative sentiment
    - Low confidence (0.5-0.85): Text containing both praise and complaints
    > Example: *"The crew was amazing but the 5-hour delay was unacceptable"*
    
    Our approach:
    
    - Treat low-confidence predictions as potential mixed sentiment
    - Manual validation using randome sampling confirmed tone were indeed mixed (not neutral)
    - Formalized as third class for fine-tuning

 #### **Produced clean, domain-specific dataset without exhaustive manual labeling**

**2. Fine-Tuning with Focal Loss**
- 3-class classification: positive, negative, mixed
- Custom Focal Loss with class-balanced α values (not external weighting)
- Addressed severe imbalance (70%/20%/10%) by focusing on hard examples
- WandB tracking for loss curves, metrics, and hyperparameter comparisons

**3. Topic Modeling + Impact Index & Temporal Analysis**
- BERTopic applied on transformer embeddings to extract themes:
  
```Impact Index = Topic Frequency (%) × Net Sentiment Score```





**Why This Approach?**
- Semi-supervised: Scalable labeling for domain-specific applications
- Focal Loss: Superior for imbalanced datasets compared to weighted CE
- WandB: Reproducible experiments, easy comparison of DistilBERT vs. RoBERTa
- BERTopic + Impact Index & Temporal: Moves beyond accuracy to actionable business intelligence



- ---
## 🧱 Technical Architecture
- ---
```
Raw Reviews (10K+)
     ↓
Data Cleaning & Aviation-Specific Normalization
     ↓
Semi-Supervised Labeling Pipeline
     ├── Zero-Shot Classification (MPNet, confidence > 0.85)
     ├── Manual Validation for Mixed/Borderline Cases
     └── 8,200 High-Quality Training Samples
     ↓
Transformer Fine-Tuning (WandB Tracking)
     ├── DistilBERT + Focal Loss (α=[0.3,1.5,2.0], γ=2)
     ├── RoBERTa Baseline Comparison
     └── Hyperparameter Optimization
     ↓
BERTopic Analysis
     ├── MPNet Embeddings + UMAP + HDBSCAN
     ├── 12 Consolidated Business Themes
     └── Temporal Trend Analysis (2018-2024)
     ↓
Impact Index Calculation & Business Reporting
```

