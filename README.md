# Singapore Airlines NLP Review Analysis
End-to-end NLP pipeline analyzing 10K Singapore Airlines customer reviews using BERTopic, DistilBERT/RoBERTa sentiment models, temporal analysis, cohort tracking, and custom Impact Index scoring.

### Core Concept

This project implements a complete NLP pipeline to decode customer sentiment from 10,000 Singapore Airlines reviews. It replaces unreliable star ratings with a transformer-based sentiment engine built through semi-supervised labeling, Focal Loss fine-tuning, and BERTopic for theme extraction. The system identifies the strongest drivers of loyalty and dissatisfaction using a custom “Impact Index,” revealing insights such as the decline in refund-related satisfaction during the pandemic and the persistent strength of the in-flight cabin experience.


<p align="center"> <img src="assets/banner_sia_nlp.png" width="700"> </p> <p align="center"> <a href="YOUR_COLAB_LINK"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a> <img src="https://img.shields.io/badge/Python-3.10+-brightgreen.svg"> <img src="https://img.shields.io/badge/Transformers-HuggingFace-orange.svg"> <img src="https://img.shields.io/badge/BERTopic-Enabled-purple.svg"> </p>

