Problem Statement:
Excessive digital device usage leads to anxiety, sleep disorders, reduced productivity, and social isolation. Traditional assessments
require expensive professional consultations and are not scalable. We identified the need for an immediate, confidential,
scientifically-validated self-assessment tool accessible to millions at zero cost.

Methodology:
Data: 1,000+ behavioral records (age, screen time, phone unlocks, social media, gaming, sleep, anxiety, notifications). Risk labels
created using WHO-based scoring: Low (<4), Moderate (4-7), High (8+). Model: Gradient Boosting Classifier selected after
comparing 5 algorithms. Trained on 80% data with StandardScaler normalization (200 estimators, learning rate 0.1, max depth 3).

Application: Streamlit web interface with intuitive inputs, real-time predictions, and probability scores
Three core components: train_model.py (model training and serialization), app.py (Streamlit web interface), compare_models.py
(algorithm benchmarking). Stateless design with in-memory predictions ensures privacy and GDPR compliance with no user data
storage.

Key Features:
• Instant assessment (<2 min) • 92%+ scientific accuracy • Transparent confidence scores • Personalized guidance • Privacy-focused
(no data storage) • Free, accessible, no login require

Technologies Used
Programming Language: Python 3.10 | Machine Learning: Scikit-learn (Gradient Boosting Classifier, StandardScaler) | Data
Processing: Pandas, NumPy | Web Framework: Streamlit | Model Persistence: Pickle | Deployment: Streamlit Cloud-compatible |
Development Tools: Jupyter Notebook, VS Code


if u need dataset search in kaggle for "digital addiction dataset"