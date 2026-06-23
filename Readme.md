# SkillPathBot

ML-powered Telegram bot that predicts IT career direction based on skills.

## Project Overview

SkillPathBot analyzes a user's technical and soft skills
and recommends the most suitable IT career direction
from four categories: Engineering, Analytics,
Operations & Support, and Architecture & Design.

The model was trained on LinkedIn job postings dataset
from Kaggle (January 2024, 6 days of data,
12,217 vacancies across the United States,
United Kingdom, Canada and Australia).

## Dataset

Source: Kaggle — LinkedIn Job Postings
Period: January 13-17, 2024
Original size: 12,217 job postings
After cleaning: 7,897 job postings
Features used: 47 skill features + country

## ML Pipeline

Data preprocessing:
- Merged 3 datasets by job_link key
- Removed parser artifacts and irrelevant columns
- Filtered IT/Data vacancies by job_title keywords
- Normalized 6,000+ unique job titles
- Grouped titles into 10 detailed categories
- Consolidated into 4 target groups

Feature engineering:
- Top-47 skills extracted from job_skills column
- One-Hot Encoding for each skill
- Label Encoding for country and job_type
- Removed duplicate skills (communication/communication_skills,
  problem_solving/problemsolving, data_analysis/data_analytics)

Model:
- Algorithm: Logistic Regression (Pipeline with StandardScaler)
- Train/Test split: 80/20 with stratify
- Cross-validation: 3 folds

Results:
- Accuracy: 79.0%
- F1 macro: 0.703
- CV mean: 0.781, std: 0.001

Target classes:
- Engineering: Data Engineer, ML Engineer, MLOps/DevOps, Software Engineer
- Analytics: Data Analyst, Data Scientist
- Operations & Support: Database Administrator, Data Management, Data Center
- Architecture & Design: Data Architect

## Bot Features

- 7 skill groups with inline keyboard selection
- Checkboxes toggle on/off
- Skip option for each group
- Top-4 results with probability bars
- Role breakdown within each direction
- Restart option

## Tech Stack

Python 3.x
scikit-learn
python-telegram-bot 20.7
joblib
numpy
Railway (deployment)
GitHub (version control)

## Project Structure

папка проекта/
├── bot.py
├── SkillPathBot.ipynb
├── requirements.txt
├── Procfile
├── .gitignore
├── lr_model.pkl
├── label_encoder.pkl
├── le_country.pkl
└── data/
    

## How to Run Locally

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Create .env file:
   BOT_TOKEN=your_token_here
4. Run:
   py bot.py

## Author

Gulnara
Economics background, retraining in AI/ML
Tomorrow School — 01EDU program, Astana, Kazakhstan
2026