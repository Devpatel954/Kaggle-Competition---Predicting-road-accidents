# 🚗 Predicting Accident Risk (Playground Series S5E10)

This project focuses on **predicting accident risk levels** using advanced machine learning models.  
The dataset is from the **Kaggle Playground Series (Season 5, Episode 10)** competition.  
Our goal is to build and blend multiple models (CatBoost, LightGBM, and XGBoost) to achieve the best RMSE score.

---

## 📘 Project Overview

Accident prediction plays a crucial role in road safety analytics.  
By leveraging structured data on road, weather, and traffic conditions, this project predicts the **probability of an accident risk score** between 0 and 1.

The notebook covers:
- Data preprocessing and feature engineering
- Model training (CatBoost, LightGBM, XGBoost)
- K-Fold cross-validation
- Model blending and stacking
- Final submission generation for Kaggle

---

## 🧠 Key Features

### 🔹 Feature Engineering
- Combined categorical features like:
  - `road_type + lighting` → `road_lighting`
  - `road_type + weather + lighting` → `danger_combo`
- Created binary indicators:
  - `bad_weather_flag`
  - `holiday_school_flag`
  - `rush_hour_flag`
- Encoded cyclic time features (`time_sin`, `time_cos`) to capture periodic patterns.

### 🔹 Models Used
| Model | Library | Purpose |
|--------|----------|----------|
| CatBoost | `catboost` | Handles categorical features efficiently |
| LightGBM | `lightgbm` | Fast gradient boosting framework |
| XGBoost | `xgboost` | Robust and widely used boosting model |
| Ridge Regression | `sklearn.linear_model` | Used for stacking meta-model |

### 🔹 Blending Techniques
- **2-way blend:** CatBoost + LightGBM  
- **3-way blend:** CatBoost + LightGBM + XGBoost  
- **Stacking:** Ridge regression meta-model trained on OOF predictions  
- **Rank blending:** Robust rank-based averaging for better generalization



