# California Housing Price Prediction & Comprehensive EDA

![keras california housing project](https://github.com/adelabbaszare/keras-california-housing-project/blob/main/plots/3_geo_price_map.png)

![Top Language](https://img.shields.io/github/languages/top/adelabbaszare/keras-california-housing-project)
![Languages Count](https://img.shields.io/github/languages/count/adelabbaszare/keras-california-housing-project)
[![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Frameworks](https://img.shields.io/badge/Keras-red?logo=keras&logoColor=white)](https://keras.io/)
[![Frameworks](https://img.shields.io/badge/TensorFlow-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

---

This project develops a deep learning model using **Keras/TensorFlow** to predict median house values in California, based on the built-in `california_housing` dataset. Beyond core model training, it features a complete **Exploratory Data Analysis (EDA)** and visualization component using **Pandas, Matplotlib, and Seaborn**.

---

## 🎯 Project Goals

1. **Data Analysis:** Perform comprehensive EDA on the housing dataset, including feature distribution, correlation analysis, and geographical plotting.

2. **Model Development:** Build, train, and evaluate a sequential **Neural Network (NN)** for the regression task.

3. **Visualization:** Generate static plots to monitor model training history (Loss/MAE) and compare actual vs. predicted prices.

4. **Interactive Prep:** Prepare and export clean data for powerful interactive visualization using the **SandDance** tool.

---

## 💾 Dataset Overview

The dataset includes aggregated data for districts across California, focusing on eight key features:

| Feature Name | Description |
|:---:|:---|
| **MedInc** | Median income for households within a block group (in tens of thousands of dollars). |
| **HouseAge** | Median house age within a block group (years). |
| **AveRooms** | Average number of rooms per household. |
| **AveBedrms** | Average number of bedrooms per household. |
| **Population** | Block group population. |
| **AveOccup** | Average house occupancy. |
| **Latitude** | Block group latitude (Y-axis). |
| **Longitude** | Block group longitude (X-axis). |
| **MedHouseVal** | (Target) Median house value (in $100,000s). |

---

## ⚙️ Setup and Installation

### 1. Clone or Download the Repository

First, clone the repository from GitHub to your local machine:

```bash
git clone https://github.com/adelabbaszare/Keras-california-housing-project.git
cd Keras-california-housing-project
```

If you prefer not to use Git, you can manually download the ZIP archive and extract it into a folder.

---

### 2. Python Environment Setup

**IMPORTANT:** It is strongly recommended to use a stable Python version, such as **Python 3.11 or 3.12**, as newer versions (like 3.13) may lack stable TensorFlow binaries.

Create and activate a virtual environment:

```bash
# Create environment
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
.\venv\Scripts\activate
```

---

### 3. Install Dependencies

Install all necessary libraries, including TensorFlow, Pandas, and visualization tools, using the updated `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution

Run the main training and analysis script from your active virtual environment:

```bash
python train.py
```

The script will handle the entire workflow:
- Data loading and preprocessing (standardisation)
- Train/validation/test splitting
- Neural network architecture definition (using Keras Sequential API)
- Model training with history logging
- Evaluation on the test set
- Generation of all static plots and saving them in the `plots/` directory
- Export of cleaned data for interactive visualisation and saving the final model

---

## 📊 Results and Outputs

Upon successful execution, the following outputs will be generated:

### 1. Static Plots (`plots/` folder)

A new directory named `plots/` will be created, containing several key visualisations that help you understand the data and evaluate model performance.

#### 1.1 Feature Distributions (`1_feature_distributions.png`)

Histograms showing the distribution of all eight input features plus the target variable. This visualisation helps identify skewness, outliers, and the overall range of each variable.

![Feature Distributions](https://github.com/adelabbaszare/Keras-california-housing-project/blob/main/plots/1_feature_distributions.png)

---

#### 1.2 Correlation Heatmap (`2_correlation_heatmap.png`)

A heatmap of the correlation matrix between all variables (features and target). It clearly highlights which features have the strongest linear relationships with the median house value. For instance, `MedInc` (median income) shows the highest positive correlation with price.

![Correlation Heatmap](https://github.com/adelabbaszare/Keras-california-housing-project/blob/main/plots/2_correlation_heatmap.png)

---

#### 1.3 Geographical Price Map (`3_geo_price_map.png`)

A scatter plot of the data points based on **Longitude** (X-axis) and **Latitude** (Y-axis), with points coloured by the median house value. This map reveals clear spatial patterns, such as higher prices along the coast and lower prices inland.

![Geographical Price Map](https://github.com/adelabbaszare/Keras-california-housing-project/blob/main/plots/3_geo_price_map.png)

---

#### 1.4 Training History (`4_training_history.png`)

**Crucial for Model Monitoring.** This plot shows the training and validation loss (MSE) and MAE over the 100 training epochs. It helps detect overfitting (when validation metrics start to degrade) or underfitting, and allows you to assess whether the model has converged properly.

![Training History](https://github.com/adelabbaszare/Keras-california-housing-project/blob/main/plots/4_training_history.png)

---

#### 1.5 Predictions vs. Actual (`5_predictions_vs_actual.png`)

A scatter plot comparing the model's predictions (y-axis) against the actual target values (x-axis) on the test set. Points close to the diagonal line (y = x) indicate accurate predictions. This visualisation gives an intuitive sense of the model's performance and reveals any systematic biases.

![Predictions vs Actual](https://github.com/adelabbaszare/Keras-california-housing-project/blob/main/plots/5_predictions_vs_actual.png)

---

### 2. Exported Data and Model

- **`california_housing_for_sanddance.csv`** – A cleaned CSV file containing the training data (with original unscaled values for interpretability), exported specifically for interactive visualisation tools like SandDance.

- **`california_housing_model.keras`** – The final trained Keras model, saved in the standard Keras format for future inference or fine‑tuning.

---

### 3. Performance Metrics

After training, the model is evaluated on the test set (approximately 20% of the data). Typical metrics include:

- **MSE (Mean Squared Error):** Lower is better – measures the average squared difference between predictions and actuals.
- **MAE (Mean Absolute Error):** Interpretable error in the same units as the target (hundreds of thousands of dollars). With proper tuning, the model often achieves an MAE of around **0.30–0.40** (i.e., an error of $30,000–$40,000).
- **R² Score:** Indicates the proportion of variance explained by the model. Values typically exceed **0.70**, which is satisfactory for this dataset.

---

## 🌐 Interactive Visualization with SandDance

The `train.py` script prepares and exports the data but does not run SandDance itself. You can use the exported CSV to create powerful, interactive 3D visualisations.

**Steps to Visualise:**

1. Ensure you have run `train.py` to generate the file **`california_housing_for_sanddance.csv`**.

2. Open the **SandDance Viewer** in your browser:  
   [SandDance Viewer](https://microsoft.github.io/SandDance/app/)

3. Drag and drop the `california_housing_for_sanddance.csv` file onto the page.

4. Explore the data interactively – for example, create a 3D scatter plot with **Longitude** on X, **Latitude** on Y, and colour the points by **MedHouseVal**. You can also group by other features, apply filters, and switch between different chart types.

---

## 📝 Summary

This project offers a complete, end‑to‑end pipeline for a regression task using neural networks with Keras. It covers:

- **Data exploration** through statistical summaries and rich visualisations.
- **Preprocessing** (standardisation) and robust splitting of data.
- **Model building** with a flexible sequential network.
- **Evaluation** using multiple metrics and diagnostic plots.
- **Export** of clean data for external interactive visualisation tools.

The code is modular and well‑commented, making it easy to adapt for other datasets or to experiment with different architectures and hyperparameters.

---

## 📌 Additional Notes

- All plots are automatically saved to the `plots/` directory upon script execution.
- To modify the model, edit the `train.py` file – you can change the number of layers, neurons, activation functions, or even the optimiser.
- You can integrate TensorBoard for more detailed logging by uncommenting the relevant callback in the training script.

---

**License:** MIT  
**Last Updated:** August 2026
