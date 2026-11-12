# Customer Lifetime Value Prediction & Customer Segmentation

An end-to-end Machine Learning project that combines customer segmentation and predictive modeling to analyze purchasing behavior and estimate future customer value. The project uses transactional retail data to identify customer groups, predict their spending over the next 90 days, and present actionable insights through an interactive Streamlit dashboard.

## Project Overview

Understanding customer purchasing behavior is essential for effective marketing, customer retention, and resource allocation.

This project addresses two main tasks:

- **Customer Segmentation:** Identify distinct customer groups based on their purchasing behavior using RFM analysis and K-Means clustering.
- **Customer Lifetime Value Prediction:** Estimate each customer's spending over the following 90 days using a supervised regression model.

The results are presented through an interactive dashboard that allows users to explore customer profiles, compare segments, and examine predicted customer value.

## Key Features

- **Data Cleaning & Preprocessing:** Handle missing customer identifiers, remove cancelled transactions and returns, and validate transaction quantities and prices.
- **RFM Analysis:** Extract Recency, Frequency, and Monetary features to represent customer purchasing behavior.
- **Customer Segmentation:** Apply K-Means clustering to group customers into four behavioral segments.
- **Dimensionality Reduction:** Use PCA to visualize customer clusters in a two-dimensional space.
- **CLV Prediction:** Train a Random Forest Regressor to predict customer spending over the next 90 days.
- **Time-Based Validation:** Use a temporal cutoff to separate historical customer behavior from future spending targets, reducing the risk of data leakage.
- **Interactive Dashboard:** Explore customer segments, performance metrics, individual customer profiles, and predicted spending using Streamlit and Plotly.

## Dataset

**Online Retail II Dataset — UCI Machine Learning Repository**

[Dataset Source](https://archive.ics.uci.edu/dataset/502)

The dataset contains approximately one million transactional records from a UK-based online retailer, covering the period from December 2009 to December 2011.

It includes two sheets:
- `Year 2009-2010`
- `Year 2010-2011`

Key attributes include invoice numbers, product descriptions, quantities, transaction dates, prices, customer IDs, and countries.

## Methodology

### 1. Data Preprocessing

- Remove records with missing Customer IDs.
- Exclude cancelled invoices and returned transactions.
- Remove records with non-positive quantities or prices.
- Convert transaction dates and prepare the dataset for temporal analysis.

### 2. RFM Feature Engineering

Calculate three behavioral features for each customer:

| Feature | Description |
|---|---|
| Recency | Days since the customer's last purchase |
| Frequency | Number of unique purchases |
| Monetary | Total amount spent |

These features provide a behavioral representation of each customer for segmentation.

### 3. Customer Segmentation

Apply K-Means clustering with **K = 4** to group customers according to their RFM characteristics.

PCA is then used to reduce the feature space to two dimensions for cluster visualization.

The resulting segments are interpreted based on their purchasing behavior:

- **Champions / VIP:** High spending and frequent purchasing activity.
- **Loyal Customers:** Consistent purchasing behavior and repeat transactions.
- **New / Recent Customers:** Relatively recent customers with limited purchasing history.
- **At-Risk / Inactive Customers:** Customers with declining or inactive purchasing behavior.

### 4. CLV Prediction

A Random Forest Regressor is trained to estimate each customer's spending over the next 90 days.

The target variable is constructed using a time-based cutoff:

- **Observation Window:** Historical transactions available before the cutoff date.
- **Prediction Window:** Transactions occurring during the following 90 days.
- **Target Variable:** Total customer spending within the prediction window.

This setup simulates a real-world prediction scenario in which only historical information is available when generating predictions.

## Model Performance

| Component | Model / Technique | Result |
|---|---|---|
| Customer Segmentation | K-Means | 4 clusters |
| Dimensionality Reduction | PCA | 2D visualization |
| CLV Prediction | Random Forest Regressor | R² = 0.9046 |

The regression model achieved an R² score of approximately **0.9046** on the evaluation data.

## Interactive Dashboard

The Streamlit dashboard provides an interactive interface for exploring the results of the analysis.

Key capabilities include:
- Customer segment distribution and KPIs.
- RFM-based customer insights.
- PCA cluster visualization.
- Individual customer profile lookup.
- Predicted 90-day customer spending.
- Segment-level marketing recommendations.

## Project Structure

```text
Customer-Lifetime-Value-Segmentation-Engine/
│
├── app.py                            # Streamlit dashboard
├── train_and_save.py                 # Data processing and model training
├── clv_segmentation_artifacts.pkl    # Saved model artifacts (generated locally)
├── requirements.txt                  # Project dependencies
├── .gitignore                        # Excluded files
└── README.md                         # Project documentation
```

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Generate Model Artifacts

Download the dataset from the UCI repository and place it in the expected location used by the training script.

Run:

```bash
python train_and_save.py
```

This generates the processed data and trained model artifacts required by the dashboard.

### 4. Launch the Dashboard

```bash
python -m streamlit run app.py
```

Open the local URL provided by Streamlit to access the dashboard.

## Technologies Used

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Dimensionality Reduction:** PCA
- **Dashboard:** Streamlit
- **Model Persistence:** Pickle

## Future Improvements

- Experiment with alternative clustering algorithms and customer segmentation strategies.
- Evaluate additional regression models and feature engineering approaches.
- Incorporate customer retention and churn prediction.
- Improve model interpretability using feature importance and SHAP.
- Deploy the dashboard using Streamlit Community Cloud.

## License

This project is licensed under the MIT License.
