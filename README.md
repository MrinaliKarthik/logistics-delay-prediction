# Uncovering Delay Drivers in Logistics Using Predictive Modeling

**Goal**  
Predict shipment delays and identify operational drivers (traffic, waiting time, inventory signals) so dispatch and inventory decisions reduce late deliveries.

---

## Data  
- **Source:** Kaggle (link in `data/external_link.txt`)  
- **Scope:** ~1,000 rows, logistics features (traffic status, waiting time, shipment status, temperature, humidity, etc.)  
- **Target:** `Logistics_Delay` (0/1)  
- **Samples included:**  
  - `logistics_raw_sample.csv` → before preprocessing  
  - `logistics_clean_sample.csv` → after cleaning and feature engineering  
- **Privacy:** Only samples are included; full dataset remains external.

---

## Tools & Methods  
- **Tools:** Python (pandas, numpy, scikit-learn, matplotlib, seaborn), mlxtend  
- **Models:** Logistic Regression (tuned), Decision Tree, Random Forest  
- **Techniques:** data cleaning, feature engineering, one-hot encoding, sequential feature selection (SFS), threshold tuning, F1/recall evaluation

---

## Key Findings (Business-facing)
- **Top drivers of delay:** **Waiting Time**, **Traffic Status**, and **Inventory Level**. Delays are strongly associated with heavy/detour traffic and long waiting times. Weather (temperature/humidity) was comparatively weak in this dataset.  
- **Best model for risk control:** Tuned **Logistic Regression** with a **0.3 threshold** minimized false negatives (only 6 missed delays).  
- **Operational moves:** Prioritize shipments with high waiting time, reroute around heavy traffic, and maintain buffer inventory during peak months.

![Key Findings](assets/figures/key_findings.png)

---

## Results (Model comparison)

| Model                     | F1  | Recall | Precision | False Negatives |
|---------------------------|-----|--------|-----------|-----------------|
| Logistic Regression (default) | 0.80 | 0.72   | 0.91      | 38              |
| **Logistic Regression (tuned)** | **0.87** | **0.95** | 0.80      | **6**              |
| Decision Tree             | 0.80 | 0.74   | 0.91      | 34              |
| Random Forest             | 0.82 | 0.78   | 0.87      | 32              |

> Lowering the decision threshold (0.5 → 0.3) significantly reduced false negatives, which is critical when the business cost of missing a true delay is high.

---

## Repository Structure

logistics-delay-prediction/
├─ README.md
├─ data/
│ ├─ sample/
│ │ ├─ logistics_raw_sample.csv
│ │ └─ logistics_clean_sample.csv
│ └─ external_link.txt
├─ code/
│ └─ Final Project.py
├─ docs/
│ └─ Data mining Project_Group 2.pdf
├─ assets/
│ └─ figures/ (screenshots/plots)
├─ requirements.txt
├─ .gitignore
└─ LICENSE

## How to Reproduce
1. **Clone/download** this repo.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
3. Place the **full dataset** locally (if available) and update the path in `code/Final Project.py`.  
4. Run:  
   ```bash
   python code/Final Project.py

## Results & Next Steps
- **What to do now:**  
  - Flag shipments with high waiting times and heavy traffic for proactive intervention.  
  - Use live traffic feeds to update ETAs and dispatching.  
  - Keep safety stock in peak months.  

- **Future improvements:**  
  - Integrate live traffic & ETA APIs.  
  - Add SHAP/partial dependence plots for feature transparency.  
  - Expand dataset with weather & routing details.  

## Acknowledgements
This project was completed as part of the Data Mining & Machine Learning coursework at Northeastern University.  
Dataset: Kaggle (Logistics Delay Prediction).  


Finalize README with reproduction steps and results

