# CodeAlpha Data Analytics Internship: Titanic Analysis

## Tasks Completed
- Task 2: Exploratory Data Analysis (EDA)
- Task 3: Data Visualization

## Tools
Python, pandas, seaborn, matplotlib, scipy

## Dataset
Titanic passenger data (891 rows, 15 columns), loaded from seaborn.

## Questions Explored
1. What share of passengers survived?
2. Did gender affect survival?
3. Did passenger class affect survival?
4. How are age and fare distributed, and are there outliers?

## Data Cleaning
- Dropped `deck` (688 of 891 values missing)
- Filled missing `age` with the median (177 values)
- Filled missing `embarked` and `embark_town` with the most common value (2 each)

## Key Findings
- Overall survival rate was 38.4%.
- Women survived at 74.2% versus 18.9% for men. A chi-square test gave p < 0.05, so the difference is statistically significant.
- Survival fell with class: 63.0% in first, 47.3% in second, 24.2% in third.
- First-class women were the most likely to survive (96.8%) and third-class men the least (13.5%).
- Fare is highly skewed, with outliers found using the IQR method.

## Visualizations
- `survival_count.png`: more passengers died than survived.
- `survival_class_sex.png`: women survived far more in every class, and survival fell from first to third.
- `age_dist.png`: most passengers were aged 20 to 40, with fewer children and elderly.
- `heatmap.png`: fare correlates positively with survival, and pclass correlates negatively.

## How to Run
pip install pandas seaborn matplotlib scipy
python analysis.py