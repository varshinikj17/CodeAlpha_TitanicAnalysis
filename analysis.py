import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

sns.set_theme(style="whitegrid")

# ---------- 1. LOAD ----------
df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)
print("QUESTIONS")
print("1. What share of passengers survived?")
print("2. Did gender affect survival?")
print("3. Did passenger class affect survival?")
print("4. How are age and fare distributed, and are there outliers?")

# ---------- 2. EXPLORE ----------
print("\nSHAPE (rows, columns):", df.shape)
print("\nDATA TYPES:\n", df.dtypes)
print("\nMISSING VALUES:\n", df.isnull().sum())
print("\nDUPLICATES:", df.duplicated().sum())
print("\nSUMMARY STATISTICS:\n", df.describe())

# ---------- 3. CLEAN ----------
df = df.drop(columns=["deck"])
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
print("\nAFTER CLEANING, missing values:", df.isnull().sum().sum())
print("Rows remaining:", len(df))

# ---------- 4. ANALYSIS ----------
print("\nOverall survival rate:", round(df.survived.mean() * 100, 1), "%")
print("\nSurvival rate by sex:\n", df.groupby("sex")["survived"].mean())
print("\nSurvival rate by class:\n", df.groupby("pclass")["survived"].mean())
print("\nSurvival by class and sex:\n",
      df.groupby(["pclass", "sex"])["survived"].mean().unstack())

q1, q3 = df.fare.quantile([0.25, 0.75])
iqr = q3 - q1
n_out = ((df.fare < q1 - 1.5 * iqr) | (df.fare > q3 + 1.5 * iqr)).sum()
print("\nFare outliers (IQR method):", n_out)

chi2, p, _, _ = chi2_contingency(pd.crosstab(df.sex, df.survived))
print("\nChi-square test (sex vs survival) p-value:", p)
if p < 0.05:
    print("Result: survival differed significantly by gender.")

# ---------- 5. VISUALIZATIONS ----------
sns.countplot(x="survived", data=df)
plt.title("Survival Count (0 = No, 1 = Yes)")
plt.savefig("survival_count.png", dpi=150, bbox_inches="tight")
plt.close()

sns.barplot(x="pclass", y="survived", hue="sex", data=df)
plt.title("Survival Rate by Class and Gender")
plt.savefig("survival_class_sex.png", dpi=150, bbox_inches="tight")
plt.close()

sns.histplot(df.age, bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("age_dist.png", dpi=150, bbox_inches="tight")
plt.close()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nDONE. Saved 4 charts and titanic.csv in this folder.")