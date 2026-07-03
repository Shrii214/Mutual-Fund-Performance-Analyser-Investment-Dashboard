import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("mutual_funds_india.csv")
print(df.shape) #checks rows and columns 
print(df.head()) #prints first 5 rows of the dataset
print(df.dtypes) #prints the data types of each column
print(df.columns.tolist()) #prints the column names of the dataset
print(df.info()) #prints the summary of the dataset
print(df.isnull().sum()) #checks for missing values in the dataset

#dropping rows with missing values in critical columns for analysis
df.dropna(subset=["returns_3yr", "returns_5yr", "expense_ratio"], inplace=True)

numeric_cols = [
    "min_sip", "min_lumpsum", "expense_ratio", "fund_size_cr",
    "fund_age_yr", "sortino", "alpha", "sd", "beta", "sharpe",
    "risk_level", "rating", "returns_1yr", "returns_3yr", "returns_5yr"
] #list of numeric columns

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median()) #fill missing values with median for numeric columns


#clean text columns by stripping whitespace and converting to title case
text_cols = df.select_dtypes(include="object").columns

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

#risk adjusted return calculation
df["risk_adjusted_return"] = df["returns_3yr"] / df["risk_level"]

#High return with low expense ratio is better for investors.
df["expense_efficiency"] = df["returns_3yr"] - df["expense_ratio"]

#Shows average performance across different time periods.
df["return_consistency"] = (df["returns_1yr"] + df["returns_3yr"] + df["returns_5yr"]) / 3

#Useful for beginner investors who want low monthly investment options.
df["sip_type"] = np.where(df["min_sip"] <= 500, "Affordable SIP", "High SIP")


#normalize the high good columns using MinMaxScaler(NORMALIZATION)
scaler = MinMaxScaler()

high_good_cols = ["returns_3yr", "returns_5yr", "sharpe", "alpha"] #high good columns

for col in high_good_cols:
    df[col + "_score"] = scaler.fit_transform(df[[col]])

#Lower expense ratio = lower cost to investor
# Lower beta = less market volatility/risk
#expense_ratio and beta: 
low_good_cols = ["expense_ratio", "beta"]

for col in low_good_cols:
    df[col + "_score"] = 1 - scaler.fit_transform(df[[col]])

#return score calculation
df["return_score"] = (df["returns_3yr_score"] * 0.50 + df["returns_5yr_score"] * 0.50)

#alpha/beta score calculation
df["alpha_beta_score"] = (df["alpha_score"] * 0.60 + df["beta_score"] * 0.40)
# Alpha shows fund manager performance.
# Beta shows market risk.
# We give alpha slightly more weight because this project includes fund manager performance.

df["final_score"] = (
    df["return_score"] * 0.40 +
    df["expense_ratio_score"] * 0.25 +
    df["sharpe_score"] * 0.20 +
    df["alpha_beta_score"] * 0.15
)  
# Final Score =
# Return Score × 40%
# Expense Score × 25%
# Sharpe Score × 20%
# Alpha/Beta Score × 15%

#Highest final_score gets rank 1.
# Second highest gets rank 2.
df["rank"] = df["final_score"].rank(ascending=False, method="dense") 

top_30 = df.sort_values("final_score", ascending=False).head(30)
#top 30 funds based on final score


#EXPORTING FILES
df.to_csv("cleaned_mutual_funds.csv", index=False)
top_30.to_csv("top_30_mutual_funds.csv", index=False)

print(top_30[["scheme_name", "returns_3yr", "returns_5yr",
              "expense_ratio", "sharpe", "alpha", "beta", "final_score", "rank"]])

# ----------------------------------------------------------------------------------------------
import os
os.makedirs("images", exist_ok=True)

#Chart 1: Average 3yr return by category 
avg_cat = df.groupby("category")["returns_3yr"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(avg_cat.index, avg_cat.values,
              color=["#274368","#1E4E5E","#1E265E","#1E5E5A","#B6D4DE"],
              edgecolor="white")
for bar, val in zip(bars, avg_cat.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Average 3-Year Return by Fund Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category"); ax.set_ylabel("Avg 3yr Return (%)")
plt.tight_layout()
plt.savefig("images/01_return_by_category.png", dpi=150, bbox_inches="tight")
plt.show()



#Chart 2: Expense ratio vs returns scatter 
plt.figure(figsize=(10,6))
colors = {"Equity":"#3B82F6","Debt":"#10B981","Hybrid":"#F59E0B",
          "Other":"#6B7280","Solution Oriented":"#8B5CF6"}
for cat, grp in df.groupby("category"):
    plt.scatter(grp["expense_ratio"], grp["returns_3yr"],
                label=cat, alpha=0.5, s=30, color=colors.get(cat,"gray"))
plt.xlabel("Expense Ratio (%)"); plt.ylabel("3-Year Return (%)")
plt.title("Does Higher Cost = Better Return?", fontsize=14, fontweight="bold")
plt.legend(title="Category")
plt.tight_layout()
plt.savefig("images/02_expense_vs_return.png", dpi=150, bbox_inches="tight")
plt.show()


# Chart 3: Correlation heatmap 
corr_cols = ["returns_1yr","returns_3yr","returns_5yr","expense_ratio",
             "sharpe","alpha","beta","final_score","return_consistency"]
plt.figure(figsize=(10,8))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Between All Metrics", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("images/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

#  Chart 4: Top 10 funds bar chart by final_score
top10 = df.sort_values("final_score", ascending=False).head(10)
plt.figure(figsize=(11,6))
bars = plt.barh(top10["scheme_name"].str[:40],
                top10["final_score"],
                color="#334665", edgecolor="white")
for bar, val in zip(bars, top10["final_score"]):
    plt.text(val+0.002, bar.get_y()+bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=10)
plt.xlabel("Final Score"); plt.title("Top 10 Funds by Composite Score",
                                      fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("images/04_top10_funds.png", dpi=150, bbox_inches="tight")
plt.show()

# Chart 5: Return consistency vs final_score 
plt.figure(figsize=(9,6))
scatter = plt.scatter(df["return_consistency"], df["final_score"],
                      c=df["risk_level"], cmap="RdYlGn_r",
                      alpha=0.6, s=40)
plt.colorbar(scatter, label="Risk Level (1=Low, 6=Very High)")
plt.xlabel("Return Consistency (Avg of 1yr+3yr+5yr)")
plt.ylabel("Final Score")
plt.title("Consistent Returns vs Final Score (coloured by Risk)",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("images/05_consistency_vs_score.png", dpi=150, bbox_inches="tight")
plt.show()

print("✅ All 5 charts saved to images/ folder")

# ----------------------------------------------------------------------------------------------
cat_summary = df.groupby("category").agg(
    fund_count         = ("scheme_name", "count"),
    avg_return_1yr     = ("returns_1yr",  "mean"),
    avg_return_3yr     = ("returns_3yr",  "mean"),
    avg_return_5yr     = ("returns_5yr",  "mean"),
    avg_expense_ratio  = ("expense_ratio","mean"),
    avg_sharpe         = ("sharpe",       "mean"),
    avg_final_score    = ("final_score",  "mean"),
    affordable_sip_count = ("sip_type", lambda x: (x=="Affordable Sip").sum())
).round(3).reset_index()

cat_summary.to_csv("category_summary.csv", index=False)
print("✅ category_summary.csv saved")
print(cat_summary.to_string())

# ==============================================================================================

with pd.ExcelWriter("mutual_fund_analysis.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer,          sheet_name="All_Funds",    index=False)
    top_30.to_excel(writer,      sheet_name="Top30",        index=False)
    cat_summary.to_excel(writer, sheet_name="Cat_Summary",  index=False)

print("✅ mutual_fund_analysis.xlsx saved with 3 sheets:")
print("   Sheet 1: All_Funds    →", len(df), "rows")
print("   Sheet 2: Top30        →", len(top_30), "rows")
print("   Sheet 3: Cat_Summary  →", len(cat_summary), "rows")