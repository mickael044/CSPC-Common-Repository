import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro


#----------------------TASK 1----------------------#
# read Heart_disease_cleveland_new.csv
df = pd.read_csv("Heart_disease_cleveland_new.csv")

# print the shapes (mötərizələr silindi)
print("The size of the information: ", df.shape) 
print("\nfirst rows:\n", df.head())

# seperate 4 continuous column
continuous_df = df[['age', 'chol', 'trestbps', 'thalach']]
print("\nselected continuous variables: \n", continuous_df.head())


#----------------------TASK 2----------------------#
# hipogram for continuous_df
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

continuous_df.hist(ax=axes, bins=20, edgecolor='black', color='skyblue')
plt.suptitle("distributions of contunious variables", fontsize=16)
plt.tight_layout()
plt.savefig("distributions.png")
plt.show()


#----------------------TASK 3----------------------#

print("\n--- Task 3: Normality Test (Shapiro-Wilk) ---")

#apply shapiro wilk test to each column
for col in continuous_df.columns:
    stat, p_value = shapiro(continuous_df[col].dropna())

    # If the p-value is greater than 0.05, normality is not rejected
    if p_value > 0.05:
        print(f"  Result: '{col}' is approximately NORMALLY distributed.\n")
    else:
        print(f"  Result: '{col}' is NOT NORMALLY distributed.\n")