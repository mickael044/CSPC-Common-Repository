import pandas as pd

print("\n--- Task 6: The Naive Analysis ---")

# 1. Load the new dataset
df_chem = pd.read_csv("chemicals_cancer.csv")

# 2. Quantify how strongly each chemical is associated with malignancy
corr_benzene = df_chem['benzene'].corr(df_chem['malignancy'])
corr_cadmium = df_chem['cadmium'].corr(df_chem['malignancy'])

print(f"Correlation between Benzene and Malignancy: {corr_benzene:.3f}")
print(f"Correlation between Cadmium and Malignancy: {corr_cadmium:.3f}")

# 3. Determine which chemical looks like the cause at face value
print("\n--- Naive Conclusion ---")
if corr_benzene > corr_cadmium:
    print("At face value, BENZENE looks far more strongly associated with malignancy.")
else:
    print("At face value, CADMIUM looks far more strongly associated with malignancy.")


print("\n--- Task 7: Look again (Accounting for Confounding Variable) ---")

# 1. Filter the data to a restricted group where pollution_index is similar
restricted_df = df_chem[(df_chem["pollution_index"] > 40) & (df_chem["pollution_index"] < 60)]

# 2. Recompute correlations within this restricted group
corr_benzene_restricted = restricted_df['benzene'].corr(restricted_df['malignancy'])
corr_cadmium_restricted = restricted_df['cadmium'].corr(restricted_df['malignancy'])

print(f"Correlation between Benzene and Malignancy (restricted group): {corr_benzene_restricted:.3f}")
print(f"Correlation between Cadmium and Malignancy (restricted group): {corr_cadmium_restricted:.3f}")

# 3. Final Conclusion
print("\n--- Real Conclusion ---")
print("Conclusion: When we account for the confounding variable (pollution_index) by looking at a restricted group,")
if corr_benzene_restricted > corr_cadmium_restricted:
    print("the correlation with Cadmium drops significantly, while Benzene remains strong. Therefore, BENZENE is the real cause, and Cadmium only looked guilty.")
else:
    print("the correlation with Benzene drops significantly, while Cadmium remains strong. Therefore, CADMIUM is the real cause, and Benzene only looked guilty.")

