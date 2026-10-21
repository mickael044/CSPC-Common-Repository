import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


#reload data froom this data 
df = pd.read_csv("Heart_disease_cleveland_new.csv")

# separete the dat ainto two groups: disease (target = 1) and health )target = 0) 

print("\n--- Task 4: Heart Disease and Maximum Heart Rate (thalach) ---")
healthy_thalach = df[df['target'] == 0]['thalach']
disease_thalach = df[df['target'] == 1]['thalach']

# calculate the mean and uncertainty (standard error) for both groups with .mean() and .sem()
mean_healthy = healthy_thalach.mean()
sem_healthy = healthy_thalach.sem()

mean_disease = disease_thalach.mean()
sem_disease = disease_thalach.sem()

# apply a t-test to check whether there is a significant difference between the two groups
t_stat, p_val = ttest_ind(healthy_thalach, disease_thalach, equal_var=False)
print(f"T-test p-value: {p_val:.5e}")

if p_val < 0.05:
    print("Result: There IS a statistically significant difference between the groups.")
else:
    print("Result: There is NO statistically significant difference between the groups.")

# visualize the results using a bar chart with error bars
plt.figure(figsize=(6, 5))
plt.bar(['Healthy (0)', 'Disease (1)'], [mean_healthy, mean_disease],
        yerr=[sem_healthy, sem_disease], capsize=10, color=['lightblue', 'salmon'], edgecolor='black')
plt.title("Mean 'thalach' in Healthy and Disease Groups")
plt.ylabel("Maximum Heart Rate (thalach)")

# Save the plot
plt.savefig("thalach_comparison.png")
plt.show()


print("\n--- Task 5: Age and Maximum Heart Rate (thalach) ---")

# 1. Calculate the correlation
correlation = df['age'].corr(df['thalach'])
print(f"Correlation between Age and Thalach: {correlation:.3f}")

# Interpretation of the result (1-2 sentences)
if correlation < 0:
    print("Interpretation: Age and maximum heart rate move in opposite directions. As age increases, maximum heart rate generally decreases.")
elif correlation > 0:
    print("Interpretation: Age and maximum heart rate move in the same direction. As age increases, heart rate also increases.")
else:
    print("Interpretation: There is no significant linear relationship between these two variables.")

# 2. Visualize the two variables in a scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(df['age'], df['thalach'], alpha=0.6, color='purple', edgecolors='white')
plt.title("Age and Maximum Heart Rate")
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate (thalach)")
plt.grid(True, linestyle='--', alpha=0.5)

# Save the plot
plt.savefig("age_vs_thalach.png")
plt.show()