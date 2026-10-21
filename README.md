# CSPC — Computer Science for Physics and Chemistry

My coursework repository for the course.
Each practical lives under `PW<n>/Lab <X>/`.

## Setup

Create and activate the environment for a given lab:

```bash
conda env create -f "PW<n>/Lab <X>/environment.yml"
conda activate cspc
```

Run the tests for a lab from inside its folder:

```bash
cd "PW<n>/Lab <X>"
pytest -v
```

---

## PW1 — Lab A: Reproducible Foundations

**What I built:**
- <one or two lines: the CSPC repo, the environment, the decay simulation, the tests>

**Speed comparison (loop vs NumPy):**

| version | time (s) |
|---------|----------|
| pure-Python loop | ... |
| NumPy (vectorised) | ... |

- Speed-up: **... × faster**

**Tests:** all passing? (yes / no)

**Conclusion:**
- <2–3 sentences: what worked, what you learned, any problems you hit and how you solved them>

---

<!-- Future sessions: add a new "## PW<n> — Lab <X>" section below. -->

## PW1 - Lab B

- **Data Analysis:** The observed decay data matches the exponential analytical law ($N(t) = N_0 e^{-\lambda t}$) very closely.
- **Workflow Automation:** A `Snakefile` was created to automatically execute `plot.py` and generate `figure.png` using Snakemake.


# CSPC Project
## PW2 -- Lab A Report

### Results
- **Mean Acceleration:** -8.58 m/s²
- **Standard Deviation (Noise):** 28.72
- **Maximum Recovery Difference:** 0.7846 m

### Why is acceleration so noisy while position looks smooth?
Differentiation acts as a high-pass filter that magnifies high-frequency noise present in the original measurements. Taking the derivative twice amplifies this noise significantly, making acceleration look very rough, even though the original position data appears visually smooth.

### Integration Result
Integrating the noisy acceleration back suppresses the noise (integration cancels out random variations) and successfully recovers the original position within less than a metre of accuracy (0.78 m).

## PW2 --- Lab B Report

### Optimization Methods Comparison (Part 2)
- **Easy Convex Function $f(x)$:** Gradient Descent, Newton's method, and SLSQP all converged smoothly to the global minimum at $x \approx 3.0000$.
- **Harder Landscape $g(x)$:** The starting point significantly affects the result. From $x_0 = 0$, Newton's method converges to a local stationary point ($x \approx 1.1389$), whereas SLSQP finds the minimum. Starting from different points highlights how local curvature and initial guesses impact optimization algorithms.

### Reaction Kinetics (Part 3)
- Fitted rate constant: $k \approx 0.25$ (or your calculated $k$ value).
- Generated plot: `kinetics.png`.

### Chemical Equilibrium (Part 4)
- Both Newton's method and SLSQP agree on the equilibrium extent $x \approx 0.66$.
- Equilibrium composition: $\text{H}_2 \approx 0.33\text{ mol}$, $\text{I}_2 \approx 0.33\text{ mol}$, and $\text{HI} \approx 1.33\text{ mol}$.
- Generated plot: `equilibrium.png`.

### Titration Equivalence Point (Part 5)
- Computed equivalence point: $V \approx 50.00\text{ mL}$ (where the slope $d(\text{pH})/dV$ peaks).
- Generated plot: `titration.png`.

## PW3 — Data, Distributions, Testing, and Causation Report

### Session 1: Normality Verdicts
- Analyzed the continuous variables (`age`, `chol`, `trestbps`, `thalach`) using histograms and the Shapiro-Wilk test.
- **Verdict:** According to the visual shape of the distributions and the Shapiro-Wilk p-values, the variables show deviations from a perfect normal distribution, requiring careful consideration for statistical testing.

### Session 2: Heart Disease Analysis
- **Heart-rate comparison:** A t-test was conducted to compare the maximum heart rate (`thalach`) between healthy patients (target=0) and those with heart disease (target=1). 
- **Conclusion:** There is a statistically significant difference between the two groups. The plotted data with error bars confirms this difference.
- **Age and Heart Rate Relationship:** The Pearson correlation between age and maximum heart rate is **-0.394**. This negative correlation indicates that they move in opposite directions: as age increases, the maximum heart rate generally decreases.

### Session 2: Chemical Exposure Mystery (Correlation vs. Causation)
- **Naive Analysis:** At face value, Cadmium appeared to be the primary driver of malignancy with a very high correlation (**0.880**), while Benzene had a much lower correlation (**0.385**).
- **Accounting for the Confounding Variable:** When restricting the dataset to a specific group where the `pollution_index` was similar (between 40 and 60), the association completely flipped. The correlation for Benzene jumped to **0.702**, while the correlation for Cadmium dropped significantly to **0.316**.
- **Final Conclusion:** **Benzene** is the actual chemical driving the malignancy. Cadmium only looked guilty in the naive analysis because of the hidden confounding variable (`pollution_index`).