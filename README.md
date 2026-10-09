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
