Below is the **Instructions.md** file generated from the project documentation.

---

# Instructions.md

## Project Overview: Data Protection 2025-2026

**Lecturers:** Côme Frappé, Vialatoux
**Institution:** Télécom Physique Strasbourg / Université de Strasbourg

The objective of this project is to apply a full data analysis chain to a **cyber-physical dataset**. You must analyze the data using two distinct approaches:

1. Using only network data.
2. Using only physical data.

---

## 📅 Important Dates

* **Kickoff:** 16/12/2025
* **Defense:** 22/01/2026

---

## 👥 Organization & Technical Constraints

* **Groups:** Work in groups of 4 (7-8 groups total for the promotion).
* **Task Allocation:** Work must be shared, and specific allocations must be mentioned in the report.
* **Hardware:** Code must run on an average laptop (**16GB RAM, no GPU**) in a reasonable time.
* **Optimization:** GPU support and memory optimizations are welcome.
* **Data Note:** Network data files are heavy.

---

## 📊 Requirements & Evaluation Tasks

### 1. Algorithm Comparison

You must compare the following algorithms:
* KNN
* CART
* Random Forest
* XGBoost
* MLP

*Note:* You may replace one model with a different choice if explained in the report.

### 2. Metrics & Analysis

For each algorithm and attack class, evaluate:

* **Balanced Data:** Precision, Recall, TPR, TNR, Accuracy.
* **Unbalanced Data:** F1-score, Balanced Accuracy, Matthews Correlation Coefficient.

* **Visuals:** Confusion matrices for all algorithms.
* **Resource Consumption:** Fit time, prediction time, and RAM usage.
* **Benchmarking:** Compare your results to the models published in the associated research paper.

### 3. Data Handling

* You are free to use **oversampling or undersampling** to improve performance.

* **BONUS:** Find a way to combine physical and network datasets.
* **Innovation:** Novel detection methods are encouraged, even if results are poor.

---

## 📂 Deliverables

### 1. Streamlit Webapp

An interactive interface to explore:

* Exploratory Data Analysis (EDA).
* Results (models, metrics, comparisons, and visualizations).

### 2. Notebooks

Include notebooks for treatments performed outside the webapp (e.g., heavy model training or data prep).

### 3. Project Report (10-20 pages)

Must include:

* EDA insights and how they influenced data handling.
* Data preparation steps.
* Training details: Parameter tuning, architectures, and resources used.
* Performance analysis.

* **Personal Sections:** 1 page per member detailing contributions and takeaways.
* **Bonus Section:** Details on combined datasets or novel methods.

---

## 🎤 Final Presentation

* **Duration:** 10 minutes + ~3 minutes for questions.
* **Content:** EDA highlights, data prep, algorithm comparison, and conclusion.
* **Format:** It is recommended to use the Streamlit app as your presentation support. If using slides, a live demo of the webapp is mandatory during the time slot.
* **Advice:** Rehearse! Timing is strictly enforced.

---

## 📝 Grading Rubric

| Category         | Criterion                                                       | Points      |
| ---------------- | --------------------------------------------------------------- | ----------- |
| **Presentation** | Timing, Fluidity, Clarity                                       | 3           |
| **Results**      | EDA, Algorithm Application, Performance Analysis, Data Handling | 11          |
| **Webapp**       | Completeness, User Experience, Pertinence                       | 6           |
| **Bonus**        | Combined info / Novel methods                                   | 1           |
| **Total**        |                                                                 | **20 (+1)** |

Note: The global mark is conditioned on the submission of all deliverables.

