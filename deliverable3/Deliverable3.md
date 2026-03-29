Project MAI643

Project - Deliverable 1: Dataset selection and brief problem descriptionAssignment
Task:
Submit a word/pdf file document (feel free to use any template) that includes the following:

1. Project Title and Team Members

2. Brief description of the problem – description of the medical domain and the specific problem (2-3 paragraphs)

3. Dataset description – explain how, where, and from whom the dataset was collected (if this information is available), describe its features and identify the target variable (the quantity you need to predict)

4. 3-5 references should be given (hint: identify similar papers/reports published relevant to the dataset, the medical domain. You can find some papers cited in the dataset link).

Feedback:

Team 6: Stroke prediction
==================================================
Title and members:
Provided
Brief description of the problem (medical domain), goal/objective of the project:
Clear description of the problem as well as the purpose of the project.
Dataset description (how, where, from whom was collected, features, target variable):
General description of the dataset including where the dataset was collected and from which website will be obtained.Short description per feature was given. You should extend it a bit more in the next deliverable to include the feature type (numerical, categorical). This is important as a first step to start getting familiarized with the data and their types.Clear description of the target variable which consists of 2 values (binary classification problem)
References:
Provided
Comments:
The datatset has rather not very high number of features but you can experiment with feature selection/extraction techniques. There are missing values so you can apply imputation techniques. There are categorical (non numerical techniques) so you can apply encoding techniques to convert text-based data to numerical data so prepare it for machine learning techniques. The next deliverable starts with EDA. It is about understanding your data before modeling, exploring distributions, relationships, missing values, outliers, and how features relate to the target. Below are suggested techniques tailored to this dataset.
1. Inspect Structure & Missing Values: Check shape (number of rows & columns), Check missing values: bmi has missing entries so you’ll need to handle these (imputation).
Data types: Separate numerical vs categorical. This gives you a foundation before visualizing anything.
2. Univariate Analysis
Numerical Features: Histograms for distributions (e.g., age, glucose, bmi), Boxplots to spot outliers (e.g., extreme bmi or glucose values), Summary statistics (mean, median, std) Watch for long tails in glucose or bmi, unrealistic values (e.g., age = 0), so check and possibly handle (e.g remove if small number)
Categorical Features: Bar charts / count plots for: gender, work_type, smoking_status, ever_married, Residence_type. These show categorical distributions and are useful for spotting rare categories.
3. Target Class Distribution
Since stroke is imbalanced (small % of stroke cases vs non-stroke), this impacts modeling later, you’ll want to quantify that imbalance.
4. Compare Features vs Target to understand how features relate to stroke: 
A) Numerical vs Target Boxplots / violin plots grouped by stroke for: age, avg_glucose_level, bmi
This helps see if, for example, older ages or higher glucose levels correlate with stroke.
B) Categorical vs Target Stacked bar charts: Compare proportions by stroke status e.g., what % of smokers had a stroke vs non-smokers?
These comparisons help identify potential signal in features.
5. Correlation & Relationships
Correlation heatmap for numerical features. Even though target is binary, look at correlations between numerical predictors. (Use Pearson correlation for numeric variables.) For categorical features, check cross-tabulations with the target.
6. Class Imbalance & Resampling
Because stroke cases are rare (≈5–8% of the data), you’ll need to consider resampling techniques: if oversampling use SMOTE.
Class weights in modeling: Visualize imbalance with pie charts or bar charts showing proportions of stroke vs non-stroke.
7. Handle Data Quality. Missing values: Impute bmi (median or KNN)
Outliers: Remove or cap extreme values
Encoding categorical features: One-hot encoding or label encoding e.g., gender, smoking_status, work_type
These steps prepare data for modeling and improve the quality of your EDA insights.
After EDA you need to apply prep-processing techniques (missing values imputation, encoding, resampling, scaling, feature selection/extraction) and create different versions of the dataset. See the project description for more details.

Project - Deliverable 2: Data exploration and pre-processing
Task:
Submit a word/pdf document including:

1. Project Title and Team Members

2. Data understanding: Provide an overview of the dataset using tables, graphs, and short explanations. Include key details such as: dataset shape (#rows, #columns), Preview and statistical summary of features, number of classes (binary vs. multiclass classification), identification of imbalanced data and missing values.

3. Exploratory data analysis: use various visualization techniques to explore the dataset, such as: histograms (feature distributions), bar plots, count plots (categorical features), violin plots, box plots (feature variability), any other relevant techniques for deeper insights.

4. Data pre-processing: Describe the data cleaning steps taken (handling missing values, outliers, inconsistencies), Explain the encoding methods used for categorical variables (if applicable), Discuss feature scaling techniques (normalization, standardization), Mention any feature selection or dimensionality reduction methods applied, Outline strategies for handling imbalanced data (e.g., over-sampling, under-sampling, SMOTE), Justify the choices made for each preprocessing step.

5. Pre-processed dataset versions: Present different versions of the dataset based on applied preprocessing techniques, by highlight variations in data transformations that will be tested in ML model training.

6. Selected machine learning algorithms: List at least two classification algorithms to be explored (e.g., KNN, Logistic Regression, Decision Trees, Random Forest, Boosting methods, Naïve Bayes, SVM), and provide a brief explanation of each algorithm and why it is relevant for the problem.

7. References: Cite relevant sources that support the selected preprocessing methods and machine learning algorithms under investigation.

The source code (e.g. Jupyter Notebook file(s)) should be submitted separately and not included in document.

Feedback:
Team 6: Stroke prediction
==================================================
Data Understanding:
Information provided is adequate. However, I would like you to include some more details regarding the statistics of each feature, especially for the numerical features, such as the mean, standard deviation, and median. These statistics can easily be obtained using the .describe() function.
The target variable was properly described in terms of class imbalance.
In addition, domain knowledge plays an important role in data analysis. Understanding the role and meaning of each feature helps in making informed assumptions and validating insights. For example, it can assist in selecting relevant features and identifying those that may be irrelevant or redundant.
Exploratory Data Analysis:
You did a good job in the source code, but not all figures were included in the documentation. Since you have done good job in the Notebook file please include all figures in the documentation as well.
Overall, the EDA should aim not only to present visualizations but also to extract insights that may guide feature selection and preprocessing decisions in the next stages of the project. 
Therefore, please include all figures in the documentation and provide detailed explanation and outcomes on the basis of each figure. Apply those guidelines in the next deliverable.
Pre-processing:
All pre-processing steps were reasonably explained.
Dataset versions:
Adequate dataset versions provided.
Machine Learning Algorithms:
You did a good job by including four machine learning algorithms with different underlying approaches. I would also recommend trying boosting algorithms, such as XGBoost or LightGBM.
References:
Adequate references were provided.
Source code:
Source code seems ok. Consult the last notebook file I provide for the next part of the project. Firstly, you need measure the performance of the selected machine learning algorithms (with default hyperparameters) on all dataset versions so as to experiment and see which datasets and models are performing better. Then, you will choose 2 best performing models and 2 best performing datasets and proceed to hyperparameter tuning using pipelines. The involved pre-processing techniques in the choses dataset version will be described as pipelines and be given to gridsearchCV. The pre-processing techniques will be applied only on the training dataset within the GridSearchCV process which does the splitting internally.

Project - Deliverable 3: Predictive Model Development and Performance Evaluation
Task:
The goal of this deliverable is to develop and evaluate machine learning classification models using the pre-processed datasets and compare findings with existing literature. Submit a word/pdf document including the following details:

1. Project Title and Team Members

2. Revisions from previous deliverable (if any)

Highlight any updates or modifications to the previous deliverable.
If no changes were made, proceed directly to the next step
3. Model evaluation metrics: Define the evaluation metrics to be used, such as Accuracy, Recall, Precision, (weighted) F1 Score. Justify the choice of metrics based on dataset characteristics (e.g. imbalanced data) and project objectives.

4. Initial model experimentation: Conduct preliminary training and validation experiments using cross validation on the pre-processed dataset versions. Test at least four classification techniques (as identified in the previous deliverable) using their default hyperparameters to understand which dataset-model combinations yield the best results.

5. Selection of best-performing models and pre-processed datasets: Based on initial experimentation, identify the top 2-3 models and the most effective pre-processing techniques.

6. Pipeline Definition: Construct pipelines that integrate the best pre-processing techniques with the best-performing models for systematic experimentation.

7. Hyper-parameter tuning: Perform GridSearchCV to fine-tune hyperparameters for the selected models. Use a well-defined hyperparameter grid.

8. Final model evaluation: Present the final evaluation results using tables and visualizations (e.g., performance metrics, confusion matrices). Report the best-performing model with its optimal hyperparameter values and the best pre-processed dataset version.

9. Discussion:

Key Findings: Summarize the most significant observations from the results.
Comparison with Literature: Prepare a table comparing your results with previous studies using the same dataset. Compare key aspects such as selected features, algorithms used, and achieved performance metrics.
Domain-Specific Validation: Discuss the relevance and validity of the findings within the medical context of the dataset.
 Study Limitations: Highlight any challenges or constraints (e.g., dataset size, computational limitations, class imbalance issues).
The source code (e.g. Jupyter Notebook file(s)) should be submitted separately and not included in document.