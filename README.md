# Travel Price Prediction Project Overview

This is a comprehensive overview of a complete MLOps project! Here is the updated `README.md` content, structured to showcase the entire lifecycle, from EDA and model selection to full deployment and MLOps practices.

---

# Travel Price Prediction Project Overview

This project implements an end-to-end Machine Learning solution to **predict flight prices** based on features like departure/arrival city, distance, flight class, and agency. The solution encompasses the entire Data Science lifecycle, from exploratory analysis and model training to deployment via a **containerized REST API**.

## I. Data Analysis and Feature Engineering (Sections 1-4)

This phase focused on understanding the `flights.csv` dataset and preparing features for optimal model performance.

* ### **Data Collection & Initial Assessment**
    * Gathered relevant travel datasets (271,888 rows, 10 columns).
    * Identified data types and confirmed **no missing values** were present.
    * **Dropped non-informative features** (`travelCode`, `userCode`) as they lacked predictive power.

* ### **Exploratory Data Analysis (EDA)**
    * Visualized distributions of **Price**, **Time**, and **Distance** using histograms and scatter plots.
    * Analyzed relationships, noting that **Time and Distance showed perfect correlation (1.0)**, confirming redundancy.
    * Visualized price distribution across **categorical features** (`flightType`, `agency`, `from`, `to`) using box plots to identify influential variables.

* ### **Data Transformation & Wrangling**
    * **Dropped the highly correlated `time` column** to avoid multicollinearity.
    * Converted the `date` attribute to the proper datetime format.
    * Applied a **ColumnTransformer** for feature preprocessing:
        * **One-Hot Encoding** on nominal categories (`flightType`, `agency`).
        * **Ordinal Encoding** on high-cardinality categories (`from`, `to`).
    * The fitted transformer was saved (`transformer.pkl`) for consistent deployment.

## II. Model Development and Evaluation (Section 5)

This section focused on building a robust regression model to achieve the highest predictive accuracy.

* ### **Regression Analysis & Model Benchmarking**
    * Trained and evaluated **10 different regression models** (including Linear, Ridge, Lasso, Decision Trees, and various Ensemble methods).
    * Benchmarked performance using key metrics: **R2 Score**, **MSE**, **MAE**, and **RMSE**.

* ### **Optimal Model Selection**
    * The **Decision Tree Regressor** was identified as the best-performing model based on the highest R2 score and lowest error metrics.
    * The final optimized model was **saved** (`decision_tree_regressor_model.pkl`) using Python's `pickle` library for immediate deployment.

## III. Productionization and Deployment (Sections 2 & 3)

The trained model was deployed as a real-time prediction service using modern MLOps practices.

* ### **Creating REST API with Flask**
    * Developed a **RESTful API endpoint (`/predict`)** using **Flask** to handle user inputs (via a web form or JSON request).
    * The API loads the saved `model.pkl` and `preprocessor.pkl` to transform new input data and serve predictions in real-time.
    * Integrated a simple **Jinja2 template (`index.html`)** to provide a user-friendly prediction interface.

* ### **Deployment with Docker**
    * **Containerized the entire Flask application** and all dependencies (including the saved models and preprocessor) using a **Dockerfile**.
    * This ensures the application is **portable** and runs consistently across any environment (local, testing, or cloud).

## IV. Advanced MLOps & Future Scope (Sections 4-9)

Future work planned to enhance scalability, reliability, and business utility:

* **Scalability & Orchestration:** Planning deployment via **Kubernetes** and orchestrating complex ML pipelines (data validation, retraining) using **Apache Airflow**.
* **CI/CD Pipeline:** Implementing a **Jenkins** CI/CD pipeline for automated testing and reliable deployment.
* **Model Tracking:** Utilizing **MLFlow** for systematic tracking, versioning, and management of model experiments.
* **New ML Applications:** Developing and deploying additional models, including a **Gender Classification Model** and a **Travel Recommendation Model** (displayed via Streamlit) to further enhance business offerings.
