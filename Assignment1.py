# CSC4600 Assignment 1, Lecturer: PROF. TS. DR. NURFADHLINA BINTI MOHD SHAREF
# This program includes Python codes which outlines a comprehensive methodology
# for analyzing and modeling a dataset to achieve classification and clustering objectives
# written by 216465, 216427, 216263, 214301

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Test-Set.csv'
data = pd.read_csv(file_path)

# ---------------------------
# 1. Data Preprocessing
# ---------------------------

# Separate features and target
X = data.drop('OutletType', axis=1)
y = data['OutletType']

# ---------------------------
# Missing Values Report
# ---------------------------
print("\nMissing Values Report:")
missing_values = X.isnull().sum()
print(missing_values[missing_values > 0])

# ---------------------------
# Outlier Detection, Reporting, and Visualization
# ---------------------------

def detect_outliers(df, column):
    """Detect and visualize outliers in the given column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    print(f"\nOutliers in '{column}': {outliers.sum()} detected.")
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

    # Visualization: Scatter Plot Only
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(df)), df[column], c=outliers, cmap='coolwarm', label='Outliers')
    plt.axhline(lower_bound, color='red', linestyle='--', label='Lower Bound')
    plt.axhline(upper_bound, color='green', linestyle='--', label='Upper Bound')
    plt.title(f'Scatter Plot Highlighting Outliers in {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.legend()
    plt.show()

    return df

# Detect and visualize outliers for 'ProductVisibility'
X = detect_outliers(X, 'ProductVisibility')

# Apply outlier capping
def cap_outliers(df, column):
    """Cap outliers to within the IQR bounds."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df

X = cap_outliers(X, 'ProductVisibility')

# Visualize outliers before capping for 'ProductVisibility'
plt.figure(figsize=(12, 6))
sns.boxplot(x=X['ProductVisibility'], color='skyblue')
plt.title("Boxplot of ProductVisibility (Before Capping)")
plt.show()

# Visualize outliers after capping for 'ProductVisibility'
plt.figure(figsize=(12, 6))
sns.boxplot(x=X['ProductVisibility'], color='lightgreen')
plt.title("Boxplot of ProductVisibility (After Capping)")
plt.show()

# ---------------------------
# Inconsistent Data Formats Report
# ---------------------------
# Standardize inconsistent formats in categorical features
if 'FatContent' in X.columns:
    original_counts = X['FatContent'].value_counts()
    X['FatContent'] = X['FatContent'].str.lower().replace({'lf': 'low fat', 'low fat': 'low fat', 'reg': 'regular'})

    print(
        "\nInconsistent Data Formats: The FatContent attribute in the dataset contained inconsistent formats, such as:")
    for value, count in original_counts.items():
        print(f"  - {value}: {count} occurrences")

    print("\nThese were standardized to:")
    print(X['FatContent'].value_counts())

    print(
        "\nThis standardization ensures consistent representation of categories like 'low fat' and 'regular', improving data interpretation and model performance.")

# ---------------------------
# Bias Detection: Target Variable Distribution
# ---------------------------
print("\nBias Detection - Target Variable Distribution (OutletType):")
print(y.value_counts())

# ---------------------------
# Numerical and Categorical Features
# ---------------------------
num_features = ['ProductVisibility', 'MRP', 'Weight']  # Add numerical columns as required
cat_features = ['ProductType', 'LocationType', 'FatContent']  # Add categorical columns as required

# Preprocessing pipelines for numerical and categorical data
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Encode target variable
y_encoded = pd.factorize(y)[0]

# ---------------------------
# 2. Classification Task
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(rf_model, X_preprocessed, y_encoded, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", np.mean(cv_scores))

# ---------------------------
# 3. Clustering Task
# ---------------------------

# Ensure columns relevant to clustering are selected
clustering_data = data[['LocationType', 'OutletSize', 'ProductType', 'ProductVisibility']].copy()

# Map LocationType to numeric values
location_type_map = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3}
clustering_data['LocationType'] = clustering_data['LocationType'].map(location_type_map)

# Encode OutletSize as numeric if necessary
outlet_size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
clustering_data['OutletSize'] = clustering_data['OutletSize'].map(outlet_size_map)

# Encode ProductType as numeric (if applicable, or use one-hot encoding for categorical data)
clustering_data['ProductType'] = pd.factorize(clustering_data['ProductType'])[0]

# Impute missing values instead of dropping them
imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'
clustering_data_imputed = pd.DataFrame(imputer.fit_transform(clustering_data), columns=clustering_data.columns)

# Scale the data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data_imputed)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Adjust n_clusters as needed
clustering_data_imputed['Cluster'] = kmeans.fit_predict(scaled_data)

# Calculate Silhouette Score
silhouette_avg = silhouette_score(scaled_data, clustering_data_imputed['Cluster'])
print(f'\nSilhouette Score for Clustering: {silhouette_avg}')

# Analyze the clusters
cluster_summary = clustering_data_imputed.groupby('Cluster').mean()
print("\nCluster Summary:\n", cluster_summary)

# ---------------------------
# Visualizations for Clusters and Feature Distributions
# ---------------------------

# Visualize the distribution of clusters
sns.countplot(data=clustering_data_imputed, x='Cluster')
plt.title('Distribution of Clusters')
plt.show()

# Visualize correlations for numerical features
sns.heatmap(clustering_data_imputed.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# ---------------------------
# End of Methodology
# ---------------------------