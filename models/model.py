import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/preprocessed_files/all_years/mental_health.csv')
target = 'Você atualmente tem um distúrbio de saúde mental?'
y = df[target]
X = df.drop(columns=target)

# Step 1: Splitting the Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Constructing the Pipeline
pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', VotingClassifier(
        estimators=[('xgb', XGBClassifier()), ('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier())],
        voting='hard')
    )
])

# Step 3: Performing Stratified k-fold Cross-Validation
classifiers = ['XGBoost', 'Random Forest', 'KNN']
accuracies = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in pipeline.named_steps['classifier'].estimators:
    scores = cross_val_score(clf, X_train, y_train, cv=skf)
    accuracy = np.mean(scores)
    accuracies.append(accuracy)
    print(f'{name} Accuracy: {accuracy:.4f}')

# Step 4: Fitting the Pipeline on the Training Data
pipeline.fit(X_train, y_train)

# Step 5: Getting Transformed Data after PCA
X_train_transformed = pipeline.named_steps['pca'].transform(X_train)
X_test_transformed = pipeline.named_steps['pca'].transform(X_test)

# Step 6: Saving Columns Used in PCA Analysis
columns_used = [f'PC{i}' for i in range(1, X_train_transformed.shape[1] + 1)]
print(f'Columns used in PCA analysis: {columns_used}')

# Step 7: Generating the Chart
fig, ax = plt.subplots()
y_pos = np.arange(len(classifiers))
ax.barh(y_pos, accuracies, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(classifiers)
ax.invert_yaxis()
ax.set_xlabel('Accuracy')
ax.set_title('Classifier Performance')

plt.show()

# Step 8: Create DataFrame with Transformed Data and Columns Used in PCA Analysis
X_train_pca_df = pd.DataFrame(X_train_transformed, columns=columns_used)
X_test_pca_df = pd.DataFrame(X_test_transformed, columns=columns_used)

import joblib

# Step 9: Save the Pipeline
pipeline_path = 'pipeline.pkl'
joblib.dump(pipeline, pipeline_path)

