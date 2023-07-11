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

# Step 1: Constructing the Pipeline
pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', VotingClassifier(
        estimators=[('xgb', XGBClassifier()), ('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier())],
        voting='hard')
    )
])

# Step 2: Performing Stratified k-fold Cross-Validation
classifiers = ['XGBoost', 'Random Forest', 'KNN']
accuracies = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in pipeline.named_steps['classifier'].estimators:
    scores = cross_val_score(clf, X, y, cv=skf)
    accuracy = np.mean(scores)
    accuracies.append(accuracy)
    print(f'{name} Accuracy: {accuracy:.4f}')

# Step 3: Generating the Chart
fig, ax = plt.subplots()
y_pos = np.arange(len(classifiers))
ax.barh(y_pos, accuracies, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(classifiers)
ax.invert_yaxis()
ax.set_xlabel('Accuracy')
ax.set_title('Classifier Performance')

plt.show()
