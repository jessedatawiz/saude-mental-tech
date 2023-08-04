import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Constants and file paths
MAIN_FILE_PATH = 'data/data_to_model/mental_health.csv'
DIR_PATH = 'data/data_to_model'
METRICS_PATH = os.path.join(DIR_PATH, 'metrics')
IMAGES_PATH = os.path.join(DIR_PATH, 'images')
PIPELINE_PATH = os.path.join(DIR_PATH, 'pipelines')

XGBOOST_THRESHOLD = 0.02

# Plot function
def plot_data(data, plot_type, file_name, features_importances=None):

    if plot_type == 'correlation_matrix':
        corr_matrix = data.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Set the figure size
        plt.figure(figsize=(20, 20))

        sns.heatmap(corr_matrix, annot=False, annot_kws={"size": 12, "ha": 'center'},
                    fmt=".1f", cmap="Purples", mask=mask, linewidths=0.5)
        
        # Rotates the ticks, to better visualization
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, ha='right')

        plt.title('Correlation Matrix')

    if plot_type == 'confusion_matrix':
        sns.heatmap(data, annot=True, fmt="d", cmap="Purples", xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

    elif plot_type == 'feature_importance':
        color = '#8B008B'
        width = 0.5
        plt.bar(data.columns, features_importances, color=color, width=width)
        plt.axhline(y=XGBOOST_THRESHOLD, color='green', linestyle='--')
        plt.xlabel('Features')
        plt.ylabel('Feature Importance Score')
        plt.title('Feature Importance XGBoost')
        plt.xticks(rotation=90)
        plt.tight_layout()

    image_path = os.path.join(IMAGES_PATH, file_name)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path)
    plt.close()


def save_metrics_to_file(metrics_dict, file_name):
    file_path = os.path.join(METRICS_PATH, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write('XGBoost -> base test: \n')
        for key, value in metrics_dict.items():
            file.write(f"{key}: {value}\n")

"""
Faz o encodamento das variáveis, equanto preserva as variáveis nulas/faltantes.
"""
def label_encode_df(df):
    encoders = {}  # Dicionário para armazenar os codificadores de rótulos para cada coluna

    for col_name in df.columns:
        series = df[col_name]  # Obtém a série da coluna atual
        label_encoder = LabelEncoder()  # Cria uma nova instância de LabelEncoder

        # Aplica a codificação de rótulos aos valores não nulos da série
        df[col_name] = pd.Series(
            label_encoder.fit_transform(series[series.notnull()]),
            index=series[series.notnull()].index
        )

        encoders[col_name] = label_encoder  # Armazena o codificador de rótulos para a coluna

    return df, encoders  # Retorna o DataFrame codificado e o dicionário de codificadores

def save_pipeline(model, file_name) -> None:
        
    pipeline_path = os.path.join(PIPELINE_PATH, file_name)
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    
    joblib.dump(model, pipeline_path)
    print(f"Pipeline salva com sucesso: {pipeline_path}")

    return None

def process_mental_health_data(df):

    # Retorna o df e a lista de encoders
    encoded_df, _ = label_encode_df(df)

    """Já que a biblioteca do PyCaret não faz o 'imputation' dos valores NaN na variável
    target, é usado a KNNImputer um passo antes de usar o PyCaret. """
    imputer = KNNImputer(n_neighbors=5)  
    imputed_values = imputer.fit_transform(encoded_df)
    # Retorna o dataframe com os valores 'imputados'
    imputed_df = pd.DataFrame(imputed_values, columns=df.columns)
    return imputed_df

def xgboost_classification(txt_file_name, best_params, feature_threshold=None, value_threshold=None, test_size=0.3):
    
    df = pd.read_csv(MAIN_FILE_PATH)

    # Faz o encoding e imputation do dataframe pré-processado antes
    imputed_df = process_mental_health_data(df)

    target = 'disturbio_saude_mental_atual'
    y = imputed_df[target].astype(int)
    X = imputed_df.drop(columns=target)

    # Se houver selação de feature_importances
    if feature_threshold is not None and value_threshold is not None:
        threshold = value_threshold
        selected_columns = X.columns[feature_threshold >= threshold]
        X = X[selected_columns]
    

    # Alreay tested model, não havendo features
    param_grid = {
    'model__learning_rate': [1, 0.1, 0.01, 0],
    'model__max_depth': [1, 2, 5, 10, 15],
    'model__n_estimators': [2, 5, 10, 20, 100],
    'model__alpha': [0, 0.05, 0.1, 0.5],
    'model__eval_metric': ['auc'],
    }

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=42)

    # Define a Pipeline para pré-processamento e XGBoost
    pipeline = Pipeline(steps=[
        ('model', xgb.XGBClassifier())
    ])    


    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    # GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=param_grid, 
                               cv=cv)
    
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    # Save the best parameters to a text file
    output_file = best_params
    with open(output_file, 'w') as f:
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"Best Score: {best_score}\n")


    # Fazer previsões no conjunto de teste
    y_pred = best_model.predict(X_test)

    # Obter as importâncias das features
    feature_importances = best_model.named_steps['model'].feature_importances_

    # Avaliar o desempenho do modelo
    """
    # Weighted: Calcula o recall/f1 para cada classe independentemente e tira a média 
    ponderada com base no número de amostras em cada classe.
    """

    # Avaliar o desempenho do modelo
    metrics_dict = {}
    metrics_dict['Recall'] = recall_score(y_test, y_pred, average='weighted')
    metrics_dict['F1-score'] = f1_score(y_test, y_pred, average='weighted')
        
    save_metrics_to_file(metrics_dict, txt_file_name)

    # Criar a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    # Retornar o melhor modelo
    return best_model, cm, feature_importances, X


# Main function
def data_main(txt_file, corr_matrix_file, cm_file, features_file, pipeline_path, best_params, feature_threshold=None, value_threshold=None):
    # Roda o modelo
    best_model_, best_params, cm_, feature_importances_, X_ = xgboost_classification(txt_file, best_params, feature_threshold, value_threshold)
    # Salva o modelo
    save_pipeline(best_model_, pipeline_path)

    # Correlation Matrix
    plot_data(X_, 'correlation_matrix', corr_matrix_file)
    # Matriz de confusão
    plot_data(cm_, 'confusion_matrix', cm_file)
    # Features importance
    plot_data(X_, 'feature_importance', features_file, features_importances = feature_importances_)

    if feature_threshold is None and value_threshold is None:
        return feature_importances_
    else:
        return None

# Runs the main function
def call_data_main():
    
    # Main function
    features_ = data_main('before_featurization/metrics_before_features.txt',
                        'before_featurization/corr_matrix_before_features.png',
                        'before_featurization/confusion_matrix_before_features.png',
                        'before_featurization/feature_importance_before_features',
                        'before_featurization/xgboost_pipeline.pkl',
                        'data/data_to_model/metrics/before_featurization/best_params.txt'
            )

    """
    Após um primeiro modelo com o XGBoost para calcular as feature importances, um novo modelo
    é feito apenas com o threshold das feature_importances >= 0.02. 
    Testes já realizados apontam que não há perda de eficiência no modelo de predição
    """
    _ = data_main('after_featurization/metrics_after_features.txt',
                'after_featurization/corr_matrix_after_features.png',
                'after_featurization/confusion_matrix_after_features.png',
                'after_featurization/feature_importance_after_features',
                'after_featurization/xgboost_pipeline.pkl',
                'data/data_to_model/metrics/after_featurization/best_params.txt',
            features_,
            0.02
            )
