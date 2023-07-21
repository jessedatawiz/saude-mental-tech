import os
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the confusion matrix
def plot_confusion_matrix(cm, file_name):

    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    dir_path = 'data/data_ready_to_model/images/'
    file_name = file_name
    image_path = os.path.join(dir_path, file_name)
    plt.savefig(image_path, dpi=400)

# plot feature importance
def plot_feature_importances(df, feature_importance, file_name):

    # Configurações estéticas (cor darkmagenta)
    color = '#8B008B'
    largura_barras = 0.5
    xgboost_threshold = 0.02

    plt.bar(df.columns, feature_importance, color=color, width=largura_barras)
    plt.axhline(y=xgboost_threshold, color='green', linestyle='--')
    plt.xlabel('Features')
    plt.ylabel('Feature Importance Score')
    plt.title('Feature Importance using XGBoost')
    plt.xticks(rotation=90)

    dir_path = 'data/data_ready_to_model/images/'
    file_name = file_name
    image_path = os.path.join(dir_path, file_name)
    plt.savefig(image_path, dpi=400)


def save_metrics_to_file(metrics_dict, file_name):
    with open(file_name, 'w') as file:
        for key, value in metrics_dict.items():
            file.write("{}: {}\n".format(key, value))

"""
Faz o encodamento das variáveis, equanto preserva as variáveis nulas/faltantes.
"""
def label_encode_dataframe(df):
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


"""
    Já que a biblioteca do PyCaret não faz o 'imputation' dos valores NaN na variável target, é usado a KNNImputer
    um passo antes de usar o PyCaret.
"""
def preprocess_mental_health_data(df):

    # Faz o encode das variáveis categóricas usando Label Encoding
    encoded_df, _ = label_encode_dataframe(df)

    # Inicializar o imputer com KNNImputer
    imputer = KNNImputer(n_neighbors=5)  
    imputed_values = imputer.fit_transform(encoded_df)

    # Retorna o dataframe com os valores 'imputados'
    imputed_df = pd.DataFrame(imputed_values, columns=df.columns)
    #imputed_df.drop(columns='ano', inplace=True)

    return imputed_df

def xgboost_classification(txt_file_name, feature_threshold=None, value_threshold=None, test_size=0.3):
    file_path = 'data/data_ready_to_model/mental_health.csv'
    df = pd.read_csv(file_path)

    # Faz o encoding e imputation do dataframe pré-processado antes
    imputed_df = preprocess_mental_health_data(df)

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
    'model__learning_rate': [1],
    'model__max_depth': [2],
    'model__n_estimators': [5],
    'model__alpha': [0.05],
    'model__eval_metric': ['auc'],
    }

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=42)

    # Define a Pipeline para pré-processamento e XGBoost
    xgboost_pipeline = Pipeline(steps=[
        ('model', xgb.XGBClassifier())
    ])    


    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    # GridSearchCV
    grid_search = GridSearchCV(estimator=xgboost_pipeline, 
                               param_grid=param_grid, 
                               cv=cv)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
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
    return best_model, cm, feature_importances, df



# Chamar a função
txt_file_name = 'metrics_1.txt'
best_model_1, cm_1, feature_importances_1, df = xgboost_classification(txt_file_name)

cm_file_name = 'cm_1.png'
#plot_confusion_matrix(cm_1, cm_file_name)

fi_file_name = 'feature_importances_1.png'
#plot_feature_importances(df, feature_importances_1, fi_file_name)

"""
Após um primeiro modelo com o XGBoost para calcular as feature importances, um novo modelo
é feito apenas com o threshold das feature_importances >= 0.02. 
Testes já realizados apontam que não há perda de eficiência no modelo de predição
"""

# Chamar a função
txt_file_name = 'metrics_2.txt'
best_model_featured, cm_featured, feature_importances_2, df = xgboost_classification(txt_file_name, 
                                                            feature_threshold=feature_importances_1, 
                                                            value_threshold=0.02)

cm_file_name = 'cm_2.png'
plot_confusion_matrix(cm_featured, cm_file_name)

fi_file_name = 'feature_importances_2.png'
#plot_feature_importances(df, feature_importances_2, fi_file_name)



