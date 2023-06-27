from googletrans import Translator
import pandas as pd

def traduz_cols(df, destination_language='pt'):
    # Inicializa o tradutor
    translator = Translator(service_urls=['translate.google.com'])

    # Traduz os nomes das colunas
    translated_columns = [translator.translate(column, dest=destination_language).text for column in df.columns]
    df.columns = translated_columns

    return df
