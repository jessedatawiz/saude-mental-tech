import os
import sys
import shutil
import time
from data.preprocessing_data import call_preprocess_data, process_data
from data.edit_data import edit_dataframe
from models.model import call_data_main

"""
Função que exclui o cache do python apos rodar.
"""
def get_main_directory():
    # Get the absolute path of the directory containing this script (cleanup.py)
    current_script = os.path.abspath(sys.argv[0])

    # Get the directory of the main script (main.py)
    main_directory = os.path.dirname(current_script)

    return main_directory

def remove_pycache(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs[:]:
            if dir_name == "__pycache__":
                pycache_dir = os.path.join(root, dir_name)
                # print(f"Removing {pycache_dir}")
                try:
                    # Remove the __pycache__ directory and its contents recursively
                    shutil.rmtree(pycache_dir)
                except OSError as e:
                    print(f"Error while removing {pycache_dir}: {e}")


def main():
    start_time = time.time()

    # Corpo da função main()

    # # Executa a primeira parte do pré-processamento
    # call_preprocess_data()

    # elapsed_time = time.time() - start_time
    # print(f"Parte 1 - Pre-process v1: completa. Tempo: {elapsed_time:.2f}")

    # # Executa a segunda parte do processamento dos dados
    # process_data()

    # elapsed_time = time.time() - start_time
    # print(f"Parte 2 - Pre-process v2: completa. Tempo: {elapsed_time:.2f}")
    
    # # Executa a edição das colunas e variáveis
    # edit_dataframe()

    # elapsed_time = time.time() - start_time
    # print(f"Parte 3 - Edicao dos dados: completa. Tempo: {elapsed_time:.2f}")

    # PCA -> XGBoost -> Feature Selection -> XGBoost
    call_data_main()

    elapsed_time = time.time() - start_time
    print(f"Parte 4 - ML model: completa. Tempo: {elapsed_time:.2f}")

    # Calcula o tempo de execução
    elapsed_time = time.time() - start_time

    # Caminho do arquivo de relatório
    report_path = os.path.join(os.getcwd(), 'utils', 'performance_report.txt')

    # Salva o relatório de desempenho
    with open(report_path, 'w') as file:
        file.write(f"Tempo total de execução: {elapsed_time:.2f} segundos\n")

    # Imprime mensagem de conclusão
    print("Relatório de desempenho salvo com sucesso!")

if __name__ == '__main__':
    main()
    main_directory = get_main_directory()
    print(f"Main directory: {main_directory}")
    remove_pycache(main_directory)
