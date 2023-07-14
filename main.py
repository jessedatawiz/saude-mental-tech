import os
import time
from data.preprocessing_data import call_preprocess_data, process_data
from data.edit_data import edit_dataframe

def main():
    start_time = time.time()

    # Corpo da função main()

    # Executa a primeira parte do pré-processamento
    call_preprocess_data()

    # Executa a segunda parte do processamento dos dados
    process_data()
    
    # Executa a edição das colunas e variáveis
    edit_dataframe()

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
