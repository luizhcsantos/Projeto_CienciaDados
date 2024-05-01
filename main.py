from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from utilities import preprocess_text, create_wordcloud, freq_words, ordenar_serie_por_frequencia
import time

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def main():
   # df_original=pd.read_csv('data/fake_and_real_news.csv')
   
   # inicio = time.time()
   # print("Tempo de início:", inicio)

   # # cria um dataframe vazio apenas com as colunas 'Text' e 'label'
   # # este dataframe receberá a coluna 'text' do dataframe original após a limpeza dos dados 
   # df_clean = pd.DataFrame({'Text': [], 'label': []})

   # # função que realiza a limpeza dos dados da coluna 'text'
   # df_clean['Text'] = df_original['Text'].apply(preprocess_text)
   
   # # salva o novo dataframe contendo o texto limpo no disco 
   # df_clean.to_csv('dados_limpos2.csv', index=False)  # index=False para não salvar o índice

   # fim = time.time()
   # tempo_total = fim - inicio
   # print("Tempo de término:", fim)
   # print("Tempo total de execução:", tempo_total)

   # # leitura e normalização da coluna 'label' do dataframe original 
   # df_clean=pd.read_csv('cleaned_data/dados_limpos.csv')
   # mapeamento = {'Real': 1, 'Fake': 0}
   
   # # cópia e escita em disco da coluna 'label' normalizada para o dataframe que contém os dados limpos 
   # df_clean['label'] = df_original['label'].map(mapeamento)
   # df_clean.to_csv('cleaned_data/dados_limpos1.csv', index=False)

   # text = df_clean['Text']
   # print(text)
   # wordcloud = create_wordcloud(text)

   df_clean=pd.read_csv('cleaned_data/dados_limpos.csv')

   fake_news = df_clean[df_clean['label'] == 0]['Text']
   real_news = ' '.join(df_clean[df_clean['label'] == 1]['Text'])

   # Criar uma lista com todas as palavras das fake news
   all_words = ' '.join(fake_news).split()

   # Contar a frequência das palavras
   word_freq = Counter(all_words)

   # Filtrar as palavras com frequência maior que 5
   filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq > 5}

   # Ordenar as palavras pela frequência
   sorted_word_freq = sorted(filtered_word_freq.items(), key=lambda x: x[1], reverse=True)

   # Mostrar as palavras ordenadas pela frequência
   for word, freq in sorted_word_freq:
      print(f'{word}: {freq}')

   # Converter o dicionário de palavras frequentes ordenadas em um DataFrame
   df_palavras_frequentes = pd.DataFrame(sorted_word_freq, columns=['Word', 'Frequency'])

   # Salvar o DataFrame em um arquivo CSV
   df_palavras_frequentes.to_csv('cleaned_data/palavras_frequentes2.csv', index=False)

if __name__ == "__main__":
    main()