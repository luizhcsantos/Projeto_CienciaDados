from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import time
import utilities.utilities as u
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from matplotlib import colormaps
import matplotlib.animation as animation

def main():

   # limpeza do texto 
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

   # ************************************************

   # # leitura e normalização da coluna 'label' do dataframe original 
   # df_clean=pd.read_csv('cleaned_data/dados_limpos.csv')
   # mapeamento = {'Real': 1, 'Fake': 0}
   
   # # cópia e escita em disco da coluna 'label' normalizada para o dataframe que contém os dados limpos 
   # df_clean['label'] = df_original['label'].map(mapeamento)
   # df_clean.to_csv('cleaned_data/dados_limpos1.csv', index=False)

   # text = df_clean['Text']
   # print(text)
   # wordcloud = create_wordcloud(text)

   # *********************************************************
   # ordenação das palavras das noticias por frequencia 
   # df_clean=pd.read_csv('cleaned_data/dados_limpos.csv')

   # fake_news = df_clean[df_clean['label'] == 0]['Text']
   # real_news = ' '.join(df_clean[df_clean['label'] == 1]['Text'])

   # resultado_fake = ordenar_palavras_por_frequencia(df_clean, 'label', 'Text', 0)
   # # Converter o dicionário de palavras frequentes ordenadas em um DataFrame
   # df_palavras_frequentes = pd.DataFrame(resultado_fake, columns=['Word', 'Frequency'])
   # # Salvar o DataFrame em um arquivo CSV
   # df_palavras_frequentes.to_csv('cleaned_data/fake_palavras_frequentes.csv', index=False)

   
   # resultado_real = ordenar_palavras_por_frequencia(df_clean, 'label', 'Text', 1)
   # # Converter o dicionário de palavras frequentes ordenadas em um DataFrame
   # df_palavras_frequentes = pd.DataFrame(resultado_real, columns=['Word', 'Frequency'])
   # # Salvar o DataFrame em um arquivo CSV
   # df_palavras_frequentes.to_csv('cleaned_data/real_palavras_frequentes.csv', index=False)

   # ***********************************************************

   # contagem de palavras por cada categoria (fake, real)

   # df_clean = pd.read_csv('cleaned_data/dados_limpos.csv')

   df_clean = pd.read_csv('cleaned_data/dados_limpos.csv')

   fake_news = df_clean[df_clean['label'] == 0]['Text']
   real_news = df_clean[df_clean['label'] == 1]['Text']

   count_fake = u.contagem_palavras(fake_news)
   count_real = u.contagem_palavras(real_news)

   print("qtde palavras noticias fake: ", count_fake, "\nqtde palavras noticias reais: ", count_real)

   # ***********************************************************
   # calculo de distancias 

   # distancia de jaccard entre as palavras unicas nos conjuntos de palavras presentes nas noticias reais e fake 
   # df_fake=pd.read_csv('cleaned_data/fake_palavras_frequentes.csv')
   # df_real=pd.read_csv('cleaned_data/real_palavras_frequentes.csv')
   
   # distancia_jaccard = u.distancia_jaccard(df_fake, df_real)
   # print("Distância de Jaccard entre os conjuntos fake e real:", distancia_jaccard)

   
   # ************************************************************
   # distancia euclidiana
   # distancia_euclidiana = u.distancia_euclidiana(df_fake, df_fake)
   # print("Distância Euclidiana entre os conjuntos fake e real:", distancia_euclidiana)

   # # distancia de manhattan 
   # distancia_manhattan = u.distancia_manhattan(df_fake, df_real)
   # print("Distância de Manhattan entre os conjuntos fake e real:", distancia_manhattan)

   # # distancia de chebyshev
   # distancia_chebyshev = u.distancia_chebyshev(df_fake, df_real)
   # print("Distância de Chebyshev entre os conjuntos fake e real:", distancia_chebyshev)

   # ***********************************************************

   # Gráficos 
   # 1.Histograma - Contagem de noticias por rlabel - real/fake

   # plt.figure(figsize=(10, 6))
   # sns.countplot(data=df_clean, x="label", hue="label")
   # plt.title("Contagem de Noticias por label")
   # plt.xlabel("Real ou Fake?")
   # plt.ylabel("Contagem")
   # plt.legend(title="Veracidade da informação")
   # plt.savefig('images/cont_noticias_label.png', dpi=300)

   # **************************************
   # Gráfico de Barras - contagens da quantidade de palavras por cada categoria de noticia
   
   # contagens = [count_real, count_fake]
   # plt.bar(['Noticias Reais', 'Noticias fake'], contagens, color=['green', 'purple'], align='edge', width=1.2)
   # plt.title('Contagem de Palavras em Notícias Reais e Fakes')
   # plt.xlabel('Tipo de Notícia')
   # plt.ylabel('Contagem de Palavras')
   # plt.savefig('images/cont_palavras_label.png', dpi=300)
   # plt.show()

   # ***************************************************** 
   # Wordcloud - Nuvem de palavras para noticias reais e fake
   
   # real_news_copy = ' '.join(real_news)
   # wordcloud_real = WordCloud(width=800, height=400, background_color ='white', max_words=2000).generate(real_news_copy)

   # wordcloud para noticias reais
   # plt.figure(figsize=(10, 6))
   # plt.imshow(wordcloud_real, interpolation='bilinear')
   # plt.axis('off')
   # plt.title("Nuvem de palavras das noticias marcadas como 'Real'")
   # plt.savefig('images/wordcloud_real.png', dpi=300)
   # plt.show()

   # wordcloud para noticias fake 
   # fake_news_copy = ' '.join(fake_news)
   # wordcloud_fake = WordCloud(width=800, height=400, background_color ='white', max_words=2000).generate(fake_news_copy)

   # plt.figure(figsize=(10, 6))
   # plt.imshow(wordcloud_fake, interpolation='bilinear')
   # plt.axis('off')
   # plt.title("Nuvem de palavras das noticias marcadas como 'Fake'")
   # plt.savefig('images/wordcloud_fake.png', dpi=300)
   # plt.show()

   # *************************************************************

   # Histograma de comprimento de texto
   df_clean['Text_length'] = df_clean['Text'].apply(len)
   # df_clean.to_csv('cleaned_data/dados_limpos.csv', index=False)

   # Filtrando dados para notícias reais e falsas
   # real_len = df_clean[df_clean['label'] == 1]['Text_length']
   # fake_len = df_clean[df_clean['label'] == 0]['Text_length']

   # # ignora valores maiores que 8000 
   # real_len_limpo = real_len[real_len <= 8000]
   # fake_len_limpo = fake_len[fake_len <= 8000]

   # plt.figure(figsize=(12, 6))

   # plt.subplot(1, 2, 1)  # Subplot para notícias reais
   # plt.hist(real_len_limpo, bins=25, color='green', alpha=0.7, ec='black')
   # plt.title('Distribuição do Comprimento das Notícias Reais')
   # plt.xlabel('Comprimento do Texto')
   # plt.ylabel('Frequência')

   # plt.subplot(1, 2, 2)  # Subplot para notícias falsas
   # plt.hist(fake_len_limpo, bins=25, color='purple', alpha=0.7, ec='black', density=True)
   # plt.title('Distribuição do Comprimento das Notícias Falsas')
   # plt.xlabel('Comprimento do Texto')
   # plt.ylabel('Frequência')

   # plt.tight_layout()  # Ajusta automaticamente os subplots para evitar sobreposição
   # plt.savefig('images/dist_comp_news.png', dpi=300)
   # plt.show()

   # ***********************************************************
   # Boxplot da distribuição do comprimento das noticias

   plt.figure(figsize=(8, 6))
   sns.boxplot(x='label', y='Text_length', data=df_clean, palette={'1': 'green', '0': 'orange'}, showfliers=False)
   plt.title('Distribuição do Comprimento das Notícias')
   plt.xlabel('Categoria "Fake" ou "Real"')
   plt.ylabel('Comprimento do Texto')
   plt.savefig('images/boxplot_comp.png', dpi=300)
   plt.show()



if __name__ == "__main__":
    main()