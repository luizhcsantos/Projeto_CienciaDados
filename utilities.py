import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import os
from wordcloud import WordCloud
from collections import Counter 
from itertools import repeat, chain
from scipy.spatial.distance import cosine
import numpy as np
from scipy.stats import pearsonr

nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))
nltk.download('popular', download_dir='nltk_data')


def preprocess_text(text):

    print("Realizando limpeza de texto...")
    # Tokenização
    tokens = word_tokenize(text)

    # Conversão para minúsculas
    tokens = [w.lower() for w in tokens]

    # Remoção de pontuação
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # Remoção de tokens não alfabéticos e stopwords
    words = [word for word in stripped if word.isalpha() and word not in stopwords.words('english')]

    # Juntar as palavras novamente em uma única string
    cleaned_text = ' '.join(words)

    # stemming das palavras
    porter = PorterStemmer()
    cleaned_text = [porter.stem(word) for word in words]

    return cleaned_text



def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color ='white', max_words=200).generate(' '.join(text))
    
    return wordcloud 


def freq_words(text):

    result = list(chain.from_iterable(repeat(i, c)
         for i, c in Counter(text).most_common()))
    
    return result


def ordenar_palavras_por_frequencia(df, label_column, text_column, label_type, min_freq=5):
    # Filtrar os dados pela coluna de rótulo
    text = df[df[label_column] == label_type][text_column]

    # Criar uma lista com todas as palavras das fake news
    all_words = ' '.join(text).split()

    # Contar a frequência das palavras
    word_freq = Counter(all_words)

    # Filtrar as palavras com frequência maior que min_freq
    filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq > min_freq}

    # Ordenar as palavras pela frequência
    sorted_word_freq = sorted(filtered_word_freq.items(), key=lambda x: x[1], reverse=True)

    return sorted_word_freq


def carregar_palavras_csv(nome_arquivo):
    df = pd.read_csv(nome_arquivo)
    all_words = ' '.join(df['Text']).split()
    word_freq = Counter(all_words)
    return word_freq


def calcular_distancia_entre_palavras(fake_freq, real_freq):
    fake_words = list(fake_freq.keys())
    real_words = list(real_freq.keys())
    
    all_words = list(set(fake_words).union(set(real_words)))
    
    fake_vector = np.array([fake_freq[word] if word in fake_words else 0 for word in all_words])
    real_vector = np.array([real_freq[word] if word in real_words else 0 for word in all_words])
    
    return cosine(fake_vector, real_vector)

def calcular_correlacao_entre_conjuntos(df_fake, df_real):

    # Fundir os DataFrames usando a coluna "Word" como índice
    df_fake.set_index('Word', inplace=True)
    df_real.set_index('Word', inplace=True)

    # Encontrar as palavras comuns entre os dois conjuntos
    palavras_comuns = df_fake.index.intersection(df_real.index)

    # Calcular a correlação de Pearson entre as frequências das palavras
    correlacoes = {}
    for palavra in palavras_comuns:
        freq_fake = df_fake.loc[palavra, 'Frequency']
        freq_real = df_real.loc[palavra, 'Frequency']
        correlacao, _ = pearsonr(freq_fake, freq_real)
        correlacoes[palavra] = correlacao

    return correlacoes