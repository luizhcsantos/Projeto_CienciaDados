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


def ordenar_palavras_frequencia(df, label_column, text_column, label_type, min_freq=5):
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

def correlacao_conjuntos(df_fake, df_real):

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
        
        # Converter os valores de frequência para arrays numpy
        freq_fake_array = np.array(freq_fake)
        freq_real_array = np.array(freq_real)
        
        # Verificar se os arrays têm pelo menos dois elementos antes de calcular a correlação
        if len(freq_fake_array) > 1 and len(freq_real_array) > 1:
            # Calcular a correlação de Pearson
            correlacao, _ = pearsonr(freq_fake_array, freq_real_array)
            correlacoes[palavra] = correlacao
        else:
            print(f'Não foi possível calcular a correlação para a palavra "{palavra}": arrays de frequência insuficientes')

    return correlacoes



def calcular_distancia_entre_palavras(df_fake, df_real, word):

    # Encontrar as frequências da palavra nos conjuntos fake e real
    freq_fake = df_fake.loc[df_fake['Word'] == word, 'Frequency'].values
    freq_real = df_real.loc[df_real['Word'] == word, 'Frequency'].values

    # Se a palavra não estiver presente em um dos conjuntos, retornar NaN
    if len(freq_fake) == 0 or len(freq_real) == 0:
        return np.nan

    # Calcular a distância euclidiana entre as frequências
    distancia = np.linalg.norm(freq_fake - freq_real)

    return distancia


def mesclar_csv(df_fake, df_real):

    # Mesclar os dataframes usando a coluna 'word' como índice
    df_merged = pd.merge(df_fake, df_real, on='Word', suffixes=('_fake', '_real'))
    
    return df_merged


def distancia_jaccard(df_fake, df_real):
    # Criar conjuntos de palavras únicas
    fake_word_set = set(df_fake['Word'])
    real_word_set = set(df_real['Word'])

    # Calcular a interseção e a união dos conjuntos
    intersection = len(fake_word_set.intersection(real_word_set))
    union = len(fake_word_set.union(real_word_set))

    # Calcular a distância de Jaccard
    jaccard_distance = 1 - (intersection / union)

    return jaccard_distance


def distancia_euclidiana(df_fake, df_real):
    # Criar conjuntos de palavras únicas
    fake_word_set = set(df_fake['Word'])
    real_word_set = set(df_real['Word'])

    # Calcular a diferença ao quadrado entre as frequências das palavras
    squared_differences = [(df_fake.loc[df_fake['Word'] == word, 'Frequency'].iloc[0] - 
                            df_real.loc[df_real['Word'] == word, 'Frequency'].iloc[0])**2
                           for word in fake_word_set.intersection(real_word_set)]

    # Calcular a soma das diferenças ao quadrado
    euclidean_distance = sum(squared_differences) ** 0.5

    return euclidean_distance

def distancia_manhattan(df_fake, df_real):
    # Criar conjuntos de palavras únicas
    fake_word_set = set(df_fake['Word'])
    real_word_set = set(df_real['Word'])

    # Calcular a diferença absoluta entre as frequências das palavras
    absolute_differences = [abs(df_fake.loc[df_fake['Word'] == word, 'Frequency'].iloc[0] - 
                                 df_real.loc[df_real['Word'] == word, 'Frequency'].iloc[0])
                            for word in fake_word_set.intersection(real_word_set)]

    # Calcular a soma das diferenças absolutas
    manhattan_distance = sum(absolute_differences)

    return manhattan_distance

def distancia_chebyshev(df_fake, df_real):
    # Criar conjuntos de palavras únicas
    fake_word_set = set(df_fake['Word'])
    real_word_set = set(df_real['Word'])

    # Calcular a máxima diferença absoluta entre as frequências das palavras
    max_difference = max(abs(df_fake.loc[df_fake['Word'] == word, 'Frequency'].iloc[0] - 
                             df_real.loc[df_real['Word'] == word, 'Frequency'].iloc[0])
                         for word in fake_word_set.intersection(real_word_set))

    return max_difference