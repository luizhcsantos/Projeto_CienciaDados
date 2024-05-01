import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import os
from wordcloud import WordCloud
from collections import Counter 
from itertools import repeat, chain
import re

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


def ordenar_serie_por_frequencia(serie):
    # Contar a frequência de cada palavra na série
    contagem_palavras = Counter(serie)
    
    # Ordenar as palavras com base em suas contagens (frequência)
    palavras_ordenadas = sorted(contagem_palavras.items(), key=lambda x: x[1], reverse=True)
    
    # Salvar a frequência de cada palavra em um dicionário
    frequencia_palavras = {palavra: frequencia for palavra, frequencia in palavras_ordenadas}
    
    return palavras_ordenadas, frequencia_palavras