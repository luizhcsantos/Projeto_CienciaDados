import pandas as pd
from bs4 import BeautifulSoup
import requests


def buscar_fonte(noticia):
    try:
        # Parâmetros da solicitação para a API do Google Custom Search
        params = {
            'q': noticia,                   # Query de pesquisa
            'key': 'AIzaSyCZNVkFcK4MTEX_Kp_-oO76-7YyGKywkOM',           # Sua chave de API do Google Custom Search
            'cx': '566fbbedec6ca4479',              # Seu ID de pesquisa personalizado do Google
            'num': 1                        # Número de resultados para retornar (apenas 1 neste caso)
        }
        # Endpoint da API do Google Custom Search
        url = 'https://www.googleapis.com/customsearch/v1'
        # Fazendo a solicitação HTTP para a API
        response = requests.get(url, params=params)
        # Verificando se a solicitação foi bem-sucedida
        if response.status_code == 200:
            # Obtendo o resultado da pesquisa (apenas o primeiro resultado neste caso)
            resultado = response.json()['items'][0]
            # Extraindo informações relevantes do resultado
            titulo = resultado['title']
            link = resultado['link']
            print("titulo: ",titulo, "\tlink: ", link)
            return titulo, link
        else:
            print(f"Falha ao buscar informações sobre a notícia1: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Erro ao buscar informações sobre a notícia: {e}")
        return None, None


def main(): 

    df_original = pd.read_csv('../data/fake_and_real_news.csv')

    #columns = ['titulo_pesquisa', 'link_pesquisa']
    df_novo = pd.DataFrame(columns=['titulo_pesquisa', 'link_pesquisa'])

    noticias_reais = df_original[df_original['label'] == 'Real']

    for index, row in df_original.head(100).iterrows():
        noticia = row['Text']
        titulo, link = buscar_fonte(noticia)
        # Armazenar o título e o link do resultado da pesquisa no DataFrame
        df_novo.at[index, 'titulo_pesquisa'] = titulo
        df_novo.at[index, 'link_pesquisa'] = link

    df_novo.to_csv('../data/fake_real_nres_source_link', index=False)

if __name__ == "__main__":
    main()