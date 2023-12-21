import pandas as pd
import numpy as np
import time
import random
import plotly.express as px
import streamlit as st
import numpy as np
from scipy.optimize import minimize_scalar
import genanki
from fpdf import FPDF
import requests
from io import BytesIO
import streamlit.components.v1 as components
from PIL import Image
import string
import os
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import barcode
import zipfile
from barcode.writer import ImageWriter


def flashnamesa(SG):
    if SG == 'CN': return 'Natureza'
    elif SG == 'MT': return 'Matemática'
    elif SG == 'CH': return 'Humanas'
    else: return 'Linguagens'

def download_image(url, filename):
    # Verificar se o arquivo já existe
    if os.path.exists(filename):
        return

    # Fazer o download da imagem
    response = requests.get(url)

    if response.status_code == 200:
        # Ler a imagem a partir dos dados binários
        image = Image.open(BytesIO(response.content))

        # Salvar a imagem
        image.save(filename)

        print(f"Imagem salva como {filename}.")
    else:
        print(f"Erro ao baixar a imagem. Código de status: {response.status_code}")

# URLs das imagens
url1 = "https://niedsonemanoel.com.br/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/natureza/fundo.png"
url2 = "https://raw.githubusercontent.com/NiedsonEmanoel/NiedsonEmanoel/main/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/natureza/EneMaster.png"

# Chamar a função para fazer o download das imagens
download_image(url1, "fundo.png")
download_image(url2, "fundo2.png")

#Definindo Classe do PDF de Saída
class PDF(FPDF):
    def header(self):
       self.image('fundo.png', x=0, y=0, w=self.w, h=self.h, type='png')

    def add_my_link(self, x, y, txt, link):
        self.set_xy(x, y)
        self.set_text_color(0, 0, 0)
        self.set_font('Times', 'BI', 12)
        self.add_link()

        # obter a largura do texto
        w = self.get_string_width(txt) + 6  # adicione uma margem de 3 em cada lado

        # desenhar o retângulo em torno do texto
        self.set_fill_color(255, 112, 79)
        self.cell(w, 10, '', border=0, ln=0, fill=True, align='C', link=link)

        # adicionar o texto com o link
        self.set_xy(x, y)
        self.cell(w, 10, txt, border=0, ln=1, align='C', link=link)

    # Page footer
    def footer(self):
      if self.page_no() != 1:
        self.image("fundo2.png", x=90, y=283, h=10,type='png')
        self.set_y(0)
        self.set_font('Arial', 'BI', 8)
        self.cell(0, 8, '     '+str(self.page_no()) + '/{nb}', 0, 0, 'C')

def toYoutube(textPrompt):
    try:
      search_query = "https://www.youtube.com/results?search_query=" + "+".join(textPrompt.split())
    except:
      search_query = 'N/A'
    return(search_query)

def remover_caracteres_invalidos(texto):
        numAssc = 251
        try:
          caracteres_invalidos = [char for char in texto if ord(char) > numAssc]
          texto_substituido = ''.join('' if ord(char) > numAssc else char for char in texto)
          print(f"Caracteres inválidos substituídos: {caracteres_invalidos}")
          return texto_substituido
        except:
          print('sorry')
          return(texto)

def Capa(dItens):
  todos_itens = ' '.join(s for s in dItens['OCRSearch'].apply(str).values)
  todos_itens = todos_itens.replace(';',  ' ').replace('/',  ' ')

  all_letters = list(string.ascii_lowercase + string.ascii_uppercase)

  stop_words = all_letters +  ['a', 'A', 'b', 'B', 'c', 'C', 'd','figura', 'D', 'e', 'E', 'v', 'nan','pela', 'ser', 'de', 'etc', '(s)', 'do', 'da', 'por', 'para', 'entre', 'se', 'um', 'até', 'ele', 'ela', 'qual', 'bem', 'só', 'mesmo', 'uma', 'um', 'mais', 'menos', 'outro', 'porque', 'por que', 'cada', 'muito', 'todo', 'foram', 'tem', 'meio', 'país', 'una', 'for',
                'uma', 'na', 'su', 'with', 'no','estes','mesma', 'lá', 'that', 'vo' 'pela', 'pelo', 'h', 'H', 'CH', 'ao', 'com', 'que', 'em', 'dos', 'das', 'eu', 'lo', 'the', 'me', 'y', 'la', 'en', 'en', 'to', 'quem', 'and', 'sem', 'on', 'at', 'essa', 'sem', 'uso', 'esse', 'las', 'suas', 'el', 'poi', 'pai', 'doi', 'in', 'pois', 'con', 'of',
                'ainda', 'não', 'o', 'a', 'os','mê','próximo', 'apresenta','quando', 'meu', 'acordo', 'grande', 'saída', 'dessa', 'as', 'deve', 'Além', 'cinco', 'nessa', 'conforme', 'contendo', 'interior', 'Disponível', 'disponível', 'ocorre', 'vezes', 'através', 'grupo', 'tipo', 'algumas', 'causa', 'considerando', 'essas', 'formação', 'so', 'SO', 'pessoa', 'utilizada', 'alguns', 'quais', 'fio', 'outras', 'só', 'exemplo', 'está', 'oo','isso', 'fonte', 'durante', 'onde', 'caso', 'será', 'pelos', 'Disponível', 'duas', 'dois', 'onde', 'podem', 'apresentam', 'alguma', 'outra', 'seja', 'menor', 'Após', 'Considere', 'partir' 'aq', 'etapa', 'três', 'vez', 'pelas', 'dia', 'nova', 'Acesso', 'veículo', 'seus', 'têm', 'quadro', 'parte', 'desses', 'alguma', 'alta', 'sendo', 'eles', 'outros', 'respectivamente', 'lhe', 'ficou','desse', 'pode', 'nas', 'nem', 'nos', 'nesse', 'apenas', 'n', 'esses', 'igual', 'estão', 'br', 'L', 'questão', 'e', 'texto', 'são', 'é', 'como', 'à', 'no', 'mai', 'seu', 'sua', 'mais', '.', 'ano', 'ma', 'ou', 'foi', 'sobre', 'às', 'aos', 'mas', 'há', 'seguinte', 'já', 'maior', 'era', 'desde', 'diferente', 'forma', 'também']

  wc = WordCloud(background_color='black',
                stopwords=stop_words,
                collocations=False,
                colormap = 'copper',
                width=2480, height=3508, contour_width=0)  # Defina a largura e altura desejadas

  wordcloud = wc.generate(todos_itens)

  # Plotar a nuvem de palavras
  plt.figure(figsize=(10, 10))  # Ajuste o tamanho da figura conforme necessário

  a4_width_inches = 8.27
  a4_height_inches = 11.69
  dpi = 300  # Ajuste a resolução conforme necessário

  # Criar a figura com o tamanho A4
  fig, ax = plt.subplots(figsize=(a4_width_inches, a4_height_inches), dpi=dpi)

  # Plotar a nuvem de palavras
  ax.imshow(wordcloud, interpolation='bilinear')
  ax.axis("off")

  # Salvar a figura em tamanho A4
  plt.savefig("wordcloud_a4.png", bbox_inches='tight', pad_inches=0)

def generate_random_number():
    # Gerar um número inteiro aleatório entre 0 e 100000
    return random.randint(0, 100000)


def questHab(dfResult_CN, name):
    try:
        cols_to_drop = ['TP_LINGUA', 'TX_MOTIVO_ABAN', 'IN_ITEM_ABAN', 'IN_ITEM_ADAPTADO', 'NU_PARAM_A', 'NU_PARAM_B', 'NU_PARAM_C']
        dfResult.drop(cols_to_drop, axis=1, inplace=True)
        Capa(dfResult_CN)
    except:
        pass

    flashnames = name
    dfResult_CN.sort_values('theta_065', ascending=True, inplace=True)
    dfResult_CN['indexacao'] = dfResult_CN.reset_index().index + 1

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_title(flashnames)

    pdf.add_page()
    pdf.image("wordcloud_a4.png", x=0, y=0, w=pdf.w, h=pdf.h, type='png')
    pdf.add_page()

    pdf.set_font('Times', 'B', 12)
    img_dir = 'images/'  # Diretório local para salvar as imagens

    # Criar diretório se não existir
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)


    for i in dfResult_CN.index:
        print("N"+str(dfResult_CN.loc[i, 'indexacao'])+"/"+str(len(dfResult_CN)))
        strCN ="N"+str(dfResult_CN.loc[i, 'indexacao'])+" - Q" + str(dfResult_CN.loc[i, "CO_POSICAO"])+':'+str(dfResult_CN.loc[i, "ANO"]) + ' - H'+str(dfResult_CN.loc[i, "CO_HABILIDADE"].astype(int))+ " - Proficiência: " + str(dfResult_CN.loc[i, "theta_065"].round(2))
        if 'dtype:' in strCN:
            print("...")
        else:
            try:
                pdf.ln(15)  # adicionar espaço entre o texto e a imagem
                img_filename = f"{dfResult_CN.loc[i, 'CO_ITEM']}.png"
                img_path = os.path.join(img_dir, img_filename)

                codestr = f"{dfResult_CN.loc[i, 'CO_ITEM']}"

                img_pathax = os.path.join(img_dir, str('xa'+codestr))

                code128 = barcode.get("code128", codestr, writer=ImageWriter())
                filename = code128.save(img_pathax)
                img_pathax = img_pathax+'.png'

                # Verificar se a imagem já foi baixada
                if not os.path.exists(img_path):
                    url = 'https://niedsonemanoel.com.br/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/1.%20Itens%20BNI_/'+ str(dfResult_CN.loc[i, "CO_ITEM"]) + '.png'
                    response = requests.get(url)

                    with open(img_path, 'wb') as img_file:
                        img_file.write(response.content)
                        print(img_path)

                # Abrir a imagem do diretório local
                with Image.open(img_path) as img:
                    img.thumbnail((160, 160))
                    width, height = img.size

                pdf.set_fill_color(255, 112, 79)
             #   pdf.ln(15)
                pdf.cell(0, 10, strCN, 0, 1, 'C', 1)
                pdf.ln(10)   # adicionar espaço entre o texto e a imagem

                # caCNular a posição y para centralizar a imagem
                y = pdf.get_y()

                # ajustar as coordenadas de posição e o tamanho da imagem
                pdf.image(img_path, x=pdf.w / 2 - width / 2, y=y, w=width, h=height)
                pdf.image(img_pathax, x=3, y=-3,  h=25) #w=45,
                pdf.ln(10)

                link = toYoutube(remover_caracteres_invalidos(dfResult_CN.loc[i, "OCRSearch"]))
                pdf.add_my_link(170, 25, "RESOLUÇÃO", link)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font('Times', 'B', 12)

                # adicionar quebra de página
                pdf.add_page()
            except FileNotFoundError:
                print(strCN)
                continue

    #GAB
    page_width = 190
    cell_width = 19
    max_cols = int(page_width / cell_width)

    # Junta as colunas do dataframe
    dfResult_CN['merged'] = dfResult_CN['indexacao'].astype(str) + ' - ' + dfResult_CN['TX_GABARITO']

    # Divide os dados em grupos de até max_cols colunas
    data = [dfResult_CN['merged'][i:i+max_cols].tolist() for i in range(0, len(dfResult_CN), max_cols)]

    # CaCNula a largura das células de acordo com o número de colunas
    cell_width = page_width / max_cols

    # Cria a tabela
    pdf.set_fill_color(89, 162, 165)
    # Title
    pdf.ln(15)
    pdf.cell(0, 10, str('GABARITO'), 0, 1, 'C', 1)
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)

    for row in data:
        for col in row:
            pdf.cell(cell_width, 10, col, 1, 0, 'C')
        pdf.ln() # quebra de linha para a próxima linha da tabela

    pdf.ln(5)
    pdf.set_font('Arial', 'BI', 8)

    strOut =str(name)+ '.pdf'

    pdf.output(strOut, 'F')

    return strOut


def calcular_probabilidade(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))


def calcular_verossimilhanca(theta, a, b, c, x):
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    num_itens, num_candidatos = x.shape
    verossimilhanca = np.ones(num_candidatos)
    for i, item in enumerate(x):
        p = calcular_probabilidade(theta, a[i], b[i], c[i])
        verossimilhanca *= np.power(p, item) * np.power(1 - p, 1 - item)
    return np.prod(verossimilhanca)

def encontrar_theta_max(a, b, c, x):
    if len(a) != x.shape[0] or len(b) != x.shape[0] or len(c) != x.shape[0]:
        raise ValueError("O comprimento das listas de parâmetros a, b e c deve corresponder ao número de itens em x.")

    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)

    theta_max_list = []
    for i in range(x.shape[1]):
        result = minimize_scalar(lambda theta: -calcular_verossimilhanca(theta, a, b, c, x[:, i]), bounds=(-3, 5), method='bounded')
        theta_max_list.append(result.x * 100 + 500)
    return theta_max_list

modelo = genanki.Model(
    187333333,
    'enemaster',
    fields=[
        {'name': 'MyMedia'},
        {'name': 'Questão'},
        {'name': 'Resposta'},
        {'name': 'Image'}
    ],
    templates=[
        {
            'name': 'Cartão 1',
            'qfmt': '<b>{{Questão}}</b><hr>{{MyMedia}}',
            'afmt': '{{FrontSide}}<br><hr><b>{{Resposta}}<hr></b></b>{{Image}}',
        },
    ])

def questionBalance_99(name, sg, nota_CN, dfResult):
    nota_CNMaior = nota_CN + 130
    nota_CNMenor = nota_CN - 80

    dfResult = dfResult.query("IN_ITEM_ABAN == 0 and TP_LINGUA not in [0, 1]")
    try:
        cols_to_drop = ['TP_LINGUA', 'TX_MOTIVO_ABAN', 'IN_ITEM_ABAN', 'IN_ITEM_ADAPTADO', 'NU_PARAM_A', 'NU_PARAM_B', 'NU_PARAM_C']
        dfResult.drop(cols_to_drop, axis=1, inplace=True)
    except:
        pass
    # Para a área de Natureza (CN)
    dfResult_CN = dfResult[dfResult['SG_AREA'] == sg]
    dfResultInterc = dfResult_CN[dfResult_CN['theta_065'] <= nota_CN+200]
    dfResultInterc = dfResult_CN[dfResult_CN['theta_065'] >= nota_CN-30]
    dfResult_CN = dfResult_CN[dfResult_CN['theta_099'] <= nota_CNMaior]
    dfResult_CN = dfResult_CN[dfResult_CN['theta_099'] >= nota_CNMenor]
    dfResult_CN = dfResult_CN[~dfResult_CN['theta_065'].isin(dfResultInterc['theta_065'])]
    dfResult_CN.sort_values('theta_065', ascending=True, inplace=True)
    dfResult_CN['indexacao'] = dfResult_CN.reset_index().index + 1

    # Criar um baralho para armazenar os flashcards
    baralho = genanki.Deck(
        generate_random_number(), # Um número aleatório que identifica o baralho
        str('TRI::Revisão::'+str(name)) # O nome do baralho
    )

    # Criar uma lista para armazenar as informações dos flashcards
    flashcards = []

    # Percorrer as linhas do dataframe dfResult_CN
    for i in dfResult_CN.index:
        # Obter o nome do arquivo de imagem da questão
        imagem = str(dfResult_CN.loc[i, "CO_ITEM"]) + '.png'
        imagemQ = str(dfResult_CN.loc[i, "CO_ITEM"]) + '.gif'

        # Obter a resposta da questão
        resposta =str('Gabarito: ')+ str(dfResult_CN.loc[i, 'TX_GABARITO'])
        inic = "Q" + str(dfResult_CN.loc[i, "CO_POSICAO"]) + ':' + str(dfResult_CN.loc[i, "ANO"]) + ' - H' + str(dfResult_CN.loc[i, "CO_HABILIDADE"].astype(int)) + " - Proficiência: " + str(dfResult_CN.loc[i, "theta_065"].round(2))

        # Criar um flashcard com a imagem e a resposta
        flashcard = genanki.Note(
            model=modelo,
            fields=[inic, '<img src="https://niedsonemanoel.com.br/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/1.%20Itens%20BNI_/' + imagem + '"]', resposta,  '<img src="https://niedsonemanoel.com.br/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/1.%20Itens%20BNI_/Correcao/' + imagemQ + '"]']
        )

        # Adicionar o flashcard à lista de flashcards
        flashcards.append(flashcard)

    for flashcard in flashcards:
        baralho.add_note(flashcard)

    # Criar um pacote com o baralho e as imagens
    pacote = genanki.Package(baralho)

    ileo = True
    if dfResult_CN.empty:
        print("O DataFrame está vazio.")
        ileo = False
    else:
        questHab(dfResult_CN, str(name)+'_revisao')


    pacote.write_to_file(str(name)+'_revisao.apkg')
    return str(name)+'_revisao', ileo

def questionBalance_65(name, sg, nota_CN, dfResult):
    nota_CNMaior = nota_CN + 200
    nota_CNMenor = nota_CN - 30

    dfResult = dfResult.query("IN_ITEM_ABAN == 0 and TP_LINGUA not in [0, 1]")
    try:
        cols_to_drop = ['TP_LINGUA', 'TX_MOTIVO_ABAN', 'IN_ITEM_ABAN', 'IN_ITEM_ADAPTADO', 'NU_PARAM_A', 'NU_PARAM_B', 'NU_PARAM_C']
        dfResult.drop(cols_to_drop, axis=1, inplace=True)
    except:
        pass
    # Para a área de Natureza (CN)
    dfResult_CN = dfResult[dfResult['SG_AREA'] == sg]
    dfResult_CN = dfResult_CN[dfResult_CN['theta_065'] <= nota_CNMaior]
    dfResult_CN = dfResult_CN[dfResult_CN['theta_065'] >= nota_CNMenor]
    dfResult_CN.sort_values('theta_065', ascending=True, inplace=True)
    dfResult_CN['indexacao'] = dfResult_CN.reset_index().index + 1

    # Criar um baralho para armazenar os flashcards
    baralho = genanki.Deck(
        generate_random_number(), # Um número aleatório que identifica o baralho
        str('TRI::Treino::'+str(name)) # O nome do baralho
    )

    # Criar uma lista para armazenar as informações dos flashcards
    flashcards = []

    # Percorrer as linhas do dataframe dfResult_CN
    for i in dfResult_CN.index:
        # Obter o nome do arquivo de imagem da questão
        imagem = str(dfResult_CN.loc[i, "CO_ITEM"]) + '.png'
        imagemQ = str(dfResult_CN.loc[i, "CO_ITEM"]) + '.gif'

        # Obter a resposta da questão
        resposta =str('Gabarito: ')+ str(dfResult_CN.loc[i, 'TX_GABARITO'])
        inic = "Q" + str(dfResult_CN.loc[i, "CO_POSICAO"]) + ':' + str(dfResult_CN.loc[i, "ANO"]) + ' - H' + str(dfResult_CN.loc[i, "CO_HABILIDADE"].astype(int)) + " - Proficiência: " + str(dfResult_CN.loc[i, "theta_065"].round(2))

        # Criar um flashcard com a imagem e a resposta
        flashcard = genanki.Note(
            model=modelo,
            fields=[inic, '<img src="https://niedsonemanoel.com.br/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/1.%20Itens%20BNI_/' + imagem + '"]', resposta,  '<img src="https://niedsonemanoel.com.br/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/1.%20Itens%20BNI_/Correcao/' + imagemQ + '"]']
        )

        # Adicionar o flashcard à lista de flashcards
        flashcards.append(flashcard)

    for flashcard in flashcards:
        baralho.add_note(flashcard)

    # Criar um pacote com o baralho e as imagens
    pacote = genanki.Package(baralho)

    ileo = True
    if dfResult_CN.empty:
        print("O DataFrame está vazio.")
        ileo = False
    else:
        questHab(dfResult_CN, str(name)+'_treino')


    pacote.write_to_file(str(name)+'_treino.apkg')
    return str(name)+'_treino', ileo

st.set_page_config(layout='wide', page_title='Enemaster.app', initial_sidebar_state="expanded", page_icon="🧊",    menu_items={
        'About': "# Feito por *enemaster.app*"
    })

def main():
    gerated = False
    dItens = pd.DataFrame()
    st.sidebar.markdown(f'<img width="100%" src="https://raw.githubusercontent.com/NiedsonEmanoel/NiedsonEmanoel/main/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/natureza/EneMaster.png">',unsafe_allow_html=True)
    st.sidebar.markdown(f"<hr>",unsafe_allow_html=True)
    option = st.sidebar.selectbox(
    "Para qual prova gerar o simulado?",
    ("Linguagens (sem idiomas)", "Humanas", "Natureza", "Matemática"),
    index=None,
    placeholder="Selecione uma prova...",
    )
    if st.sidebar.button('Gerar!', type='primary'):
        if option is not None:
            urlItens = "https://github.com/NiedsonEmanoel/NiedsonEmanoel/raw/main/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/gerador/provasOrdernadasPorTri.csv"
            dLeso = pd.read_csv(urlItens, encoding='utf-8', decimal=',')
            disciplina = ''

            if option=="Linguagens (sem idiomas)":
                disciplina = 'LC'
            elif option=="Humanas":
                disciplina = 'CH'
            elif option =='Natureza':
                disciplina='CN'
            else:
                disciplina='MT'
            
            dLeso = dLeso[dLeso['SG_AREA'] == disciplina]
            dLeso = dLeso[dLeso['CO_HABILIDADE'].between(1, 30)]
            dLeso = dLeso[dLeso['IN_ITEM_ABAN'] == 0]

            dLeso.sort_values('theta_065', ascending=True, inplace=True)

            if disciplina == 'LC':
                dLeso = dLeso[~dLeso['CO_HABILIDADE'].isin([5, 6, 7, 8])]
            #
            habilidades_unicas = dLeso.groupby('CO_HABILIDADE').sample(1)
            habilidades_repetidas = dLeso.groupby('CO_HABILIDADE').apply(lambda x: x.sample(min(len(x), 3)))
            habilidades_repetidas = habilidades_repetidas.sample(n=12, replace=True)
            resultado = pd.concat([habilidades_unicas, habilidades_repetidas])
            habilidades_presentes = resultado['CO_HABILIDADE'].unique()
            if disciplina != 'LC':
                if len(habilidades_presentes) < 30:
                    # Calcular o número de habilidades faltantes
                    habilidades_faltantes = np.setdiff1d(range(1, 31), habilidades_presentes)
                    num_faltantes = 30 - len(habilidades_presentes)

                    # Selecionar itens adicionais para as habilidades faltantes
                    itens_faltantes = dLeso[dLeso['CO_HABILIDADE'].isin(habilidades_faltantes)].sample(n=num_faltantes, replace=True)

                    # Combinar os itens faltantes com os resultados atuais
                    resultado = pd.concat([resultado, itens_faltantes])
            # Verificar o número de itens atual
            num_itens = len(resultado)

            # Remover itens extras se o número atual for maior que 45
            if num_itens > 45:
                resultado = resultado.sample(n=45)

            # Preencher com itens adicionais se o número atual for menor que 45
            if num_itens < 45:
                num_adicionais = 45 - num_itens
                itens_adicionais = dLeso.sample(n=num_adicionais, replace=True)
                resultado = pd.concat([resultado, itens_adicionais])
            
            csv = resultado.to_csv(index=False, encoding='utf-8', decimal=',')
            st.sidebar.download_button(
                label="Download .csv Simulado",
                data=csv,
                #type='primary',
                file_name='simulado'+disciplina+'.csv',
                mime='text/csv',
            )

    st.sidebar.markdown(f"<hr>",unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Envie o arquivo do Simulado", type=['csv'])
    if uploaded_file is not None:
        dItens = pd.read_csv(uploaded_file, encoding='utf-8', decimal=',')
        a, b, c = zip(*dItens[['NU_PARAM_A', 'NU_PARAM_B', 'NU_PARAM_C']].values.tolist())
        total = len(a)
        st.title("Simulado "+str(flashnamesa(dItens['SG_AREA'][0]))+':')
    else:
        st.subheader("Adicione o arquivo .csv do seu simulado ao lado ou gere seu simulado.")


    anwers = []
    erradas = []
    with st.form("form", clear_on_submit=False):

        if uploaded_file is not None :
            nu = 1
            for i in dItens.index:
                strCN ="Nº"+str(nu)+" - Q" + str(dItens.loc[i, "CO_POSICAO"])+':'+str(dItens.loc[i, "ANO"]) 
                st.caption(strCN)

                st.image(dItens.loc[i, "imagAPI"])

                #st.markdown(f"<br>",unsafe_allow_html=True)
                Resposta = st.radio(
                        'Questão ' + str(nu) + ' - Resposta:',
                        ['A', 'B', 'C', 'D', 'E'],
                        horizontal=True,
                    )
                if dItens.loc[i, "TX_GABARITO"] == Resposta:
                    anwers.append(1)
                else:
                    anwers.append(0)

                    erradas.append(dItens.loc[i])

                st.markdown(f"<hr>",unsafe_allow_html=True)
                nu+=1

            submitted = st.form_submit_button("Concluir!", type="primary")
            if submitted:
                x = np.array(anwers)
                with st.spinner("Estimando sua nota TRI..."):
                    time.sleep(2)
                    theta_max_list = encontrar_theta_max(a, b, c, x)
                tri = round(theta_max_list[0],2)
                st.success('Sua nota aproximada é: '+str(tri), icon="✅")
                erradas = pd.DataFrame(erradas)
                with st.spinner("Gerando material de estudo..."):

                    questHab(erradas, 'erradas')

                    flTreino, ileo65 = questionBalance_65(flashnamesa(str((dItens['SG_AREA'][0]))), dItens['SG_AREA'][0], tri, dItens)
                    namesFile = ['erradas.pdf']
                    if ileo65 == True:
                        namesFile.append(flTreino+'.pdf')
                        namesFile.append(flTreino+'.apkg')
                    else:
                        namesFile.append(flTreino+'.apkg')

                    flRevisa, ileo99 = questionBalance_99(flashnamesa(str((dItens['SG_AREA'][0]))), dItens['SG_AREA'][0], tri, dItens)

                    if ileo99 == True:
                        namesFile.append(flRevisa+'.pdf')
                        namesFile.append(flRevisa+'.apkg')
                    else:
                        namesFile.append(flRevisa+'.apkg')

                    zip_filename = 'materialestudo.zip'

                    # Crie um objeto ZipFile em modo de escrita
                    with zipfile.ZipFile(zip_filename, 'w') as zipf:
                        # Adicione cada arquivo à archive
                        for file in namesFile:
                            # Certifique-se de que o arquivo exista antes de adicioná-lo ao zip
                            if os.path.exists(file):
                                zipf.write(file, os.path.basename(file))

                    # Apague os arquivos originais após zipar
                    for file in namesFile:
                        if os.path.exists(file):
                            os.remove(file)
                            print(f'O arquivo {file} foi removido com sucesso.')

                    print(f'Arquivos foram zipados para {zip_filename} e os originais foram removidos.')

                    with open(zip_filename, "rb") as fp:
                        st.sidebar.markdown(f"<hr>",unsafe_allow_html=True)
                        st.info('Baixe seu material de estudo ao lado.', icon="ℹ️")
                        st.balloons()
                        st.sidebar.download_button(
                            label="Download Material de Estudo",
                            type='primary',
                            data=fp,
                            file_name=zip_filename,
                            mime='application/zip',
                        )

if __name__ == "__main__":
    main()
            
