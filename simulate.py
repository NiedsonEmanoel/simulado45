import pandas as pd
import numpy as np
import time
import random
import plotly.express as px
import streamlit as st
import numpy as np
from scipy.optimize import minimize_scalar


def generate_random_number():
    # Gerar um número inteiro aleatório entre 0 e 100000
    return random.randint(0, 100000)

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

st.set_page_config(layout='wide', page_title='Enemaster.app', initial_sidebar_state="expanded", page_icon="🧊",    menu_items={
        'About': "# Feito por *enemaster.app*"
    })

def main():
    gerated = False
    dItens = pd.DataFrame()
    st.sidebar.markdown(f'<img width="100%" src="https://raw.githubusercontent.com/NiedsonEmanoel/NiedsonEmanoel/main/enem/An%C3%A1lise%20de%20Itens/OrdenarPorTri/natureza/EneMaster.png">',unsafe_allow_html=True)
    st.sidebar.markdown(f"<br><hr>",unsafe_allow_html=True)
    email = st.sidebar.text_input('Email')
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
        st.title("Simulado:")
    else:
        st.subheader("Adicione o arquivo .csv do seu simulado ao lado ou gere seu simulado.")


    anwers = []

    with st.form("form", clear_on_submit=False):

        if uploaded_file is not None :
            nu = 1
            for i in dItens.index:
                #st.caption()
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
                st.balloons()

if __name__ == "__main__":
    main()
            
