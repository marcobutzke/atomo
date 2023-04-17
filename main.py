import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from matplotlib import pyplot as plt
import altair as alt

st.set_page_config(layout="wide")

@st.cache_data
def load_databases():
    base = pd.read_excel('dados/original/Contabil.xlsx')
    base['Dia'] = 1
    base['Data'] = pd.to_datetime(dict(year=base.Ano, month=base.Mes, day=base.Dia))
    base['periodo'] = ((base['Ano'] -  base['Ano'].min()) * 12) + base['Mes']
    return base
    
    # , pd.read_excel('dados/original/Contabil_Contas_Resultado.xlsx')

# contabil, resultado = load_databases()
contabil = load_databases()
st.title('Átomo - protótipos')

abas = ['Regressão 1', 'Regressão 2', 'Regressão 3', 'Regressão 4', 'Regressão 5', 'Cluster', 'Orçamento', 'Evolução Mensal']

with st.expander('Contábil'):
    # st.dataframe(contabil)
    regr1, regr2, regr3, regr4, regr5, clstr, orcam, evlmn = st.tabs(abas)
    with regr1:
        st.header('Contas Nível 1')
        conta1 = st.selectbox('Conta Nível 1', contabil['Conta_1_Des'].unique())
        contabil_conta1 = contabil[contabil['Conta_1_Des'] == conta1]
        contabil_conta1 = contabil_conta1.groupby(['Data'])['Valor'].sum().reset_index()
        contabil_conta1 = contabil_conta1.rename(columns={'Data': 'ds', 'Valor': 'y'})
        m = Prophet().fit(contabil_conta1)
        future = m.make_future_dataframe(periods=21, freq='MS')
        forecast = m.predict(future)
        st.pyplot(m.plot(forecast))
    with regr2:
        st.header('Contas Nível 2')
        conta2 = st.selectbox('Conta Nível 2', contabil['Conta_2_Des'].unique())
        contabil_conta2 = contabil[contabil['Conta_2_Des'] == conta2]
        contabil_conta2 = contabil_conta2.groupby(['Data'])['Valor'].sum().reset_index()
        contabil_conta2 = contabil_conta2.rename(columns={'Data': 'ds', 'Valor': 'y'})
        m = Prophet().fit(contabil_conta2)
        future = m.make_future_dataframe(periods=21, freq='MS')
        forecast = m.predict(future)
        st.pyplot(m.plot(forecast))
    with regr3:
        st.header('Contas Nível 3')
        conta3 = st.selectbox('Conta Nível 3', contabil['Conta_3_Des'].unique())
        contabil_conta3 = contabil[contabil['Conta_3_Des'] == conta3]
        contabil_conta3 = contabil_conta3.groupby(['Data'])['Valor'].sum().reset_index()
        contabil_conta3 = contabil_conta3.rename(columns={'Data': 'ds', 'Valor': 'y'})
        m = Prophet().fit(contabil_conta3)
        future = m.make_future_dataframe(periods=21, freq='MS')
        forecast = m.predict(future)
        st.pyplot(m.plot(forecast))
    with regr4:
        st.header('Contas Nível 4')
        conta4 = st.selectbox('Conta Nível 4', contabil['Conta_4_Des'].unique())
        contabil_conta4 = contabil[contabil['Conta_4_Des'] == conta4]
        contabil_conta4 = contabil_conta4.groupby(['Data'])['Valor'].sum().reset_index()
        contabil_conta4 = contabil_conta4.rename(columns={'Data': 'ds', 'Valor': 'y'})
        m = Prophet().fit(contabil_conta4)
        future = m.make_future_dataframe(periods=21, freq='MS')
        forecast = m.predict(future)
        st.pyplot(m.plot(forecast))
    with regr5:
        st.header('Contas Nível 5')
        conta5 = st.selectbox('Conta Nível 5', contabil['Conta_5_Des'].unique())
        contabil_conta5 = contabil[contabil['Conta_5_Des'] == conta5]
        contabil_conta5 = contabil_conta5.groupby(['Data'])['Valor'].sum().reset_index()
        contabil_conta5 = contabil_conta5.rename(columns={'Data': 'ds', 'Valor': 'y'})
        m = Prophet().fit(contabil_conta5)
        future = m.make_future_dataframe(periods=21, freq='MS')
        forecast = m.predict(future)
        st.pyplot(m.plot(forecast))
    with evlmn:
        st.header('Contas Nível 5')
        col1, col2 = st.columns(2)
        evconta5 = col1.selectbox('Conta Nível 5x', contabil['Conta_5_Des'].unique())
        natureza = col2.selectbox('Natureza', ['D', 'C'])
        evcontabil_conta5 = contabil[(contabil['Conta_5_Des'] == evconta5) & (contabil['D_C'] == natureza)]
        md = evcontabil_conta5['Valor'].mean()
        li = md - (1.96 * evcontabil_conta5['Valor'].std() / np.sqrt(evcontabil_conta5['Valor'].count()))
        ls = md + (1.96 * evcontabil_conta5['Valor'].std() / np.sqrt(evcontabil_conta5['Valor'].count()))
        outmin = evcontabil_conta5['Valor'].quantile(0.25) - ((evcontabil_conta5['Valor'].quantile(0.75) - evcontabil_conta5['Valor'].quantile(0.25)) * 1.5)
        outmax = evcontabil_conta5['Valor'].quantile(0.75) + ((evcontabil_conta5['Valor'].quantile(0.75) - evcontabil_conta5['Valor'].quantile(0.25)) * 1.5)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Outlier Mínimo', round(outmin,2), delta=None)
        c2.metric('Limite Inferior', round(li,2), delta=None)
        c3.metric('Média', round(md,2), delta=None)
        c4.metric('Limite Superior', round(ls,2), delta=None)
        c5.metric('Outlier Máximo', round(outmax,2), delta=None)
        evcontabil_conta5['classe'] = evcontabil_conta5['Valor'].apply(
            lambda x : 'outmin' if x < outmin else(
                'abaixo' if x < li else (
                    'media' if x < ls else (
                        'acima' if x < outmax else 'outmax'
                    )
                )
            )
        )
        # st.dataframe(evcontabil_conta5.sort_values(by='periodo'))
        st.altair_chart(alt.Chart(evcontabil_conta5.sort_values(by='periodo')).mark_bar().encode(x='periodo:O', y="Valor:Q", color='classe:N').properties(width=1200, height=500))

with st.expander('Resultado'):
    st.subheader('Resultados')
    # st.dataframe(resultado)    