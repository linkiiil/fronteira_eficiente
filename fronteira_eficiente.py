import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página 
st.set_page_config(page_title="Otimizador de Carteiras - Fronteira Eficiente", layout="wide")

# Customização estética básica via CSS para manter o design minimalista
st.markdown("""
    <style>
    .reportview-container { background: #fafafa; }
    .metric-box { padding: 15px; border-radius: 5px; background-color: #f0f2f6; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Otimizador Estatístico de Portfólios")
st.subheader("Algoritmo Vetorizado Baseado na Teoria Moderna do Portfólio (Markowitz)")

# --- PAINEL LATERAL DE CONFIGURAÇÕES ---
st.sidebar.header("⚙️ Parâmetros de Entrada")

# Entrada de ativos flexível
tickers_input = st.sidebar.text_input("Ativos (Separados por vírgula):", "AAPL, MSFT, NVDA, GOOG, SPY")
ativos = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Período de análise histórica
col_data1, col_data2 = st.sidebar.columns(2)
with col_data1:
    data_inicio = st.date_input("Data de Início:", pd.to_datetime("2021-01-01"))
with col_data2:
    data_fim = st.date_input("Data de Fim:", pd.to_datetime("today"))

# Parâmetros financeiros do investidor
valor_disponivel = st.sidebar.number_input("Capital Total para Alocação ($):", min_value=1000, value=10000, step=1000)

# Procurar dinamicamente a Taxa Livre de Risco (10-Year Treasury Yield ^TNX)
try:
    tnx = yf.download("^TNX", period="1d", progress=False)['Close']
    tlr_atual = float(tnx.iloc[-1]) / 100 if not tnx.empty else 0.04
except Exception:
    tlr_atual = 0.04  # Fallback caso a API falhe ou esteja fora do horário

taxa_livre_risco = st.sidebar.slider("Taxa Livre de Risco Anual (TLR):", min_value=0.0, max_value=0.15, value=tlr_atual, step=0.0025, format="%.4f")
num_simulacoes = st.sidebar.slider("Número de Simulações de Monte Carlo:", min_value=2000, max_value=25000, value=10000, step=1000)

# Botão para executar a lógica
if st.sidebar.button("⚙️ Executar Otimização Vetorizada", use_container_width=True):
    if len(ativos) < 2:
        st.error("Por favor, insira pelo menos 2 ativos válidos para otimização.")
    else:
        with st.spinner("A processar dados de mercado e a executar simulação linear..."):
            try:
                # 1. Extração de Dados Históricos
                dados_mercado = yf.download(ativos, start=data_inicio, end=data_fim, progress=False)
                
                # Tratamento de multi-index da API recente do yfinance, se necessário
                if isinstance(dados_mercado.columns, pd.MultiIndex):
                    precos_df = dados_mercado['Close']
                else:
                    precos_df = dados_mercado['Close'] if 'Close' in dados_mercado else dados_mercado
                
                precos_df = precos_df[ativos].dropna()
                
                if precos_df.empty:
                    st.error("Não foi possível encontrar dados válidos para os ativos e datas selecionadas.")
                    st.stop()
                
                # 2. Processamento Quantitativo
                retornos_ln = np.log(precos_df / precos_df.shift(1)).dropna()
                cov_matrix = retornos_ln.cov() * 252
                retornos_anuais = retornos_ln.mean() * 252
                
                num_ativos = len(ativos)
                
                # 3. Motor Estatístico Vetorizado (Álgebra Linear com NumPy Pura)
                pesos_aleatorios = np.random.random((num_simulacoes, num_ativos))
                pesos_matriz = pesos_aleatorios / np.sum(pesos_aleatorios, axis=1, keepdims=True)
                
                retornos_portfolios = np.dot(pesos_matriz, retornos_anuais)
                variancias_portfolios = np.einsum('ij,jk,ik->i', pesos_matriz, cov_matrix.values, pesos_matriz)
                volatilidades_portfolios = np.sqrt(variancias_portfolios)
                
                sharpe_ratios = (retornos_portfolios - taxa_livre_risco) / volatilidades_portfolios
                
                # Consolidação dos resultados das simulações
                df_simulacoes = pd.DataFrame({
                    'Retorno': retornos_portfolios,
                    'Volatilidade': volatilidades_portfolios,
                    'Sharpe': sharpe_ratios
                })
                for idx, acao in enumerate(ativos):
                    df_simulacoes[acao] = pesos_matriz[:, idx]
                
                # Identificação das Carteiras Notáveis
                idx_max_sharpe = df_simulacoes['Sharpe'].idxmax()
                carteira_ideal = df_simulacoes.loc[idx_max_sharpe]
                
                # --- VISUALIZAÇÃO DOS RESULTADOS ---
                st.success("Cálculos concluídos com sucesso!")
                
                # Métricas Principais da Carteira Ótima
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Retorno Esperado do Portfólio", f"{carteira_ideal['Retorno']:.2%}")
                with c2:
                    st.metric("Risco Anualizado (Volatilidade)", f"{carteira_ideal['Volatilidade']:.2%}")
                with c3:
                    st.metric("Índice Sharpe Máximo", f"{carteira_ideal['Sharpe']:.4f}")
                
                st.markdown("---")
                
                # Organização do Layout Central
                col_esquerda, col_direita = st.columns([1.2, 1])
                
                with col_esquerda:
                    st.subheader("🎯 A Fronteira Eficiente de Markowitz")
                    fig_fronteira = px.scatter(
                        df_simulacoes, x='Volatilidade', y='Retorno', color='Sharpe',
                        labels={'Volatilidade': 'Volatilidade Anualizada (Risco)', 'Retorno': 'Retorno Esperado'},
                        color_continuous_scale='Viridis', title="Mapeamento de Alocações Possíveis"
                    )
                    # Destacar a carteira de Máximo Sharpe
                    fig_fronteira.add_trace(go.Scatter(
                        x=[carteira_ideal['Volatilidade']], y=[carteira_ideal['Retorno']],
                        mode='markers', marker=dict(color='red', size=14, symbol='star'),
                        name='Máximo Sharpe'
                    ))
                    fig_fronteira.update_layout(template="plotly_white")
                    st.plotly_chart(fig_fronteira, use_container_width=True)
                    
                with col_direita:
                    st.subheader("💰 Distribuição Sugerida do Aporte")
                    pesos_otimos = [carteira_ideal[acao] for acao in ativos]
                    alocacao_dinheiro = [peso * valor_disponivel for peso in pesos_otimos]
                    
                    df_alocacao = pd.DataFrame({
                        'Ativo': ativos,
                        'Peso (%)': [f"{p:.2%}" for p in pesos_otimos],
                        'Aporte Sugerido': [f"$ {v:,.2f}" for v in alocacao_dinheiro]
                    })
                    
                    st.dataframe(df_alocacao, use_container_width=True, hide_index=True)
                    
                    # Gráfico de Distribuição Percentual
                    fig_rosca = px.pie(
                        names=ativos, values=pesos_otimos, hole=0.4,
                        title="Composição Percentual da Carteira",
                        color_discrete_sequence=px.colors.sequential.Slate
                    )
                    fig_rosca.update_layout(template="plotly_white")
                    st.plotly_chart(fig_rosca, use_container_width=True)
                
                st.markdown("---")
                
                # Evolução Histórica Base-100 (Uso dos pesos ótimos aplicados ao histórico)
                st.subheader("📈 Crescimento Histórico Simulado da Carteira Ótima (Base 100)")
                precos_normalizados = precos_df / precos_df.iloc[0]
                carteira_base_100 = (precos_normalizados * np.array(pesos_otimos)).sum(axis=1) * 100
                
                fig_linha = go.Figure()
                fig_linha.add_trace(go.Scatter(
                    x=carteira_base_100.index, y=carteira_base_100.values,
                    mode='lines', name='PORTFÓLIO ÓTIMO', line=dict(color='#1f77b4', width=3.5)
                ))
                
                for acao in ativos:
                    fig_linha.add_trace(go.Scatter(
                        x=precos_normalizados.index, y=precos_normalizados[acao].values * 100,
                        mode='lines', name=acao, line=dict(width=1, dash='dash')
                    ))
                    
                    fig_linha.update_layout(
                        title="Evolução Comparativa de um Aporte Inicial de $100",
                        xaxis_title="Data", yaxis_title="Valor do Patrimônio Equivalente ($)",
                        template="plotly_white"
                    )
                st.plotly_chart(fig_linha, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro na execução dos cálculos ou download dos ativos: {str(e)}")
                st.info("Dica: Verifique se os códigos dos tickers estão corretos e no padrão do Yahoo Finance (ex: ativos brasileiros exigem '.SA').")
