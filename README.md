# Otimizador Estatístico de Portfólios: Fronteira Eficiente (MPT)

Este repositório contém uma aplicação analítica desenvolvida em **Python** e **Streamlit**, desenhada para calcular a alocação ideal de capital baseada na **Teoria Moderna do Portfólio (MPT)** de Harry Markowitz.

O motor do projeto substitui iterações lentas por uma arquitetura **100% vetorizada via NumPy**, calculando o retorno esperado, a volatilidade e o Índice de Sharpe de milhares de simulações.

## 🚀 Arquitetura e Funcionalidades
* **Motor Matemático Vetorizado:** Processamento matricial com `np.einsum` para otimização de risco/retorno sem o uso de estruturas de repetição, garantindo altíssima performance.
* **Mapeamento Interativo:** Visualização da Fronteira Eficiente utilizando `Plotly`, isolando matematicamente a carteira de tangência (Máximo Índice Sharpe).
* **Ancoragem Macroeconômica:** Coleta automática do *10-Year Treasury Yield* (`^TNX`) para parametrizar a Taxa Livre de Risco ($R_f$) dinamicamente, substituindo variáveis estáticas.
* **Backtest (Base 100):** Projeção histórica comparativa injetando os pesos ótimos na linha do tempo real dos ativos.

## ⚠️ Limitações do Modelo e Cuidados Analíticos

Embora o algoritmo aplique um rigor matemático estrito na extração e processamento da matriz de covariância, a Teoria Moderna do Portfólio possui limitações intrínsecas que devem ser respeitadas para garantir a integridade da análise:

### 1. A Natureza Errática e Não-Estacionária dos Retornos
O motor calcula os pesos ótimos assumindo que os retornos médios históricos e a correlação entre os ativos se manterão constantes. Na realidade, os mercados financeiros operam com volatilidade não-estacionária. O algoritmo foi propositalmente desenhado para **modelar a realidade do dataset exatamente como ela se apresenta**, sem aplicar suavizações ou reamostragens artificiais. Consequentemente, o modelo tenderá a alocar mais peso nos ativos que mais performaram no passado recente. Este painel deve ser usado como uma bússola de risco, não como um preditor definitivo de rentabilidade futura.

### 2. Distorção Cambial e Mistura de Mercados
O cálculo de covariância assume que todos os ativos estão expostos à mesma base monetária e inflacionária. **Não é recomendado inserir ativos de diferentes jurisdições e moedas no mesmo cálculo** (ex: misturar ações brasileiras cotadas em BRL com ativos americanos). 
A ausência de conversão cambial ou *hedge* distorce severamente a matriz de risco. Para resultados íntegros, recomenda-se analisar ecossistemas isolados — por exemplo, inserindo apenas *tickers* do mercado americano (como `GOOG`, `NVDA`, `SPY` ou ETFs setoriais) para descobrir a alocação ótima em dólar.

### 3. A Maldição da Dimensionalidade no Espaço de Busca
O motor de otimização deste projeto utiliza a simulação estocástica (Monte Carlo) para gerar milhares de combinações aleatórias de pesos e mapear a Fronteira Eficiente. Este método é extremamente rápido e visualmente didático para portfólios focados (idealmente entre 5 e 15 ativos). 
No entanto, ao inserir um número massivo de ativos (ex: 50 ações), o algoritmo esbarra na "Maldição da Dimensionalidade". O espaço geométrico de combinações possíveis cresce de forma exponencial, fazendo com que 25.000 simulações se tornem uma amostra estatisticamente irrelevante para encontrar o verdadeiro cume do Índice de Sharpe. Para otimização de dezenas ou centenas de ativos, a abordagem ideal exigiria a substituição deste motor estocástico por um solucionador de programação quadrática (convex optimization), garantindo a convergência matemática exata.

## 🛠️ Como Executar Localmente

1. **Clone o repositório:**
```bash
git clone [https://github.com/SEU_USUARIO/otimizador-portfolio.git](https://github.com/SEU_USUARIO/otimizador-portfolio.git)
cd otimizador-portfolio
