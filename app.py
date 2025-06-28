import os
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

# =======================
# Funções Auxiliares
# =======================

def _read_close_prices(path_csv: str) -> pd.DataFrame:
    """Lê CSV de preços. Tenta MultiIndex, se não, usa header simples."""
    try:
        df = pd.read_csv(path_csv, header=[0,1], index_col=0, parse_dates=True)
        if 'Close' in df.columns.get_level_values(1):
            return df.xs('Close', axis=1, level=1).copy()
    except Exception as e:
        pass

    df2 = pd.read_csv(path_csv, index_col=0, parse_dates=True)
    close_cols = [c for c in df2.columns if 'Close' in c]
    if not close_cols:
        raise RuntimeError(f"Nenhuma coluna 'Close' em {path_csv}")
    return df2[close_cols].copy()

def filtrar_tickers(prices: pd.DataFrame, tickers: list, min_obs: int = 200):
    """
    Retorna duas listas: tickers válidos (com pelo menos min_obs observações válidas)
    e tickers problemáticos (que não atendem a esse critério).
    """
    tickers_validos = []
    tickers_problema = []
    for t in tickers:
        if t in prices.columns:
            n_obs = prices[t].dropna().shape[0]
            if n_obs >= min_obs:
                tickers_validos.append(t)
            else:
                st.write(f"[INFO] {t} possui apenas {n_obs} observações; removendo da simulação.")
                tickers_problema.append(t)
        else:
            st.write(f"[INFO] {t} não encontrado no DataFrame.")
            tickers_problema.append(t)
    return tickers_validos, tickers_problema

def filtrar_fronteira_eficiente(vols, rets):
    """
    Remove pontos ineficientes da fronteira: mantém apenas os com retorno máximo
    para cada nível crescente de volatilidade.
    """
    idx_sort = np.argsort(vols)
    vols_sorted = vols[idx_sort]
    rets_sorted = rets[idx_sort]

    efficient_vols = []
    efficient_rets = []
    current_max = -np.inf

    for v, r in zip(vols_sorted, rets_sorted):
        if r > current_max:
            efficient_vols.append(v)
            efficient_rets.append(r)
            current_max = r

    return np.array(efficient_vols), np.array(efficient_rets)

# =======================
# Funções de Simulação
# =======================

def simulate_portfolios(prices: pd.DataFrame, tickers: list, n_sim: int,
                        min_assets: int, max_assets: int, min_w: float,
                        max_w: float, seed: int, alpha_dirichlet: float = 0.3):
    """
    Simula carteiras via Monte Carlo utilizando distribuição Dirichlet
    para gerar os pesos, calcula retorno e risco, e filtra carteiras
    que atendem a:
        - cardinalidade entre min_assets e max_assets (via min_w)
        - peso máximo <= max_w

    Parâmetro alpha_dirichlet controla a concentração: quanto menor,
    maior a chance de carteiras mais concentradas (sugestão: 0.1 a 0.5).
    """
    # Cálculo dos log-retornos e anualização
    rets = np.log(prices / prices.shift(1)).dropna()
    mu   = rets.mean() * 252
    cov  = rets.cov()  * 252

    # Geração dos pesos com Dirichlet esparsa
    rng = np.random.default_rng(seed)
    sim_w = rng.dirichlet(alpha_dirichlet * np.ones(len(tickers)), size=n_sim)

    # Cálculo das métricas
    sim_ret = sim_w.dot(mu.values)
    sim_vol = np.sqrt(np.einsum('ij,jk,ik->i', sim_w, cov.values, sim_w))

    # Filtros
    card   = (sim_w >= min_w).sum(axis=1)
    maxpos = sim_w.max(axis=1)
    mask = (card >= min_assets) & (card <= max_assets) & (maxpos <= max_w)
    valid_idx = np.where(mask)[0]

    return sim_ret[valid_idx], sim_vol[valid_idx]

def simulate_portfolios_cardinalidade_controlada(
    prices: pd.DataFrame,
    tickers: list,
    n_sim: int,
    min_assets: int,
    max_assets: int,
    min_w: float,
    max_w: float,
    seed: int,
    alpha_dirichlet: float = 0.3
):
    rets = np.log(prices / prices.shift(1)).dropna()
    mu = rets.mean() * 252
    cov = rets.cov() * 252

    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    # Ajusta max_assets se for maior que o número de ativos disponíveis
    max_assets = min(max_assets, n_assets)
    min_assets = min(min_assets, max_assets)

    sim_ret = []
    sim_vol = []
    sim_w = []
    ativos_usados = []

    for _ in range(n_sim):
        k = rng.integers(min_assets, max_assets + 1)
        ativos_escolhidos = rng.choice(n_assets, size=k, replace=False)

        pesos = np.zeros(n_assets)
        pesos_ativos = rng.dirichlet(alpha_dirichlet * np.ones(k))
        pesos[ativos_escolhidos] = pesos_ativos

        if pesos.max() > max_w or (pesos >= min_w).sum() < min_assets:
            continue

        ret = np.dot(mu.values, pesos)
        vol = np.sqrt(np.dot(pesos.T, np.dot(cov.values, pesos)))

        sim_ret.append(ret)
        sim_vol.append(vol)
        sim_w.append(pesos)
        ativos_usados.append(ativos_escolhidos)

    return (
        np.array(sim_ret),
        np.array(sim_vol),
        np.array(sim_w),
        ativos_usados
    )

# =======================
# Funções de Otimização
# =======================

def negative_sharpe(w, mu, cov):
    ret = np.dot(mu, w)
    vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    return -ret / vol
    
def optimize_max_sharpe(mu, cov, min_w=0.0, max_w=1.0):
    n = len(mu)
    init = np.repeat(1/n, n)
    bounds = [(min_w, max_w)] * n
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    res = minimize(negative_sharpe, init, args=(mu, cov), method='SLSQP', bounds=bounds, constraints=cons)
    return res.x, -res.fun

def portfolio_return(w, mu):
    return np.dot(mu, w)

def portfolio_volatility(w, cov):
    return np.sqrt(np.dot(w.T, np.dot(cov, w)))

def minimize_volatility_for_target(mu, cov, target, bounds=None):
    n = len(mu)
    init = np.repeat(1/n, n)
    bounds = bounds or [(0, 1)] * n
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - target}
    )
    res = minimize(portfolio_volatility, init, args=(cov,), method='SLSQP', bounds=bounds, constraints=cons)
    return res.x, portfolio_volatility(res.x, cov)

def compute_efficient_frontier(mu, cov, n_points=50):
    targets = np.linspace(min(mu), max(mu), n_points)
    vols, rets = [], []
    for t in targets:
        try:
            w, vol = minimize_volatility_for_target(mu, cov, t)
            vols.append(vol)
            rets.append(t)
        except:
            continue
    return np.array(vols), np.array(rets)

def rebalance_weights(weights, min_w=0.03):
    """
    Zera os pesos abaixo de min_w e rebalanceia os restantes para somar 100%.
    """
    weights = np.where(weights < min_w, 0.0, weights)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights

def normalizar_tickers(lista):
    return [ticker.strip().upper() + ".SA" if not ticker.strip().upper().endswith(".SA") else ticker.strip().upper() for ticker in lista]

def otimizar_carteira_hibrida(
    tickers_man: list,
    valores_man: list,
    ativos_sugeridos: list,
    mu_comb: pd.Series,
    cov_comb: pd.DataFrame,
    percentual_adicional: float = 0.3
):
    # 1) Calcular o "total híbrido" (= 100% da carteira manual + pct adicional)
    total_hibrido = sum(valores_man) * (1 + percentual_adicional)

    # 2) Pesos mínimos obrigatórios para cada ativo manual
    pesos_minimos = {
        t: v / total_hibrido
        for t, v in zip(tickers_man, valores_man)
    }

    if not ativos_sugeridos:
        # monta a lista de candidatos: tudo em mu_comb, menos os manuais
        ativos_sugeridos = [t for t in mu_comb.index if t not in tickers_man]

    # 3) Combinar tickers
    tickers_hibrida = tickers_man + ativos_sugeridos

    # 4) Extrair mu e cov para todos
    mu_h    = mu_comb.loc[tickers_hibrida].values
    cov_h   = cov_comb.loc[tickers_hibrida, tickers_hibrida].values

    # 5) Índices para pesos
    idx_map = {t: i for i, t in enumerate(tickers_hibrida)}
    idx_manual    = [idx_map[t] for t in tickers_man]
    idx_sugeridos = [idx_map[t] for t in ativos_sugeridos]

    # 6) Montar constraints:
    cons = []
    # 6.1 soma dos pesos = 1
    cons.append({'type': 'eq', 'fun': lambda w: w.sum() - 1})

    # 6.2 cada manual w[i] >= peso_minimo[i]
    for t, min_w in pesos_minimos.items():
        i = idx_map[t]
        cons.append({
            'type': 'ineq',
            'fun': lambda w, i=i, min_w=min_w: w[i] - min_w
        })

    # 6.3 soma dos pesos de ativos_sugeridos ≤ percentual_adicional / (1 + percentual_adicional)
    max_extra = percentual_adicional / (1 + percentual_adicional)
    cons.append({
        'type': 'ineq',
        'fun': lambda w: max_extra - w[idx_sugeridos].sum()
    })

    # 7) Bounds [0,1] para cada w
    bounds = [(0.0, 1.0)] * len(tickers_hibrida)

    # 8) Chute inicial (distribuição uniforme)
    init = np.repeat(1 / len(tickers_hibrida), len(tickers_hibrida))

    # 9) Executar otimização de Sharpe (negativo)
    res = minimize(
        negative_sharpe,
        init,
        args=(mu_h, cov_h),
        method='SLSQP',
        bounds=bounds,
        constraints=cons
    )

    w_hibrida = res.x
    ret_h = w_hibrida.dot(mu_h)
    vol_h = np.sqrt(w_hibrida.dot(cov_h).dot(w_hibrida))
    sharpe_h = ret_h / vol_h

    return tickers_hibrida, w_hibrida, ret_h, vol_h, sharpe_h

# =======================
# Funções de Plotagem
# =======================

def plot_results(sim_vol_aco, sim_ret_aco, ef_vol_aco_opt, ef_ret_aco_opt, vol_aco, ret_aco,
                 sim_vol_fii, sim_ret_fii, ef_vol_fii_opt, ef_ret_fii_opt, vol_fii, ret_fii,
                 sim_vol_comb, sim_ret_comb, ef_vol_comb_opt, ef_ret_comb_opt, vol_comb, ret_comb,
                 vol_man, ret_man, vol_opt_manual, ret_opt_manual, vol_hibrida, ret_hibrida):
    plt.figure(figsize=(12,8))

    # Monte Carlo e Fronteiras
    plt.scatter(sim_vol_fii, sim_ret_fii, s=8, alpha=0.12, color='green', label='FIIs (Monte Carlo)')
    plt.plot(ef_vol_fii_opt, ef_ret_fii_opt, 'g-', lw=2, label='Fronteira FIIs')
    plt.scatter(vol_fii, ret_fii, color='green', marker='*', s=180, label='Sharpe Máx FIIs')

    plt.scatter(sim_vol_comb, sim_ret_comb, s=8, alpha=0.12, color='red', label='Combinado (Monte Carlo)')
    plt.plot(ef_vol_comb_opt, ef_ret_comb_opt, 'r-', lw=2, label='Fronteira Combinada')
    plt.scatter(vol_comb, ret_comb, color='red', marker='*', s=180, label='Sharpe Máx Combinado')

    plt.scatter(sim_vol_aco, sim_ret_aco, s=8, alpha=0.12, color='blue', label='Ações (Monte Carlo)')
    plt.plot(ef_vol_aco_opt, ef_ret_aco_opt, 'b-', lw=2, label='Fronteira Ações')
    plt.scatter(vol_aco, ret_aco, color='blue', marker='*', s=180, label='Sharpe Máx Ações')

    # Carteiras manuais
    plt.scatter(vol_man, ret_man, c="black", s=80, marker="X", label="Carteira Manual")
    plt.scatter(vol_opt_manual, ret_opt_manual, c="orange", s=80, marker="D", label="Manual Otimizada")
    plt.scatter(vol_hibrida, ret_hibrida, c="purple", s=80, marker="P", label="Carteira Híbrida")

    plt.xlabel("Volatilidade Anualizada")
    plt.ylabel("Retorno Anualizado")
    plt.title("Fronteira Eficiente – Ações x FIIs x Combinado")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    st.pyplot(plt)

# =======================
# Bloco Principal
# =======================

def main():
    st.title("Simulação de Carteiras e Fronteira Eficiente: v12")
    # Upload do arquivo CSV
    url = "https://raw.githubusercontent.com/dcecagno/Optimize-portfolio/main/all_precos.csv"
    prices_read_original = _read_close_prices(url)
    prices_read = prices_read_original.copy()
    
    # Período de análise
    anos = st.slider("Anos de análise", 1, 10, 5)
    time_end = pd.Timestamp.now().normalize()
    time_start = time_end - pd.DateOffset(years=anos)
    prices_read = prices_read.loc[time_start:time_end]
    
    # Parâmetros para a simulação de Monte Carlo
    n_sim = 100_000
    seed = 42
    alpha_dirichlet = 1
    min_assets = st.number_input("Número mínimo de ativos", min_value=1, max_value=20, value=6)
    max_assets = st.number_input("Número máximo de ativos", min_value=1, max_value=20, value=15)
    min_w_percent = st.number_input("Peso mínimo por ativo (%)", min_value=0, max_value=100, value=3, step=1)
    max_w_percent = st.number_input("Peso máximo por ativo (%)", min_value=0, max_value=100, value=30, step=1)

    # Converte para proporção (0 a 1)
    min_w = min_w_percent / 100
    max_w = max_w_percent / 100

    # Carteira manual
    st.subheader("Carteira Manual")
    
    # Entrada da carteira manual em valores monetários
    num_ativos = st.number_input("Número de ativos na carteira manual", min_value=1, max_value=20, value=4)
    tickers_man = []
    valores_man = []

    cols = st.columns(2)
    for i in range(num_ativos):
        with cols[0]:
            ticker = st.text_input(f"Ticker {i+1}", key=f"ticker_{i}")
        with cols[1]:
            valor_str = st.text_input(f"Valor investido (R$) {i+1}", key=f"valor_{i}")
        
        tickers_man.append(ticker.strip().upper())

        try:
            valor = float(valor_str.replace(",", ".")) if valor_str else 0.0
        except ValueError:
            st.warning(f"Valor inválido no ativo {i+1}. Digite um número válido.")
            valor = 0.0

        valores_man.append(valor)

    # Filtra tickers não vazios e valores positivos
    tickers_man = [t for t, v in zip(tickers_man, valores_man) if t and v > 0]
    valores_man = [v for t, v in zip(tickers_man, valores_man) if t and v > 0]

    # Normaliza os tickers para garantir o formato correto (ex: PETR3 → PETR3.SA)
    tickers_man = normalizar_tickers(tickers_man)

    st.subheader("Carteira Otimizada")
    percentual_adicional = st.slider(
                    "Percentual adicional para otimização da carteira híbrida (%)",
                    min_value=0,
                    max_value=100,
                    value=30,
                    step=5
                ) / 100.0

    # Botão para iniciar a simulação:
    if st.button("Rodar simulação"):
    
        # Listas de ativos
        acoes_input = st.text_area("Lista de ações (separadas por vírgula)", value="ABEV3.SA, AGRO3.SA, BBAS3.SA, BBDC3.SA, BBSE3.SA, BMOB3.SA, BPAC11.SA, BRAV3.SA, BRBI11.SA, BRSR6.SA, CBAV3.SA, CGRA4.SA, CMIG4.SA, CPFE3.SA, CPLE6.SA, CSAN3.SA, CSMG3.SA, CSUD3.SA, CXSE3.SA, EGIE3.SA, ELET3.SA, ENEV3.SA, ENGI11.SA, EQTL3.SA, FLRY3.SA, GGBR4.SA, GRND3.SA, IRBR3.SA, ISAE4.SA, ITUB4.SA, JBSS3.SA, JHSF3.SA, KEPL3.SA, KLBN11.SA, NEOE3.SA, ODPV3.SA, PETR4.SA, PNVL3.SA, POMO4.SA, PSSA3.SA, PRIO3.SA, RANI3.SA, RECV3.SA, RENT3.SA, RNEW4.SA, SANB11.SA, SAPR4.SA, SBSP3.SA, SUZB3.SA, TAEE11.SA, TIMS3.SA, VALE3.SA, VIVR3.SA, VULC3.SA, WEGE3.SA, WIZC3.SA")
        acoes = normalizar_tickers([x.strip() for x in acoes_input.split(",")]) # acoes = [x.strip() for x in acoes.split(",")]

        fii_input = st.text_area("Lista de FIIs (separadas por vírgula)", value="BLMG11.SA, BRCO11.SA, KNIP11.SA, LVBI11.SA, MXRF11.SA, BRCR11.SA, BTLG11.SA, CNES11.SA, CPSH11.SA, GARE11.SA, HGLG11.SA, HGRU11.SA, HSML11.SA, PATL11.SA, RBRP11.SA, RBRR11.SA, XPML11.SA")
        fii = normalizar_tickers([x.strip() for x in fii_input.split(",")]) # fii = [x.strip() for x in fii.split(",")]

        # Filtra os tickers com base em um mínimo desejado de observações (por exemplo, 200)
        acoes_validos, acoes_problema = filtrar_tickers(prices_read, acoes, min_obs=200)
        fii_validos, fii_problema     = filtrar_tickers(prices_read, fii, min_obs=200)

        st.write("[LOG] Ações carregadas:", acoes_validos)
        if acoes_problema:
            st.warning(f"[LOG] Ações indisponíveis ou com poucos dados: {acoes_problema}")
        st.write("[LOG] FIIs carregados:", fii_validos)
        if fii_problema:
            st.warning(f"[LOG] FIIs indisponíveis ou com poucos dados: {fii_problema}")
        st.write("[LOG] Carregando o gráfico. Aguarde alguns minutos!")

        # Cria os DataFrames filtrados para as simulações
        prices_aco  = prices_read[acoes_validos]
        prices_fii  = prices_read[fii_validos]
        prices_comb = prices_read[acoes_validos + fii_validos]

        # Prepara dados
        rets_aco = np.log(prices_aco / prices_aco.shift(1)).dropna()
        mu_aco = rets_aco.mean() * 252
        cov_aco = rets_aco.cov() * 252

        rets_fii = np.log(prices_fii / prices_fii.shift(1)).dropna()
        mu_fii = rets_fii.mean() * 252
        cov_fii = rets_fii.cov() * 252

        rets_comb = np.log(prices_comb / prices_comb.shift(1)).dropna()
        mu_comb = rets_comb.mean() * 252
        cov_comb = rets_comb.cov() * 252

        if "simulacoes_realizadas" not in st.session_state:
            st.session_state.simulacoes_realizadas = False

        if not st.session_state.simulacoes_realizadas:
            # Simulação para ações
            st.session_state.sim_ret_aco, st.session_state.sim_vol_aco, st.session_state.sim_pesos_aco, st.session_state.ativos_aco = simulate_portfolios_cardinalidade_controlada(
                prices_aco,
                acoes_validos,
                n_sim,
                min_assets,
                max_assets,
                min_w,
                max_w,
                seed,
                alpha_dirichlet
            )

            # Simulação para FIIs
            st.session_state.sim_ret_fii, st.session_state.sim_vol_fii, st.session_state.sim_pesos_fii, st.session_state.ativos_fii = simulate_portfolios_cardinalidade_controlada(
                prices_fii,
                fii_validos,
                n_sim,
                min_assets,
                max_assets,
                min_w,
                max_w,
                seed,
                alpha_dirichlet
            )

            # Simulação para ações + FIIs
            st.session_state.sim_ret_comb, st.session_state.sim_vol_comb, st.session_state.sim_pesos_comb, st.session_state.ativos_comb = simulate_portfolios_cardinalidade_controlada(
                prices_comb,
                acoes_validos + fii_validos,
                n_sim,
                min_assets,
                max_assets,
                min_w,
                max_w,
                seed,
                alpha_dirichlet
            )

            st.session_state.simulacoes_realizadas = True

        # Recupera os dados simulados do session_state
        sim_ret_aco = st.session_state.sim_ret_aco
        sim_vol_aco = st.session_state.sim_vol_aco
        sim_pesos_aco = st.session_state.sim_pesos_aco
        ativos_aco = st.session_state.ativos_aco

        sim_ret_fii = st.session_state.sim_ret_fii
        sim_vol_fii = st.session_state.sim_vol_fii
        sim_pesos_fii = st.session_state.sim_pesos_fii
        ativos_fii = st.session_state.ativos_fii

        sim_ret_comb = st.session_state.sim_ret_comb
        sim_vol_comb = st.session_state.sim_vol_comb
        sim_pesos_comb = st.session_state.sim_pesos_comb
        ativos_comb = st.session_state.ativos_comb

        # Fronteiras eficientes via SLSQP
        ef_vol_aco_opt, ef_ret_aco_opt = compute_efficient_frontier(mu_aco.values, cov_aco.values)
        ef_vol_aco_opt, ef_ret_aco_opt = filtrar_fronteira_eficiente(ef_vol_aco_opt, ef_ret_aco_opt)

        ef_vol_fii_opt, ef_ret_fii_opt = compute_efficient_frontier(mu_fii.values, cov_fii.values)
        ef_vol_fii_opt, ef_ret_fii_opt = filtrar_fronteira_eficiente(ef_vol_fii_opt, ef_ret_fii_opt)

        ef_vol_comb_opt, ef_ret_comb_opt = compute_efficient_frontier(mu_comb.values, cov_comb.values)
        ef_vol_comb_opt, ef_ret_comb_opt = filtrar_fronteira_eficiente(ef_vol_comb_opt, ef_ret_comb_opt)

        # Carteiras de Sharpe máximo
        w_sharpe_aco, sharpe_aco = optimize_max_sharpe(mu_aco.values, cov_aco.values, 0.0, max_w)
        w_sharpe_aco = rebalance_weights(w_sharpe_aco, min_w)

        w_sharpe_fii, sharpe_fii = optimize_max_sharpe(mu_fii.values, cov_fii.values, 0.0, max_w)
        w_sharpe_fii = rebalance_weights(w_sharpe_fii, min_w)

        w_sharpe_comb, sharpe_comb = optimize_max_sharpe(mu_comb.values, cov_comb.values, 0.0, max_w)
        w_sharpe_comb = rebalance_weights(w_sharpe_comb, min_w)

        ret_aco = np.exp(portfolio_return(w_sharpe_aco, mu_aco.values)) - 1
        vol_aco = portfolio_volatility(w_sharpe_aco, cov_aco.values)

        ret_fii = np.exp(portfolio_return(w_sharpe_fii, mu_fii.values)) - 1
        vol_fii = portfolio_volatility(w_sharpe_fii, cov_fii.values)

        ret_comb = np.exp(portfolio_return(w_sharpe_comb, mu_comb.values)) - 1
        vol_comb = portfolio_volatility(w_sharpe_comb, cov_comb.values)

        # Verifica se há tickers da carteira manual que não estão em prices_comb
        tickers_faltando = [t for t in tickers_man if t not in prices_comb.columns]

        if tickers_faltando:
            st.warning(f"Buscando dados no Yahoo Finance para: {', '.join(tickers_faltando)}")
            try:
                novos_dados = yf.download(
                    tickers_faltando,
                    start=prices_comb.index.min(),
                    end=prices_comb.index.max()
                )['Close']

                if isinstance(novos_dados, pd.Series):
                    novos_dados = novos_dados.to_frame()
                    novos_dados.columns = tickers_faltando
                else:
                    novos_dados = novos_dados[tickers_faltando]

                prices_comb = pd.concat([prices_comb, novos_dados], axis=1)
                prices_comb = prices_comb.sort_index()
            except Exception as e:
                st.error(f"Erro ao buscar dados no Yahoo Finance: {e}")

                # Alinha datas e concatena com prices_comb
                prices_comb = pd.concat([prices_comb, novos_dados], axis=1)
                prices_comb = prices_comb.sort_index()
            except Exception as e:
                st.error(f"Erro ao buscar dados no Yahoo Finance: {e}")

        if tickers_man:
            pares = list(dict.fromkeys(
                zip(normalizar_tickers(tickers_man), valores_man)
            ))
            tickers_man, valores_man = zip(*pares)
            tickers_man, valores_man = list(tickers_man), list(valores_man)


            # 2) Filtra SÓ os tickers que realmente estão em prices_comb.columns
            disponiveis = set(prices_comb.columns)
            filt_tickers = []
            filt_valores = []
            for t, v in zip(tickers_man, valores_man):
                if t in disponiveis:
                    filt_tickers.append(t)
                    filt_valores.append(v)
                else:
                    st.warning(f"Sem dados para {t}, removido da carteira manual.")

            if not filt_tickers:
                st.error("Após o filtro, não sobrou nenhum ticker válido.")
                st.stop()

            tickers_man, valores_man = filt_tickers, filt_valores

            # Agora **tickers_man** e **valores_man** têm o mesmo comprimento e só contêm ativos válidos!
            # 4) Vai para os cálculos sem risco de index error

            total_man     = sum(valores_man)
            w_man         = np.array([v/total_man for v in valores_man])
            tickers_hibrida = []
            w_hibrida       = np.array([])     # array vazio por padrão
            ret_hibrida     = vol_hibrida = sharpe_hibrida = np.nan

            # Prepara DataFrame alinhado
            prices_manual = prices_comb[tickers_man].dropna()
            rets_manual   = np.log(prices_manual / prices_manual.shift(1)).dropna()

            mu_manual  = rets_manual.mean() * 252
            cov_manual = rets_manual.cov()  * 252

            mu_vec  = mu_manual.loc[tickers_man].values
            cov_mat = cov_manual.loc[tickers_man, tickers_man].values

            try:
                # Carteira manual original
                ret_man = np.exp(np.dot(w_man, mu_vec)) - 1
                vol_man = np.sqrt(np.dot(w_man.T, np.dot(cov_mat, w_man)))
                sharpe_man = ret_man / vol_man

                # Carteira manual otimizada
                w_opt_manual, sharpe_opt_manual = optimize_max_sharpe(mu_vec, cov_mat, min_w, max_w)
                w_opt_manual = rebalance_weights(w_opt_manual, min_w)
                ret_opt_manual = np.exp(np.dot(w_opt_manual, mu_vec)) - 1
                vol_opt_manual = np.sqrt(np.dot(w_opt_manual.T, np.dot(cov_mat, w_opt_manual)))

                # Carteira Híbrida – via SLSQP, mantendo pesos mínimos manuais e teto extra  
                tickers_hibrida, w_hibrida, ret_hibrida, vol_hibrida, sharpe_hibrida = \
                    otimizar_carteira_hibrida(
                        tickers_man,          # 1) lista de manuais
                        valores_man,          # 2) valores correspondentes
                        [],                   # 3) ativos_sugeridos → vazio faz a função escolher
                        mu_comb,              # 4) pd.Series de retornos combinados
                        cov_comb,             # 5) pd.DataFrame de covariâncias combinadas
                        percentual_adicional  # 6) float em [0,1]
                    )
                
            except Exception as e:
                st.error(f"Erro ao processar carteira manual: {e}")
                ret_man = vol_man = ret_opt_manual = vol_opt_manual = ret_hibrida = vol_hibrida = 0.0
        else:
            st.warning("Nenhum ticker válido foi inserido na carteira manual.")
            ret_man = vol_man = ret_opt_manual = vol_opt_manual = ret_hibrida = vol_hibrida = 0.0

        # Plotagem
        plot_results(
            sim_vol_aco, np.exp(sim_ret_aco) - 1, ef_vol_aco_opt, np.exp(ef_ret_aco_opt) - 1, vol_aco, ret_aco,
            sim_vol_fii, np.exp(sim_ret_fii) - 1, ef_vol_fii_opt, np.exp(ef_ret_fii_opt) - 1, vol_fii, ret_fii,
            sim_vol_comb, np.exp(sim_ret_comb) - 1, ef_vol_comb_opt, np.exp(ef_ret_comb_opt) - 1, vol_comb, ret_comb,
            vol_man, ret_man, vol_opt_manual, ret_opt_manual, vol_hibrida, ret_hibrida
        )

        st.subheader("Carteira de Sharpe Máximo – AÇÕES")
        serie_aco = pd.Series(w_sharpe_aco, index=acoes_validos)
        serie_aco = serie_aco[serie_aco > 0.001].sort_values(ascending=False)
        st.dataframe(serie_aco.apply(lambda x: f"{x:.2%}"))
        st.write(f"**Sharpe:** {sharpe_aco:.4f} | **Retorno:** {ret_aco:.2%} | **Volatilidade:** {vol_aco:.2%}")

        st.subheader("Carteira de Sharpe Máximo – FIIs")
        serie_fii = pd.Series(w_sharpe_fii, index=fii_validos)
        serie_fii = serie_fii[serie_fii > 0.001].sort_values(ascending=False)
        st.dataframe(serie_fii.apply(lambda x: f"{x:.2%}"))
        st.write(f"**Sharpe:** {sharpe_fii:.4f} | **Retorno:** {ret_fii:.2%} | **Volatilidade:** {vol_fii:.2%}")

        st.subheader("Carteira de Sharpe Máximo – AÇÕES E FIIs")
        tickers_comb = acoes_validos + fii_validos
        serie_comb = pd.Series(w_sharpe_comb, index=tickers_comb)
        serie_comb = serie_comb[serie_comb > 0.001].sort_values(ascending=False)
        st.dataframe(serie_comb.apply(lambda x: f"{x:.2%}"))
        st.write(f"**Sharpe:** {sharpe_comb:.4f} | **Retorno:** {ret_comb:.2%} | **Volatilidade:** {vol_comb:.2%}")

        # Composição por classe
        idx_acoes = [i for i, tk in enumerate(tickers_comb) if tk in acoes_validos]
        idx_fii = [i for i, tk in enumerate(tickers_comb) if tk in fii_validos]
        pct_acoes = w_sharpe_comb[idx_acoes].sum()
        pct_fii = w_sharpe_comb[idx_fii].sum()
        st.write(f"**Composição por classe:** Ações: {pct_acoes:.2%} | FIIs: {pct_fii:.2%}")

        st.subheader("Carteira Manual")
        serie_man = pd.Series(w_man, index=tickers_man)
        serie_man = serie_man[serie_man > 0.001].sort_values(ascending=False)
        st.dataframe(serie_man.apply(lambda x: f"{x:.2%}"))
        st.write(f"**Sharpe:** {sharpe_man:.4f} | **Retorno:** {ret_man:.2%} | **Volatilidade:** {vol_man:.2%}")

        st.subheader("Carteira Otimizada")
        serie_opt = pd.Series(w_opt_manual, index=tickers_man)
        serie_opt = serie_opt[serie_opt > 0.001].sort_values(ascending=False)
        st.dataframe(serie_opt.apply(lambda x: f"{x:.2%}"))
        st.write(f"**Sharpe:** {sharpe_opt_manual:.4f} | **Retorno:** {ret_opt_manual:.2%} | **Volatilidade:** {vol_opt_manual:.2%}")

        # Se a otimização retornou algo válido, exiba a tabela
        if w_hibrida.size:
            st.subheader(f"Carteira Híbrida Otimizada (com {int(percentual_adicional*100)}% adicionais)")
            serie_hibrida = pd.Series(w_hibrida, index=tickers_hibrida).sort_values(ascending=False)
            st.dataframe(serie_hibrida.apply(lambda x: f"{x:.2%}"))
            st.write(
                f"**Sharpe:** {sharpe_hibrida:.4f} | "
                f"**Retorno:** {ret_hibrida:.2%} | "
                f"**Volatilidade:** {vol_hibrida:.2%}"
            )
        else:
            st.warning("Não foi possível otimizar a carteira híbrida sob essas restrições.")
     
if __name__ == "__main__":
    main()