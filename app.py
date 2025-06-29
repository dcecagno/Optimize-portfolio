import os
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
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

    limiar = 1e-6
    ativos_nonzero = [(t, w) for t, w in zip(tickers_hibrida, w_hibrida) if w > limiar]
    if ativos_nonzero:
        tickers_hibrida, w_hibrida = zip(*ativos_nonzero)
    else:
        tickers_hibrida, w_hibrida = [], np.array([])

    return list(tickers_hibrida), np.array(w_hibrida), ret_h, vol_h, sharpe_h

# =======================
# Funções de Plotagem
# =======================

def plot_results(sim_vol_aco, sim_ret_aco, ef_vol_aco_opt, ef_ret_aco_opt, vol_aco, ret_aco,
                 sim_vol_fii, sim_ret_fii, ef_vol_fii_opt, ef_ret_fii_opt, vol_fii, ret_fii,
                 sim_vol_comb, sim_ret_comb, ef_vol_comb_opt, ef_ret_comb_opt, vol_comb, ret_comb,
                 vol_man, ret_man, vol_opt_manual, ret_opt_manual, vol_hibrida, ret_hibrida):
    plt.figure(figsize=(12,8))

    # Monte Carlo e Fronteiras
    plt.scatter(sim_vol_fii, sim_ret_fii, s=8, alpha=0.12, color='green', label='Simulações - FII')
    plt.plot(ef_vol_fii_opt, ef_ret_fii_opt, 'g-', lw=2, label='Fronteira Eficiente - FIIs')
    plt.scatter(vol_fii, ret_fii, color='green', marker='*', s=180, label='Sharpe Máx - FII')

    plt.scatter(sim_vol_aco, sim_ret_aco, s=8, alpha=0.12, color='blue', label='Simulações - Ações')
    plt.plot(ef_vol_aco_opt, ef_ret_aco_opt, 'b-', lw=2, label='Fronteira Eficiente - Ações')
    plt.scatter(vol_aco, ret_aco, color='blue', marker='*', s=180, label='Sharpe Máx - Ações')

    plt.scatter(sim_vol_comb, sim_ret_comb, s=8, alpha=0.12, color='red', label='Simulações - Ações + FII')
    plt.plot(ef_vol_comb_opt, ef_ret_comb_opt, 'r-', lw=2, label='Fronteira Eficiente - Ações + FII')
    plt.scatter(vol_comb, ret_comb, color='red', marker='*', s=180, label='Sharpe Máx - Ações + FII')

    # Carteiras manuais
    plt.scatter(vol_man, ret_man, c="black", s=80, marker="X", label="Carteira Manual")
    plt.scatter(vol_opt_manual, ret_opt_manual, c="orange", s=80, marker="D", label="Manual Otimizada")
    plt.scatter(vol_hibrida, ret_hibrida, c="purple", s=80, marker="P", label="Carteira Manual com Inclusão")

    plt.xlabel("Volatilidade Anualizada")
    plt.ylabel("Retorno Anualizado")
    plt.title("Fronteira Eficiente – Ações x FIIs x (Ações + FII)")
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    max_vol = max(sim_vol_aco.max(), sim_vol_fii.max(), sim_vol_comb.max())
    max_ret = max(sim_ret_aco.max(), sim_ret_fii.max(), sim_ret_comb.max())
    min_ret = min(sim_ret_aco.min(), sim_ret_fii.min(), sim_ret_comb.min())

    ax.set_xlim(0, max_vol * 1.15)
    ax.set_ylim(min_ret * 0.85, max_ret * 1.15)

    st.pyplot(plt)

def plot_correlation_heatmap(
    cov_df: pd.DataFrame,
    weights: np.ndarray,
    tickers: list[str],
    min_weight: float = 0.001,
    title: str = None,
    clean_suffix: str = ".SA",
    cbar: bool = False,         # <— mostrar colorbar?
    show_title: bool = False    # <— desenhar title no ax?

):
    """
    1) Monta Series de pesos indexada pelos tickers
    2) Filtra só os ativos com peso > min_weight
    3) Extrai sub-DataFrame de covariância
    4) Calcula correlação
    5) Ajusta figsize e fontsize dinamicamente e plota o heatmap
    """
    title = title or "Matriz de Correlação"

    # 1) Series de pesos
    serie_w = pd.Series(weights, index=tickers)

    # 2) Filtra ativos relevantes
    serie_sel = serie_w[serie_w > min_weight]
    tickers_sel = serie_sel.index.tolist()
    if not tickers_sel:
        st.warning(f"Nenhum ativo acima do limiar de {min_weight:.1%} em {title}.")
        return

    # 3) Sub-matriz de covariância
    cov_sub = cov_df.loc[tickers_sel, tickers_sel]

    # 4) Calcula a correlação
    std = np.sqrt(np.diag(cov_sub))
    corr_mat = cov_sub.values / np.outer(std, std)
    corr_df = pd.DataFrame(corr_mat, index=tickers_sel, columns=tickers_sel)

    # limpa sufixos nos rótulos, se houver
    if clean_suffix:
        labels = [t.replace(clean_suffix, "") for t in corr_df.columns]
        corr_df.index = labels
        corr_df.columns = labels

    # 5) Ajusta figure size e font size
    n = corr_df.shape[0]
    # Define um tamanho mínimo e cresce 0.5" por ativo
    fig_side = max(6, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))

    # Fonte entre 6pt e 20pt; quanto mais ativos, menor a fonte
    min_fs, max_fs = 6, 10
    font_size = int(max(min_fs, min(max_fs, 150 / n)))

    # 6) Desenha o heatmap
    sns.heatmap(
        corr_df,
        annot=corr_df.applymap(lambda x: f"{x:.0%}"),
        fmt="", 
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"fontsize": font_size},
        cbar=cbar             
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    ax.set_title(title)

    st.pyplot(fig)

def render_portfolio_section(
    name: str,
    weights: np.ndarray,
    tickers: list[str],
    cov_df: pd.DataFrame,
    sharpe: float,
    ret: float,
    vol: float,
    min_weight: float = 0.001
):
    """
    1) Cria a Series de participação
    2) Exibe subheader, dataframe e métricas
    3) Plota heatmap de correlação só para ativos com peso > min_weight
    """
    # 1) Série de participação
    serie = (
        pd.Series(weights, index=tickers)
          .loc[lambda s: s > min_weight]
          .sort_values(ascending=False)
          .rename(index=lambda x: x.replace(".SA", ""))
          .rename_axis(index="Ticker")
          .rename("Participação")
    )
    if serie.empty:
        st.warning(f"Nenhum ativo acima de {min_weight:.1%} em {name}.")
        return

    # 2) Tabela + métricas
    st.markdown(
        f"<h3 style='text-align: center;'>{name}</h3>",
        unsafe_allow_html=True
    )

    st.dataframe(serie.apply(lambda x: f"{x:.2%}"), use_container_width=True)
    st.write(
        f"**Sharpe:** {sharpe:.2f} | "
        f"**Retorno:** {ret:.2%} | "
        f"**Volatilidade:** {vol:.2%}"
    )

    # 3) Heatmap
    #st.subheader(f"Matriz de Correlação — {name}")
    plot_correlation_heatmap(
        cov_df=cov_df,
        weights=weights,
        tickers=tickers,
        min_weight=min_weight,
        cbar        = False,
        show_title  = False
    )

# =======================
# Bloco Principal
# =======================

def main():
    st.title("Simulação de Carteiras e Fronteira Eficiente: v19")
    # Upload do arquivo CSV
    url = "https://raw.githubusercontent.com/dcecagno/Optimize-portfolio/main/all_precos.csv"
    prices_read = _read_close_prices(url)

    # Dicionários de classificação
    SECTOR_MAP_ACOES = {
        'AERI3.SA': 'Bens industriais',
        'AGRX11.SA': 'Bens industriais',
        'ALLD3.SA': 'Bens industriais',
        'ALPK3.SA': 'Bens industriais',
        'AMBP3.SA': 'Bens industriais',
        'AMOB3.SA': 'Bens industriais',
        'ARML3.SA': 'Bens industriais',
        'ASAI3.SA': 'Bens industriais',
        'ATMP3.SA': 'Bens industriais',
        'AVLL3.SA': 'Bens industriais',
        'BBGO11.SA': 'Bens industriais',
        'BHIA3.SA': 'Bens industriais',
        'BLAU3.SA': 'Bens industriais',
        'BMLC11.SA': 'Bens industriais',
        'BMOB3.SA': 'Bens industriais',
        'BRBI11.SA': 'Bens industriais',
        'CASH3.SA': 'Bens industriais',
        'CBAV3.SA': 'Bens industriais',
        'CJCT11.SA': 'Bens industriais',
        'CMIN3.SA': 'Bens industriais',
        'CPTR11.SA': 'Bens industriais',
        'CRAA11.SA': 'Bens industriais',
        'CSED3.SA': 'Bens industriais',
        'CURY3.SA': 'Bens industriais',
        'CXSE3.SA': 'Bens industriais',
        'DEVA11.SA': 'Bens industriais',
        'DMVF3.SA': 'Bens industriais',
        'DOTZ3.SA': 'Bens industriais',
        'EGAF11.SA': 'Bens industriais',
        'ELMD3.SA': 'Bens industriais',
        'ENJU3.SA': 'Bens industriais',
        'ESPA3.SA': 'Bens industriais',
        'FGAA11.SA': 'Bens industriais',
        'FRAS3.SA': 'Bens industriais',
        'GCRA11.SA': 'Bens industriais',
        'GGPS3.SA': 'Bens industriais',
        'GMAT3.SA': 'Bens industriais',
        'GOLL3.SA': 'Bens industriais',
        'GRWA11.SA': 'Bens industriais',
        'HBSA3.SA': 'Bens industriais',
        'HGAG11.SA': 'Bens industriais',
        'HODL11.SA': 'Bens industriais',
        'INTB3.SA': 'Bens industriais',
        'ISAE4.SA': 'Bens industriais',
        'JALL3.SA': 'Bens industriais',
        'JSLG3.SA': 'Bens industriais',
        'KEPL3.SA': 'Bens industriais',
        'KNCA11.SA': 'Bens industriais',
        'KRSA3.SA': 'Bens industriais',
        'LAVV3.SA': 'Bens industriais',
        'LJQQ3.SA': 'Bens industriais',
        'LSAG11.SA': 'Bens industriais',
        'MATD3.SA': 'Bens industriais',
        'MBLY3.SA': 'Bens industriais',
        'MELK3.SA': 'Bens industriais',
        'MLAS3.SA': 'Bens industriais',
        'MTRE3.SA': 'Bens industriais',
        'NGRD3.SA': 'Bens industriais',
        'OPCT3.SA': 'Bens industriais',
        'ORVR3.SA': 'Bens industriais',
        'PETZ3.SA': 'Bens industriais',
        'PGMN3.SA': 'Bens industriais',
        'PLCA11.SA': 'Bens industriais',
        'PLPL3.SA': 'Bens industriais',
        'POMO4.SA': 'Bens industriais',
        'RECR11.SA': 'Bens industriais',
        'RURA11.SA': 'Bens industriais',
        'RZAT11.SA': 'Bens industriais',
        'SEQL3.SA': 'Bens industriais',
        'SNAG11.SA': 'Bens industriais',
        'TFCO4.SA': 'Bens industriais',
        'TTEN3.SA': 'Bens industriais',
        'URPR11.SA': 'Bens industriais',
        'VAMO3.SA': 'Bens industriais',
        'VCRA11.SA': 'Bens industriais',
        'VGIA11.SA': 'Bens industriais',
        'VITT3.SA': 'Bens industriais',
        'WEGE3.SA': 'Bens industriais',
        'WEST3.SA': 'Bens industriais',
        'ZAMP3.SA': 'Bens industriais',
        'ALOS3.SA': 'Consumo cíclico',
        'AMAR3.SA': 'Consumo cíclico',
        'AZZA3.SA': 'Consumo cíclico',
        'CEAB3.SA': 'Consumo cíclico',
        'CEDO4.SA': 'Consumo cíclico',
        'CGRA4.SA': 'Consumo cíclico',
        'COGN3.SA': 'Consumo cíclico',
        'CTKA4.SA': 'Consumo cíclico',
        'CTNM3.SA': 'Consumo cíclico',
        'CVCB3.SA': 'Consumo cíclico',
        'CYRE3.SA': 'Consumo cíclico',
        'DIRR3.SA': 'Consumo cíclico',
        'ESTR3.SA': 'Consumo cíclico',
        'ESTR4.SA': 'Consumo cíclico',
        'EVEN3.SA': 'Consumo cíclico',
        'EZTC3.SA': 'Consumo cíclico',
        'GFSA3.SA': 'Consumo cíclico',
        'GRND3.SA': 'Consumo cíclico',
        'HBOR3.SA': 'Consumo cíclico',
        'HOOT4.SA': 'Consumo cíclico',
        'JHSF3.SA': 'Consumo cíclico',
        'LEVE3.SA': 'Consumo cíclico',
        'LREN3.SA': 'Consumo cíclico',
        'MGLU3.SA': 'Consumo cíclico',
        'MNDL3.SA': 'Consumo cíclico',
        'MOVI3.SA': 'Consumo cíclico',
        'MRVE3.SA': 'Consumo cíclico',
        'MYPK3.SA': 'Consumo cíclico',
        'PDGR3.SA': 'Consumo cíclico',
        'RDNI3.SA': 'Consumo cíclico',
        'RENT3.SA': 'Consumo cíclico',
        'RSID3.SA': 'Consumo cíclico',
        'SBFG3.SA': 'Consumo cíclico',
        'SEER3.SA': 'Consumo cíclico',
        'SLED3.SA': 'Consumo cíclico',
        'SLED4.SA': 'Consumo cíclico',
        'TCSA3.SA': 'Consumo cíclico',
        'VIVR3.SA': 'Consumo cíclico',
        'VSTE3.SA': 'Consumo cíclico',
        'VULC3.SA': 'Consumo cíclico',
        'WHRL4.SA': 'Consumo cíclico',
        'YDUQ3.SA': 'Consumo cíclico',
        'ABEV3.SA': 'Consumo não cíclico',
        'AGRO3.SA': 'Consumo não cíclico',
        'BEEF3.SA': 'Consumo não cíclico',
        'BRFS3.SA': 'Consumo não cíclico',
        'CAML3.SA': 'Consumo não cíclico',
        'CRFB3.SA': 'Consumo não cíclico',
        'JBSS3.SA': 'Consumo não cíclico',
        'MDIA3.SA': 'Consumo não cíclico',
        'MRFG3.SA': 'Consumo não cíclico',
        'NTCO3.SA': 'Consumo não cíclico',
        'PCAR3.SA': 'Consumo não cíclico',
        'SMTO3.SA': 'Consumo não cíclico',
        'VIVA3.SA': 'Consumo não cíclico',
        'B3SA3.SA': 'Financeiro',
        'BBAS3.SA': 'Financeiro',
        'BBDC3.SA': 'Financeiro',
        'BBDC4.SA': 'Financeiro',
        'BBSE3.SA': 'Financeiro',
        'BMGB4.SA': 'Financeiro',
        'BMIN3.SA': 'Financeiro',
        'BPAC11.SA': 'Financeiro',
        'BPAN4.SA': 'Financeiro',
        'BPAR3.SA': 'Financeiro',
        'BRPR3.SA': 'Financeiro',
        'BRSR6.SA': 'Financeiro',
        'BSLI3.SA': 'Financeiro',
        'CSUD3.SA': 'Financeiro',
        'GSHP3.SA': 'Financeiro',
        'IGTI11.SA': 'Financeiro',
        'IRBR3.SA': 'Financeiro',
        'ITSA4.SA': 'Financeiro',
        'ITUB4.SA': 'Financeiro',
        'LOGG3.SA': 'Financeiro',
        'LPSB3.SA': 'Financeiro',
        'MERC4.SA': 'Financeiro',
        'MULT3.SA': 'Financeiro',
        'NDIV11.SA': 'Financeiro',
        'PDTC3.SA': 'Financeiro',
        'PSSA3.SA': 'Financeiro',
        'SANB11.SA': 'Financeiro',
        'SCAR3.SA': 'Financeiro',
        'SYNE3.SA': 'Financeiro',
        'TRAD3.SA': 'Financeiro',
        'WIZC3.SA': 'Financeiro',
        'AZTE3.SA': 'Materiais básicos',
        'BRAP4.SA': 'Materiais básicos',
        'BRKM5.SA': 'Materiais básicos',
        'CSNA3.SA': 'Materiais básicos',
        'DEXP3.SA': 'Materiais básicos',
        'DXCO3.SA': 'Materiais básicos',
        'EUCA4.SA': 'Materiais básicos',
        'FESA4.SA': 'Materiais básicos',
        'GGBR4.SA': 'Materiais básicos',
        'GOAU4.SA': 'Materiais básicos',
        'KLBN11.SA': 'Materiais básicos',
        'LAND3.SA': 'Materiais básicos',
        'MMXM11.SA': 'Materiais básicos',
        'NEMO3.SA': 'Materiais básicos',
        'PMAM3.SA': 'Materiais básicos',
        'RANI3.SA': 'Materiais básicos',
        'SUZB3.SA': 'Materiais básicos',
        'UNIP6.SA': 'Materiais básicos',
        'VALE3.SA': 'Materiais básicos',
        'AFHI11.SA': 'Outros',
        'AGXY3.SA': 'Outros',
        'ALUG11.SA': 'Outros',
        'BDOM11.SA': 'Outros',
        'BIME11.SA': 'Outros',
        'CACR11.SA': 'Outros',
        'CRPG5.SA': 'Outros',
        'CXAG11.SA': 'Outros',
        'CYCR11.SA': 'Outros',
        'EQIR11.SA': 'Outros',
        'GTLG11.SA': 'Outros',
        'HSRE11.SA': 'Outros',
        'HUSI11.SA': 'Outros',
        'JGPX11.SA': 'Outros',
        'JSAF11.SA': 'Outros',
        'MORC11.SA': 'Outros',
        'PORT3.SA': 'Outros',
        'PPLA11.SA': 'Outros',
        'PURB11.SA': 'Outros',
        'ROOF11.SA': 'Outros',
        'RZAG11.SA': 'Outros',
        'SMAB11.SA': 'Outros',
        'SMFT3.SA': 'Outros',
        'SOJA3.SA': 'Outros',
        'SPXB11.SA': 'Outros',
        'USTK11.SA': 'Outros',
        'VTRU3.SA': 'Outros',
        'WRLD11.SA': 'Outros',
        'YDRO11.SA': 'Outros',
        'CSAN3.SA': 'Petróleo, Gás e Biocombustíveis',
        'OSXB3.SA': 'Petróleo, Gás e Biocombustíveis',
        'PETR4.SA': 'Petróleo, Gás e Biocombustíveis',
        'PRIO3.SA': 'Petróleo, Gás e Biocombustíveis',
        'RAIZ4.SA': 'Petróleo, Gás e Biocombustíveis',
        'RECV3.SA': 'Petróleo, Gás e Biocombustíveis',
        'RPMG3.SA': 'Petróleo, Gás e Biocombustíveis',
        'SRNA3.SA': 'Petróleo, Gás e Biocombustíveis',
        'UGPA3.SA': 'Petróleo, Gás e Biocombustíveis',
        'VBBR3.SA': 'Petróleo, Gás e Biocombustíveis',
        'AALR3.SA': 'Saúde',
        'BALM4.SA': 'Saúde',
        'BIOM3.SA': 'Saúde',
        'FLRY3.SA': 'Saúde',
        'HYPE3.SA': 'Saúde',
        'ODPV3.SA': 'Saúde',
        'OFSA3.SA': 'Saúde',
        'ONCO3.SA': 'Saúde',
        'PFRM3.SA': 'Saúde',
        'PNVL3.SA': 'Saúde',
        'QUAL3.SA': 'Saúde',
        'RADL3.SA': 'Saúde',
        'VVEO3.SA': 'Saúde',
        'IFCM3.SA': 'Tecnologia da Informação',
        'LVTC3.SA': 'Tecnologia da Informação',
        'LWSA3.SA': 'Tecnologia da Informação',
        'POSI3.SA': 'Tecnologia da Informação',
        'TOTS3.SA': 'Tecnologia da Informação',
        'DESK3.SA': 'Telecomunicações',
        'FIQE3.SA': 'Telecomunicações',
        'OIBR3.SA': 'Telecomunicações',
        'TELB3.SA': 'Telecomunicações',
        'TIMS3.SA': 'Telecomunicações',
        'VIVT3.SA': 'Telecomunicações',
        'AFLT3.SA':  'Utilidade pública',
        'ALUP11.SA': 'Utilidade pública',
        'AURE3.SA':  'Utilidade pública',
        'BRAV3.SA':  'Utilidade pública',
        'CASN3.SA':  'Utilidade pública',
        'CASN4.SA':  'Utilidade pública',
        'CEBR6.SA':  'Utilidade pública',
        'CEED3.SA':  'Utilidade pública',
        'CEGR3.SA':  'Utilidade pública',
        'CGAS5.SA':  'Utilidade pública',
        'CLSC4.SA':  'Utilidade pública',
        'CMIG4.SA':  'Utilidade pública',
        'COCE5.SA':  'Utilidade pública',
        'CPFE3.SA':  'Utilidade pública',
        'CPLE6.SA':  'Utilidade pública',
        'CSMG3.SA':  'Utilidade pública',
        'EGIE3.SA':  'Utilidade pública',
        'ELET3.SA':  'Utilidade pública',
        'ENEV3.SA':  'Utilidade pública',
        'ENGI11.SA':  'Utilidade pública',
        'EQTL3.SA':  'Utilidade pública',
        'GEPA4.SA':  'Utilidade pública',
        'NEOE3.SA':  'Utilidade pública',
        'RNEW4.SA':  'Utilidade pública',
        'SAPR4.SA': 'Utilidade pública',
        'SAPR11.SA': 'Utilidade pública',
        'SBSP3.SA':  'Utilidade pública',
        'TAEE11.SA': 'Utilidade pública' 
    }

    SECTOR_MAP_FII = {
        'BBPO11.SA': 'Agências de Bancos',
        'BBRC11.SA': 'Agências de Bancos',
        'BNFS11.SA': 'Agências de Bancos',
        'CXAG11.SA': 'Agências de Bancos',
        'TVRI11.SA': 'Agências de Bancos',
        'NEXG11.SA': 'Agricultura',
        'FAED11.SA': 'Educacional',
        'FCFL11.SA': 'Educacional',
        'RBED11.SA': 'Educacional',
        'ZIFI11.SA': 'Fundo de Desenvolvimento',
        'BRIM11.SA': 'Fundo de Desenvolvimento',
        'BRIP11.SA': 'Fundo de Desenvolvimento',
        'BTWR11.SA': 'Fundo de Desenvolvimento',
        'CFHI11.SA': 'Fundo de Desenvolvimento',
        'FLFL11.SA': 'Fundo de Desenvolvimento',
        'HRDF11.SA': 'Fundo de Desenvolvimento',
        'INRD11.SA': 'Fundo de Desenvolvimento',
        'KEVE11.SA': 'Fundo de Desenvolvimento',
        'KINP11.SA': 'Fundo de Desenvolvimento',
        'KNRE11.SA': 'Fundo de Desenvolvimento',
        'LOFT11B.SA': 'Fundo de Desenvolvimento',
        'MFII11.SA': 'Fundo de Desenvolvimento',
        'PABY11.SA': 'Fundo de Desenvolvimento',
        'PATC11.SA': 'Fundo de Desenvolvimento',
        'PNDL11.SA': 'Fundo de Desenvolvimento',
        'PNPR11.SA': 'Fundo de Desenvolvimento',
        'RBDS11.SA': 'Fundo de Desenvolvimento',
        'RBIR11.SA': 'Fundo de Desenvolvimento',
        'RBRI11.SA': 'Fundo de Desenvolvimento',
        'RBRM11.SA': 'Fundo de Desenvolvimento',
        'RBRS11.SA': 'Fundo de Desenvolvimento',
        'RBTS11.SA': 'Fundo de Desenvolvimento',
        'ROOF11.SA': 'Fundo de Desenvolvimento',
        'RSPD11.SA': 'Fundo de Desenvolvimento',
        'SNEL11.SA': 'Fundo de Desenvolvimento',
        'STRX11.SA': 'Fundo de Desenvolvimento',
        'TGAR11.SA': 'Fundo de Desenvolvimento',
        'TRXB11.SA': 'Fundo de Desenvolvimento',
        'VXXV11.SA': 'Fundo de Desenvolvimento',
        'YUFI11.SA': 'Fundo de Desenvolvimento',
        'ALZM11.SA': 'Fundo de Fundos',
        'BBFO11.SA': 'Fundo de Fundos',
        'BCIA11.SA': 'Fundo de Fundos',
        'BICE11.SA': 'Fundo de Fundos',
        'BLMR11.SA': 'Fundo de Fundos',
        'BPFF11.SA': 'Fundo de Fundos',
        'CLIN11.SA': 'Fundo de Fundos',
        'CPFF11.SA': 'Fundo de Fundos',
        'CRFF11.SA': 'Fundo de Fundos',
        'CXRI11.SA': 'Fundo de Fundos',
        'DVFF11.SA': 'Fundo de Fundos',
        'GCFF11.SA': 'Fundo de Fundos',
        'HFOF11.SA': 'Fundo de Fundos',
        'HGFF11.SA': 'Fundo de Fundos',
        'IBFF11.SA': 'Fundo de Fundos',
        'ITIT11.SA': 'Fundo de Fundos',
        'JCIN11.SA': 'Fundo de Fundos',
        'JSAF11.SA': 'Fundo de Fundos',
        'KFOF11.SA': 'Fundo de Fundos',
        'KISU11.SA': 'Fundo de Fundos',
        'MORE11.SA': 'Fundo de Fundos',
        'OUFF11.SA': 'Fundo de Fundos',
        'RBFF11.SA': 'Fundo de Fundos',
        'RBRF11.SA': 'Fundo de Fundos',
        'RCFF11.SA': 'Fundo de Fundos',
        'RECX11.SA': 'Fundo de Fundos',
        'RFOF11.SA': 'Fundo de Fundos',
        'RVBI11.SA': 'Fundo de Fundos',
        'SNFF11.SA': 'Fundo de Fundos',
        'TMPS11.SA': 'Fundo de Fundos',
        'VIFI11.SA': 'Fundo de Fundos',
        'XPSF11.SA': 'Fundo de Fundos',
        'HCRI11.SA': 'Hospitalar',
        'HUCG11.SA': 'Hospitalar',
        'HUSC11.SA': 'Hospitalar',
        'HUSI11.SA': 'Hospitalar',
        'NSLU11.SA': 'Hospitalar',
        'NVHO11.SA': 'Hospitalar',
        'BTHI11.SA': 'Hotéis',
        'HTMX11.SA': 'Hotéis',
        'MGHT11.SA': 'Hotéis',
        'XPHT11.SA': 'Hotéis',
        'AURB11.SA': 'Imóveis Comerciais - Outros',
        'JASC11.SA': 'Imóveis Comerciais - Outros',
        'LIFE11.SA': 'Imóveis Comerciais - Outros',
        'SOLR11.SA': 'Imóveis Comerciais - Outros',
        'TOPP11.SA': 'Imóveis Comerciais - Outros',
        'TRXF11.SA': 'Imóveis Comerciais - Outros',
        'AROA11.SA': 'Imóveis Industriais e Logísticos',
        'BLCP11.SA': 'Imóveis Industriais e Logísticos',
        'BLMG11.SA': 'Imóveis Industriais e Logísticos',
        'BRCO11.SA': 'Imóveis Industriais e Logísticos',
        'BTAL11.SA': 'Imóveis Industriais e Logísticos',
        'BTLG11.SA': 'Imóveis Industriais e Logísticos',
        'BTSG11.SA': 'Imóveis Industriais e Logísticos',
        'CXTL11.SA': 'Imóveis Industriais e Logísticos',
        'EURO11.SA': 'Imóveis Industriais e Logísticos',
        'FIIB11.SA': 'Imóveis Industriais e Logísticos',
        'GARE11.SA': 'Imóveis Industriais e Logísticos',
        'GGRC11.SA': 'Imóveis Industriais e Logísticos',
        'GLOG11.SA': 'Imóveis Industriais e Logísticos',
        'GTLG11.SA': 'Imóveis Industriais e Logísticos',
        'HDEL11.SA': 'Imóveis Industriais e Logísticos',
        'HGLG11.SA': 'Imóveis Industriais e Logísticos',
        'HLOG11.SA': 'Imóveis Industriais e Logísticos',
        'HSLG11.SA': 'Imóveis Industriais e Logísticos',
        'INLG11.SA': 'Imóveis Industriais e Logísticos',
        'LVBI11.SA': 'Imóveis Industriais e Logísticos',
        'NEWL11.SA': 'Imóveis Industriais e Logísticos',
        'OULG11.SA': 'Imóveis Industriais e Logísticos',
        'PATL11.SA': 'Imóveis Industriais e Logísticos',
        'PQAG11.SA': 'Imóveis Industriais e Logísticos',
        'RBLG11.SA': 'Imóveis Industriais e Logísticos',
        'RBRL11.SA': 'Imóveis Industriais e Logísticos',
        'RELG11.SA': 'Imóveis Industriais e Logísticos',
        'RZAT11.SA': 'Imóveis Industriais e Logísticos',
        'RZZR11.SA': 'Imóveis Industriais e Logísticos',
        'SDIL11.SA': 'Imóveis Industriais e Logísticos',
        'SJAU11.SA': 'Imóveis Industriais e Logísticos',
        'SNLG11.SA': 'Imóveis Industriais e Logísticos',
        'TRBL11.SA': 'Imóveis Industriais e Logísticos',
        'VILG11.SA': 'Imóveis Industriais e Logísticos',
        'VTLT11.SA': 'Imóveis Industriais e Logísticos',
        'XPIN11.SA': 'Imóveis Industriais e Logísticos',
        'XPLG11.SA': 'Imóveis Industriais e Logísticos',
        'APTO11.SA': 'Imóveis Residenciais',
        'HOSI11.SA': 'Imóveis Residenciais',
        'HRES11.SA': 'Imóveis Residenciais',
        'JFLL11.SA': 'Imóveis Residenciais',
        'LTMT11.SA': 'Imóveis Residenciais',
        'OBAL11.SA': 'Imóveis Residenciais',
        'PNCR11.SA': 'Imóveis Residenciais',
        'VCRR11.SA': 'Imóveis Residenciais',
        'BTYU11.SA': 'Incorporações',
        'SMRE11.SA': 'Incorporações',
        'AAGR11.SA': 'Indefinido',
        'AAZQ11.SA': 'Indefinido',
        'AGRX11.SA': 'Indefinido',
        'ASRF11.SA': 'Indefinido',
        'BBGO11.SA': 'Indefinido',
        'BDIF11.SA': 'Indefinido',
        'BDIV11.SA': 'Indefinido',
        'BIDB11.SA': 'Indefinido',
        'BIME11.SA': 'Indefinido',
        'BIPD11.SA': 'Indefinido',
        'BLCA11.SA': 'Indefinido',
        'BODB11.SA': 'Indefinido',
        'BTAG11.SA': 'Indefinido',
        'BTRA11.SA': 'Indefinido',
        'CCME11.SA': 'Indefinido',
        'CCVA11.SA': 'Indefinido',
        'CDII11.SA': 'Indefinido',
        'CFII11.SA': 'Indefinido',
        'CPSH11.SA': 'Indefinido',
        'CPTI11.SA': 'Indefinido',
        'CPTR11.SA': 'Indefinido',
        'CRAA11.SA': 'Indefinido',
        'CXCI11.SA': 'Indefinido',
        'CYCR11.SA': 'Indefinido',
        'DCRA11.SA': 'Indefinido',
        'DPRO11.SA': 'Indefinido',
        'EGAF11.SA': 'Indefinido',
        'ENDD11.SA': 'Indefinido',
        'EQIR11.SA': 'Indefinido',
        'ESUD11.SA': 'Indefinido',
        'ESUT11.SA': 'Indefinido',
        'ESUU11.SA': 'Indefinido',
        'EXES11.SA': 'Indefinido',
        'FGAA11.SA': 'Indefinido',
        'FPOR11.SA': 'Indefinido',
        'FTCA11.SA': 'Indefinido',
        'FZDA11.SA': 'Indefinido',
        'FZDB11.SA': 'Indefinido',
        'GAME11.SA': 'Indefinido',
        'GCOI11.SA': 'Indefinido',
        'GCRA11.SA': 'Indefinido',
        'GRWA11.SA': 'Indefinido',
        'GZIT11.SA': 'Indefinido',
        'HBCR11.SA': 'Indefinido',
        'HCRA11.SA': 'Indefinido',
        'HGAG11.SA': 'Indefinido',
        'HILG11.SA': 'Indefinido',
        'IAAG11.SA': 'Indefinido',
        'IAGR11.SA': 'Indefinido',
        'IBBP11.SA': 'Indefinido',
        'IDFI11.SA': 'Indefinido',
        'IFRA11.SA': 'Indefinido',
        'INFB11.SA': 'Indefinido',
        'IRIM11.SA': 'Indefinido',
        'JGPX11.SA': 'Indefinido',
        'JURO11.SA': 'Indefinido',
        'KCRE11.SA': 'Indefinido',
        'KDIF11.SA': 'Indefinido',
        'KFEN11.SA': 'Indefinido',
        'KNCA11.SA': 'Indefinido',
        'KNHF11.SA': 'Indefinido',
        'KNOX11.SA': 'Indefinido',
        'KNUQ11.SA': 'Indefinido',
        'KOPA11.SA': 'Indefinido',
        'LPLP11.SA': 'Indefinido',
        'LRDI11.SA': 'Indefinido',
        'LSAG11.SA': 'Indefinido',
        'MANA11.SA': 'Indefinido',
        'MATV11.SA': 'Indefinido',
        'MMPD11.SA': 'Indefinido',
        'NUIF11.SA': 'Indefinido',
        'NVRP11.SA': 'Indefinido',
        'OGHY11.SA': 'Indefinido',
        'OIAG11.SA': 'Indefinido',
        'PFIN11.SA': 'Indefinido',
        'PICE11.SA': 'Indefinido',
        'PLAG11.SA': 'Indefinido',
        'PLCA11.SA': 'Indefinido',
        'PNRC11.SA': 'Indefinido',
        'PPEI11.SA': 'Indefinido',
        'PRIF11.SA': 'Indefinido',
        'RBIF11.SA': 'Indefinido',
        'RBRX11.SA': 'Indefinido',
        'RIFF11.SA': 'Indefinido',
        'RINV11.SA': 'Indefinido',
        'RURA11.SA': 'Indefinido',
        'RZAG11.SA': 'Indefinido',
        'RZEO11.SA': 'Indefinido',
        'SEED11.SA': 'Indefinido',
        'SNID11.SA': 'Indefinido',
        'SNME11.SA': 'Indefinido',
        'SPDE11.SA': 'Indefinido',
        'SPMO11.SA': 'Indefinido',
        'SPXS11.SA': 'Indefinido',
        'SRVD11.SA': 'Indefinido',
        'TELM11.SA': 'Indefinido',
        'TJKB11.SA': 'Indefinido',
        'VANG11.SA': 'Indefinido',
        'VCRA11.SA': 'Indefinido',
        'VGIA11.SA': 'Indefinido',
        'VIGT11.SA': 'Indefinido',
        'VVRI11.SA': 'Indefinido',
        'WHGR11.SA': 'Indefinido',
        'WSEC11.SA': 'Indefinido',
        'XPCA11.SA': 'Indefinido',
        'XPID11.SA': 'Indefinido',
        'XPIE11.SA': 'Indefinido',
        'ZAVC11.SA': 'Indefinido',
        'ZAVI11.SA': 'Indefinido',
        'AIEC11.SA': 'Lajes Corporativas',
        'ALMI11.SA': 'Lajes Corporativas',
        'ASMT11.SA': 'Lajes Corporativas',
        'BLMO11.SA': 'Lajes Corporativas',
        'BMLC11.SA': 'Lajes Corporativas',
        'BRCR11.SA': 'Lajes Corporativas',
        'BREV11.SA': 'Lajes Corporativas',
        'BROF11.SA': 'Lajes Corporativas',
        'BTML11.SA': 'Lajes Corporativas',
        'CBOP11.SA': 'Lajes Corporativas',
        'CEOC11.SA': 'Lajes Corporativas',
        'CJCT11.SA': 'Lajes Corporativas',
        'CNES11.SA': 'Lajes Corporativas',
        'CTXT11.SA': 'Lajes Corporativas',
        'CXCO11.SA': 'Lajes Corporativas',
        'EDGA11.SA': 'Lajes Corporativas',
        'ERPA11.SA': 'Lajes Corporativas',
        'FATN11.SA': 'Lajes Corporativas',
        'FISC11.SA': 'Lajes Corporativas',
        'FLMA11.SA': 'Lajes Corporativas',
        'FMOF11.SA': 'Lajes Corporativas',
        'FPAB11.SA': 'Lajes Corporativas',
        'GTWR11.SA': 'Lajes Corporativas',
        'HAAA11.SA': 'Lajes Corporativas',
        'HGPO11.SA': 'Lajes Corporativas',
        'HGRE11.SA': 'Lajes Corporativas',
        'HOFC11.SA': 'Lajes Corporativas',
        'KORE11.SA': 'Lajes Corporativas',
        'MBRF11.SA': 'Lajes Corporativas',
        'NEWU11.SA': 'Lajes Corporativas',
        'ONEF11.SA': 'Lajes Corporativas',
        'PRSV11.SA': 'Lajes Corporativas',
        'PVBI11.SA': 'Lajes Corporativas',
        'RBCO11.SA': 'Lajes Corporativas',
        'RBOP11.SA': 'Lajes Corporativas',
        'RBRP11.SA': 'Lajes Corporativas',
        'RCRB11.SA': 'Lajes Corporativas',
        'RECT11.SA': 'Lajes Corporativas',
        'RMAI11.SA': 'Lajes Corporativas',
        'RNGO11.SA': 'Lajes Corporativas',
        'SPTW11.SA': 'Lajes Corporativas',
        'TEPP11.SA': 'Lajes Corporativas',
        'TRNT11.SA': 'Lajes Corporativas',
        'TSER11.SA': 'Lajes Corporativas',
        'VINO11.SA': 'Lajes Corporativas',
        'VPPR11.SA': 'Lajes Corporativas',
        'VVCO11.SA': 'Lajes Corporativas',
        'VVMR11.SA': 'Lajes Corporativas',
        'XPCM11.SA': 'Lajes Corporativas',
        'GRUL11.SA': 'Logística',
        'ALZR11.SA': 'Misto',
        'BLOG11.SA': 'Misto',
        'BRFT11.SA': 'Misto',
        'BTSI11.SA': 'Misto',
        'CARE11.SA': 'Misto',
        'CPLG11.SA': 'Misto',
        'CPOF11.SA': 'Misto',
        'DAMA11.SA': 'Misto',
        'HGBL11.SA': 'Misto',
        'HGRU11.SA': 'Misto',
        'HSRE11.SA': 'Misto',
        'ICRI11.SA': 'Misto',
        'IDGR11.SA': 'Misto',
        'IFRI11.SA': 'Misto',
        'JSRE11.SA': 'Misto',
        'KDOL11.SA': 'Misto',
        'KNRI11.SA': 'Misto',
        'LAFI11.SA': 'Misto',
        'LLAO11.SA': 'Misto',
        'OGIN11.SA': 'Misto',
        'OURE11.SA': 'Misto',
        'PATA11.SA': 'Misto',
        'PMFO11.SA': 'Misto',
        'PMIS11.SA': 'Misto',
        'RZTR11.SA': 'Misto',
        'SAPI11.SA': 'Misto',
        'SARE11.SA': 'Misto',
        'SEQR11.SA': 'Misto',
        'SNFZ11.SA': 'Misto',
        'TORD11.SA': 'Misto',
        'VGHF11.SA': 'Misto',
        'VGRI11.SA': 'Misto',
        'VIUR11.SA': 'Misto',
        'VRTM11.SA': 'Misto',
        'VVPR11.SA': 'Misto',
        'JMBI11.SA': 'Outros',
        'TRXY11.SA': 'Outros',
        'URHF11.SA': 'Outros',
        'AFHI11.SA': 'Papéis',
        'ALZC11.SA': 'Papéis',
        'ARRI11.SA': 'Papéis',
        'ARXD11.SA': 'Papéis',
        'BARI11.SA': 'Papéis',
        'BBIM11.SA': 'Papéis',
        'BCRI11.SA': 'Papéis',
        'BINC11.SA': 'Papéis',
        'BLMC11.SA': 'Papéis',
        'BLUR11.SA': 'Papéis',
        'BTCI11.SA': 'Papéis',
        'BTCR11.SA': 'Papéis',
        'BTHF11.SA': 'Papéis',
        'CACR11.SA': 'Papéis',
        'CCRF11.SA': 'Papéis',
        'CPTS11.SA': 'Papéis',
        'CVBI11.SA': 'Papéis',
        'DEVA11.SA': 'Papéis',
        'FLCR11.SA': 'Papéis',
        'GCRI11.SA': 'Papéis',
        'HABT11.SA': 'Papéis',
        'HCHG11.SA': 'Papéis',
        'HCTR11.SA': 'Papéis',
        'HGCR11.SA': 'Papéis',
        'HGIC11.SA': 'Papéis',
        'HREC11.SA': 'Papéis',
        'HSAF11.SA': 'Papéis',
        'IBCR11.SA': 'Papéis',
        'IRDM11.SA': 'Papéis',
        'IRIF11.SA': 'Papéis',
        'ISCJ11.SA': 'Papéis',
        'ITIP11.SA': 'Papéis',
        'JBFO11.SA': 'Papéis',
        'JCCJ11.SA': 'Papéis',
        'JPPA11.SA': 'Papéis',
        'JSCR11.SA': 'Papéis',
        'KIVO11.SA': 'Papéis',
        'KNCR11.SA': 'Papéis',
        'KNHY11.SA': 'Papéis',
        'KNIP11.SA': 'Papéis',
        'KNSC11.SA': 'Papéis',
        'LFTT11.SA': 'Papéis',
        'LSPA11.SA': 'Papéis',
        'MCCI11.SA': 'Papéis',
        'MCRE11.SA': 'Papéis',
        'MGCR11.SA': 'Papéis',
        'MORC11.SA': 'Papéis',
        'MXRF11.SA': 'Papéis',
        'NAVT11.SA': 'Papéis',
        'NCHB11.SA': 'Papéis',
        'NCRI11.SA': 'Papéis',
        'OCRE11.SA': 'Papéis',
        'OUJP11.SA': 'Papéis',
        'PEMA11.SA': 'Papéis',
        'PLCR11.SA': 'Papéis',
        'PLRI11.SA': 'Papéis',
        'PORD11.SA': 'Papéis',
        'PULV11.SA': 'Papéis',
        'QAMI11.SA': 'Papéis',
        'QIRI11.SA': 'Papéis',
        'RBHG11.SA': 'Papéis',
        'RBHY11.SA': 'Papéis',
        'RBRR11.SA': 'Papéis',
        'RBRY11.SA': 'Papéis',
        'RECD11.SA': 'Papéis',
        'RECM11.SA': 'Papéis',
        'RECR11.SA': 'Papéis',
        'REIT11.SA': 'Papéis',
        'RNDP11.SA': 'Papéis',
        'RPRI11.SA': 'Papéis',
        'RRCI11.SA': 'Papéis',
        'RZAK11.SA': 'Papéis',
        'RZLC11.SA': 'Papéis',
        'SADI11.SA': 'Papéis',
        'SNAG11.SA': 'Papéis',
        'SNCI11.SA': 'Papéis',
        'URPR11.SA': 'Papéis',
        'VCJR11.SA': 'Papéis',
        'VCRI11.SA': 'Papéis',
        'VGIP11.SA': 'Papéis',
        'VGIR11.SA': 'Papéis',
        'VJFD11.SA': 'Papéis',
        'VOTS11.SA': 'Papéis',
        'VRTA11.SA': 'Papéis',
        'VSLH11.SA': 'Papéis',
        'VTPL11.SA': 'Papéis',
        'VVCR11.SA': 'Papéis',
        'XPCI11.SA': 'Papéis',
        'AZPL11.SA': 'Serviços Financeiros Diversos',
        'BBIG11.SA': 'Serviços Financeiros Diversos',
        'RENV11.SA': 'Serviços Financeiros Diversos',
        'ZAGH11.SA': 'Serviços Financeiros Diversos',
        'ABCP11.SA': 'Shoppings',
        'AJFI11.SA': 'Shoppings',
        'APXM11.SA': 'Shoppings',
        'ATSA11.SA': 'Shoppings',
        'BPML11.SA': 'Shoppings',
        'FIGS11.SA': 'Shoppings',
        'FLRP11.SA': 'Shoppings',
        'FVPQ11.SA': 'Shoppings',
        'GSFI11.SA': 'Shoppings',
        'HGBS11.SA': 'Shoppings',
        'HPDP11.SA': 'Shoppings',
        'HSML11.SA': 'Shoppings',
        'ITRI11.SA': 'Shoppings',
        'LASC11.SA': 'Shoppings',
        'MALL11.SA': 'Shoppings',
        'MCEM11.SA': 'Shoppings',
        'PQDP11.SA': 'Shoppings',
        'RBGS11.SA': 'Shoppings',
        'SCPF11.SA': 'Shoppings',
        'SHOP11.SA': 'Shoppings',
        'SHPH11.SA': 'Shoppings',
        'VISC11.SA': 'Shoppings',
        'VSHO11.SA': 'Shoppings',
        'WPLZ11.SA': 'Shoppings',
        'XPML11.SA': 'Shoppings',
        'VGII11.SA': 'Tecidos. Vestuário e Calçados',
        'ERCR11.SA': 'Varejo',
        'MAXR11.SA': 'Varejo',
        'RBRD11.SA': 'Varejo',
        'RBVA11.SA': 'Varejo'   
    }

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

        st.write("**Esta não é uma recomendação de investimento.**")
        st.write("**Rentabilidade passada não é garantia de rentabilidade futura.**")
        st.write("**Utilize para fins de estudo.**")

        # Listas de ativos
        all_tickers = prices_read.columns.tolist()

        # 4) Filtra Ações, FIIs e “não localizados”
        acoes_detectadas = [
            t.replace(".SA","").strip().upper()
            for t in all_tickers 
            if t in SECTOR_MAP_ACOES]
        acoes_detectadas = list(dict.fromkeys(acoes_detectadas))
        acoes = normalizar_tickers(acoes_detectadas)

        fiis_detectados = [
            t.replace(".SA","").strip().upper()
            for t in all_tickers 
            if t in SECTOR_MAP_FII]
        fiis_detectados = list(dict.fromkeys(fiis_detectados))
        fii = normalizar_tickers(fiis_detectados)

        nao_localizados = [
            t.replace(".SA","").strip().upper()
            for t in all_tickers 
            if t not in SECTOR_MAP_ACOES 
            and t not in SECTOR_MAP_FII]

        if nao_localizados:
            st.subheader("Tickers não localizados")
            st.write(nao_localizados)

        # Filtra os tickers com base em um mínimo desejado de observações (por exemplo, 200)
        acoes_validos, acoes_problema = filtrar_tickers(prices_read, acoes, min_obs=200)
        fii_validos, fii_problema     = filtrar_tickers(prices_read, fii, min_obs=200)

        st.write("[LOG] Ações carregadas:", acoes_validos)

        st.write("[LOG] FIIs carregados:", fii_validos)

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
        tickers_comb = acoes_validos + fii_validos

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
                cov_opt_manual = cov_manual

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
                cov_hibrida = cov_comb.loc[tickers_hibrida, tickers_hibrida]

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

        cenarios = [
            ("Carteira de Sharpe Máximo – AÇÕES", w_sharpe_aco, acoes_validos, cov_aco, sharpe_aco, ret_aco, vol_aco),
            ("Carteira de Sharpe Máximo – FIIs", w_sharpe_fii, fii_validos, cov_fii, sharpe_fii, ret_fii, vol_fii),
            ("Carteira de Sharpe Máximo – AÇÕES E FIIs", w_sharpe_comb, tickers_comb, cov_comb, sharpe_comb, ret_comb, vol_comb),
            ("Carteira Manual", w_man, tickers_man, cov_manual, sharpe_man, ret_man, vol_man),
            ("Carteira Manual Otimizada", w_opt_manual, tickers_man, cov_opt_manual, sharpe_opt_manual, ret_opt_manual, vol_opt_manual),
            (f"Carteira Híbrida Otimizada (com {int(percentual_adicional*100)}% adicionais)", w_hibrida, tickers_hibrida, cov_hibrida, sharpe_hibrida, ret_hibrida, vol_hibrida),
        ]

        for (nome, w, ticks, cov, s, r, v) in cenarios:
            render_portfolio_section(
                name=nome,
                weights=w,
                tickers=ticks,
                cov_df=cov,
                sharpe=s,
                ret=r,
                vol=v,
                min_weight=0.001
            )

if __name__ == "__main__":
    main()