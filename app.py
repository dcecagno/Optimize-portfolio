import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import streamlit as st

# =======================
# Funções Auxiliares
# =======================

def _read_close_prices(path_csv: str) -> pd.DataFrame:
    """
    Lê CSV com cabeçalho em duas linhas (ticker / Price,Open,…)
    e pula a terceira linha que só traz 'Date' e vírgulas.
    Retorna só o Close.
    """
    try:
        df = pd.read_csv(
            path_csv,
            header=[0,1],
            skiprows=[2],        # ignora a linha “Date,,,,”
            index_col=0,
            parse_dates=True
        )
        lvl1 = df.columns.get_level_values(1)

        # 1) Se existe 'Adj Close', retorna só esse slice
        if 'Close' in lvl1:
            return df.xs('Close', level=1, axis=1).copy()

    except Exception:
        pass

    # fallback para CSV sem multiindex
    df2 = pd.read_csv(path_csv, index_col=0, parse_dates=True)

    close_cols = [c for c in df2.columns if 'Close' in c]

    if not close_cols:
        raise RuntimeError(f"Nenhuma coluna 'Close' em {path_csv}")

    return df2[close_cols].copy()

def filtrar_valid_tickers(prices: pd.DataFrame, tickers: list, min_obs: int = 200):
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

# =======================
# Funções de Simulação
# =======================

def dynamic_compound_portfolio_metrics(prices, weights, tickers):
    # 1) pega retornos diários e transforma em log-returns
    rets = prices[tickers].pct_change().dropna()
    log_rets = np.log1p(rets)

    # 2) anualiza média e covariância
    mu_log  = log_rets.mean() * 252
    cov_log = log_rets.cov()  * 252

    # 3) portfólio em log-retorno + converte pra composto
    port_log_ret = np.dot(weights, mu_log.values)
    port_ret     = np.expm1(port_log_ret)

    # 4) volatilidade anual
    port_vol     = np.sqrt(weights @ cov_log.values @ weights)

    return port_ret, port_vol

def simulate_portfolios(
    prices: pd.DataFrame,
    tickers: list[str],
    n_sim: int,
    min_assets: int,
    max_assets: int,
    min_w: float,
    max_w: float,
    seed: int = 42,
    alpha_dirichlet: float = 1.0,
    acoes: set[str] = set(),
    fiis: set[str] = set()
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[list[str]]]:
    """
    Simula n_sim carteiras respeitando:
      - Número de ativos entre min_assets e max_assets
      - Cada peso w_i em [min_w, max_w]
      - Soma dos pesos = 1 (sem clip/remap)
      - Se `acoes` e `fiis` não-vazios, exige ao menos um de cada classe
      - Retorno composto anualizado via log-returns
      - Volatilidade anualizada via log-returns
    Retorna:
      sim_ret      : np.ndarray dos retornos a.a.
      sim_vol      : np.ndarray das volatilidades a.a.
      sim_w        : lista de arrays de pesos
      sim_tickers  : lista de listas de tickers correspondentes
    """

    np.random.seed(seed)

    # 1) Calcular log-returns diários e anualizar
    rets     = prices[tickers].pct_change().dropna()
    log_rets = np.log1p(rets)
    mu_log   = log_rets.mean() * 252       # expectativa de log-retorno a.a.
    cov_log  = log_rets.cov()  * 252       # covariância de log-returns a.a.

    sim_ret, sim_vol, sim_w, sim_tickers = [], [], [], []
    attempts, max_attempts = 0, n_sim * 10

    # 2) Loop até ter n_sim carteiras válidas ou estourar tentativas
    while len(sim_ret) < n_sim and attempts < max_attempts:
        attempts += 1

        # 2.1) Escolhe quantos e quais ativos
        n_assets = np.random.randint(min_assets, max_assets + 1)
        chosen   = np.random.choice(tickers, n_assets, replace=False)

        # 2.2) Se exigido mix de ações + FIIs
        if acoes and fiis:
            if not (set(chosen) & acoes and set(chosen) & fiis):
                continue

        # 2.3) Gera pesos via Dirichlet e rejeita fora dos limites
        w = np.random.dirichlet([alpha_dirichlet] * n_assets)
        if (w < min_w).any() or (w > max_w).any():
            continue

        # 3) Cálculo de retorno composto e volatilidade anualizados
        mu_vec   = mu_log.loc[chosen].values
        cov_mat  = cov_log.loc[chosen, chosen].values

        port_log = np.dot(w, mu_vec)           # log-retorno anual
        port_ret = np.expm1(port_log)          # converte para retorno a.a.
        port_vol = np.sqrt(w @ cov_mat @ w)    # volatilidade a.a.

        # 4) Armazena
        sim_ret.append(port_ret)
        sim_vol.append(port_vol)
        sim_w.append(w)
        sim_tickers.append(list(chosen))

    if attempts >= max_attempts:
        print(f"[WARNING] atingido {max_attempts} tentativas; geradas {len(sim_ret)} carteiras.")

    return (
        np.array(sim_ret),
        np.array(sim_vol),
        sim_w,
        sim_tickers
    )

def filtrar_por_composicao(ativos_simulados: list[list[str]], acoes: set, fiis: set):
    """
    Classifica as carteiras simuladas em:
    - só ações
    - só FIIs
    - mistas
    """
    so_acoes = []
    so_fiis = []
    mistos = []

    for i, ativos in enumerate(ativos_simulados):
        ativos_set = set(ativos)
        contem_acao = any(a in acoes for a in ativos_set)
        contem_fii = any(f in fiis for f in ativos_set)

        if contem_acao and not contem_fii:
            so_acoes.append(i)
        elif contem_fii and not contem_acao:
            so_fiis.append(i)
        elif contem_acao and contem_fii:
            mistos.append(i)

    return so_acoes, so_fiis, mistos

# =======================
# Funções de Otimização
# =======================

def optimize_max_sharpe(mu_vec, cov_mat, min_w, max_w, rf):
    from scipy.optimize import minimize
    n_assets = len(mu_vec)

    def neg_sharpe(w):
        port_ret = np.dot(w, mu_vec)
        port_vol = np.sqrt(w.T @ cov_mat @ w)
        if port_vol == 0:
            return np.inf
        return -(port_ret - rf) / port_vol

    bounds = [(min_w, max_w)] * n_assets

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    x0 = np.array([1 / n_assets] * n_assets)

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    if result.success:
        w_opt = result.x
        port_ret = np.dot(w_opt, mu_vec)
        port_vol = np.sqrt(w_opt.T @ cov_mat @ w_opt)
        sharpe = (port_ret - rf) / port_vol
        return w_opt, sharpe, True, "Sucesso"
    else:
        return x0, 0.0, False, result.message

def normalizar_tickers(lista):
    return [ticker.strip().upper() + ".SA" if not ticker.strip().upper().endswith(".SA") else ticker.strip().upper() for ticker in lista]

def otimizar_carteira_hibrida(
    tickers_man: list[str],
    valores_man: list[float],
    prices: pd.DataFrame,
    percentual_adicional: float,
    rf: float,
    min_w: float,
    max_w: float,
    eps: float = 1e-6
) -> tuple[list[str], np.ndarray, float, float, float]:
    """
    Otimiza carteira híbrida:
      - Dilui pesos manuais em frac_man = 1/(1+p_add)
      - Otimiza só pesos dos novos ativos:
         bounds: [0, max_w], e sum = rest = p_add/(1+p_add)
      - Filtra novos w < min_w → zero + renormaliza o restante
      - Retorno/vol anualizados via log-returns + expm1
    Retorna: (tickers_incl, pesos_incl, ret, vol, sharpe)
    """

    # 1) cálculo dos alvos manuais diluídos e capital novo
    total_man    = sum(valores_man)
    w_man_orig   = np.array([v/total_man for v in valores_man])
    frac_man     = 1.0 / (1.0 + percentual_adicional)
    w_man_target = frac_man * w_man_orig
    rest         = 1.0 - w_man_target.sum()  # capital para novos

    # 2) lista completa de tickers e índices
    tickers_total = tickers_man + [t for t in prices.columns if t not in tickers_man]
    tickers_total = list(dict.fromkeys(tickers_total))
    idx_man       = [tickers_total.index(t) for t in tickers_man]
    idx_new       = [i for i in range(len(tickers_total)) if i not in idx_man]

    # 3) estatísticas de log-returns anuais
    rets     = prices[tickers_total].pct_change().dropna()
    log_rets = np.log1p(rets)
    mu_log   = log_rets.mean() * 252
    cov_log  = log_rets.cov()  * 252
    μ, Σ     = mu_log.values, cov_log.values

    # 4) função objetivo (negativo do Sharpe)
    def neg_sharpe_new(w_new):
        # monta vetor completo de pesos
        w_full = np.zeros_like(μ)
        # manual
        for j, im in enumerate(idx_man):
            w_full[im] = w_man_target[j]
        # novos
        for k, inew in enumerate(idx_new):
            w_full[inew] = w_new[k]
        # portfólio em log-retorno + composto
        port_log = w_full @ μ
        port_ret = np.expm1(port_log)
        port_vol = np.sqrt(w_full @ Σ @ w_full)
        return -(port_ret - rf) / port_vol

    # 5) bounds e constraint para w_new
    n_new  = len(idx_new)
    bounds = [(0.0, max_w)] * n_new
    cons   = [{"type": "eq", "fun": lambda w: np.sum(w) - rest}]

    # 6) chute inicial: distribui rest igualmente
    x0 = np.array([rest / n_new] * n_new)

    # 7) otimização
    res = minimize(
        neg_sharpe_new,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons
    )
    if not res.success:
        raise RuntimeError("Otimização híbrida falhou: " + res.message)
    w_new_opt = res.x

    # 8) filtra novos abaixo de min_w e renormaliza o capital rest
    mask_new = w_new_opt >= min_w
    if mask_new.any():
        w_new_opt[~mask_new] = 0.0
        s = w_new_opt.sum()
        if s > 0:
            w_new_opt[mask_new] *= rest / s
    else:
        # nenhum novo acima de min_w → retorna só manual
        w_new_opt[:] = 0.0

    # 9) monta vetor final de pesos e métricas
    w_full = np.zeros_like(μ)
    for j, im in enumerate(idx_man):
        w_full[im] = w_man_target[j]
    for k, inew in enumerate(idx_new):
        w_full[inew] = w_new_opt[k]

    port_log   = w_full @ μ
    ret_opt    = float(np.expm1(port_log))
    vol_opt    = float(np.sqrt(w_full @ Σ @ w_full))
    sharpe_opt = (ret_opt - rf) / vol_opt

    # 10) filtra ativos não-nulos para retorno
    mask      = w_full > eps
    tickers_incl = [t for t, m in zip(tickers_total, mask) if m]
    w_incl       = w_full[mask]
    # já somam 1, pois manual+novos_rest =1

    return tickers_incl, w_incl, ret_opt, vol_opt, sharpe_opt

def pick_best_sim(
        sim_ret, 
        sim_vol, 
        sim_w, 
        rf=0.0):
    sharpe = (sim_ret - rf) / sim_vol
    idx = np.nanargmax(sharpe)
    return sim_w[idx], sim_ret[idx], sim_vol[idx], sharpe[idx]

def convex_frontier_with_indices(vols: np.ndarray, rets: np.ndarray):
    """
    Retorna
      vol_front: volatilidades dos vértices do upper hull
      ret_front: retornos correspondentes
      idxs: índices originais em vols/rets desses vértices
    """
    if len(vols) < 2:
        return np.array([]), np.array([]), []

    pts = np.column_stack((vols, rets))
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([]), np.array([]), []

    verts = sorted(hull.vertices, key=lambda i: vols[i])
    envelope, idxs = [], []
    cur_max = -np.inf
    for i in verts:
        r, v = rets[i], vols[i]
        if r >= cur_max:
            envelope.append((v, r))
            idxs.append(i)
            cur_max = r

    if not envelope:
        return np.array([]), np.array([]), []

    vol_f, ret_f = map(np.array, zip(*envelope))
    return vol_f, ret_f, idxs

# =======================
# Funções de Execução
# =======================

def ensure_simulations(
    prices_comb, tickers_comb,
    prices_aco,  tickers_aco,
    prices_fii,  tickers_fii,
    n_sim, min_assets, max_assets,
    min_w, max_w, seed, alpha_dirichlet
):
    """
    Garante que as simulações (combinado, só ações e só FIIs) sejam executadas
    apenas uma vez, armazenadas em session_state, e retorna todas as tuplas.
    """
    if "sim_done" not in st.session_state:
        st.session_state.sim_comb = simulate_portfolios(
            prices_comb, tickers_comb,
            n_sim, min_assets, max_assets,
            min_w, max_w, seed, alpha_dirichlet,
            acoes=set(tickers_aco), fiis=set(tickers_fii)
        )
        st.session_state.sim_aco = simulate_portfolios(
            prices_aco, tickers_aco,
            n_sim, min_assets, max_assets,
            min_w, max_w, seed, alpha_dirichlet,
            acoes=set(), fiis=set()
        )
        st.session_state.sim_fii = simulate_portfolios(
            prices_fii, tickers_fii,
            n_sim, min_assets, max_assets,
            min_w, max_w, seed, alpha_dirichlet,
            acoes=set(), fiis=set()
        )
        st.session_state.sim_done = True

    sim_ret_comb, sim_vol_comb, sim_w_comb, sim_tk_comb = st.session_state.sim_comb
    sim_ret_aco,  sim_vol_aco,  sim_w_aco,  sim_tk_aco  = st.session_state.sim_aco
    sim_ret_fii,  sim_vol_fii,  sim_w_fii,  sim_tk_fii  = st.session_state.sim_fii

    return (
        sim_ret_comb, sim_vol_comb, sim_w_comb, sim_tk_comb,
        sim_ret_aco,  sim_vol_aco,  sim_w_aco,  sim_tk_aco,
        sim_ret_fii,  sim_vol_fii,  sim_w_fii,  sim_tk_fii
    )

def compute_dynamic_cloud(
    sim_ret: np.ndarray,
    sim_vol: np.ndarray,
    sim_w: list[np.ndarray],
    sim_tickers: list[list[str]],
    prices: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dada a simulação bruta (sim_ret, sim_vol, sim_w, sim_tickers)
    e o DataFrame de preços, retorna duas arrays:
      - sim_ret_dyn: retornos compostos via log-returns
      - sim_vol_dyn: volatilidades compostas
    """
    n = sim_ret.shape[0]
    sim_ret_dyn = np.empty(n, dtype=float)
    sim_vol_dyn = np.empty(n, dtype=float)

    for i, (w, ticks) in enumerate(zip(sim_w, sim_tickers)):
        r, v = dynamic_compound_portfolio_metrics(prices, w, ticks)
        sim_ret_dyn[i] = r
        sim_vol_dyn[i] = v

    return sim_ret_dyn, sim_vol_dyn

def compute_frontier_and_sharpe(
    sim_ret_dyn: np.ndarray,
    sim_vol_dyn: np.ndarray,
    sim_w: list[np.ndarray],
    sim_tickers: list[list[str]],
    prices: pd.DataFrame,
    rf: float
) -> dict:
    """
    Dada a nuvem dinâmica e os pesos/tickers originais, constrói:
      - frontier_vol, frontier_ret: envelope convexo da nuvem
      - hull_idxs: índices dos vértices no array dinâmico
      - w_sh, ticks_sh, ret_sh, vol_sh, sharpe_sh: ponto de Sharpe Máx
    Retorna um dict com todas essas informações.
    """
    # 1) envelope convexo
    vol_front, ret_front, hull_idxs = convex_frontier_with_indices(
        sim_vol_dyn, sim_ret_dyn
    )

    # 2) extrai ponto de Sharpe na fronteira
    if vol_front.size > 0:
        sharpe_vals = (ret_front - rf) / vol_front
        best        = np.nanargmax(sharpe_vals)
        idxf        = hull_idxs[best]

        w_sh    = sim_w[idxf]
        ticks_sh= sim_tickers[idxf]
        # garante métricas compostas exatas no ponto ótimo
        ret_sh, vol_sh = dynamic_compound_portfolio_metrics(
            prices, w_sh, ticks_sh
        )
        sharpe_sh = (ret_sh - rf) / vol_sh
    else:
        # fallback
        idxf = None
        w_sh, ret_sh, vol_sh, sharpe_sh = pick_best_sim(
            sim_ret_dyn, sim_vol_dyn, sim_w, rf
        )
        ticks_sh = []

    return {
        "front_vol":   vol_front,
        "front_ret":   ret_front,
        "hull_idxs":   hull_idxs,
        "idx_sh":      idxf,
        "w_sh":        w_sh,
        "ticks_sh":    ticks_sh,
        "ret_sh":      ret_sh,
        "vol_sh":      vol_sh,
        "sharpe_sh":   sharpe_sh
    }

def portfolio_metrics_matrix(
    sim_w: list[np.ndarray],
    sim_tickers: list[list[str]],
    tickers_all: list[str],
    mu_log_vec: np.ndarray,
    cov_log_mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vetoriza o cálculo de retorno composto e volatilidade anual
    para uma lista de carteiras (sim_w) e seus tickers.
    """
    n_sim = len(sim_w)
    N     = len(tickers_all)
    # Monta matriz (n_sim × N) de pesos
    W = np.zeros((n_sim, N))
    for i, (w, ticks) in enumerate(zip(sim_w, sim_tickers)):
        idxs = [tickers_all.index(t) for t in ticks]
        W[i, idxs] = w

    # retorno: expm1(W @ mu_log_vec)
    ret = np.expm1(W.dot(mu_log_vec))
    # volatilidade: sqrt( w_i^T @ cov @ w_i ) para cada sim
    vol = np.sqrt(np.einsum('ij,jk,ik->i', W, cov_log_mat, W))
    return ret, vol

# =======================
# Funções de Plotagem
# =======================

def plot_results(
    sim_vol_dyn_aco, sim_ret_dyn_aco,
    vol_lin_dyn_aco, ret_lin_dyn_aco,
    vol_sh_aco, ret_sh_aco,
    sim_vol_dyn_fii, sim_ret_dyn_fii,
    vol_lin_dyn_fii, ret_lin_dyn_fii,
    vol_sh_fii, ret_sh_fii,
    sim_vol_dyn_comb, sim_ret_dyn_comb,
    vol_lin_dyn_comb, ret_lin_dyn_comb,
    vol_sh_comb, ret_sh_comb,
    vol_man, ret_man,
    vol_opt_manual, ret_opt_manual,
    vol_hibrida, ret_hibrida,
    tickers_man,
    ret_anual_ibov, vol_anual_ibov
):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Monte Carlo clouds (compostos)
    if sim_vol_dyn_comb.size > 0:
        ax.scatter(sim_vol_dyn_comb, sim_ret_dyn_comb,
                   s=8, alpha=0.12, c='red')
    if sim_vol_dyn_aco.size > 0:
        ax.scatter(sim_vol_dyn_aco, sim_ret_dyn_aco,
                   s=8, alpha=0.12, c='blue')
    if sim_vol_dyn_fii.size > 0:
        ax.scatter(sim_vol_dyn_fii, sim_ret_dyn_fii,
                   s=8, alpha=0.12, c='green')

    # Efficient frontier lines (compostos)
    if vol_lin_dyn_comb.size > 0:
        ax.plot(vol_lin_dyn_comb, ret_lin_dyn_comb,
                '-', c='red', lw=2, label='Fronteira – Ações+FIIs')
    if vol_lin_dyn_aco.size > 0:
        ax.plot(vol_lin_dyn_aco, ret_lin_dyn_aco,
                '-', c='blue', lw=2, label='Fronteira – Ações')
    if vol_lin_dyn_fii.size > 0:
        ax.plot(vol_lin_dyn_fii, ret_lin_dyn_fii,
                '-', c='green', lw=2, label='Fronteira – FIIs')

    # Sharpe max stars (compostos)
    if not np.isnan(vol_sh_comb):
        ax.scatter(vol_sh_comb, ret_sh_comb,
                   marker='*', c='red', s=180,
                   edgecolors='black', linewidths=1.0,
                   label='Sharpe Máx – Ações+FIIs')
    if not np.isnan(vol_sh_aco):
        ax.scatter(vol_sh_aco, ret_sh_aco,
                   marker='*', c='blue', s=180,
                   edgecolors='black', linewidths=1.0,
                   label='Sharpe Máx – Ações')
    if not np.isnan(vol_sh_fii):
        ax.scatter(vol_sh_fii, ret_sh_fii,
                   marker='*', c='green', s=180,
                   edgecolors='black', linewidths=1.0,
                   label='Sharpe Máx – FIIs')

    # Ibovespa
    ax.scatter(vol_anual_ibov, ret_anual_ibov,
               marker='*', c='white', s=180,
               edgecolors='black', linewidths=1.0,
               label='Ibovespa')

    # Manual portfolios
    if tickers_man:
        ax.scatter(vol_man, ret_man,
                   c='orange', s=180, marker='*',
                   edgecolors='black', linewidths=1.0,
                   label='Carteira Manual')
        ax.scatter(vol_opt_manual, ret_opt_manual,
                   c='orange', s=80, marker='D',
                   edgecolors='black', linewidths=1.0,
                   label='Manual Otimizada')
        ax.scatter(vol_hibrida, ret_hibrida,
                   c='orange', s=120, marker='P',
                   edgecolors='black', linewidths=1.0,
                   label='Carteira Híbrida')

    # Labels, title, legend
    ax.set_xlabel("Volatilidade Composta Anualizada")
    ax.set_ylabel("Retorno Composto Anualizado")
    ax.set_title("Fronteira Eficiente (Retorno Composto)")
    ax.legend(loc='best')
    ax.grid(True)

    # Percent format
    fmt = mtick.PercentFormatter(xmax=1.0, decimals=0)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    # Axis limits com base na nuvem dinâmica
    all_vols = []
    all_rets = []
    for arr in (sim_vol_dyn_aco, sim_vol_dyn_fii, sim_vol_dyn_comb):
        if hasattr(arr, 'size') and arr.size > 0:
            all_vols.append(arr.max())
    for arr in (sim_ret_dyn_aco, sim_ret_dyn_fii, sim_ret_dyn_comb):
        if hasattr(arr, 'size') and arr.size > 0:
            all_rets.append(arr.max())
            all_rets.append(arr.min())

    if all_vols and all_rets:
        max_v = max(all_vols) * 1.15
        min_r = min(all_rets) * 0.85
        max_r = max(all_rets) * 1.15
        ax.set_xlim(0, max_v)
        ax.set_ylim(min_r, max_r)
    else:
        st.warning(
            "Não foi possível calcular limites do gráfico. "
            "Verifique parâmetros de simulação."
        )

    st.pyplot(fig)

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
    st.title("Simulação de Carteiras Eficientes")
    # Upload do arquivo CSV
    url = "https://raw.githubusercontent.com/dcecagno/Optimize-portfolio/main/all_precos.csv"
    prices_read = _read_close_prices(url)
    url2 = "https://raw.githubusercontent.com/dcecagno/Optimize-portfolio/main/ibovespa_precos.csv"
    ibov_read = _read_close_prices(url2)

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
    ibov_read = ibov_read.loc[time_start:time_end]
    
    # Parâmetros para a simulação de Monte Carlo
    n_sim = 20_000
    seed = 42
    alpha_dirichlet = 1
    min_assets = st.number_input("Número mínimo de ativos", min_value=1, max_value=20, value=6)
    max_assets = st.number_input("Número máximo de ativos", min_value=1, max_value=20, value=15)
    min_w_percent = st.number_input("Peso mínimo por ativo (%)", min_value=0, max_value=100, value=3, step=1)
    max_w_percent = st.number_input("Peso máximo por ativo (%)", min_value=0, max_value=100, value=30, step=1)
    #rf_raw = st.number_input("Taxa livre de risco anual (%)", min_value=0, max_value=100, value=5)

    # Converte para proporção (0 a 1)
    min_w = min_w_percent / 100
    max_w = max_w_percent / 100
    #rf_percent = rf_raw / 100
    #rf = (1 + rf_percent) ** anos - 1
    rf = 0

    # Carteira manual
    st.subheader("Carteira Manual (opcional)")
    
    # Entrada da carteira manual em valores monetários
    num_ativos = st.number_input("Número de ativos na sua carteira", min_value=1, max_value=20, value=4)
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

    st.subheader("Carteira Híbrida (opcional)")
    percentual_adicional = st.slider(
                    "Percentual adicional para otimização da carteira híbrida (%). " \
                    "Insira quantos % você gostaria de aportar na carteira",
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
        
        if '^BVSP' in ibov_read.columns:
            ibovespa = ibov_read['^BVSP']
        else:
            st.error("Coluna '^BVSP' não encontrada no arquivo de preços do Ibovespa.")
            st.stop()

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
        acoes_validos, acoes_problema = filtrar_valid_tickers(prices_read, acoes, min_obs=200)
        fii_validos, fii_problema     = filtrar_valid_tickers(prices_read, fii, min_obs=200)
        
        acoes_validos.sort()
        fii_validos.sort()

        acoes_display = [t.removesuffix(".SA") for t in acoes_validos]
        fii_display = [t.removesuffix(".SA") for t in fii_validos]

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("✅ **Ações carregadas:**", expanded=True):
                for i, ticker in enumerate(acoes_display, start=1):
                    st.markdown(f"{i}. {ticker}")

        with col2:
            with st.expander("✅ **FIIs carregados:**", expanded=True):
                for i, ticker in enumerate(fii_display, start=1):
                    st.markdown(f"{i}. {ticker}")

        st.write("[LOG] Carregando o gráfico. Aguarde alguns minutos!")
                    
        # Cria os DataFrames filtrados para as simulações
        prices_aco  = prices_read[acoes_validos]
        prices_fii  = prices_read[fii_validos]
        prices_comb = prices_read[acoes_validos + fii_validos]

        # Usa retorno simples composto (pct_change) para tudo
        rets_aco  = prices_aco.pct_change().dropna()
        cov_aco = rets_aco.cov() * 252

        rets_fii  = prices_fii.pct_change().dropna()
        cov_fii = rets_fii.cov() * 252

        rets_comb = prices_comb.pct_change().dropna()
        mu_comb = (1 + rets_comb).prod() ** (252 / len(rets_comb)) - 1
        cov_comb = rets_comb.cov() * 252
                
        rets_ibov = ibovespa.pct_change().dropna()

        # Retorno anualizado
        ret_anual_ibov = rets_ibov.mean() * 252

        # Volatilidade anualizada
        vol_anual_ibov = rets_ibov.std() * np.sqrt(252)

        # 1) garante e carrega as 3 simulações
        (
            sim_ret_comb, sim_vol_comb, sim_w_comb, sim_tickers_comb,
            sim_ret_aco,  sim_vol_aco,  sim_w_aco,  sim_tickers_aco,
            sim_ret_fii,  sim_vol_fii,  sim_w_fii,  sim_tickers_fii
        ) = ensure_simulations(
            prices_comb, acoes_validos + fii_validos,
            prices_aco,  acoes_validos,
            prices_fii,  fii_validos,
            n_sim, min_assets, max_assets,
            min_w, max_w, seed, alpha_dirichlet
        )

        # 2) computa nuvens compostas
        sim_ret_dyn_aco, sim_vol_dyn_aco   = compute_dynamic_cloud(sim_ret_aco, sim_vol_aco, sim_w_aco, sim_tickers_aco, prices_aco)
        sim_ret_dyn_fii, sim_vol_dyn_fii   = compute_dynamic_cloud(sim_ret_fii, sim_vol_fii, sim_w_fii, sim_tickers_fii, prices_fii)
        sim_ret_dyn_comb, sim_vol_dyn_comb = compute_dynamic_cloud(sim_ret_comb, sim_vol_comb, sim_w_comb, sim_tickers_comb, prices_comb)

        # --- pré-cálculo para Ações ---
        rets_aco     = prices_aco.pct_change().dropna()
        log_rets_aco = np.log1p(rets_aco)
        mu_log_aco   = log_rets_aco.mean() * 252         # vetor (N_aco,)
        cov_log_aco  = log_rets_aco.cov()  * 252         # matriz (N_aco×N_aco)

        # --- pré-cálculo para FIIs ---
        rets_fii     = prices_fii.pct_change().dropna()
        log_rets_fii = np.log1p(rets_fii)
        mu_log_fii   = log_rets_fii.mean() * 252
        cov_log_fii  = log_rets_fii.cov()  * 252

        # --- pré-cálculo para Combinado ---
        rets_comb     = prices_comb.pct_change().dropna()
        log_rets_comb = np.log1p(rets_comb)
        mu_log_comb   = log_rets_comb.mean() * 252
        cov_log_comb  = log_rets_comb.cov()  * 252

        # 2) Nuvens compostas vetorizadas
        sim_ret_dyn_aco,  sim_vol_dyn_aco  = portfolio_metrics_matrix(
            sim_w_aco, sim_tickers_aco, acoes_validos,
            mu_log_aco.values, cov_log_aco.values
        )
        sim_ret_dyn_fii,  sim_vol_dyn_fii  = portfolio_metrics_matrix(
            sim_w_fii, sim_tickers_fii, fii_validos,
            mu_log_fii.values, cov_log_fii.values
        )
        sim_ret_dyn_comb, sim_vol_dyn_comb = portfolio_metrics_matrix(
            sim_w_comb, sim_tickers_comb, acoes_validos + fii_validos,
            mu_log_comb.values, cov_log_comb.values
        )

        # 3) constrói fronteira + ponto de Sharpe
        aco_res  = compute_frontier_and_sharpe(sim_ret_dyn_aco, sim_vol_dyn_aco, sim_w_aco, sim_tickers_aco, prices_aco, rf)
        fii_res  = compute_frontier_and_sharpe(sim_ret_dyn_fii, sim_vol_dyn_fii, sim_w_fii, sim_tickers_fii, prices_fii, rf)
        comb_res = compute_frontier_and_sharpe(sim_ret_dyn_comb, sim_vol_dyn_comb, sim_w_comb, sim_tickers_comb, prices_comb, rf)

        # ——— Ações ———
        w_sharpe_aco      = aco_res["w_sh"]
        ret_sh_aco        = aco_res["ret_sh"]
        vol_sh_aco        = aco_res["vol_sh"]
        sharpe_liquida_aco= aco_res["sharpe_sh"]
        idx_sharpe_aco    = aco_res["idx_sh"]

        # ——— FIIs ———
        w_sharpe_fii      = fii_res["w_sh"]
        ret_sh_fii        = fii_res["ret_sh"]
        vol_sh_fii        = fii_res["vol_sh"]
        sharpe_liquida_fii= fii_res["sharpe_sh"]
        idx_sharpe_fii    = fii_res["idx_sh"]

        # ——— Combinado ———
        w_sharpe_comb      = comb_res["w_sh"]
        ticks_comb         = comb_res["ticks_sh"]
        ret_sh_comb        = comb_res["ret_sh"]
        vol_sh_comb        = comb_res["vol_sh"]
        sharpe_liquida_comb= comb_res["sharpe_sh"]
        idx_sharpe_comb= comb_res["idx_sh"]

        tickers_man = normalizar_tickers(tickers_man)

        # Verifica se há tickers da carteira manual que não estão em prices_comb
        tickers_faltando = [t for t in tickers_man if t not in prices_comb.columns]
        
        if tickers_faltando:
            st.warning(f"Buscando dados no Yahoo Finance para: {', '.join(tickers_faltando)}")
            
            novos_list = []
            baixados   = []

            for ticker in tickers_faltando:
                try:
                    df_multi = yf.download(
                        ticker,
                        start=prices_comb.index.min(),
                        end=prices_comb.index.max(),
                        auto_adjust= True,
                        group_by   = 'ticker',
                        progress=False
                    )

                    if ticker not in df_multi.columns.levels[0]:
                        st.error(f"Ticker {ticker} não existe no Yahoo Finance. Removido da carteira.")
                        continue

                    # 3) Extrai série de fechamento
                    if 'Close' in df_multi[ticker].columns:
                        ser_close = df_multi[ticker]['Close']

                    # 4) Se a série veio, mas está totalmente vazia
                    if ser_close.empty:
                        st.error(f"Ticker {ticker} não existe ou não tem histórico no Yahoo. Removido da carteira.")
                        continue

                    # 5) Transforma em DF de 1 coluna e acumula na lista
                    df_close = ser_close.to_frame(name=ticker)
                    novos_list.append(df_close)
                    baixados.append(ticker)

                except Exception:
                    st.error(f"Erro ao buscar {ticker}. Verifique o nome do ativo e a conexão.")
                    continue

            if novos_list:
                novos_dados = pd.concat(novos_list, axis=1)
                prices_comb = pd.concat([prices_comb, novos_dados], axis=1)
                prices_comb.sort_index(inplace=True)
                prices_comb.columns = prices_comb.columns.map(str).str.strip()

                rets_comb = prices_comb.pct_change().dropna()
                mu_comb   = rets_comb.mean() * 252
                cov_comb  = rets_comb.cov()  * 252


            pares_filtrados = [
                (t, v) for t, v in zip(tickers_man, valores_man)
                if (t not in tickers_faltando) or (t in baixados)
            ]
            
            tickers_man = [t.strip() for t in tickers_man]
            prices_comb.columns = prices_comb.columns.map(str).str.strip()
            tickers_man = [str(t).strip() for t in tickers_man]

            faltantes = [t for t in tickers_man if t not in prices_comb.columns]
            if faltantes:
                st.error(
                    f"Não foi encontrado no Yahoo Finance: "
                    f"{', '.join(faltantes)}. "
                    "Verifique se você digitou o ticker corretamente e remova-o ou corrija."
                )
                # filtra só os que existem, mantendo os valores alinhados
                pares_ok = [(t, v) for t, v in zip(tickers_man, valores_man) if t in prices_comb.columns]
                if not pares_ok:
                    st.error("Nenhum ticker válido sobrou na carteira manual. Reinicie e insira tickers válidos.")
                    st.stop()
                tickers_man, valores_man = zip(*pares_ok)
                tickers_man, valores_man = list(tickers_man), list(valores_man)


            # descompacta de volta (ou vazio, se não sobrou ninguém)
            if pares_filtrados:
                tickers_man, valores_man = zip(*pares_filtrados)
                tickers_man, valores_man = list(tickers_man), list(valores_man)
            else:
                st.warning("Nenhum ticker válido sobrou na carteira manual.")
                st.stop()

        falt = [t for t in tickers_man if t not in prices_comb.columns]
        if falt:
            st.error(f"Não encontrei colunas para {falt}. Removido.")
            tickers_man = [t for t in tickers_man if t in prices_comb.columns]
        
        tickers_hibrida = []
        w_hibrida       = np.array([])
        ret_hibrida     = 0.0
        vol_hibrida     = 0.0
        sharpe_hibrida  = 0.0

        if tickers_man:
            total_man       = sum(valores_man)
            w_man           = np.array([v/total_man for v in valores_man])
            tickers_hibrida = []
            w_hibrida       = np.array([])      # array vazio por padrão
            ret_hibrida     = vol_hibrida = sharpe_hibrida = np.nan

            # Série de preços limpa para o cálculo composto
            prices_manual = prices_comb[tickers_man].dropna()

            try:
                # 1) Carteira manual original (1 ativo) — composto exato
                ret_man, vol_man = dynamic_compound_portfolio_metrics(
                    prices_manual,
                    w_man,
                    tickers_man
                )
                sharpe_man = (ret_man - rf) / vol_man

                # 2) Carteira manual otimizada (Sharpe) — composto exato
                #    usamos mu_comb e cov_comb para otimização
                mu_vec  = mu_comb.loc[tickers_man].values
                cov_mat = cov_comb.loc[tickers_man, tickers_man].values

                w_opt_manual, sharpe_opt_manual, ok_opt, msg_opt = optimize_max_sharpe(
                    mu_vec, cov_mat, min_w, max_w, rf
                )
                if not ok_opt:
                    st.warning(
                        "Não foi possível otimizar a carteira manual dentro dos limites de peso mínimo, "
                        "peso máximo e número de ativos definidos. Tente relaxar algum desses parâmetros "
                        "e execute novamente."
                    )
                    w_opt_manual = w_man
                    sharpe_opt_manual = sharpe_man

                ret_opt_manual, vol_opt_manual = dynamic_compound_portfolio_metrics(
                    prices_manual,
                    w_opt_manual,
                    tickers_man
                )
                sharpe_opt_manual = (ret_opt_manual - rf) / vol_opt_manual

                # 3) Carteira Híbrida – otimiza só os pesos
                tickers_hibrida, w_hibrida, ret_hibrida, vol_hibrida, sharpe_hibrida = otimizar_carteira_hibrida(
                    tickers_man,          # lista de manuais
                    valores_man,          # valores correspondentes
                    prices_comb,          # DataFrame de preços ajustados do universo combinado
                    percentual_adicional, # float em [0,1]
                    rf,                   # taxa livre de risco
                    min_w,                # ex: 0.0 ou 0.03
                    max_w
                )
                if tickers_hibrida:
                    ret_hibrida, vol_hibrida = dynamic_compound_portfolio_metrics(
                        prices_comb, w_hibrida, tickers_hibrida
                    )
                    sharpe_hibrida = (ret_hibrida - rf) / vol_hibrida
                else:
                    # fallback caso não haja carteira híbrida
                    ret_hibrida = vol_hibrida = sharpe_hibrida = np.nan

            except Exception as e:
                st.error(f"Erro ao processar carteira manual: {e}")
                # zera tudo
                ret_man = vol_man = ret_opt_manual = vol_opt_manual = ret_hibrida = vol_hibrida = 0.0

        else:
            st.warning("Nenhum ticker válido foi inserido na carteira manual.")
            ret_man = vol_man = ret_opt_manual = vol_opt_manual = ret_hibrida = vol_hibrida = 0.0

        # 12) Estatísticas individuais por ticker
        stats = []

        # Para ações: usa mu_aco e cov_aco
        for t in acoes_validos:
            try:
                ret_i, vol_i = dynamic_compound_portfolio_metrics(
                    prices_aco, np.array([1.0]), [t]
                )
                if np.isnan(ret_i) or np.isnan(vol_i) or vol_i == 0:
                    continue
                sharpe_i = (ret_i - rf) / vol_i
                stats.append({
                    "Ticker":      t.replace(".SA", ""),
                    "Ativo":       "Ação",
                    "Sharpe":      sharpe_i,
                    "Retorno":     ret_i,
                    "Volatilidade": vol_i
                })
            except:
                continue

        # Para FIIs
        for t in fii_validos:
            try:
                ret_i, vol_i = dynamic_compound_portfolio_metrics(
                    prices_fii, np.array([1.0]), [t]
                )
                if np.isnan(ret_i) or np.isnan(vol_i) or vol_i == 0:
                    continue
                sharpe_i = (ret_i - rf) / vol_i
                stats.append({
                    "Ticker":      t.replace(".SA", ""),
                    "Ativo":       "FII",
                    "Sharpe":      sharpe_i,
                    "Retorno":     ret_i,
                    "Volatilidade": vol_i
                })
            except:
                continue


        # Converte em DataFrame e ordena
        df_stats = pd.DataFrame(stats) \
            .sort_values("Sharpe", ascending=False) \
            .reset_index(drop=True)         # <-- aí o índice vira 0,1,2,…

        df_stats.index.name = "Rank"       # opcional: dá um nome à coluna de índice
        df_stats.index += 1

        # Plotagem
        plot_results(
            # Ações (dinâmico)
            sim_vol_dyn_aco, sim_ret_dyn_aco,
            aco_res["front_vol"],   aco_res["front_ret"],
            aco_res["vol_sh"],      aco_res["ret_sh"],

            # FIIs (dinâmico)
            sim_vol_dyn_fii, sim_ret_dyn_fii,
            fii_res["front_vol"],   fii_res["front_ret"],
            fii_res["vol_sh"],      fii_res["ret_sh"],

            # Combinado (dinâmico)
            sim_vol_dyn_comb, sim_ret_dyn_comb,
            comb_res["front_vol"],  comb_res["front_ret"],
            comb_res["vol_sh"],     comb_res["ret_sh"],

            # Carteira Manual e Otimizada
            vol_man, ret_man,
            vol_opt_manual, ret_opt_manual,

            # Carteira Híbrida
            vol_hibrida, ret_hibrida,

            # Manual tickers (para mostrar os pontos laranja)
            tickers_man,

            # Ibovespa
            ret_anual_ibov, vol_anual_ibov
        )

        # ================================
        # 1) Montagem de todos os cenários
        # ================================
        cenarios = []

        # Sharpe Máx – Ações
        if not np.isnan(vol_sh_aco):
            ticks = sim_tickers_aco[idx_sharpe_aco]
            # Debug: aritmético vs composto vs passado
            rets_aco = prices_aco[ticks].pct_change().dropna()
            mu_arith = rets_aco.mean() * 252
            w_dot_mu = np.dot(w_sharpe_aco, mu_arith.values)
            ret_dyn, vol_dyn = dynamic_compound_portfolio_metrics(
                prices_aco, w_sharpe_aco, ticks
            )
            st.write("🏷️ Sharpe Máx – Ações")
            st.write(f"   w@μ_arith   = {w_dot_mu:.2%}")
            st.write(f"   ret_passado = {ret_sh_aco:.2%} | vol_passado = {vol_sh_aco:.2%}")
            st.write(f"   ret_dyn     = {ret_dyn:.2%} | vol_dyn     = {vol_dyn:.2%}")
            cov_sub = cov_aco.loc[ticks, ticks]
            cenarios.append((
                "Sharpe Máx – Ações",
                w_sharpe_aco,
                ticks,
                cov_sub,
                sharpe_liquida_aco,
                ret_sh_aco,
                vol_sh_aco
            ))

        # Sharpe Máx – FIIs
        if not np.isnan(vol_sh_fii):
            ticks = sim_tickers_fii[idx_sharpe_fii]
            # Debug: aritmético vs composto vs passado
            rets_fii = prices_fii[ticks].pct_change().dropna()
            mu_arith = rets_fii.mean() * 252
            w_dot_mu = np.dot(w_sharpe_fii, mu_arith.values)
            ret_dyn, vol_dyn = dynamic_compound_portfolio_metrics(
                prices_fii, w_sharpe_fii, ticks
            )
            st.write("🏷️ Sharpe Máx – FIIs")
            st.write(f"   w@μ_arith   = {w_dot_mu:.2%}")
            st.write(f"   ret_passado = {ret_sh_fii:.2%} | vol_passado = {vol_sh_fii:.2%}")
            st.write(f"   ret_dyn     = {ret_dyn:.2%} | vol_dyn     = {vol_dyn:.2%}")
            cov_sub = cov_fii.loc[ticks, ticks]
            cenarios.append((
                "Sharpe Máx – FIIs",
                w_sharpe_fii,
                ticks,
                cov_sub,
                sharpe_liquida_fii,
                ret_sh_fii,
                vol_sh_fii
            ))

        # Sharpe Máx – Ações + FIIs
        if not np.isnan(vol_sh_comb):
            ticks = ticks_comb
            # Debug: aritmético vs composto vs passado
            rets_comb = prices_comb[ticks].pct_change().dropna()
            mu_arith  = rets_comb.mean() * 252
            w_dot_mu  = np.dot(w_sharpe_comb, mu_arith.values)
            ret_dyn, vol_dyn = dynamic_compound_portfolio_metrics(
                prices_comb, w_sharpe_comb, ticks
            )
            st.write("🏷️ Sharpe Máx – Ações + FIIs")
            st.write(f"   w@μ_arith   = {w_dot_mu:.2%}")
            st.write(f"   ret_passado = {ret_sh_comb:.2%} | vol_passado = {vol_sh_comb:.2%}")
            st.write(f"   ret_dyn     = {ret_dyn:.2%} | vol_dyn     = {vol_dyn:.2%}")
            cov_sub = cov_comb.loc[ticks, ticks]
            cenarios.append((
                "Sharpe Máx – Ações + FIIs",
                w_sharpe_comb,
                ticks,
                cov_sub,
                sharpe_liquida_comb,
                ret_sh_comb,
                vol_sh_comb
            ))

        # Carteira Manual (original)
        if tickers_man:
            cov_sub = cov_comb.loc[tickers_man, tickers_man]
            cenarios.append((
                "Carteira Manual",
                w_man,
                tickers_man,
                cov_sub,
                sharpe_man,
                ret_man,
                vol_man
            ))

        # Carteira Manual Otimizada
        if tickers_man and isinstance(w_opt_manual, np.ndarray):
            cov_sub = cov_comb.loc[tickers_man, tickers_man]
            cenarios.append((
                "Carteira Manual Otimizada",
                w_opt_manual,
                tickers_man,
                cov_sub,
                sharpe_opt_manual,
                ret_opt_manual,
                vol_opt_manual
            ))

        # Carteira Híbrida
        if tickers_hibrida and isinstance(w_hibrida, np.ndarray):
            cov_sub = cov_comb.loc[tickers_hibrida, tickers_hibrida]
            nome    = f"Carteira Híbrida (+{int(percentual_adicional*100)}% novos)"
            cenarios.append((
                nome,
                w_hibrida,
                tickers_hibrida,
                cov_sub,
                sharpe_hibrida,
                ret_hibrida,
                vol_hibrida
            ))

        # ================================
        # 2) Renderização sequencial
        # ================================
        st.divider()
        for name, weights, ticks, cov_df, sharpe, ret, vol in cenarios:
            render_portfolio_section(
                name=name,
                weights=weights,
                tickers=ticks,
                cov_df=cov_df,
                sharpe=sharpe,
                ret=ret,
                vol=vol,
                min_weight=0.001
            )

            # se for o combinado, exibe a composição por classes
            if name == "Sharpe Máx – Ações + FIIs":
                pct_aco = sum(weights[i] for i,t in enumerate(ticks) if t in acoes_validos)
                pct_fii = sum(weights[i] for i,t in enumerate(ticks) if t in fii_validos)
                st.markdown("**Composição por classe:**")
                st.markdown(f"- Ações: {pct_aco:.2%} | FIIs: {pct_fii:.2%}")

            st.divider()


                # Exibe no Streamlit
        st.subheader("📊 Métricas Individuais dos Ativos")
        st.dataframe(
            df_stats.style.format({
                "Sharpe":       "{:.2f}",
                "Retorno":      "{:.2%}",
                "Volatilidade": "{:.2%}"
            }),
            use_container_width=True
        )

if __name__ == "__main__":
    main()