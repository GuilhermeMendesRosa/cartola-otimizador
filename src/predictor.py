import pandas as pd
import numpy as np

from .api import Atleta, Partida
from .data_collector import load_atletas_csv


FORBIDDEN_STATUSES = {2, 3, 6, 8, 9, 10, 13}

HOME_BONUS = 1.2
AWAY_PENALTY = -0.5


def compute_expected_scores(
    atletas: list[Atleta],
    partidas: list[Partida],
    recent_rounds: int = 5,
    alpha: float = 0.42,
    beta: float = 0.18,
    gamma: float = 1.0,
    delta: float = 1.0,
    epsilon: float = 1.0,
    df: pd.DataFrame | None = None,
) -> dict[int, float]:
    atletas_df = _load_or_build_df(df)
    club_strength = _build_club_strength(atletas_df)
    home_map = _build_home_map(partidas)
    pos_bonus = _build_position_bonus(partidas)
    scores: dict[int, float] = {}
    historical = _get_historical_by_atleta(atletas_df, recent_rounds * 2)

    for atleta in atletas:
        sid = atleta.status_id
        if sid in FORBIDDEN_STATUSES:
            scores[atleta.atleta_id] = 0.0
            continue

        hist = historical.get(atleta.atleta_id, pd.DataFrame())

        if hist.empty or len(hist) < 3:
            base = atleta.media if atleta.media > 0 else atleta.pontos
            if base == 0 and not hist.empty:
                base = hist["pontos"].mean()
            if base == 0 and atleta.jogos > 0:
                base = 0.5
            momentum = 0.0
        else:
            hist_recent = hist.nlargest(recent_rounds, "rodada")
            hist_recent = hist_recent.sort_values("rodada")

            w = _exponential_weights(len(hist_recent))
            weighted_avg = np.average(hist_recent["pontos"].values, weights=w)
            median = hist_recent["pontos"].median()
            base = alpha * weighted_avg + beta * median

            momentum = _compute_momentum(hist_recent)

        home_factor = 0.0
        if home_map and atleta.clube_id in home_map:
            home_factor = HOME_BONUS if home_map[atleta.clube_id] else AWAY_PENALTY

        opponent_factor = 0.0
        if home_map and atleta.clube_id in home_map:
            opponent_factor = _compute_opponent_factor(
                atleta, partidas, club_strength, home_map
            )

        table_pos_bonus = pos_bonus.get(atleta.clube_id, 0.0)

        expected = base + gamma * momentum + delta * home_factor + epsilon * opponent_factor + 0.3 * table_pos_bonus
        expected = max(expected, 0.0)

        if atleta.media > 0 and atleta.jogos > 0:
            expected = 0.85 * expected + 0.15 * atleta.media

        scores[atleta.atleta_id] = round(expected, 2)

    return scores


def _compute_momentum(hist_recent: pd.DataFrame) -> float:
    if len(hist_recent) < 4:
        return 0.0

    pontos = hist_recent["pontos"].values
    half = len(pontos) // 2
    older_avg = np.mean(pontos[:half])
    recent_avg = np.mean(pontos[half:])

    if older_avg == 0 and recent_avg == 0:
        return 0.0
    if older_avg == 0:
        return min(recent_avg / 5.0, 1.5)

    pct_change = (recent_avg - older_avg) / older_avg
    momentum = pct_change * 2.0
    return round(np.clip(momentum, -1.5, 1.5), 3)


def _load_or_build_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is not None:
        return df
    try:
        return load_atletas_csv()
    except FileNotFoundError:
        return pd.DataFrame()


def _exponential_weights(n: int, decay: float = 0.7) -> np.ndarray:
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    return weights / weights.sum()


def _get_historical_by_atleta(df: pd.DataFrame, recent_rounds: int) -> dict[int, pd.DataFrame]:
    if df.empty:
        return {}
    max_round = df["rodada"].max()
    min_round = max(1, max_round - recent_rounds)
    recent = df[df["rodada"] >= min_round]
    return {aid: group for aid, group in recent.groupby("atleta_id")}


def _build_club_strength(df: pd.DataFrame) -> dict[int, float]:
    if df.empty:
        return {}
    club_scores = df.groupby("clube_id")["pontos"].agg(["mean", "std"]).fillna(0)
    strength: dict[int, float] = {}
    for club_id, row in club_scores.iterrows():
        strength[club_id] = row["mean"]
    return strength


def _build_home_map(partidas: list[Partida]) -> dict[int, bool]:
    home_map: dict[int, bool] = {}
    for p in partidas:
        if p.valida:
            home_map[p.clube_casa_id] = True
            home_map[p.clube_visitante_id] = False
    return home_map


def _build_position_bonus(partidas: list[Partida]) -> dict[int, float]:
    pos_map: dict[int, float] = {}
    for p in partidas:
        if p.valida:
            if p.clube_casa_posicao:
                pos_map[p.clube_casa_id] = (10.5 - p.clube_casa_posicao) / 6.0
            if p.clube_visitante_posicao:
                pos_map[p.clube_visitante_id] = (10.5 - p.clube_visitante_posicao) / 6.0
    return pos_map


def _compute_opponent_factor(
    atleta: Atleta,
    partidas: list[Partida],
    club_strength: dict[int, float],
    home_map: dict[int, bool],
) -> float:
    is_home = home_map.get(atleta.clube_id, False)
    opponent_id = None

    for p in partidas:
        if not p.valida:
            continue
        if p.clube_casa_id == atleta.clube_id:
            opponent_id = p.clube_visitante_id
            break
        if p.clube_visitante_id == atleta.clube_id:
            opponent_id = p.clube_casa_id
            break

    if opponent_id is None or opponent_id not in club_strength:
        return 0.0

    all_means = list(club_strength.values())
    if not all_means:
        return 0.0

    global_mean = np.mean(all_means)
    global_std = max(np.std(all_means), 0.01)
    opp_z = (club_strength[opponent_id] - global_mean) / global_std

    if atleta.posicao_id == 5:
        opp_z *= -1
    elif atleta.posicao_id == 1 or atleta.posicao_id == 3:
        pass
    else:
        opp_z *= -0.5

    return round(np.clip(opp_z * 1.5, -2.0, 2.0), 2)
