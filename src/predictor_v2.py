import pandas as pd
import numpy as np

from .api import Atleta, Partida
from .data_collector import load_atletas_csv

FORBIDDEN_STATUSES = {2, 3, 6, 8, 9, 10, 13}

EXPERIMENTS = {
    "r6_form_bonus": {
        "alpha": 0.42, "beta": 0.18, "momentum_w": 1.0,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 1.0,
        "decay": 0.7, "blend": 0.15, "consistency_w": 0.0,
        "table_pos_w": 0.0, "form_w": 0.4, "home_bonus": 1.2,
    },
    "r7_pos_bonus": {
        "alpha": 0.42, "beta": 0.18, "momentum_w": 1.0,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 1.0,
        "decay": 0.7, "blend": 0.15, "consistency_w": 0.0,
        "table_pos_w": 0.3, "form_w": 0.0, "home_bonus": 1.2,
    },
    "f1_combined": {
        "alpha": 0.42, "beta": 0.18, "momentum_w": 1.0,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 1.0,
        "decay": 0.7, "blend": 0.15, "consistency_w": 0.0,
        "table_pos_w": 0.0, "form_w": 0.4, "home_bonus": 1.2,
    },
    "r7_pos_bonus": {
        "alpha": 0.42, "beta": 0.18, "momentum_w": 1.0,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 1.0,
        "decay": 0.7, "blend": 0.15, "consistency_w": 0.0,
        "table_pos_w": 0.3, "form_w": 0.0, "home_bonus": 1.2,
    },
    "f1_combined": {
        "alpha": 0.38, "beta": 0.22, "momentum_w": 0.8,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 0.9,
        "decay": 0.72, "blend": 0.12, "consistency_w": 0.10,
        "table_pos_w": 0.25, "form_w": 0.35, "home_bonus": 1.1,
    },
    "f2_median_heavy": {
        "alpha": 0.30, "beta": 0.30, "momentum_w": 0.7,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 0.85,
        "decay": 0.75, "blend": 0.18, "consistency_w": 0.15,
        "table_pos_w": 0.2, "form_w": 0.3, "home_bonus": 1.0,
    },
    "f3_max_lineup": {
        "alpha": 0.42, "beta": 0.18, "momentum_w": 1.1,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 1.05,
        "decay": 0.7, "blend": 0.10, "consistency_w": 0.05,
        "table_pos_w": 0.35, "form_w": 0.4, "home_bonus": 1.3,
    },
    "f4_high_spearman": {
        "alpha": 0.35, "beta": 0.25, "momentum_w": 0.6,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 0.75,
        "decay": 0.78, "blend": 0.20, "consistency_w": 0.18,
        "table_pos_w": 0.15, "form_w": 0.45, "home_bonus": 1.0,
    },
    "f5_lowest_mae": {
        "alpha": 0.35, "beta": 0.28, "momentum_w": 0.5,
        "home_w": 0.9, "away_w": -0.4, "opponent_w": 0.7,
        "decay": 0.8, "blend": 0.22, "consistency_w": 0.2,
        "table_pos_w": 0.1, "form_w": 0.3, "home_bonus": 1.0,
    },
    "f6_pos_form": {
        "alpha": 0.42, "beta": 0.18, "momentum_w": 1.0,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 1.0,
        "decay": 0.7, "blend": 0.15, "consistency_w": 0.0,
        "table_pos_w": 0.3, "form_w": 0.4, "home_bonus": 1.2,
    },
    "f7_boosted": {
        "alpha": 0.44, "beta": 0.16, "momentum_w": 1.1,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 1.0,
        "decay": 0.68, "blend": 0.10, "consistency_w": 0.0,
        "table_pos_w": 0.35, "form_w": 0.45, "home_bonus": 1.3,
    },
    "f8_lineup_king": {
        "alpha": 0.45, "beta": 0.15, "momentum_w": 1.2,
        "home_w": 1.0, "away_w": -0.6, "opponent_w": 1.1,
        "decay": 0.65, "blend": 0.08, "consistency_w": 0.0,
        "table_pos_w": 0.4, "form_w": 0.5, "home_bonus": 1.4,
    },
    "f9_opt_balance": {
        "alpha": 0.40, "beta": 0.20, "momentum_w": 0.9,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 0.95,
        "decay": 0.7, "blend": 0.10, "consistency_w": 0.0,
        "table_pos_w": 0.4, "form_w": 0.35, "home_bonus": 1.2,
    },
    "f10_r7_plus": {
        "alpha": 0.42, "beta": 0.18, "momentum_w": 1.0,
        "home_w": 1.0, "away_w": -0.5, "opponent_w": 1.0,
        "decay": 0.7, "blend": 0.05, "consistency_w": 0.0,
        "table_pos_w": 0.4, "form_w": 0.2, "home_bonus": 1.2,
    },
}


def compute_expected_scores_v2(
    atletas: list[Atleta],
    partidas: list[Partida],
    config: dict,
    recent_rounds: int = 5,
    df: pd.DataFrame | None = None,
) -> dict[int, float]:
    atletas_df = _load_or_build_df(df)
    historical = _get_historical_by_atleta(atletas_df, recent_rounds * 2)
    home_map = _build_home_map(partidas)
    pos_map = _build_position_map(partidas)
    form_map = _build_form_map(partidas)
    club_strength = _build_club_strength_from_df(atletas_df)

    alpha = config["alpha"]
    beta = config["beta"]
    decay = config["decay"]
    blend = config["blend"]

    scores: dict[int, float] = {}

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
            consistency_penalty = 0.0
        else:
            hist_recent = hist.nlargest(recent_rounds, "rodada")
            hist_recent = hist_recent.sort_values("rodada")

            w = _exponential_weights(len(hist_recent), decay)
            weighted_avg = np.average(hist_recent["pontos"].values, weights=w)
            median = hist_recent["pontos"].median()
            base = alpha * weighted_avg + beta * median

            momentum = _compute_momentum(hist_recent)
            hist_std = float(hist_recent["pontos"].std())
            consistency_penalty = hist_std if not np.isnan(hist_std) else 0.0

        home_factor = 0.0
        is_home = home_map.get(atleta.clube_id)
        if is_home is True:
            home_factor = config["home_bonus"] * config["home_w"]
        elif is_home is False:
            home_factor = config["away_w"]

        opponent_factor = _compute_opponent_v2(
            atleta, partidas, club_strength, home_map, config
        )

        table_pos_factor = _compute_table_pos_factor(
            atleta, partidas, pos_map, config["table_pos_w"]
        )

        team_form_factor = _compute_team_form_factor(
            atleta, partidas, form_map, home_map, config["form_w"]
        )

        expected = (
            base
            + config["momentum_w"] * momentum
            + home_factor
            + config["opponent_w"] * opponent_factor
            + table_pos_factor
            + team_form_factor
            - config["consistency_w"] * consistency_penalty
        )

        expected = max(expected, 0.0)

        if atleta.media > 0 and atleta.jogos > 0 and blend > 0:
            expected = (1.0 - blend) * expected + blend * atleta.media

        scores[atleta.atleta_id] = round(expected, 2)

    return scores


def _compute_opponent_v2(
    atleta: Atleta,
    partidas: list[Partida],
    club_strength: dict[int, float],
    home_map: dict[int, bool],
    config: dict,
) -> float:
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


def _compute_table_pos_factor(
    atleta: Atleta,
    partidas: list[Partida],
    pos_map: dict[int, int],
    weight: float,
) -> float:
    if weight == 0 or atleta.clube_id not in pos_map:
        return 0.0

    club_pos = pos_map[atleta.clube_id]
    z = (10.5 - club_pos) / 6.0
    return round(z * weight, 2)


def _compute_team_form_factor(
    atleta: Atleta,
    partidas: list[Partida],
    form_map: dict[int, float],
    home_map: dict[int, bool],
    weight: float,
) -> float:
    if weight == 0:
        return 0.0

    form_score = form_map.get(atleta.clube_id, 0.0)

    is_home = home_map.get(atleta.clube_id)
    if is_home:
        form_score *= 1.2
    elif is_home is False:
        form_score *= 0.8

    return round(form_score * weight, 2)


def _load_or_build_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is not None:
        return df
    try:
        return load_atletas_csv()
    except FileNotFoundError:
        return pd.DataFrame()


def _exponential_weights(n: int, decay: float) -> np.ndarray:
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    s = weights.sum()
    return weights / s if s > 0 else np.ones(n) / n


def _get_historical_by_atleta(df: pd.DataFrame, recent_rounds: int) -> dict[int, pd.DataFrame]:
    if df.empty:
        return {}
    max_round = df["rodada"].max()
    min_round = max(1, max_round - recent_rounds)
    recent = df[df["rodada"] >= min_round]
    return {aid: group for aid, group in recent.groupby("atleta_id")}


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
    return round(np.clip(pct_change * 2.0, -1.5, 1.5), 3)


def _build_home_map(partidas: list[Partida]) -> dict[int, bool]:
    home_map: dict[int, bool] = {}
    for p in partidas:
        if p.valida:
            home_map[p.clube_casa_id] = True
            home_map[p.clube_visitante_id] = False
    return home_map


def _build_position_map(partidas: list[Partida]) -> dict[int, int]:
    pos_map: dict[int, int] = {}
    for p in partidas:
        if p.valida:
            if p.clube_casa_posicao:
                pos_map[p.clube_casa_id] = p.clube_casa_posicao
            if p.clube_visitante_posicao:
                pos_map[p.clube_visitante_id] = p.clube_visitante_posicao
    return pos_map


def _build_form_map(partidas: list[Partida]) -> dict[int, float]:
    VALUE = {"v": 3, "e": 1, "d": 0}
    form_map: dict[int, float] = {}

    for p in partidas:
        if not p.valida:
            continue

        casa_form = p.aproveitamento_mandante or []
        if casa_form:
            scores = [VALUE.get(r, 1) for r in casa_form[-5:]]
            weights = np.array([0.6, 0.7, 0.8, 0.9, 1.0][-len(scores):])
            form = float(np.average(scores, weights=weights))
            form_map[p.clube_casa_id] = (form - 1.0) * 1.5

        fora_form = p.aproveitamento_visitante or []
        if fora_form:
            scores = [VALUE.get(r, 1) for r in fora_form[-5:]]
            weights = np.array([0.6, 0.7, 0.8, 0.9, 1.0][-len(scores):])
            form = float(np.average(scores, weights=weights))
            form_map[p.clube_visitante_id] = (form - 1.0) * 1.5

    return form_map


def _build_club_std(df: pd.DataFrame) -> dict[int, float]:
    if df.empty:
        return {}
    return dict(df.groupby("clube_id")["pontos"].std().fillna(2.0))


def _build_club_strength_from_df(df: pd.DataFrame) -> dict[int, float]:
    if df.empty:
        return {}
    club_scores = df.groupby("clube_id")["pontos"].agg(["mean", "std"]).fillna(0)
    strength: dict[int, float] = {}
    for club_id, row in club_scores.iterrows():
        strength[club_id] = row["mean"]
    return strength
