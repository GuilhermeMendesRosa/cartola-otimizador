import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from .api import Atleta, Partida, get_atletas_mercado, get_partidas
from .data_collector import PROCESSED_DIR, RAW_DIR, build_atletas_csv
from .predictor import compute_expected_scores
from .optimizer import optimize_lineup
from .formatter import console
from rich.table import Table
from rich.panel import Panel
from rich import box


def _load_raw_round(rodada: int) -> dict:
    filepath = RAW_DIR / f"rodada_{rodada}.json"
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_historical_df(up_to_round: int) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "atletas.csv")
    return df[df["rodada"] <= up_to_round].copy()


def run_backtest(min_round: int = 6, max_round: int = 13, cartoletas: float = 120.0, formacao: str = "4-3-3"):
    current_atletas, clubes, posicoes = get_atletas_mercado()
    price_map: dict[int, float] = {a.atleta_id: a.preco for a in current_atletas}
    atleta_info: dict[int, dict] = {}
    for a in current_atletas:
        atleta_info[a.atleta_id] = {"nome": a.apelido, "clube_id": a.clube_id, "posicao_id": a.posicao_id, "media": a.media}

    full_df = pd.read_csv(PROCESSED_DIR / "atletas.csv")

    results = []
    metrics = defaultdict(list)

    if max_round is None:
        max_round = int(full_df["rodada"].max()) - 1
    max_round = min(max_round, int(full_df["rodada"].max()) - 1)

    with console.status("[bold green]Rodando backtest..."):
        for rodada in range(min_round, max_round + 1):
            actual_data = _load_raw_round(rodada)
            if not actual_data:
                continue

            actual_atletas = actual_data.get("atletas", {})
            if not actual_atletas:
                continue

            hist_df = _build_historical_df(rodada - 1)

            try:
                partidas = get_partidas(rodada)
            except Exception:
                partidas = []

            atletas_previsao = []
            for aid_str, a in actual_atletas.items():
                aid = int(aid_str)
                price = price_map.get(aid, 5.0)
                info = atleta_info.get(aid, {})
                atletas_previsao.append(Atleta(
                    atleta_id=aid,
                    nome=info.get("nome", a.get("apelido", "")),
                    apelido=a.get("apelido", ""),
                    slug=a.get("apelido", "").lower().replace(" ", "-"),
                    posicao_id=a.get("posicao_id", 0),
                    clube_id=a.get("clube_id", 0),
                    status_id=7,
                    preco=price,
                    pontos=0,
                    media=info.get("media", 0),
                    variacao=0,
                    jogos=0,
                    entrou_em_campo=a.get("entrou_em_campo", True),
                ))

            expected = compute_expected_scores(
                atletas=atletas_previsao,
                partidas=partidas,
                df=hist_df,
            )

            actual_scores: dict[int, float] = {}
            for aid_str, a in actual_atletas.items():
                actual_scores[int(aid_str)] = float(a.get("pontuacao", 0))

            # Prediction accuracy
            pred_actual = []
            for aid, pred in expected.items():
                actual = actual_scores.get(aid, None)
                if actual is not None and pred > 0:
                    pred_actual.append((pred, actual))

            if len(pred_actual) < 10:
                continue

            preds = np.array([p[0] for p in pred_actual])
            actuals = np.array([p[1] for p in pred_actual])

            mae = float(np.mean(np.abs(preds - actuals)))
            rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
            r2 = float(1 - np.sum((actuals - preds) ** 2) / np.sum((actuals - actuals.mean()) ** 2))

            if len(preds) > 1:
                corr_matrix = np.corrcoef(preds, actuals)
                pearson = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else 0.0
            else:
                pearson = 0.0

            try:
                from scipy.stats import spearmanr
                spearman, _ = spearmanr(preds, actuals)
            except ImportError:
                spearman = 0.0

            # Per-position accuracy
            pos_metrics = {}
            for pos_name in ["goleiro", "zagueiro", "lateral", "meia", "atacante", "tecnico"]:
                pos_preds = []
                pos_actuals = []
                for aid, pred in expected.items():
                    actual = actual_scores.get(aid, None)
                    if actual is not None and pred > 0:
                        info = atleta_info.get(aid, {})
                        pos_id = info.get("posicao_id", 0)
                        pos_map = {1: "goleiro", 2: "lateral", 3: "zagueiro", 4: "meia", 5: "atacante", 6: "tecnico"}
                        if pos_map.get(pos_id) == pos_name:
                            pos_preds.append(pred)
                            pos_actuals.append(actual)
                if len(pos_preds) > 2:
                    pos_preds_arr = np.array(pos_preds)
                    pos_actuals_arr = np.array(pos_actuals)
                    pos_mae = float(np.mean(np.abs(pos_preds_arr - pos_actuals_arr)))
                    pos_metrics[pos_name] = pos_mae

            # Lineup backtest
            lineup = optimize_lineup(
                atletas=atletas_previsao,
                expected_scores=expected,
                clubes=clubes,
                cartoletas=cartoletas,
                formacao=formacao,
            )

            lineup_score = 0.0
            lineup_player_count = 0
            if lineup:
                for p in lineup.titulares:
                    actual = actual_scores.get(p.atleta_id, 0)
                    lineup_score += actual
                    lineup_player_count += 1

            results.append({
                "rodada": rodada,
                "n_predicoes": len(pred_actual),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 3),
                "pearson": round(pearson, 3),
                "spearman": round(spearman, 3),
                "pos_metrics": pos_metrics,
                "lineup_score": round(lineup_score, 2),
                "lineup_players": lineup_player_count,
            })

            for k, v in results[-1].items():
                if k in ("mae", "rmse", "r2", "pearson", "spearman"):
                    metrics[k].append(v)

    return results, dict(metrics)


def print_backtest_results(results: list, aggregated: dict):
    console.print()
    title = "BACKTEST — Acurácia das Predições por Rodada"
    console.print(Panel(title, border_style="cyan", box=box.ROUNDED))

    table = Table(box=box.SIMPLE_HEAVY, border_style="bright_black", header_style="bold cyan")
    table.add_column("Rodada", style="bold yellow", width=8)
    table.add_column("N", justify="right", width=6)
    table.add_column("MAE", justify="right", style="red", width=8)
    table.add_column("RMSE", justify="right", style="red", width=8)
    table.add_column("R²", justify="right", style="green", width=8)
    table.add_column("Pearson", justify="right", style="blue", width=9)
    table.add_column("Spearman", justify="right", style="magenta", width=10)
    table.add_column("Time real", justify="right", style="yellow", width=10)

    for r in results:
        table.add_row(
            str(r["rodada"]),
            str(r["n_predicoes"]),
            f"{r['mae']:.2f}",
            f"{r['rmse']:.2f}",
            f"{r['r2']:.3f}",
            f"{r['pearson']:.3f}",
            f"{r['spearman']:.3f}",
            f"{r['lineup_score']:.1f}",
        )

    # Average row
    avg = {k: round(np.mean(v), 3) for k, v in aggregated.items()}

    table.add_section()
    table.add_row(
        "[bold]MÉDIA[/]",
        "",
        f"[bold red]{avg['mae']:.2f}[/]",
        f"[bold red]{avg['rmse']:.2f}[/]",
        f"[bold green]{avg['r2']:.3f}[/]",
        f"[bold blue]{avg['pearson']:.3f}[/]",
        f"[bold magenta]{avg['spearman']:.3f}[/]",
        "",
    )

    console.print(table)

    # Per-position breakdown
    console.print()
    pos_table = Table(title="MAE por Posição", box=box.SIMPLE, border_style="bright_black")
    pos_table.add_column("Rodada", style="dim", width=8)
    for pos in ["goleiro", "zagueiro", "lateral", "meia", "atacante", "tecnico"]:
        pos_table.add_column(pos.capitalize(), justify="right", width=10)

    all_pos_mae = defaultdict(list)
    for r in results:
        row = [str(r["rodada"])]
        for pos in ["goleiro", "zagueiro", "lateral", "meia", "atacante", "tecnico"]:
            val = r["pos_metrics"].get(pos, None)
            if val is not None:
                row.append(f"{val:.2f}")
                all_pos_mae[pos].append(val)
            else:
                row.append("-")
        pos_table.add_row(*row)

    pos_table.add_section()
    avg_row = ["[bold]MÉDIA[/]"]
    for pos in ["goleiro", "zagueiro", "lateral", "meia", "atacante", "tecnico"]:
        vals = all_pos_mae.get(pos, [])
        if vals:
            avg_row.append(f"[bold]{np.mean(vals):.2f}[/]")
        else:
            avg_row.append("-")
    pos_table.add_row(*avg_row)

    console.print(pos_table)

    # Interpretation
    console.print()
    interp = Table(box=box.ROUNDED, border_style="dim")
    interp.add_column("Métrica", style="bold")
    interp.add_column("Interpretação")

    interp.add_row("MAE", f"{avg['mae']:.2f} — Erro médio de ~{avg['mae']:.1f} pontos por atleta")
    interp.add_row("RMSE", f"{avg['rmse']:.2f} — Penaliza erros grandes (outliers)")

    r2_text = "excelente" if avg["r2"] > 0.5 else ("razoável" if avg["r2"] > 0.2 else "baixo")
    interp.add_row("R²", f"{avg['r2']:.3f} — Explica {avg['r2']*100:.1f}% da variância dos pontos ({r2_text})")

    spearman_text = "forte" if avg["spearman"] > 0.5 else ("moderada" if avg["spearman"] > 0.3 else "fraca")
    interp.add_row("Spearman", f"{avg['spearman']:.3f} — Capacidade de ranquear corretamente ({spearman_text})")

    console.print(Panel(interp, title="Guia de interpretação", border_style="dim"))


def run_multi_experiment_backtest(
    min_round: int = 6,
    max_round: int = None,
    cartoletas: float = 130.0,
    formacao: str = "4-3-3",
):
    from .predictor_v2 import compute_expected_scores_v2, EXPERIMENTS
    from .predictor import compute_expected_scores as baseline_predictor

    current_atletas, clubes, posicoes = get_atletas_mercado()
    price_map: dict[int, float] = {a.atleta_id: a.preco for a in current_atletas}
    atleta_info: dict[int, dict] = {}
    for a in current_atletas:
        atleta_info[a.atleta_id] = {"nome": a.apelido, "clube_id": a.clube_id, "posicao_id": a.posicao_id, "media": a.media}

    full_df = pd.read_csv(PROCESSED_DIR / "atletas.csv")

    if max_round is None:
        max_round = int(full_df["rodada"].max()) - 1
    max_round = min(max_round, int(full_df["rodada"].max()) - 1)

    rounds = list(range(min_round, max_round + 1))

    all_results: dict[str, dict] = {}

    exp_names = ["original"] + ["r7_pos_bonus", "f6_pos_form", "f9_opt_balance", "f10_r7_plus", "f4_high_spearman"]

    for exp_name in exp_names:
        console.print(f"  Rodando [cyan]{exp_name}[/]...")
        config = EXPERIMENTS.get(exp_name, {})

        mae_list = []
        spearman_list = []
        lineup_scores = []

        for rodada in rounds:
            actual_data = _load_raw_round(rodada)
            if not actual_data:
                continue
            actual_atletas = actual_data.get("atletas", {})
            if not actual_atletas:
                continue

            hist_df = _build_historical_df(rodada - 1)

            try:
                partidas = get_partidas(rodada)
            except Exception:
                partidas = []

            atletas_previsao = []
            for aid_str, a in actual_atletas.items():
                aid = int(aid_str)
                price = price_map.get(aid, 5.0)
                info = atleta_info.get(aid, {})
                atletas_previsao.append(Atleta(
                    atleta_id=aid,
                    nome=info.get("nome", a.get("apelido", "")),
                    apelido=a.get("apelido", ""),
                    slug=a.get("apelido", "").lower().replace(" ", "-"),
                    posicao_id=a.get("posicao_id", 0),
                    clube_id=a.get("clube_id", 0),
                    status_id=7,
                    preco=price,
                    pontos=0,
                    media=info.get("media", 0),
                    variacao=0,
                    jogos=0,
                    entrou_em_campo=a.get("entrou_em_campo", True),
                ))

            if exp_name == "original":
                expected = baseline_predictor(
                    atletas=atletas_previsao,
                    partidas=partidas,
                    df=hist_df,
                )
            else:
                expected = compute_expected_scores_v2(
                    atletas=atletas_previsao,
                    partidas=partidas,
                    config=config,
                    df=hist_df,
                )

            actual_scores: dict[int, float] = {}
            for aid_str, a in actual_atletas.items():
                actual_scores[int(aid_str)] = float(a.get("pontuacao", 0))

            pred_actual = []
            for aid, pred in expected.items():
                actual = actual_scores.get(aid, None)
                if actual is not None and pred > 0:
                    pred_actual.append((pred, actual))

            if len(pred_actual) < 10:
                continue

            preds = np.array([p[0] for p in pred_actual])
            actuals = np.array([p[1] for p in pred_actual])

            mae_list.append(float(np.mean(np.abs(preds - actuals))))

            try:
                from scipy.stats import spearmanr
                sp, _ = spearmanr(preds, actuals)
                spearman_list.append(float(sp) if not np.isnan(sp) else 0.0)
            except ImportError:
                spearman_list.append(0.0)

            lineup = optimize_lineup(
                atletas=atletas_previsao,
                expected_scores=expected,
                clubes=clubes,
                cartoletas=cartoletas,
                formacao=formacao,
            )
            if lineup:
                score = sum(actual_scores.get(p.atleta_id, 0) for p in lineup.titulares)
                lineup_scores.append(score)

        all_results[exp_name] = {
            "mae": round(np.mean(mae_list), 2) if mae_list else 0,
            "mae_std": round(np.std(mae_list), 2) if mae_list else 0,
            "spearman": round(np.mean(spearman_list), 3) if spearman_list else 0,
            "sp_std": round(np.std(spearman_list), 3) if spearman_list else 0,
            "lineup_avg": round(np.mean(lineup_scores), 1) if lineup_scores else 0,
            "lineup_max": round(max(lineup_scores), 1) if lineup_scores else 0,
            "lineup_min": round(min(lineup_scores), 1) if lineup_scores else 0,
        }

    return all_results


def print_multi_experiment_results(results: dict, cartoletas: float):
    console.print()
    console.print(
        Panel(
            f"MULTI-EXPERIMENTO — {len(results)} configurações testadas em 8 rodadas (C$ {cartoletas:.0f})",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    table = Table(box=box.SIMPLE_HEAVY, border_style="bright_black", header_style="bold cyan")
    table.add_column("Experimento", style="bold white", width=18)
    table.add_column("MAE", justify="right", style="red", width=8)
    table.add_column("±", justify="right", style="dim red", width=6)
    table.add_column("Spearman", justify="right", style="magenta", width=10)
    table.add_column("±", justify="right", style="dim magenta", width=6)
    table.add_column("Time Méd", justify="right", style="yellow", width=10)
    table.add_column("Máx", justify="right", style="green", width=8)
    table.add_column("Mín", justify="right", style="dim red", width=8)

    best_mae = min(r["mae"] for r in results.values())
    best_sp = max(r["spearman"] for r in results.values())
    best_lineup = max(r["lineup_avg"] for r in results.values())

    for name, r in results.items():
        mae_style = "bold red" if r["mae"] == best_mae else ""
        sp_style = "bold magenta" if r["spearman"] == best_sp else ""
        lu_style = "bold yellow" if r["lineup_avg"] == best_lineup else ""

        table.add_row(
            name,
            f"[{mae_style}]{r['mae']:.2f}[/]" if mae_style else f"{r['mae']:.2f}",
            f"{r['mae_std']:.2f}",
            f"[{sp_style}]{r['spearman']:.3f}[/]" if sp_style else f"{r['spearman']:.3f}",
            f"{r['sp_std']:.3f}",
            f"[{lu_style}]{r['lineup_avg']:.1f}[/]" if lu_style else f"{r['lineup_avg']:.1f}",
            f"{r['lineup_max']:.1f}",
            f"{r['lineup_min']:.1f}",
        )

    console.print(table)

    best = max(results.items(), key=lambda kv: (
        kv[1]["spearman"] * 0.3 + kv[1]["lineup_avg"] * 0.4 + (5.0 - kv[1]["mae"]) * 0.3
    ))
    console.print()
    console.print(f"  [bold green]Melhor configuração: {best[0]}[/]")
    console.print(f"  MAE={best[1]['mae']} | Spearman={best[1]['spearman']} | Time Méd={best[1]['lineup_avg']} pts")
