import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .api import POSITIONS, get_atletas_mercado, get_partidas, get_mercado_status

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

COLUMNS = [
    "atleta_id", "nome", "apelido", "slug", "posicao_id", "posicao",
    "clube_id", "status_id", "preco", "media", "variacao", "jogos",
    "pontos", "entrou_em_campo", "rodada",
]


def _ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def collect_current_round() -> dict:
    _ensure_dirs()
    status = get_mercado_status()
    atletas, clubes, posicoes = get_atletas_mercado()
    partidas = get_partidas()

    data = {
        "rodada": status.rodada_atual,
        "status": {
            "status_mercado": status.status_mercado,
            "cartoleta_inicial": status.cartoleta_inicial,
            "fechamento_timestamp": status.fechamento_timestamp,
            "times_escalados": status.times_escalados,
            "game_over": status.game_over,
        },
        "atletas": [
            {
                "atleta_id": a.atleta_id,
                "nome": a.nome,
                "apelido": a.apelido,
                "slug": a.slug,
                "posicao_id": a.posicao_id,
                "posicao": a.posicao,
                "clube_id": a.clube_id,
                "status_id": a.status_id,
                "preco": a.preco,
                "media": a.media,
                "variacao": a.variacao,
                "jogos": a.jogos,
                "pontos": a.pontos,
                "entrou_em_campo": a.entrou_em_campo,
            }
            for a in atletas
        ],
        "partidas": [
            {
                "partida_id": p.partida_id,
                "clube_casa_id": p.clube_casa_id,
                "clube_visitante_id": p.clube_visitante_id,
                "partida_data": p.partida_data,
                "local": p.local,
                "valida": p.valida,
                "aproveitamento_mandante": p.aproveitamento_mandante,
                "aproveitamento_visitante": p.aproveitamento_visitante,
            }
            for p in partidas
        ],
        "clubes": clubes,
        "posicoes": posicoes,
    }

    filepath = RAW_DIR / f"rodada_{status.rodada_atual}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def collect_historical(from_round: int = 1, to_round: Optional[int] = None):
    _ensure_dirs()

    if to_round is None:
        status = get_mercado_status()
        to_round = status.rodada_atual - 1

    collected = 0
    for rodada in range(from_round, to_round + 1):
        filepath = RAW_DIR / f"rodada_{rodada}.json"
        if filepath.exists():
            continue
        try:
            import requests

            resp = requests.get(
                f"https://api.cartolafc.globo.com/atletas/pontuados/{rodada}",
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            collected += 1
        except Exception:
            pass

    return collected


def build_atletas_csv() -> pd.DataFrame:
    _ensure_dirs()
    rows = []

    files = sorted(RAW_DIR.glob("rodada_*.json"))
    for filepath in files:
        try:
            rodada_num = int(filepath.stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_atletas = data.get("atletas", [])
        if not raw_atletas:
            continue

        if isinstance(raw_atletas, dict):
            for aid_str, a in raw_atletas.items():
                rows.append({
                    "atleta_id": int(aid_str),
                    "nome": a.get("nome", a.get("apelido", "")),
                    "apelido": a.get("apelido", a.get("apelido", "")),
                    "slug": a.get("slug", a.get("apelido", "").lower().replace(" ", "-")),
                    "posicao_id": a.get("posicao_id", 0),
                    "posicao": POSITIONS.get(a.get("posicao_id", 0), "desconhecido"),
                    "clube_id": a.get("clube_id", 0),
                    "status_id": a.get("status_id", 7),
                    "preco": float(a.get("preco_num", a.get("preco", 0))),
                    "media": float(a.get("media_num", a.get("media", 0))),
                    "variacao": float(a.get("variacao_num", a.get("variacao", 0))),
                    "jogos": int(a.get("jogos_num", 0)),
                    "pontos": float(a.get("pontuacao", a.get("pontos_num", a.get("pontos", 0)))),
                    "entrou_em_campo": bool(a.get("entrou_em_campo", True)),
                    "rodada": rodada_num,
                })
        else:
            for a in raw_atletas:
                rows.append({
                    "atleta_id": a["atleta_id"],
                    "nome": a.get("nome", ""),
                    "apelido": a.get("apelido", ""),
                    "slug": a.get("slug", ""),
                    "posicao_id": a.get("posicao_id", 0),
                    "posicao": POSITIONS.get(a.get("posicao_id", 0), "desconhecido"),
                    "clube_id": a.get("clube_id", 0),
                    "status_id": a.get("status_id", 7),
                    "preco": float(a.get("preco_num", a.get("preco", 0))),
                    "media": float(a.get("media_num", a.get("media", 0))),
                    "variacao": float(a.get("variacao_num", a.get("variacao", 0))),
                    "jogos": int(a.get("jogos_num", 0)),
                    "pontos": float(a.get("pontuacao", a.get("pontos_num", a.get("pontos", 0)))),
                    "entrou_em_campo": bool(a.get("entrou_em_campo", True)),
                    "rodada": rodada_num,
                })

    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        csv_path = PROCESSED_DIR / "atletas.csv"
        df.to_csv(csv_path, index=False)

    return df


def load_atletas_csv() -> pd.DataFrame:
    csv_path = PROCESSED_DIR / "atletas.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "CSV não encontrado. Execute 'python -m src.main collect' primeiro."
        )
    return pd.read_csv(csv_path)


def build_clubes_csv(clubes: dict) -> pd.DataFrame:
    _ensure_dirs()
    rows = [
        {
            "clube_id": int(pid),
            "nome": c.get("nome", ""),
            "abreviacao": c.get("abreviacao", ""),
            "slug": c.get("slug", ""),
            "apelido": c.get("apelido", ""),
            "nome_fantasia": c.get("nome_fantasia", ""),
        }
        for pid, c in clubes.items()
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(PROCESSED_DIR / "clubes.csv", index=False)
    return df
