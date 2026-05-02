from dataclasses import dataclass, field
from typing import Optional

from pulp import (
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
    PULP_CBC_CMD,
)

from .api import Atleta, FORMATION_LAYOUT


POS_LABEL_PT = {
    "goleiro": "Goleiro",
    "lateral": "Lateral",
    "zagueiro": "Zagueiro",
    "meia": "Meia",
    "atacante": "Atacante",
    "tecnico": "Técnico",
}


@dataclass
class SelectedPlayer:
    atleta_id: int
    nome: str
    apelido: str
    clube: str
    clube_id: int
    posicao: str
    preco: float
    media: float
    expected: float
    titular: bool


@dataclass
class OptimizedLineup:
    formacao: str
    cartoletas_usadas: float
    cartoletas_disponiveis: float
    pontuacao_esperada: float
    titulares: list[SelectedPlayer]
    reservas: list[SelectedPlayer]
    status: str = "optimal"

    @property
    def total_custo(self) -> float:
        return round(self.cartoletas_usadas, 2)

    @property
    def total_esperado(self) -> float:
        return round(self.pontuacao_esperada, 2)


def optimize_lineup(
    atletas: list[Atleta],
    expected_scores: dict[int, float],
    clubes: dict,
    cartoletas: float,
    formacao: str = "4-3-3",
    forced_ids: Optional[list[int]] = None,
    banned_ids: Optional[list[int]] = None,
    max_por_clube: int = 4,
) -> Optional[OptimizedLineup]:
    forced_ids = forced_ids or []
    banned_ids = banned_ids or []

    if formacao not in FORMATION_LAYOUT:
        raise ValueError(f"Formação inválida: {formacao}. Use: {list(FORMATION_LAYOUT.keys())}")

    layout = FORMATION_LAYOUT[formacao]

    by_pos = _group_by_position(atletas)

    problem = LpProblem("Cartola_Lineup", LpMaximize)
    x = {}
    score_terms = []

    for atleta in atletas:
        aid = atleta.atleta_id
        score = expected_scores.get(aid, atleta.media)

        if aid in forced_ids:
            x[aid] = 1
            score_terms.append(score * 1)
            continue

        if aid in banned_ids:
            x[aid] = 0
            continue

        if score <= 0:
            continue

        var = LpVariable(f"x_{aid}", cat="Binary")
        x[aid] = var
        score_terms.append(score * var)

    problem += lpSum(score_terms)

    for fb_id in forced_ids:
        if isinstance(x.get(fb_id), LpVariable):
            problem += x[fb_id] == 1

    for ban_id in banned_ids:
        if isinstance(x.get(ban_id), LpVariable):
            problem += x[ban_id] == 0

    variables = {aid: v for aid, v in x.items() if isinstance(v, LpVariable)}
    if not variables:
        return None

    problem += lpSum(
        float(_lookup_preco(atletas, aid)) * v for aid, v in variables.items()
    ) <= cartoletas

    for pos_name, count in layout.items():
        pos_ids = _filter_position(variables, atletas, pos_name)
        if pos_ids:
            problem += lpSum(variables[aid] for aid in pos_ids) == count

    goleiro_ids = _filter_position(variables, atletas, "goleiro")
    if goleiro_ids:
        problem += lpSum(variables[aid] for aid in goleiro_ids) == 1

    tecnico_ids = _filter_position(variables, atletas, "tecnico")
    if tecnico_ids:
        problem += lpSum(variables[aid] for aid in tecnico_ids) == 1

    for club_id in set(a.clube_id for a in atletas):
        club_vars = [v for aid, v in variables.items() if _lookup_clube(atletas, aid) == club_id]
        if club_vars:
            problem += lpSum(club_vars) <= max_por_clube

    problem.solve(PULP_CBC_CMD(msg=False))

    if LpStatus[problem.status] != "Optimal":
        return None

    selected = []
    for aid, var in variables.items():
        if value(var) and value(var) > 0.5 or aid in forced_ids:
            atleta = _find_atleta(atletas, aid)
            if not atleta:
                continue
            club_data = clubes.get(str(atleta.clube_id), {})
            selected.append(
                SelectedPlayer(
                    atleta_id=aid,
                    nome=atleta.apelido,
                    apelido=atleta.apelido,
                    clube=club_data.get("abreviacao", str(atleta.clube_id)),
                    clube_id=atleta.clube_id,
                    posicao=atleta.posicao,
                    preco=atleta.preco,
                    media=atleta.media,
                    expected=expected_scores.get(aid, atleta.media),
                    titular=True,
                )
            )

    titulares, reservas = _separate_starters(selected, layout, atletas)
    custo = sum(p.preco for p in selected)
    esperado = sum(p.expected for p in selected)

    return OptimizedLineup(
        formacao=formacao,
        cartoletas_usadas=round(custo, 2),
        cartoletas_disponiveis=round(cartoletas, 2),
        pontuacao_esperada=round(esperado, 2),
        titulares=sorted(titulares, key=lambda p: _pos_order(p.posicao)),
        reservas=sorted(reservas, key=lambda p: _pos_order(p.posicao)),
        status="optimal",
    )


def _group_by_position(atletas: list[Atleta]) -> dict[str, list[Atleta]]:
    groups: dict[str, list[Atleta]] = {}
    for a in atletas:
        groups.setdefault(a.posicao, []).append(a)
    return groups


def _filter_position(
    variables: dict[int, LpVariable],
    atletas: list[Atleta],
    posicao: str,
) -> list[int]:
    return [aid for aid in variables if _lookup_pos(atletas, aid) == posicao]


def _lookup_pos(atletas: list[Atleta], atleta_id: int) -> str:
    for a in atletas:
        if a.atleta_id == atleta_id:
            return a.posicao
    return "desconhecido"


def _lookup_preco(atletas: list[Atleta], atleta_id: int) -> float:
    for a in atletas:
        if a.atleta_id == atleta_id:
            return a.preco
    return 0.0


def _lookup_clube(atletas: list[Atleta], atleta_id: int) -> int:
    for a in atletas:
        if a.atleta_id == atleta_id:
            return a.clube_id
    return 0


def _find_atleta(atletas: list[Atleta], atleta_id: int) -> Optional[Atleta]:
    for a in atletas:
        if a.atleta_id == atleta_id:
            return a
    return None


def _separate_starters(
    selected: list[SelectedPlayer],
    layout: dict[str, int],
    atletas: list[Atleta],
) -> tuple[list[SelectedPlayer], list[SelectedPlayer]]:
    by_pos = _group_selected(selected)

    titulares = []
    reservas = []

    for pos_name, count in layout.items():
        pos_players = by_pos.get(pos_name, [])
        pos_players.sort(key=lambda p: p.expected, reverse=True)
        titulares.extend(pos_players[:count])
        reservas.extend(pos_players[count:])

    gks = by_pos.get("goleiro", [])
    tecs = by_pos.get("tecnico", [])
    gks.sort(key=lambda p: p.expected, reverse=True)
    tecs.sort(key=lambda p: p.expected, reverse=True)

    if gks:
        gk_tit = gks[0]
        if gk_tit not in titulares:
            titulares.append(gk_tit)
        reservas.extend(gks[1:])
    if tecs:
        tec_tit = tecs[0]
        if tec_tit not in titulares:
            titulares.append(tec_tit)
        reservas.extend(tecs[1:])

    for p in reservas:
        p.titular = False

    return titulares, reservas


def _group_selected(players: list[SelectedPlayer]) -> dict[str, list[SelectedPlayer]]:
    groups: dict[str, list[SelectedPlayer]] = {}
    for p in players:
        groups.setdefault(p.posicao, []).append(p)
    return groups


def _pos_order(posicao: str) -> int:
    return {
        "goleiro": 0,
        "zagueiro": 1,
        "lateral": 2,
        "meia": 3,
        "atacante": 4,
        "tecnico": 5,
    }.get(posicao, 9)
