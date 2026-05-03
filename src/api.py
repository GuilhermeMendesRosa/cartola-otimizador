import requests
from dataclasses import dataclass, field
from typing import Optional
import time

BASE_URL = "https://api.cartolafc.globo.com"
TIMEOUT = 15
RETRY_DELAY = 2

POSITIONS = {
    1: "goleiro",
    2: "lateral",
    3: "zagueiro",
    4: "meia",
    5: "atacante",
    6: "tecnico",
}

FORMATION_LAYOUT = {
    "4-3-3": {"zagueiro": 2, "lateral": 2, "meia": 3, "atacante": 3},
    "3-4-3": {"zagueiro": 2, "lateral": 1, "meia": 4, "atacante": 3},
    "3-5-2": {"zagueiro": 2, "lateral": 1, "meia": 5, "atacante": 2},
    "4-4-2": {"zagueiro": 2, "lateral": 2, "meia": 4, "atacante": 2},
    "5-3-2": {"zagueiro": 3, "lateral": 2, "meia": 3, "atacante": 2},
}


@dataclass
class Atleta:
    atleta_id: int
    nome: str
    apelido: str
    slug: str
    posicao_id: int
    clube_id: int
    status_id: int
    preco: float
    pontos: float
    media: float
    variacao: float
    jogos: int
    entrou_em_campo: bool

    @property
    def posicao(self) -> str:
        return POSITIONS.get(self.posicao_id, "desconhecido")


@dataclass
class Partida:
    partida_id: int
    clube_casa_id: int
    clube_visitante_id: int
    partida_data: str
    local: str
    valida: bool
    aproveitamento_mandante: list[str]
    aproveitamento_visitante: list[str]
    clube_casa_posicao: int | None = None
    clube_visitante_posicao: int | None = None


@dataclass
class MercadoStatus:
    rodada_atual: int
    status_mercado: int
    cartoleta_inicial: float
    fechamento_timestamp: int
    times_escalados: int
    game_over: bool


def _request(endpoint: str, retries: int = 2) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"Falha ao acessar {url} após {retries + 1} tentativas: {e}"
                ) from e


def get_mercado_status() -> MercadoStatus:
    data = _request("mercado/status")
    return MercadoStatus(
        rodada_atual=data["rodada_atual"],
        status_mercado=data["status_mercado"],
        cartoleta_inicial=data["cartoleta_inicial"],
        fechamento_timestamp=data["fechamento"]["timestamp"],
        times_escalados=data["times_escalados"],
        game_over=data["game_over"],
    )


def get_atletas_mercado() -> tuple[list[Atleta], dict, dict]:
    data = _request("atletas/mercado")
    clubes = data.get("clubes", {})
    posicoes = data.get("posicoes", {})

    atletas = []
    for raw in data.get("atletas", []):
        atletas.append(
            Atleta(
                atleta_id=raw["atleta_id"],
                nome=raw["nome"],
                apelido=raw["apelido"],
                slug=raw["slug"],
                posicao_id=raw["posicao_id"],
                clube_id=raw["clube_id"],
                status_id=raw.get("status_id", 7),
                preco=float(raw["preco_num"]),
                pontos=float(raw.get("pontos_num", 0)),
                media=float(raw.get("media_num", 0)),
                variacao=float(raw.get("variacao_num", 0)),
                jogos=int(raw.get("jogos_num", 0)),
                entrou_em_campo=raw.get("entrou_em_campo", False),
            )
        )

    return atletas, clubes, posicoes


def get_partidas(rodada: Optional[int] = None) -> list[Partida]:
    endpoint = f"partidas/{rodada}" if rodada else "partidas"
    data = _request(endpoint)

    return [
        Partida(
            partida_id=p["partida_id"],
            clube_casa_id=p["clube_casa_id"],
            clube_visitante_id=p["clube_visitante_id"],
            partida_data=p["partida_data"],
            local=p["local"],
            valida=p["valida"],
            aproveitamento_mandante=p.get("aproveitamento_mandante", []),
            aproveitamento_visitante=p.get("aproveitamento_visitante", []),
            clube_casa_posicao=p.get("clube_casa_posicao"),
            clube_visitante_posicao=p.get("clube_visitante_posicao"),
        )
        for p in data.get("partidas", [])
    ]
