from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .optimizer import OptimizedLineup, POS_LABEL_PT

console = Console()


def print_lineup(lineup: OptimizedLineup) -> None:
    title = Text("🏆  MELHOR ESCALAÇÃO", style="bold white on dark_green")
    info = Text(
        f"Cartoletas: C$ {lineup.cartoletas_disponiveis:.2f}  "
        f"|  Formação: {lineup.formacao}  "
        f"|  Status: {lineup.status}",
        style="bright_black",
    )
    panel = Panel(info, title=title, border_style="green", box=box.ROUNDED)
    console.print(panel)

    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style="bright_black",
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Posição", style="bold yellow", width=12)
    table.add_column("Jogador", style="bold white", width=28)
    table.add_column("Preço", justify="right", style="green", width=10)
    table.add_column("Média", justify="right", style="cyan", width=8)
    table.add_column("Projeção", justify="right", style="magenta", width=10)

    titulares = [p for p in lineup.titulares if p.posicao != "tecnico"]
    tecnicos = [p for p in lineup.titulares if p.posicao == "tecnico"]

    _add_section(table, titulares)
    table.add_section()
    _add_section(table, tecnicos)

    console.print(table)

    summary = Text()
    custo = lineup.total_custo
    esperado = lineup.total_esperado
    saldo = lineup.cartoletas_disponiveis - custo
    summary.append(f"Custo total: ", style="bright_black")
    summary.append(f"C$ {custo:.2f}", style="bold green")
    summary.append("  │  ", style="bright_black")
    summary.append(f"Projeção: ", style="bright_black")
    summary.append(f"{esperado:.1f} pts", style="bold magenta")
    if lineup.com_banco:
        summary.append("  │  ", style="bright_black")
        summary.append("Proteção banco: ", style="bright_black")
        summary.append(f"+{lineup.protecao_banco:.1f}", style="bold cyan")
    summary.append("  │  ", style="bright_black")
    summary.append(f"Saldo: ", style="bright_black")
    summary.append(f"C$ {saldo:.2f}", style="yellow")

    console.print(
        Panel(summary, border_style="bright_black", box=box.ROUNDED)
    )

    if lineup.reservas:
        console.print()
        res_table = Table(
            box=box.SIMPLE,
            border_style="bright_black",
            header_style="italic dim white",
        )
        res_table.add_column("Reservas", style="dim white", width=12)
        res_table.add_column("Jogador", style="dim white", width=28)
        res_table.add_column("Preço", justify="right", style="dim green", width=10)
        res_table.add_column("Projeção", justify="right", style="dim magenta", width=10)

        for p in sorted(lineup.reservas, key=lambda x: _reserve_order(x.posicao)):
            label = POS_LABEL_PT.get(p.posicao, p.posicao)
            res_table.add_row(
                label, f"{p.nome} ({p.clube})",
                f"C$ {p.preco:.2f}", f"{p.expected:.1f}",
            )

        console.print(Panel(res_table, border_style="dim", box=box.ROUNDED))


def _add_section(table: Table, players: list) -> None:
    for p in players:
        label = POS_LABEL_PT.get(p.posicao, p.posicao)
        table.add_row(
            label,
            f"{p.nome} ({p.clube})",
            f"C$ {p.preco:.2f}",
            f"{p.media:.1f}",
            f"{p.expected:.1f}",
        )


def print_status(status) -> None:
    table = Table(title="Status do Mercado", border_style="cyan", box=box.ROUNDED)
    table.add_column("Campo", style="bright_black")
    table.add_column("Valor", style="bold white")

    table.add_row("Rodada atual", str(status.rodada_atual))
    table.add_row("Mercado aberto", "Sim" if status.status_mercado == 1 else "Não")
    table.add_row("Cartoletas iniciais", f"C$ {status.cartoleta_inicial:.2f}")
    table.add_row("Times escalados", f"{status.times_escalados:,}")
    table.add_row("Game over", "Sim" if status.game_over else "Não")
    console.print(table)


def _reserve_order(posicao: str) -> int:
    return {
        "goleiro": 0,
        "zagueiro": 1,
        "lateral": 2,
        "meia": 3,
        "atacante": 4,
    }.get(posicao, 9)
