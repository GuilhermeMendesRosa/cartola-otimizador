import click

from .api import (
    get_mercado_status,
    get_atletas_mercado,
    get_partidas,
    FORMATION_LAYOUT,
)
from .data_collector import (
    collect_current_round,
    collect_historical,
    build_clubes_csv,
    build_atletas_csv,
)
from .predictor import compute_expected_scores
from .optimizer import optimize_lineup
from .formatter import print_lineup, print_status
from .backtest import run_backtest, print_backtest_results, run_multi_experiment_backtest, print_multi_experiment_results


@click.group()
def cli():
    """Cartola Otimizador - Monte a melhor escalação do Cartola FC."""


@cli.command()
def collect():
    """Coleta TODAS as rodadas do Cartola e salva localmente."""
    click.echo("Coletando dados do Cartola FC...")

    status = get_mercado_status()
    rodada_atual = status.rodada_atual

    click.echo(f"Atualizando rodadas 1 a {rodada_atual - 1}...")
    n = collect_historical(from_round=1, to_round=rodada_atual - 1)
    click.echo(f"✓ {n} novas rodadas históricas coletadas.")

    data = collect_current_round()
    click.echo(f"✓ Rodada {data['rodada']} atual salva com {len(data['atletas'])} atletas.")
    click.echo(f"✓ {len(data['partidas'])} partidas registradas.")
    build_clubes_csv(data.get("clubes", {}))
    build_atletas_csv()
    click.echo("✓ CSV consolidado gerado com sucesso.")


@cli.command()
@click.option("--cartoletas", "-c", default=100.0, type=float, help="Cartoletas disponíveis.")
@click.option(
    "--formacao", "-f", default="4-3-3", type=click.Choice(list(FORMATION_LAYOUT.keys())),
    help="Formação tática.",
)
@click.option("--incluir", "-i", multiple=True, type=str, help="Slug de atleta para forçar na escalação.")
@click.option("--excluir", "-e", multiple=True, type=str, help="Slug de atleta para banir da escalação.")
@click.option("--max-clube", "-m", default=4, type=int, help="Máximo de atletas do mesmo clube.")
def scale(cartoletas, formacao, incluir, excluir, max_clube):
    """Encontra a melhor escalação com as cartoletas disponíveis."""
    click.echo("Carregando dados do Cartola...")

    atletas, clubes, _ = get_atletas_mercado()
    partidas = get_partidas()

    forced_ids = []
    banned_ids = []
    for a in atletas:
        if a.slug in incluir:
            forced_ids.append(a.atleta_id)
        if a.slug in excluir:
            banned_ids.append(a.atleta_id)

    click.echo(f"Calculando projeções para {len(atletas)} atletas...")
    expected = compute_expected_scores(atletas, partidas)

    click.echo(f"Otimizando escalação (formação {formacao}, C$ {cartoletas:.0f})...")
    lineup = optimize_lineup(
        atletas=atletas,
        expected_scores=expected,
        clubes=clubes,
        cartoletas=cartoletas,
        formacao=formacao,
        forced_ids=forced_ids if forced_ids else None,
        banned_ids=banned_ids if banned_ids else None,
        max_por_clube=max_clube,
    )

    if lineup is None:
        click.secho(
            "Não foi possível encontrar uma escalação. Tente mais cartoletas ou outra formação.",
            fg="red",
        )
        return

    print_lineup(lineup)


@cli.command()
def status():
    """Exibe o status atual do mercado."""
    s = get_mercado_status()
    print_status(s)


@cli.command()
@click.option("--cartoletas", "-c", default=130.0, type=float, help="Cartoletas para simulação.")
@click.option("--formacao", "-f", default="4-3-3", type=click.Choice(list(FORMATION_LAYOUT.keys())), help="Formação.")
def backtest(cartoletas, formacao):
    """Backtest: testa as predições contra resultados reais de rodadas passadas."""
    results, aggregated = run_backtest(cartoletas=cartoletas, formacao=formacao)
    print_backtest_results(results, aggregated)


@cli.command()
@click.option("--cartoletas", "-c", default=130.0, type=float, help="Cartoletas para simulação.")
@click.option("--formacao", "-f", default="4-3-3", type=click.Choice(list(FORMATION_LAYOUT.keys())), help="Formação.")
def experiment(cartoletas, formacao):
    """Compara múltiplas configurações do preditor via backtest."""
    results = run_multi_experiment_backtest(cartoletas=cartoletas, formacao=formacao)
    print_multi_experiment_results(results, cartoletas)


def main():
    cli()


if __name__ == "__main__":
    main()
