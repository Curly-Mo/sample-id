"""Command line interface for sample_id"""
import click

from sample_id import main


@click.command()
@click.option('-v', '--verbose', count=True)
@click.option('--option', default='default_val', help='An example optional parameter')
def cli(option, verbose):
    click.echo(click.style('Hello sample_id!', fg='magenta', bold=True))
    click.echo('Verbosity level: {}'.format(verbose))
    click.echo('Optional param: {}'.format(option))
    result = main.main()
    click.echo('Main output: {}'.format(result))
