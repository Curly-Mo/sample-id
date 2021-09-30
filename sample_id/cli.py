"""Command line interface for sample_id"""
import logging

import click
import joblib
import matplotlib.pyplot as plt

from sample_id import main


@click.group()
@click.option("-v", "--verbose", count=True)
@click.pass_context
def cli(ctx, verbose):
    ctx.show_default = True
    initLogging(verbose)


@cli.command()
@click.argument("tracks", type=click.Path(exists=True), nargs=-1)
# @click.option("--tracks", type=click.Path(exists=True), nargs=-1, help="Either a directory or list of files")
@click.option("--hop-length", type=click.INT, default=256, help="Hop length for computing CQTgram")
@click.option("--octave-bins", type=click.INT, default=36, help="Number of bins per octave of CQT")
@click.option("--n-octaves", type=click.INT, default=7, help="Number of octaves for CQT")
@click.option("--fmin", type=click.INT, default=50, help="Starting frequency of CQT in Hz")
@click.option("--sr", type=click.INT, default=22050, help="Sampling frequency (audio will be resampled)")
@click.option("--algorithm", type=click.STRING, default="annoy", help="Approximate nearest neighbor algorithm")
@click.option("--dedupe/--no-dedupe", default=False, help="Remove similar keypoints per track")
@click.option("--contrast-thresh", type=click.FLOAT, default=5, help="Contrast threshold for SIFT detector")
@click.option("--save", type=click.Path(), default=None, help="Location to save model to disk")
@click.option("--save-spec/--no-save-spec", default=False, help="Save spectrograms with model")
def train(
    tracks,
    hop_length,
    octave_bins,
    n_octaves,
    fmin,
    sr,
    algorithm,
    dedupe,
    contrast_thresh,
    save,
    save_spec,
):
    return main.train_keypoints_for_paths(
        tracks,
        hop_length=hop_length,
        octave_bins=octave_bins,
        n_octaves=n_octaves,
        fmin=fmin,
        sr=sr,
        algorithm=algorithm,
        dedupe=dedupe,
        contrast_thresh=contrast_thresh,
        save=save,
        save_spec=save_spec,
    )


@cli.command()
@click.argument("tracks", type=click.Path(exists=True), nargs=-1)
@click.argument("model", type=click.Path(exists=True))
# @click.option("--tracks", type=click.Path(exists=True), nargs=-1, help="Either a directory or list of files")
# @click.option("--model", type=click.Path(exists=True, dir_okay=False), help="Location of saved model to query against")
@click.option("--abs-thresh", type=click.FLOAT, default=None, help="Absolute threshold for filtering matches")
@click.option("--ratio-thresh", type=click.FLOAT, default=None, help="Ratio threshold for filtering matches")
@click.option("--cluster-dist", type=click.FLOAT, default=1.0, help="Time in seconds for clustering matches")
@click.option(
    "--cluster-size",
    type=click.FLOAT,
    default=3,
    help="Minimum cluster size to be considered a sample",
)
@click.option(
    "--match-orientation",
    is_flag=True,
    default=False,
    help="Remove matches with differing orientations",
)
@click.option("--plot/--no-plot", default=True, help="Plot results")
@click.option("--plot-all-kp/--no-plot-all-kp", default=False, help="Plot all keypoints on spectrograms")
@click.option("--save/--no-save", default=False, help="Save plot")
@click.option("--save-results", type=click.Path(), default=None, help="path to save results to")
def query(
    tracks,
    model,
    abs_thresh,
    ratio_thresh,
    cluster_dist,
    cluster_size,
    match_orientation,
    plot,
    plot_all_kp,
    save,
    save_results,
):
    results = main.query_tracks_for_paths(
        tracks,
        model,
        abs_thresh=abs_thresh,
        ratio_thresh=ratio_thresh,
        cluster_dist=cluster_dist,
        cluster_size=cluster_size,
        match_orientation=match_orientation,
        plot=plot,
        plot_all_kp=plot_all_kp,
        save=save,
    )
    if save_results:
        joblib.dump(results, save_results)
    if plot:
        plt.show(block=True)
    return results


def initLogging(verbosity=0):
    """Setup logging with a given verbosity level"""
    # import logging.config
    # logging.config.fileConfig('logging_config.ini', disable_existing_loggers=False)
    logging.basicConfig()
    if verbosity == 0:
        logging.root.setLevel(logging.WARN)
    if verbosity == 1:
        logging.root.setLevel(logging.INFO)
    if verbosity > 1:
        logging.root.setLevel(logging.DEBUG)
