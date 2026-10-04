from importlib.metadata import version, PackageNotFoundError, metadata
import logging
import typer

from . import core


try:
    __version__ = version("mcce-features")
    meta = metadata("mcce-features")
    __email__ = meta.get("Author-email", "unknown")
except PackageNotFoundError:
    __author__ = "unknown"
    __email__ = "unknown"
    __version__ = "unknown"


def print_version():
    typer.echo(f"mcce-features version: {__version__}")
    typer.echo(f"mcce-features developer: {__email__}")


app = typer.Typer(
    help="mcce-features: Extract electrostatic features from MCCE output files or other sources.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True
)

# -----------------------------
# Logging setup
# -----------------------------
def setup_logging(level: str = "INFO"):
    """Configure root logger with simple text format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s]: %(message)s",
    )


# -----------------------------
# Global callback
# -----------------------------
@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
):
    """Configure logging before running any command."""
    setup_logging(log_level)

# -----------------------------
# Global version command
# -----------------------------
@app.command("version")
def version_cmd():
    """Show mcce-features version."""
    print_version()


@app.command("extract")
def extract(
    mcce_folder: str = typer.Argument(...,
                                      help="Path to the MCCE working directory.")
):
    """Extract electrostatic features from a MCCE working directory."""
    core.extract(mcce_folder)

@app.command("extract-folders")
def extract_folders_cmd(
    folder_file: str = typer.Argument(
        ...,
        help=("Path to a (space, comma, tab) delimited file with a "
              "MCCE folder path/str in the first column of each line.")
                                      ),
    output_file: str = typer.Option(core.FEATURES_TSV,
                                    help="Path to the output TSV file.")
):
    """Extract electrostatic features from multiple MCCE folders."""
    core.extract_folders(folder_file, output_file)


@app.command("extract-subfolders-with-book")
def extract_subfolders_cmd(
    simulations_folder: str = typer.Argument(
        ...,
        help=("Path to a folder with a collection of simulation folders. "
              "Those with a book.txt file will be processed."),
    ),
    subfolders_startwith: str = typer.Option(
        "",
        help=("Only process subfolders in simulations_folder whose names startwith this string. "
              "Enables collating over chunked datasets."),
    ),
):
    """Extract electrostatic features from multiple sets of MCCE folders."""
    core.extract_subfolders_with_book(simulations_folder, subfolders_startwith)

@app.command("collate-tsv")
def collate_tsv_cmd(
    simulations_folder: str = typer.Argument(
        ...,
        help=("Path to a folder with a collection of simulation folders. "),
    ),
    feat_filepaths_file: str = typer.Option(
        help="File listing the tsv filepaths (str) to collate."
    ),
    collated_features_tsv: str = typer.Option(
        core.COLLATED_FEATURES_TSV,
        help="Collated features TSV filename (not filepath)."
    ),
    batch_size: int = typer.Argument(
        10,
        help=("Batch size"),
    ),
):
    """Collate electrostatic features from multiple sets of MCCE folders."""
    core.collate_features_files(simulations_folder,
                                feat_filepaths_file,
                                collated_tsv_name=collated_features_tsv,
                                batch_size=batch_size)


if __name__ == "__main__":
    app()
