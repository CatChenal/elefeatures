"""
Implementation of core functionalities for the mcce-features package.
"""
import csv
from collections import Counter
from functools import partial
import logging
import multiprocessing as mp
from pathlib import Path
from re import split as re_split
from typing import List, Union

import pandas as pd
from rich.traceback import install

from .features import MCCEFeatureExtractor


# Configure rich to render tracebacks for cleaner CLI output without modifying click globals
install(show_locals=False)

read_tsv = partial(pd.read_csv, sep="\t")

FEATURES_TSV = "mcce_elefeatures.tsv"
COLLATED_FEATURES_TSV = "collated_mcce_elefeatures.tsv"
FEATURE_FILEPATHS_FNAME = "features_filepaths.txt"

BOOK_HEADER_LINES = 3
BOOK_FOOTER_LINES = 5


def extract(mcce_folder: str,
            verbose: bool = True) -> tuple:
    """Extract electrostatic features from MCCE output files in the specified folder."""
    logging.info(f"Starting feature extraction from MCCE folder: {mcce_folder}")

    extractor = MCCEFeatureExtractor()
    extractor.mcce_folder = mcce_folder
    if extractor.missing_sources():
        logging.critical("At least one required source file is missing.")
        return None, None
    features = extractor.extract_all_features(folder=mcce_folder)
    feature_names = extractor.feature_names
    if verbose:
        for name, feature in zip(feature_names, features):
            print(f"{name}: {feature}")

    return feature_names, features


def get_book_data_bounds(book_fp: Path) -> tuple:
    """Return the start and end indices of the body in
    a book.txt file for slicing lines.
    Usage:
        l1, l2 = get_book_data_bounds(book_fp)
        if l1 is not None:
            lines = book_fp.read_text().splitlines()[l1:l2]
        else:
            lines = book_fp.read_text().splitlines()

    Retruns a valued 2-tuple if the book has 2 separator lines 
    for the header and footer as in pro_batch book.txt, else
    the tuple values are both None.
    """
    sep_line = "--------------------------------------"
    txt = book_fp.read_text()
    if txt.count(sep_line) == 2:
        # book from pro_batch has header/footer delineated by sep_line
        return BOOK_HEADER_LINES, -BOOK_FOOTER_LINES
    return None, None


def extract_folders(
    folder_file: str,
    output_file: str = FEATURES_TSV,
):
    """
    Extract electrostatic features from multiple MCCE folders.

    Input:
        folder_file:
            Text file with one MCCE folder path per line, or a
            (space, comma, tab) delimited file with the folder 
            name in the first column.

    Output:
        TSV file with columns:
            mcce_folder, feature_1, feature_2, ...
    """
    folders_fp = Path(folder_file).resolve()
    book_parent = folders_fp.parent

    is_book = folders_fp.name == "book.txt"
    other_book = False
    if is_book:
        l1, l2 = get_book_data_bounds(folders_fp)
        other_book = l1 is None  # book not from pro_batch

    if not is_book or other_book:
        lines = folders_fp.read_text().splitlines()
    else:
        lines = folders_fp.read_text().splitlines()[l1:l2]
    folder_paths = [book_parent.joinpath(re_split(r"[ ,\t]+", line)[0])
                    for line in lines
                    if line.strip() and not line.strip().startswith("#")
                    ]

    if not folder_paths:
        logging.warning(f"No folders found in {folders_fp.parent}")
        return

    folder_name_counts = Counter(folder.name for folder in folder_paths)
    duplicate_names = {
        name: count
        for name, count in folder_name_counts.items()
        if count > 1
    }
    if duplicate_names:
        duplicate_summary = ", ".join(
            f"{name} ({count})"
            for name, count in sorted(duplicate_names.items())
        )
        logging.warning(
            "Duplicate MCCE folder names encountered; output mcce_folder values may be ambiguous: %s",
            duplicate_summary,
        )

    logging.info(f"Found {len(folder_paths)} MCCE folders to process in {folders_fp.parent}")

    rows = []
    all_feature_names = []
    rows_written = 0

    for i, mcce_folder in enumerate(folder_paths, start=1):
        logging.info(f"[{i:,}/{len(folder_paths):,}] Processing {mcce_folder}")
        try:
            feature_names, features = extract(mcce_folder, verbose=False)
            if feature_names is not None:
                feature_row = dict(zip(feature_names, features))
                rows.append((mcce_folder.name, feature_row))
                for feature_name in feature_names:
                    if feature_name not in all_feature_names:
                        all_feature_names.append(feature_name)
                rows_written += 1
        except Exception as exc:
            logging.exception(f"Failed to process {mcce_folder}: {exc}")
            continue

    if rows_written == 0:
        logging.error("No features extracted, and do not write the output file")
        return

    with open(output_file, "w", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["mcce_folder"] + all_feature_names)
        for mcce_folder_name, feature_row in rows:
            writer.writerow(
                [mcce_folder_name] +
                [feature_row.get(feature_name, "") for feature_name in all_feature_names]
            )

    logging.info(f"Wrote feature table to {output_file}\n")

    return


def get_sims_dirs(sims_dir: str=".", subfolders_startwith: str = "") -> list:
    """Get the list of subfolders paths with a book file.
    """
    dirs_lst = []
    for dir in Path(sims_dir).iterdir():
        if not dir.is_dir():
            continue
        if subfolders_startwith and not dir.name.startswith(subfolders_startwith):
            continue

        if (dir/"runs").is_dir():
            if (dir/"runs"/"book.txt").exists():
                dirs_lst.append(dir/"runs"/"book.txt")
        elif (dir/"book.txt").exists():
            dirs_lst.append(dir/"book.txt")

    return dirs_lst


def extract_subfolders_with_book(sims_dir: str = ".",
                                 subfolders_startwith: str = "",):
    """
    Run extract_folders in all subfolders of a set of simulations 
    folders (sims_dir) that have a book.txt file.
    Expected sim_dir subfolders' structure:
        sim1_dir/
          - [runs/]  # prot folders may be under runs/; used if found
          - PDB1/
          - PDB2/
          ...
          - [book.txt]
        sim2_dir/
         [same as above]

    1. Get all the dirs with book.txt
    2. Run extract_folders(book_fp)
    3. Save list of tsv files to collate in file with name ending with 'features_filepaths.txt'
       for next command 'collate-tsv'.
    """
    # list of book filepaths:
    dirs_lst = get_sims_dirs(sims_dir=sims_dir, subfolders_startwith=subfolders_startwith)
    sims_dir = Path(sims_dir).resolve()

    if not dirs_lst:
        logging.warning(f"{sims_dir.name}: No subfolders with a book.txt file.")
        return

    tsv_lst = []
    for book_fp in dirs_lst:
        tsv_fp = book_fp.parent/FEATURES_TSV
        extract_folders(book_fp, tsv_fp)

        if tsv_fp.exists():
            tsv_lst.append(str(tsv_fp))
        else:
            if book_fp.parent.name=="runs":
                missing_tsv = f"{book_fp.parent.parent.name}/runs"
            else:
                missing_tsv = str(book_fp.parent.name)
            tsv_lst.append(f"# {missing_tsv}: No features tsv file.")

    if tsv_lst:
        # save list for separate command 'collate-tsv'
        if subfolders_startwith:
            feats_filepaths_fp = sims_dir.joinpath(f"{subfolders_startwith}_features_filepaths.txt")
        else:
            feats_filepaths_fp = sims_dir.joinpath("features_filepaths.txt")

        logging.info(f"Saving list of tsv files to collate to: {feats_filepaths_fp!s}...")
        with open(feats_filepaths_fp, "w") as fh:
            fh.write("\n".join(f"{tsv!s}" for tsv in tsv_lst) + "\n")
    else:
        logging.warning("No features files found.")

    return


def _read_csv(files: List[str]):
    """Multiprocessing pooling function
    """
    return pd.concat([pd.read_csv(filename) for filename in files])


def mp_collate_features_files(tsv_lst: List[str],
                              collated_fp: Path,
                              batch: int = 10,
):
    """Collate all features files in tsv_lst into sims_dir/collated_tsv.
    """
    n_files = len(tsv_lst)
    logging.info(f"Collating {n_files:,}...")
    with mp.Pool(mp.cpu_count()) as pool:
        dfs = pool.map(_read_csv, (tsv_lst[i:i+batch] for i in range(0, n_files, batch)))

    df = pd.concat(dfs)
    df.to_csv(collated_fp, index=False, sep="\t")
    logging.info(f"Collated features into {collated_fp!s}")

    return


def collate_features_files(sims_dir: Union[str,Path],
                           feat_filepaths_file: Union[str,Path],
                           collated_tsv_name: str = COLLATED_FEATURES_TSV,
                           batch_size: int = 10,
):
    """Collate all features files listed in feat_filepaths_file into sims_dir/collated_tsv_name.
    """
    sims_dir = Path(sims_dir)
    feat_filepaths_fp = sims_dir.joinpath(feat_filepaths_file)
    if not feat_filepaths_fp.exists():
        logging.critical("The filepaths file does not exists.")
        return

    tsv_lst = [fp for fp in feat_filepaths_fp.read_text().splitlines()
               if not fp.startswith("#") and Path(fp).exists()]
    if not tsv_lst:
        logging.critical("Empty list from filepaths file. All lines are commented? Paths not found?")
        return

    mp_collate_features_files(tsv_lst, sims_dir.joinpath(collated_tsv_name), batch=batch_size)

    logging.info("Collation over.")

    return


