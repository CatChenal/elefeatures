"""
Implementation of core functionalities for the mcce-features package.
"""
import csv
from collections import Counter
from functools import partial
import logging
from pathlib import Path
from re import split as re_split
from typing import Union

from numpy import float32 as np_float32
from numpy import number as np_number
import pandas as pd
from rich.traceback import install

from .features import MCCEFeatureExtractor


# Configure rich to render tracebacks for cleaner CLI output without modifying click globals
install(show_locals=False)

read_tsv = partial(pd.read_csv, sep="\t")

FEATURES_TSV = "mcce_elefeatures.tsv"
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
        raise ValueError(f"No folders found in {folders_fp.parent}")

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


def collate_features_files(sims_dir: Union[str,Path],
                           tsv_lst: list,
                           collated_tsv: str = FEATURES_TSV
):
    """Collate all features files in tsv_lst that have data
    into sims_dir/collated_tsv.
    """
    hdr = None
    sims_dir = Path(sims_dir)
    out_path = sims_dir.joinpath(collated_tsv)
    with open(out_path, "w") as fo:
        for tsv in tsv_lst:
            lines = sims_dir.joinpath(tsv).read_text().splitlines()
            if not lines:
                continue
            if hdr is None:
                hdr = lines[0]
                fo.write(f"{hdr}\n")
    
            data_lines = lines[1:]
            if not data_lines:
                continue
            logging.info(f">> Collating {tsv!s}")
            fo.writelines("\n".join(line for line in data_lines))

    logging.info(f"Collated features into {out_path}")

    return


def extract_subfolders_with_book(sims_dir: str = ".",
                                 subfolders_startwith: str = "",
                                 collated_tsv_name: str = FEATURES_TSV):
    """
    Run extract_folders in all subfolders of a set of simulations 
    folders (sims_dir) that have a book.txt file.
    Expected sim_dir subfolders' structure:
        sim1_dir/
          - [runs/]  #prot folders may be under runs/; used if found
          - PDB1/
          - PDB2/
          ...
          - [book.txt]
        sim2_dir/
         [same as above]

    1. Get all the dirs with book.txt
    2. Run extract_folders(book_fp)
    3. Collate all mcce_elefeatures.tsv files
    """
    # list of book filepaths:
    collated_tsv_is_default = collated_tsv_name == FEATURES_TSV
    dirs_lst = get_sims_dirs(sims_dir=sims_dir, subfolders_startwith=subfolders_startwith)
    if not dirs_lst:
        msg = f"{Path(sims_dir).resolve().name}: No subfolders with a book.txt file."
        logging.info(msg)
        return

    tsv_lst = []
    for book_fp in dirs_lst:
        tsv_fp = book_fp.parent/FEATURES_TSV
        extract_folders(book_fp, tsv_fp)

        if tsv_fp.exists():
            tsv_lst.append(tsv_fp)
        else:
            if book_fp.parent.name=="runs":
                missing_tsv = f"{book_fp.parent.parent.name}/runs"
            else:
                missing_tsv = book_fp.parent.name
            tsv_lst.append(f"# {missing_tsv}: No features tsv file.")

    if tsv_lst:
        # save list
        if subfolders_startwith:
            feats_files_fp = Path(sims_dir).joinpath(f"{subfolders_startwith}_features_filepaths.txt")
            if collated_tsv_is_default:   # add prefix:
                collated_tsv_name = f"{subfolders_startwith}_{FEATURES_TSV}"
        else:
            feats_files_fp = Path(sims_dir).joinpath("features_filepaths.txt")

        with open(feats_files_fp, "w") as fh:
            fh.write("\n".join(f"{tsv!s}" for tsv in tsv_lst) + "\n")

        collate_features_files(sims_dir, tsv_lst, collated_tsv_name)
    else:
        logging.warning("No features files found.")

    return
