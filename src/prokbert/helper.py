from typing import Literal, Any

import os
import gzip
import yaml
import json
import random
import zipfile
import pathlib
import urllib.request
from functools import partial
from mimetypes import guess_type
from pydantic import TypeAdapter

import torch
import datasets
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from prokbert.types import Contig, ContigMetaData


FASTA_EXTENSIONS = ('.fasta', '.fa', '.fna', '.fasta.gz', '.fa.gz', '.fna.gz')


def set_seed(seed: int = 43) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    with open(path) as f:
        return yaml.safe_load(f)


def load_file(path: str) -> list[Any]:

    if not isinstance(path, str):
        raise ValueError(f"Expected file_path to be a string, got '{type(path)}'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")

    panda_readers = {
        '.csv': pd.read_csv,
        '.parquet': pd.read_parquet,
        '.jsonl': partial(pd.read_json, lines=True),
    }

    if path.endswith(FASTA_EXTENSIONS):
        return load_fasta(path)
    elif os.path.isdir(path):
        ds = datasets.load_from_disk(path)
        if isinstance(ds, datasets.DatasetDict):
            raise TypeError(f"{path} contains a DatasetDict with splits {list(ds)}, expected a single Dataset.")
        return convert_from(ds, format="datasets")
    elif reader := next((r for ext, r in panda_readers.items() if path.endswith(ext)), None):
        rows = reader(path).to_dict(orient="records")
        return convert_from(rows, format="pandas")
    else:
        raise ValueError(
            f"Invalid file extension for {path}. "
            f"Supported extensions are fasta, pandas and hugginface extensions."
        )


def load_fasta(path: str):
    _, encoding = guess_type(path)
    o = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
    with o(path) as f:
        data = list(SeqIO.parse(f, "fasta"))
    return data


def read_json(file_path) -> Any:
    file_path = pathlib.Path(file_path)
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {file_path}") from e


def save_json(data, dir_path, file_name: str) -> None:
    dir_path = pathlib.Path(dir_path)
    file_path = dir_path / file_name
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4, default=str)
    except OSError as e:
        raise OSError(f"Could not save file: {file_path}") from e


def save_sequence(sequence, dir_path: str, file_name: str) -> None:
    arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    path = os.path.join(dir_path, file_name)
    np.save(path, arr)


def load_sequence(path: str, to_string: bool = False) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if to_string:
        return arr.tobytes().decode("ascii")
    return arr


def convert_to(
    data: list[dict] | list[Contig] | list[ContigMetaData],
    format: Literal["pandas", "datasets"] = "pandas",
) -> pd.DataFrame | datasets.Dataset:
    if format == "pandas":
        return pd.DataFrame(data)
    elif format == "datasets":
        return datasets.Dataset.from_list(data)
    else:
        raise ValueError(
                f"Invalid value for return_as: {format}. "
                f"Supported values are 'pandas', 'datasets'."
            )


def convert_from(
    data: pd.DataFrame | datasets.Dataset,
    format: Literal["pandas", "datasets"] = "pandas",
    return_as: str = "contig"
) -> list[Contig] | list[ContigMetaData]:
    if format == "pandas" and isinstance(data, pd.DataFrame):
        metadata = data.to_dict(orient="records")
    elif format == "datasets" and isinstance(data, datasets.Dataset):
        metadata =  data.to_list()
    else:
        raise ValueError(
            f"Invalid value for format: {format}. "
            f"Supported values are 'pandas', 'datasets'."
        )
    if return_as == "contig":
        contig_adapter = TypeAdapter(list[Contig])
    elif return_as == "contigmetadata":
        contig_adapter = TypeAdapter(list[ContigMetaData])
    else:
        raise ValueError(f"Invalid value for return_as param. Got '{return_as}'. ")

    return contig_adapter.validate_python(metadata)


def download(
    url: str,
    target_dir: str,
    file_name: str | None = None,
    unzip: bool = False,
    remove_download: bool = False
) -> None:
    """
    Download a file from a URL to a local directory with a real-time progress bar.

    Creates the target directory if it doesn't exist. Streams the download in
    8 KB chunks to avoid loading the entire file into memory. Optionally extracts
    the downloaded archive and/or removes it after extraction.

    Args:
        url (str): The URL of the file to download.
        target_dir (str): Local directory path where the file will be saved.
            Created automatically if it does not already exist.
        file_name (Optional[str]): Name to save the file as. If None, the
            filename is inferred from the URL's basename.
        unzip (bool): If True, extract the downloaded file as a ZIP archive
            into `target_dir` after download. Defaults to False.
        remove_download (bool): If True, delete the downloaded file after
            extraction. Only meaningful when `unzip` is True. Defaults to False.

    Returns:
        None

    Raises:
        urllib.error.URLError: If the URL is unreachable or the request fails.
        urllib.error.HTTPError: If the server returns an HTTP error status.
        zipfile.BadZipFile: If `unzip=True` but the file is not a valid ZIP.
        OSError: If the target directory cannot be created or the file cannot
            be written or deleted.

    Example:
        >>> download(
        ...     url="https://example.com/data.zip",
        ...     target_dir="./data",
        ...     unzip=True,
        ...     remove_download=True
        ... )
    """
    os.makedirs(target_dir, exist_ok=True)

    file_name = file_name or os.path.basename(url)
    download_path = os.path.join(target_dir, file_name)

    with urllib.request.urlopen(url) as source, open(download_path, "wb") as output:
        with tqdm(
            desc="Downloading data",
            total=int(source.info().get("Content-Length") or 0),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if unzip:
        unzip_file(download_path, target_dir)
    if remove_download:
        os.remove(download_path)


def unzip_file(download_path: str, target_dir: str) -> None:
    """
    Extract all contents of a ZIP archive to a directory with a progress bar.

    Iterates over each entry in the archive individually so that tqdm can
    report per-file extraction progress. The directory structure encoded in
    the archive is preserved.

    Args:
        download_path (str): Absolute or relative path to the `.zip` file to
            extract.
        target_dir (str): Directory into which all archive contents are
            extracted. Must already exist; use `os.makedirs` beforehand if
            needed.

    Returns:
        None

    Raises:
        FileNotFoundError: If `download_path` does not point to an existing file.
        zipfile.BadZipFile: If the file at `download_path` is not a valid ZIP
            archive.
        OSError: If extraction fails due to permission errors or disk space.

    Example:
        >>> unzip_file("./data/archive.zip", "./data/")
    """
    with zipfile.ZipFile(download_path, "r") as zip_ref:

        zip_info_list = zip_ref.infolist()
        total_files = len(zip_info_list)

        with tqdm(desc="Unzipping data", total=total_files, unit='file', ncols=80) as progress_bar:
            for zip_info in zip_info_list:
                zip_ref.extract(zip_info, target_dir)
                progress_bar.update(1)


def estimate_tensor_memory(x, dtype=None, unit="GB"):
    if dtype is None:
        dtype = x.dtype
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    bytes_used = x.numel() * dtype_bytes
    units = { "B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    if unit not in units:
        raise ValueError(f"Unknown unit {unit}")
    return bytes_used / units[unit]
