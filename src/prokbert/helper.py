from typing import Optional

import os
import zipfile
import random
import urllib.request

import torch
import numpy as np
from tqdm import tqdm


def set_seed(seed: int = 43) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download(
    url: str,
    target_dir: str,
    file_name: Optional[str] = None,
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
