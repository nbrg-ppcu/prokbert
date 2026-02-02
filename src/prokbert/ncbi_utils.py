import re
import unicodedata
from typing import List

import pandas as pd


def _slugify(s: str, *, max_len: int = 40) -> str:
    """
    Convert an arbitrary string into a safe, compact, ASCII identifier.

    Normalization rules:
      - lowercase
      - strip accents / non-ASCII characters
      - replace any run of non-alphanumeric characters with '_'
      - collapse multiple '_' into one
      - trim leading/trailing '_'
      - optionally truncate to `max_len`

    Parameters
    ----------
    s : str
        Input string (e.g., taxon name). If None/empty, returns "unknown".
    max_len : int, optional (default=40)
        Maximum length of the output slug. If 0/None, no truncation.

    Returns
    -------
    str
        A slug suitable for filenames / HF repo IDs, e.g. "klebsiella_pneumoniae".
    """
    if s is None:
        s = ""
    s = str(s).strip().lower()

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "unknown"


def taxon_tag(
    taxa_df: pd.DataFrame,
    taxon_id: int,
    *,
    name_col: str = "name",
    id_col: str = "taxon_id",
    prefix: str = "tax",
    max_name_len: int = 40,
) -> str:
    """
    Build a stable tag for a taxon to embed into model/run names.

    Format:
        '{prefix}{taxon_id}_{slugified_taxon_name}'

    Example:
        taxon_tag(taxa, 573) -> 'tax573_klebsiella'

    Parameters
    ----------
    taxa_df : pd.DataFrame
        Taxonomy node table. Must contain at least `id_col` and `name_col`.
    taxon_id : int
        NCBI taxon id.
    name_col : str, optional
        Column containing the taxon name.
    id_col : str, optional
        Column containing the taxon id.
    prefix : str, optional
        Prefix prepended before the numeric id (default: "tax").
    max_name_len : int, optional
        Max length used when slugifying the name.

    Returns
    -------
    str
        Deterministic identifier string.
    """
    rows = taxa_df.loc[taxa_df[id_col] == int(taxon_id), name_col]
    if len(rows) == 0:
        name = "unknown"
    else:
        name = rows.iloc[0]

    return f"{prefix}{int(taxon_id)}_{_slugify(name, max_len=max_name_len)}"


def get_descendants_including_self(
    taxa2uppertaxa: pd.DataFrame,
    query_taxon_id: int,
) -> List[int]:
    """
    Return all taxa for which `query_taxon_id` is an ancestor, plus the query itself.

    This uses the transitive-closure table `taxa2uppertaxa`, where each row means:
        taxon_id has ancestor asc_taxon_id (at asc_rank)

    Descendants are extracted by selecting rows where:
        asc_taxon_id == query_taxon_id
    and collecting their taxon_id values.

    Parameters
    ----------
    taxa2uppertaxa : pd.DataFrame
        Must contain columns: ['taxon_id', 'asc_taxon_id'].
    query_taxon_id : int
        The taxon that defines the subtree root.

    Returns
    -------
    List[int]
        Sorted unique list of descendant taxon_ids including the query_taxon_id.
    """
    query_taxon_id = int(query_taxon_id)
    desc = (
        taxa2uppertaxa.loc[taxa2uppertaxa["asc_taxon_id"] == query_taxon_id, "taxon_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if query_taxon_id not in desc:
        desc.append(query_taxon_id)
    return sorted(set(desc))


def get_ascendants_including_self(
    taxa2uppertaxa: pd.DataFrame,
    query_taxon_id: int,
) -> List[int]:
    """
    Return all ancestors of `query_taxon_id`, plus the query itself.

    This uses the transitive-closure table `taxa2uppertaxa`, where each row means:
        taxon_id has ancestor asc_taxon_id (at asc_rank)

    Ancestors are extracted by selecting rows where:
        taxon_id == query_taxon_id
    and collecting their asc_taxon_id values.

    Parameters
    ----------
    taxa2uppertaxa : pd.DataFrame
        Must contain columns: ['taxon_id', 'asc_taxon_id'].
    query_taxon_id : int
        Taxon for which to retrieve the ancestor chain.

    Returns
    -------
    List[int]
        Sorted unique list of ancestor taxon_ids including query_taxon_id.
    """
    query_taxon_id = int(query_taxon_id)
    asc = (
        taxa2uppertaxa.loc[taxa2uppertaxa["taxon_id"] == query_taxon_id, "asc_taxon_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if query_taxon_id not in asc:
        asc.append(query_taxon_id)
    return sorted(set(asc))
