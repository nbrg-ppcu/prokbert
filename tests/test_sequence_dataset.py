"""Tests for prokbert.sequence_dataset.

Uses tests/data/extremophiles_mini.fasta, which contains 5 short contigs:

    sequence_id  contig_id    length  coordinate (half-open [start, end))
    0            OM913597.1   30      (0, 30)
    1            OM913598.1   24      (30, 54)
    2            OM913599.1   20      (54, 74)
    3            OQ832096.1   21      (74, 95)    lowercase letters and N in the FASTA
    4            JX507079.1   20      (95, 115)
"""

import pathlib

import pytest
import torch
import yaml

from prokbert.constants import BACKWARD, FORWARD
from prokbert.sequence_dataset import EmbeddingDataset, SequenceDataset


DATA_PATH = str(pathlib.Path(__file__).parent / "data" / "extremophiles_mini.fasta")

CONTIG_IDS = ["OM913597.1", "OM913598.1", "OM913599.1", "OQ832096.1", "JX507079.1"]
SEQUENCES = [
    "TCTCGACCCGTCCCACCGCAATCGCCCCCA",
    "TCTCGACCCGCCCCACCGCAACCT",
    "CTCGCCTCGCCTCGCTTTCA",
    "ACGTNACGTNGGATCCTTAAA",  # "ACGTNacgtnGGATCCttaAA" in the FASTA, uppercased on load
    "GATTACTCGCCTCGCCTCGC",
]
COORDINATES = [(0, 30), (30, 54), (54, 74), (74, 95), (95, 115)]
TOTAL_LENGTH = 115


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def seq_ds() -> SequenceDataset:
    ds = SequenceDataset()
    ds.create_dataset([DATA_PATH])
    return ds


def _write_config(path: pathlib.Path, pooling_length: int, special_tokens: list) -> str:
    config = {
        "dataset": "extremophiles_mini",
        "tokenizer": "LCATokenizer",
        "kmer": 6,
        "shift": 2,
        "model": "neuralbioinfo/prokbert-mini-long",
        "pooling_strategy": "mean",
        "pooling_length": pooling_length,
        "special_tokens": special_tokens,
    }
    config_path = path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return str(config_path)


def _write_index_embedding(path: pathlib.Path, n_rows: int, dim: int = 4) -> str:
    # every value in row i equals i, so a lookup shows exactly which rows it returned
    embedding = torch.arange(n_rows, dtype=torch.float32).unsqueeze(1).repeat(1, dim)
    embedding_path = path / "embedding.pt"
    torch.save(embedding, embedding_path)
    return str(embedding_path)


@pytest.fixture
def emb_ds(seq_ds, tmp_path) -> EmbeddingDataset:
    """kmer=6, shift=2, pooling_length=5, no special tokens -> 55 tokens, 11 pooled rows."""
    n_tokens = EmbeddingDataset.calculate_lca_num_tokens(TOTAL_LENGTH, 6, 2, special_tokens=0)
    return EmbeddingDataset(
        sequence_dataset=seq_ds,
        config_path=_write_config(tmp_path, pooling_length=5, special_tokens=[]),
        embedding_file=_write_index_embedding(tmp_path, n_tokens // 5),
    )


def _rows(embedding: torch.Tensor) -> list:
    return embedding[:, 0].int().tolist()


# --------------------------------------------------------------------------- #
# Loading contigs
# --------------------------------------------------------------------------- #

def test_load_contig_reads_all_records():
    contigs = SequenceDataset().load_contig(DATA_PATH)

    assert [c["contig_id"] for c in contigs] == CONTIG_IDS
    assert [c["sequence"] for c in contigs] == SEQUENCES
    assert all(c["orientation"] == FORWARD for c in contigs)


def test_load_contig_uppercases_sequence():
    contigs = SequenceDataset().load_contig(DATA_PATH)
    assert contigs[3]["sequence"] == "ACGTNACGTNGGATCCTTAAA"


def test_load_contig_keeps_description():
    contigs = SequenceDataset().load_contig(DATA_PATH)
    assert contigs[0]["description"] == "OM913597.1 Aeromonas phage vB_AspA_Bolek, synthetic genome"


def test_load_contigs_concatenates_files():
    contigs = SequenceDataset().load_contigs([DATA_PATH, DATA_PATH])
    assert [c["contig_id"] for c in contigs] == CONTIG_IDS * 2


# --------------------------------------------------------------------------- #
# convert_to
# --------------------------------------------------------------------------- #

def test_convert_to_list_returns_input():
    contigs = SequenceDataset().load_contig(DATA_PATH)
    assert SequenceDataset.convert_to(contigs, return_as="list") is contigs


def test_convert_to_pandas():
    contigs = SequenceDataset().load_contig(DATA_PATH)
    df = SequenceDataset.convert_to(contigs, return_as="pandas")

    assert len(df) == 5
    assert set(df.columns) == {"contig_id", "sequence", "orientation", "description"}
    assert df["sequence"].tolist() == SEQUENCES


def test_convert_to_datasets():
    contigs = SequenceDataset().load_contig(DATA_PATH)
    ds = SequenceDataset.convert_to(contigs, return_as="datasets")

    assert len(ds) == 5
    assert list(ds["contig_id"]) == CONTIG_IDS  # list(): newer `datasets` returns a Column


def test_convert_to_invalid_value_raises():
    with pytest.raises(ValueError, match="Invalid value for return_as"):
        SequenceDataset.convert_to([], return_as="numpy")


# --------------------------------------------------------------------------- #
# create_dataset
# --------------------------------------------------------------------------- #

def test_create_dataset_concatenates_sequences(seq_ds):
    assert len(seq_ds.sequence) == TOTAL_LENGTH
    assert seq_ds.sequence == "".join(SEQUENCES)


def test_create_dataset_metadata(seq_ds):
    assert [m["contig_id"] for m in seq_ds.metadata] == CONTIG_IDS
    assert [m["sequence_id"] for m in seq_ds.metadata] == list(range(5))
    assert [tuple(m["coordinate"]) for m in seq_ds.metadata] == COORDINATES


def test_create_dataset_coordinates_are_contiguous(seq_ds):
    for prev, curr in zip(seq_ds.metadata, seq_ds.metadata[1:]):
        assert prev["coordinate"][1] == curr["coordinate"][0]
    assert seq_ds.metadata[0]["coordinate"][0] == 0
    assert seq_ds.metadata[-1]["coordinate"][1] == TOTAL_LENGTH


def test_create_dataset_slices_give_back_contigs(seq_ds):
    for meta, expected in zip(seq_ds.metadata, SEQUENCES):
        start, end = meta["coordinate"]
        assert seq_ds.sequence[start:end] == expected


def test_create_dataset_sequence_ids_unique_across_files():
    ds = SequenceDataset()
    ds.create_dataset([DATA_PATH, DATA_PATH])

    assert [m["sequence_id"] for m in ds.metadata] == list(range(10))
    assert len(ds.sequence) == 2 * TOTAL_LENGTH
    assert tuple(ds.metadata[5]["coordinate"]) == (TOTAL_LENGTH, TOTAL_LENGTH + 30)


def test_create_dataset_twice_replaces_previous_data():
    ds = SequenceDataset()
    ds.create_dataset([DATA_PATH, DATA_PATH])
    ds.create_dataset([DATA_PATH])

    assert len(ds.sequence) == TOTAL_LENGTH
    assert len(ds.metadata) == 5
    assert ds._starts == [start for start, _ in COORDINATES]


# --------------------------------------------------------------------------- #
# Saving and loading
# --------------------------------------------------------------------------- #

def test_save_creates_files(tmp_path):
    save_dir = tmp_path / "dataset"
    SequenceDataset().create_dataset([DATA_PATH], save_dir=str(save_dir))

    assert (save_dir / "sequence.npy").exists()
    assert (save_dir / "metadata.json").exists()


def test_save_and_load_roundtrip_as_string(tmp_path):
    save_dir = str(tmp_path / "dataset")
    original = SequenceDataset()
    original.create_dataset([DATA_PATH], save_dir=save_dir)

    loaded = SequenceDataset()
    loaded.load_dataset(save_dir, to_string=True)

    assert loaded.sequence == original.sequence
    assert [m["contig_id"] for m in loaded.metadata] == CONTIG_IDS
    assert [m["sequence_id"] for m in loaded.metadata] == list(range(5))
    # JSON has no tuples, so coordinates come back as lists
    assert [tuple(m["coordinate"]) for m in loaded.metadata] == COORDINATES
    assert loaded._starts == [start for start, _ in COORDINATES]


def test_save_and_load_roundtrip_as_array(tmp_path):
    save_dir = str(tmp_path / "dataset")
    original = SequenceDataset()
    original.create_dataset([DATA_PATH], save_dir=save_dir)

    loaded = SequenceDataset()
    loaded.load_dataset(save_dir, to_string=False)

    assert len(loaded.sequence) == TOTAL_LENGTH
    assert loaded.sequence.tobytes().decode("ascii") == original.sequence


def test_loaded_dataset_supports_queries(tmp_path):
    save_dir = str(tmp_path / "dataset")
    SequenceDataset().create_dataset([DATA_PATH], save_dir=save_dir)

    loaded = SequenceDataset()
    loaded.load_dataset(save_dir, to_string=True)

    assert tuple(loaded.get_coordinates_from_sequence_id(1)) == (30, 54)
    assert loaded.get_sequence_id_from_start_coordinate(60) == 2
    assert loaded.get_sequence_from_metadata(0, 0, 10, BACKWARD) == "CGGGTCGAGA"


# --------------------------------------------------------------------------- #
# Coordinate lookups
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sequence_id, expected", list(enumerate(COORDINATES)))
def test_get_coordinates_from_sequence_id(seq_ds, sequence_id, expected):
    assert tuple(seq_ds.get_coordinates_from_sequence_id(sequence_id)) == expected


@pytest.mark.parametrize("sequence_id", [-1, 5, 100])
def test_get_coordinates_from_sequence_id_out_of_bounds(seq_ds, sequence_id):
    with pytest.raises(ValueError, match="out of bounds"):
        seq_ds.get_coordinates_from_sequence_id(sequence_id)


@pytest.mark.parametrize(
    "position, expected",
    [
        (0, 0),      # first position
        (29, 0),     # last position of contig 0
        (30, 1),     # first position of contig 1 (boundary)
        (53, 1),
        (54, 2),
        (74, 3),
        (95, 4),
        (114, 4),    # last position of the whole sequence
    ],
)
def test_get_sequence_id_from_start_coordinate(seq_ds, position, expected):
    assert seq_ds.get_sequence_id_from_start_coordinate(position) == expected


@pytest.mark.parametrize("position", [-1, TOTAL_LENGTH, 1000])
def test_get_sequence_id_from_start_coordinate_out_of_range(seq_ds, position):
    with pytest.raises(ValueError, match="not found"):
        seq_ds.get_sequence_id_from_start_coordinate(position)


def test_get_sequence_id_from_start_coordinate_empty_dataset():
    with pytest.raises(ValueError):
        SequenceDataset().get_sequence_id_from_start_coordinate(0)


def test_every_position_maps_to_its_contig(seq_ds):
    for meta in seq_ds.metadata:
        start, end = meta["coordinate"]
        for position in range(start, end):
            assert seq_ds.get_sequence_id_from_start_coordinate(position) == meta["sequence_id"]


# --------------------------------------------------------------------------- #
# Sequence extraction and reverse complement
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "sequence, expected",
    [
        ("ACGT", "ACGT"),                    # palindromic
        ("AAAA", "TTTT"),
        ("TCTCGACCCG", "CGGGTCGAGA"),
        ("GGATCC", "GGATCC"),                # BamHI site, its own reverse complement
        ("ACGTNACGTNGGATCCTTAAA", "TTTAAGGATCCNACGTNACGT"),  # N stays N
        ("acgt", "acgt"),                    # lowercase is kept lowercase
        ("", ""),
    ],
)
def test_reverse_complement(seq_ds, sequence, expected):
    assert seq_ds.reverse_complement(sequence) == expected


def test_reverse_complement_is_an_involution(seq_ds):
    assert seq_ds.reverse_complement(seq_ds.reverse_complement(seq_ds.sequence)) == seq_ds.sequence


@pytest.mark.parametrize("sequence_id", range(5))
def test_get_sequence_from_metadata_full_contig(seq_ds, sequence_id):
    length = len(SEQUENCES[sequence_id])

    forward = seq_ds.get_sequence_from_metadata(sequence_id, 0, length, FORWARD)
    backward = seq_ds.get_sequence_from_metadata(sequence_id, 0, length, BACKWARD)

    assert forward == SEQUENCES[sequence_id]
    assert backward == seq_ds.reverse_complement(SEQUENCES[sequence_id])


def test_get_sequence_from_metadata_region(seq_ds):
    assert seq_ds.get_sequence_from_metadata(0, 0, 10, FORWARD) == "TCTCGACCCG"
    assert seq_ds.get_sequence_from_metadata(0, 0, 10, BACKWARD) == "CGGGTCGAGA"
    assert seq_ds.get_sequence_from_metadata(0, 20, 30, BACKWARD) == "TGGGGGCGAT"


def test_get_sequence_from_metadata_uses_contig_relative_coordinates(seq_ds):
    # positions 0-5 of contig 1 and contig 4, not of the concatenated sequence
    assert seq_ds.get_sequence_from_metadata(1, 0, 5, FORWARD) == "TCTCG"
    assert seq_ds.get_sequence_from_metadata(4, 15, 20, FORWARD) == "CTCGC"


def test_get_sequence_from_metadata_single_base(seq_ds):
    assert seq_ds.get_sequence_from_metadata(3, 4, 5, FORWARD) == "N"
    assert seq_ds.get_sequence_from_metadata(3, 0, 1, BACKWARD) == "T"  # complement of A


@pytest.mark.parametrize(
    "sequence_id, start, end",
    [
        (2, 0, 50),    # longer than the contig (20 nt)
        (2, 0, 21),    # one past the end of the contig
        (0, -1, 5),    # negative start
    ],
)
def test_get_sequence_from_metadata_out_of_bounds(seq_ds, sequence_id, start, end):
    with pytest.raises(ValueError, match="out of bounds"):
        seq_ds.get_sequence_from_metadata(sequence_id, start, end, FORWARD)


def test_get_sequence_from_metadata_does_not_cross_into_next_contig(seq_ds):
    # contig 1 ends at 54 in the concatenated sequence; asking for 25 nt must not
    # return the first base of contig 2
    with pytest.raises(ValueError):
        seq_ds.get_sequence_from_metadata(1, 0, 25, FORWARD)


@pytest.mark.parametrize("start, end", [(5, 5), (10, 5)])
def test_get_sequence_from_metadata_empty_or_reversed_interval(seq_ds, start, end):
    with pytest.raises(ValueError, match="must be less than"):
        seq_ds.get_sequence_from_metadata(0, start, end, FORWARD)


def test_get_sequence_from_metadata_invalid_orientation(seq_ds):
    with pytest.raises(ValueError, match="Invalid orientation"):
        seq_ds.get_sequence_from_metadata(0, 0, 10, "sideways")


def test_get_sequence_from_metadata_invalid_sequence_id(seq_ds):
    with pytest.raises(ValueError, match="out of bounds"):
        seq_ds.get_sequence_from_metadata(5, 0, 10, FORWARD)


# --------------------------------------------------------------------------- #
# LCA token counts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "seq_len, kmer, shift, special_tokens, expected",
    [
        (12, 6, 2, 0, 4),       # AAGTCC, GTCCAG, CCAGGA, AGGATC
        (12, 6, 2, 2, 6),       # + [CLS], [SEP]
        (13, 6, 2, 0, 4),       # the last base does not fit into another k-mer
        (6, 6, 1, 0, 1),        # exactly one k-mer
        (5, 6, 2, 2, 2),        # shorter than one k-mer: only special tokens
        (0, 6, 2, 0, 0),
        (10, 1, 1, 0, 10),      # character-level tokenization
        (1000, 6, 1, 0, 995),
        (1000, 6, 2, 0, 498),
        (TOTAL_LENGTH, 6, 2, 0, 55),
    ],
)
def test_calculate_lca_num_tokens(seq_len, kmer, shift, special_tokens, expected):
    assert EmbeddingDataset.calculate_lca_num_tokens(seq_len, kmer, shift, special_tokens) == expected


def test_calculate_lca_num_tokens_matches_explicit_kmers():
    sequence = "AAGTCCAGGATCAAGATT"  # example from the ProkBERT paper
    for kmer in range(1, 8):
        for shift in range(1, 4):
            kmers = [sequence[i:i + kmer] for i in range(0, len(sequence) - kmer + 1, shift)]
            n_tokens = EmbeddingDataset.calculate_lca_num_tokens(
                len(sequence), kmer, shift, special_tokens=0
            )
            assert n_tokens == len(kmers), f"kmer={kmer}, shift={shift}"


def test_calculate_lca_num_tokens_callable_on_instance(emb_ds):
    assert emb_ds.calculate_lca_num_tokens(12, 6, 2) == 6


# --------------------------------------------------------------------------- #
# EmbeddingDataset
# --------------------------------------------------------------------------- #

def test_embedding_dataset_loads_config(emb_ds):
    assert emb_ds.config["kmer"] == 6
    assert emb_ds.config["shift"] == 2
    assert emb_ds.config["pooling_length"] == 5
    assert emb_ds.config["special_tokens"] == []


def test_embedding_dataset_loads_embedding(emb_ds):
    assert tuple(emb_ds.embedding.shape) == (11, 4)


def test_embedding_dataset_missing_config_key_raises(seq_ds, tmp_path):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"kmer": 6, "shift": 2}, f)

    with pytest.raises(KeyError):
        EmbeddingDataset(seq_ds, str(config_path), _write_index_embedding(tmp_path, 11))


@pytest.mark.parametrize(
    "sequence_id, expected_rows",
    [
        (0, [0, 1]),
        (1, [2, 3, 4]),
        (2, [5, 6]),
        (3, [7, 8]),
        (4, [9, 10]),
    ],
)
def test_get_embedding_full_contig(emb_ds, sequence_id, expected_rows):
    start, end = COORDINATES[sequence_id]
    embedding = emb_ds.get_embedding_from_sequence_id_with_coordinates(sequence_id, 0, end - start)

    assert _rows(embedding) == expected_rows
    assert embedding.shape[1] == 4


def test_get_embedding_full_contigs_cover_all_rows(emb_ds):
    rows = []
    for sequence_id, (start, end) in enumerate(COORDINATES):
        embedding = emb_ds.get_embedding_from_sequence_id_with_coordinates(sequence_id, 0, end - start)
        rows.extend(_rows(embedding))

    assert rows == list(range(11))


def test_get_embedding_region_inside_contig(emb_ds):
    # contig 1, positions [5, 20) -> [35, 50) in the concatenated sequence
    # -> 15 and 23 tokens -> pooled rows [3, 4)
    embedding = emb_ds.get_embedding_from_sequence_id_with_coordinates(1, 5, 20)
    assert _rows(embedding) == [3]


def test_get_embedding_last_partial_window_is_dropped(emb_ds):
    # both ends are rounded down to whole pooling windows, so a region shorter than one
    # pooling window (here 3 nt at the start of contig 0) gives no rows
    with pytest.raises(ValueError, match="out of bounds"):
        emb_ds.get_embedding_from_sequence_id_with_coordinates(0, 0, 3)


@pytest.mark.parametrize(
    "sequence_id, start, end",
    [
        (0, -1, 10),    # negative start
        (0, 10, 5),     # end before start
        (2, 0, 21),     # past the end of contig 2 (20 nt)
    ],
)
def test_get_embedding_invalid_coordinates(emb_ds, sequence_id, start, end):
    with pytest.raises(ValueError, match="out of bounds"):
        emb_ds.get_embedding_from_sequence_id_with_coordinates(sequence_id, start, end)


def test_get_embedding_invalid_sequence_id(emb_ds):
    with pytest.raises(ValueError, match="out of bounds"):
        emb_ds.get_embedding_from_sequence_id_with_coordinates(5, 0, 10)


def test_get_embedding_with_special_tokens_and_no_pooling(seq_ds, tmp_path):
    # with [CLS] and [SEP] every token index is shifted by 2, and without pooling
    # (pooling_length=1) there is one row per token
    n_tokens = EmbeddingDataset.calculate_lca_num_tokens(TOTAL_LENGTH, 6, 2, special_tokens=2)
    emb_ds = EmbeddingDataset(
        sequence_dataset=seq_ds,
        config_path=_write_config(tmp_path, pooling_length=1, special_tokens=["[CLS]", "[SEP]"]),
        embedding_file=_write_index_embedding(tmp_path, n_tokens),
    )

    embedding = emb_ds.get_embedding_from_sequence_id_with_coordinates(0, 0, 30)
    # contig 0 has 13 k-mers. The current code adds all special tokens to both ends of the
    # interval, so the lookup starts at row 2, although only [CLS] precedes the first k-mer.
    assert _rows(embedding) == list(range(2, 15))