import os
import sys


def profiling_enabled() -> bool:
    return os.environ.get("PROKBERT_PROFILE") == "1"


def file_size(*paths: str) -> int:
    """Total size of the given files on disk in bytes."""
    return sum(os.path.getsize(path) for path in paths)


def get_dict_size(d: dict) -> int:
    """Memory of a flat dict in bytes: the dict itself plus its values.

    sys.getsizeof(d) alone counts only the dict's own table (~200 bytes), not the
    strings it points to.
    """
    return sys.getsizeof(d) + sum(sys.getsizeof(value) for value in d.values())
