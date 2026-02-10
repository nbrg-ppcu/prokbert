from typing import Tuple, Dict

import os
import json
import time
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd


class ShardedTokenStore:
    def __init__(self, root: str, dtype=np.uint16, max_open_shards: int = 32, seed: int = 1337, verbose: bool = True):
        t0 = time.time()
        self.verbose = verbose
        d = Path(root)
        self.dtype = np.dtype(dtype)
        self.itemsize = self.dtype.itemsize

        # core arrays
        self.shard_id    = np.load(d / "shard_id.npy")        # (N,), int32
        self.offsets_tok = np.load(d / "offsets_tok.npy")     # (N,), int64
        self.lengths_tok = np.load(d / "lengths_tok.npy")     # (N,), int32
        self.acc_b       = np.load(d / "acc.npy")             # (N,), |S*
        self.shard_sizes = np.load(d / "shard_sizes.npy")     # (S,), int64
        with open(d / "id_of.json") as fh:
            self.id_of = json.load(fh)
        self.root = d

        # LRU opener
        self._open = self._make_opener(max_open_shards)

        # length index (length-proportional sampling)
        t1 = time.time()
        self._lens_all = self.lengths_tok.astype(np.int64, copy=False)
        mask = self._lens_all > 0
        if not np.any(mask):
            raise ValueError("No positive-length contigs.")
        self._ids_pos  = np.nonzero(mask)[0].astype(np.int64, copy=False)   # valid contig ids
        self._lens_pos = self._lens_all[mask]
        self._cumsum   = np.cumsum(self._lens_pos, dtype=np.int64)
        self._total    = int(self._cumsum[-1])   # total tokens (len(store))
        self._rng      = np.random.default_rng(seed)

        if self.verbose:
            print(f"[store] loaded arrays in {t1-t0:.3f}s  "
                  f"(contigs={self.n_contigs}, shards={self.num_shards()}, total_tokens={self._total})")
            print(f"[store] built length-index in {time.time()-t1:.3f}s")

    # ----------- basic info -----------
    def __len__(self) -> int:
        """Total number of tokens across all contigs (sum of lengths)."""
        return self._total

    @property
    def n_contigs(self) -> int:
        return int(self.lengths_tok.shape[0])

    def num_shards(self) -> int:
        return int(self.shard_sizes.shape[0])

    def shards_info(self) -> Dict[str, object]:
        """Quick shard stats (counts, sizes)."""
        sizes = self.shard_sizes.astype(np.int64)
        return {
            "num_shards": int(sizes.shape[0]),
            "sizes_bytes": sizes.tolist(),
            "total_bytes": int(sizes.sum()),
            "dtype": str(self.dtype),
            "itemsize": int(self.itemsize),
        }

    # ----------- open shards with small LRU -----------
    def _make_opener(self, max_open):
        @lru_cache(maxsize=max_open)
        def _open_shard(sid: int):
            p = self.root / f"tokens.{int(sid):03d}.bin"
            size = os.path.getsize(p)
            mm = np.memmap(p, dtype=self.dtype, mode="r", shape=(size // self.itemsize,))
            if self.verbose:
                print(f"[store] opened shard {sid:03d} ({size/1e6:.1f} MB)")
            return mm
        return _open_shard

    # ----------- id/key mapping -----------
    def key_to_id(self, acc: str) -> int:
        return int(self.id_of[acc])

    def id_to_key(self, i: int) -> str:
        return self.acc_b[int(i)].decode("ascii")

    def contig_metadata_df(self) -> pd.DataFrame:
        """
        Per-contig metadata:
          cid, accession, shard_id, shard_file, offset_tok, offset_bytes, length_tok
        """
        N = self.lengths_tok.shape[0]
        shard_ids = self.shard_id.astype(np.int64, copy=False)
        shard_files = [str(self.root / f"tokens.{int(s):03d}.bin") for s in shard_ids]
        accs = [self.acc_b[i].decode("ascii") for i in range(N)]
        off_tok = self.offsets_tok.astype(np.int64, copy=False)
        lens    = self.lengths_tok.astype(np.int64, copy=False)

        df = pd.DataFrame({
            "cid":         np.arange(N, dtype=np.int64),
            "accession":   accs,
            "shard_id":    shard_ids,
            "shard_file":  shard_files,
            "offset_tok":  off_tok,
            "offset_bytes": off_tok * int(self.itemsize),
            "length_tok":  lens,
        })
        return df


    # ----------- core window fetch -----------
    def window_by_id(self, i: int, start: int, L: int, pad_id: int = 0) -> np.ndarray:
        i   = int(i)
        sid = int(self.shard_id[i])
        off = int(self.offsets_tok[i])
        clen= int(self.lengths_tok[i])

        if L >= clen:
            s_eff, L_eff = 0, clen
        else:
            if start < 0:
                s_eff = 0
            elif start > clen - L:
                s_eff = clen - L
            else:
                s_eff = int(start)
            L_eff = L

        out = np.full((L,), pad_id, dtype=self.dtype)
        if L_eff > 0:
            mm = self._open(sid)
            view = mm[off + s_eff : off + s_eff + L_eff]
            out[:L_eff] = view
        return out

    def window(self, acc: str, start: int, L: int, pad_id: int = 0) -> np.ndarray:
        return self.window_by_id(self.key_to_id(acc), start, L, pad_id)

    # ----------- fair (length-proportional) sampling -----------
    def set_seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def _pick_contig_index(self) -> int:
        r = int(self._rng.integers(0, self._total, endpoint=False))
        return int(np.searchsorted(self._cumsum, r, side="right"))

    def draw_pair(self, L: int) -> Tuple[int, int, int]:
        """One sample: (contig_id, start, L_eff). No cross-contig overlap."""
        j = self._pick_contig_index()
        cid  = int(self._ids_pos[j])
        clen = int(self._lens_all[cid])
        if L >= clen:
            return cid, 0, clen
        start = int(self._rng.integers(0, clen - L + 1))
        return cid, start, L

    def draw_window(self, L: int, pad_id: int = 0):
        """One random window (np.ndarray[L]), plus (cid, start, L_eff)."""
        cid, start, L_eff = self.draw_pair(L)
        w = self.window_by_id(cid, start, L, pad_id)
        return w, cid, start, L_eff

    # ----------- batch sampling (random, unbatched semantics) -----------
    def draw_batch_pairs(self, L: int, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        k independent samples. Returns:
          cids:   (k,) int64
          starts: (k,) int64
          L_effs: (k,) int64
        """
        # sample k global positions → contig indices
        r = self._rng.integers(0, self._total, size=int(k), endpoint=False)
        idxs = np.searchsorted(self._cumsum, r, side="right").astype(np.int64)
        cids = self._ids_pos[idxs].astype(np.int64, copy=False)
        clen = self._lens_all[cids].astype(np.int64, copy=False)

        L_arr = np.full_like(clen, L, dtype=np.int64)
        L_eff = np.minimum(L_arr, clen)  # truncate where L > clen

        starts = np.zeros_like(clen, dtype=np.int64)
        mask = L < clen
        if np.any(mask):
            spans = (clen[mask] - L + 1)
            starts_masked = self._rng.integers(0, spans, size=spans.shape[0]).astype(np.int64)
            starts[mask] = starts_masked
        # where L >= clen, starts remain 0

        return cids, starts, L_eff

    def draw_batch_windows(self, L: int, k: int, pad_id: int = 0, group_by_shard: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw k samples and fetch their windows.
        Returns:
          batch:  (k, L) array
          cids:   (k,)
          starts: (k,)
          L_eff:  (k,)
        """
        t0 = time.time()
        cids, starts, L_eff = self.draw_batch_pairs(L, k)
        out = np.full((int(k), int(L)), pad_id, dtype=self.dtype)

        if group_by_shard:
            # process by shard for locality
            sids = self.shard_id[cids].astype(np.int64, copy=False)
            order = np.argsort(sids, kind="stable")
            for r in order:
                i   = int(cids[r])
                out[r, :] = self.window_by_id(i, int(starts[r]), int(L), pad_id)
        else:
            for r in range(int(k)):
                i   = int(cids[r])
                out[r, :] = self.window_by_id(i, int(starts[r]), int(L), pad_id)

        if self.verbose:
            dt = time.time() - t0
            print(f"[store] draw_batch_windows k={k} L={L} -> {out.shape}, {dt*1e3:.1f} ms")

        return out, cids, starts, L_eff


def build_segmentdb(store, cids: np.ndarray, starts: np.ndarray, L_req: int, L_eff: np.ndarray | None = None) -> pd.DataFrame:
    """
    Build a DataFrame of sampled segments from precomputed arrays.
    Inputs:
      - store: ShardedTokenStore
      - cids:  (k,) contig ids (int)
      - starts:(k,) start coords (int)
      - L_req: requested window length (int, same for all)
      - L_eff: (k,) actual copied length (int) [optional]. If None, computed as min(L_req, contig_len).
    Output:
      segmentdb: DataFrame with columns:
        cid, accession, shard_id, shard_file, contig_len, start, L_req, L_eff, end
    """
    cids   = np.asarray(cids, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)

    contig_len = store.lengths_tok[cids].astype(np.int64, copy=False)
    if L_eff is None:
        L_eff = np.minimum(contig_len, int(L_req)).astype(np.int64, copy=False)
    else:
        L_eff = np.asarray(L_eff, dtype=np.int64)

    # Clamp end inside contig (safety; your sampler should already ensure this)
    end = starts + L_eff
    over = end > contig_len
    if np.any(over):
        # shift starts left so end == contig_len where needed
        shift = (end - contig_len)
        starts = starts.copy()
        starts[over] -= shift[over]
        end = starts + L_eff

    shard_ids  = store.shard_id[cids].astype(np.int64, copy=False)
    # Map shard_id -> file path string
    shard_files = [str(store.root / f"tokens.{int(s):03d}.bin") for s in shard_ids]
    accessions  = [store.id_to_key(int(i)) for i in cids]

    segmentdb = pd.DataFrame({
        "cid":        cids,
        "accession":  accessions,
        "shard_id":   shard_ids,
        "shard_file": shard_files,
        "contig_len": contig_len,
        "start":      starts,
        "L_req":      int(L_req),
        "L_eff":      L_eff,
        "end":        end,
    })
    return segmentdb
