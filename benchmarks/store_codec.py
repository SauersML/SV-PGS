"""Read throughput of the dosage store's codecs on public imputed codes, and the chunk-size fit.

Data [semi-real]: bench-sim v7 chr22 Beagle 5.5 codes (public 1kGP haplotype mosaics), the first
``rows`` records of every finished 5,000-sample batch, side by side.  For each codec and inner
chunk size R the codes are written as one store array in ``work`` (node-local), read once to
warm the page cache, then read in sequential requests of L records:

* ``cpu``: ``CodeArray.read_rows_into`` on one thread (raw: preadv; zstd: decode; rowdict: the
  CPU reference decoder);
* ``host``: rowdict's host share of the device path alone (preadv, crc32c, frame location), on
  one thread and on every task CPU;
* ``device``: ``CodeArray.read_rows_to_device`` end to end (host share, pinned copy, decode);
  for raw, preadv into pinned memory plus the copy, which is what the device path costs today.

The host share is fitted as seconds = tau * bytes + t_chunk * chunks + t_request * requests by
least squares over every (R, L); ``docs/design/math/codec.md`` derives the chunk size from it.
argv: output json, rows, work directory.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from sv_pgs import rowdict_codec
from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.dosage_store import _PINNED_POOL, CodeArray, CodeShardWriter, _pread_exact, create_code_array

BEAGLE = Path("/scratch.global/sauer354/svpgs-team/bench-sim/v7/cohort/chr22/beagle")
CHUNK_ROWS = (1, 4, 16, 64, 256, 1024)
REQUEST_ROWS = (64, 512, 4096)


def load_codes(rows: int) -> np.ndarray:
    names = sorted(path.name for path in BEAGLE.glob("codes*.npy") if (BEAGLE / f"batch{path.name[5:-4]}.done").exists())
    return np.ascontiguousarray(np.hstack([np.load(BEAGLE / name, mmap_mode="r")[:rows] for name in names]))


def write_array(directory: Path, codes: np.ndarray, codec: str, inner_rows: int) -> tuple[CodeArray, float]:
    shutil.rmtree(directory, ignore_errors=True)
    rows, samples = codes.shape
    shard_rows = -(-rows // inner_rows) * inner_rows
    begin = time.perf_counter()
    layout = create_code_array(directory, rows, samples, codec=codec, shard_rows=shard_rows, inner_rows=inner_rows)
    with CodeShardWriter(directory, layout, 0) as writer:
        writer.write_rows(codes)
    return CodeArray(directory), time.perf_counter() - begin


def stored_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file() and path.name != "zarr.json")


def requests(rows: int, request_rows: int) -> list[tuple[int, int]]:
    return [(start, min(rows, start + request_rows)) for start in range(0, rows, request_rows)]


def timed(function, *arguments) -> float:
    begin = time.perf_counter()
    function(*arguments)
    return time.perf_counter() - begin


def cpu_pass(array: CodeArray, spans: list[tuple[int, int]], out: np.ndarray) -> None:
    for start, stop in spans:
        array.read_rows_into(start, stop, out[: stop - start])


def host_share(array: CodeArray, span: tuple[int, int]) -> None:
    shard = array._shard(0)
    encoded = np.empty(array._encoded_bytes(shard, *span), dtype=np.uint8)
    array._rowdict_frames(shard, span[0], span[1], encoded)


def host_pass(array: CodeArray, spans: list[tuple[int, int]], threads: int) -> None:
    if threads == 1:
        for span in spans:
            host_share(array, span)
        return
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(lambda span: host_share(array, span), spans))


def main() -> None:
    out_path, rows, work = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    cupy = _try_import_cupy()
    codes = load_codes(rows)
    samples = codes.shape[1]
    decoded = codes.size
    threads = len(os.sched_getaffinity(0))
    report: dict = {"data": "[semi-real] bench-sim v7 chr22 Beagle 5.5 codes (public 1kGP mosaics)", "rows": rows,
                    "samples": samples, "task_cpus": threads, "gpu": None, "arrays": []}
    if cupy is not None:
        report["gpu"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        decoder = rowdict_codec.GpuRowDecoder(cupy)
    host_rows = []
    out = np.empty((max(REQUEST_ROWS), samples), dtype=np.uint8)
    for codec in ("raw", "zstd", "rowdict"):
        for inner_rows in CHUNK_ROWS if codec != "raw" else (max(REQUEST_ROWS),):
            directory = work / f"{codec}_{inner_rows}"
            array, write_seconds = write_array(directory, codes, codec, inner_rows)
            entry = {"codec": codec, "inner_rows": inner_rows, "write_seconds": write_seconds,
                     "bits_per_code": 8 * stored_bytes(directory) / decoded, "cpu_gbps": {}, "host_gbps": {},
                     "host_all_cpus_gbps": {}, "device_gbps": {}}
            check = np.empty_like(codes)
            array.read_rows_into(0, rows, check)
            if not np.array_equal(check, codes):
                raise AssertionError(f"{codec} R={inner_rows} does not read back exactly")
            del check
            for request_rows in REQUEST_ROWS:
                spans = requests(rows, request_rows)
                cpu_pass(array, spans, out)
                entry["cpu_gbps"][request_rows] = decoded / timed(cpu_pass, array, spans, out) / 1e9
                if codec == "rowdict":
                    host_pass(array, spans, 1)
                    seconds = timed(host_pass, array, spans, 1)
                    entry["host_gbps"][request_rows] = decoded / seconds / 1e9
                    entry["host_all_cpus_gbps"][request_rows] = decoded / timed(host_pass, array, spans, threads) / 1e9
                    shard = array._shard(0)
                    chunks = sum(array._chunk_range(*span)[1] - array._chunk_range(*span)[0] for span in spans)
                    read = sum(array._encoded_bytes(shard, *span) for span in spans)
                    host_rows.append({"inner_rows": inner_rows, "request_rows": request_rows, "seconds": seconds,
                                      "bytes": read, "chunks": chunks, "requests": len(spans)})
                if cupy is None:
                    continue
                target = cupy.empty((request_rows, samples), dtype=cupy.uint8)
                if codec == "rowdict":
                    def device_pass() -> None:
                        for start, stop in spans:
                            array.read_rows_to_device(start, stop, target[: stop - start], decoder)
                        cupy.cuda.get_current_stream().synchronize()
                elif codec == "raw":
                    owner, pinned = _PINNED_POOL.acquire(cupy, request_rows * samples)

                    def device_pass() -> None:
                        for start, stop in spans:
                            staged = pinned[: (stop - start) * samples]
                            _pread_exact(array._shard(0).descriptor, [memoryview(staged)], start * samples)
                            target[: stop - start].reshape(-1).set(staged)
                        cupy.cuda.get_current_stream().synchronize()
                else:
                    continue
                device_pass()
                entry["device_gbps"][request_rows] = decoded / timed(device_pass) / 1e9
                if codec == "rowdict":
                    first = spans[-1]
                    target[: first[1] - first[0]].fill(0)
                    array.read_rows_to_device(*first, target[: first[1] - first[0]], decoder)
                    if not np.array_equal(cupy.asnumpy(target[: first[1] - first[0]]), codes[first[0] : first[1]]):
                        raise AssertionError(f"rowdict R={inner_rows} decodes wrongly on the device")
                if codec == "raw":
                    _PINNED_POOL.release(owner)
            if codec == "rowdict" and cupy is not None:
                # The kernels alone, on the whole array already on the device.
                shard = array._shard(0)
                encoded = np.empty(array._encoded_bytes(shard, 0, rows), dtype=np.uint8)
                frames = array._rowdict_frames(shard, 0, rows, encoded)
                device_bytes = cupy.asarray(encoded)
                whole = cupy.empty((rows, samples), dtype=cupy.uint8)
                decoder.decode(device_bytes, frames, samples, whole)
                start_event, stop_event = cupy.cuda.Event(), cupy.cuda.Event()
                start_event.record()
                decoder.decode(device_bytes, frames, samples, whole)
                stop_event.record()
                stop_event.synchronize()
                entry["kernel_gbps"] = decoded / (cupy.cuda.get_elapsed_time(start_event, stop_event) / 1e3) / 1e9
                if not bool(cupy.array_equal(whole, cupy.asarray(codes))):
                    raise AssertionError(f"rowdict R={inner_rows} kernels decode wrongly")
                del whole, device_bytes
            array.close()
            shutil.rmtree(directory)
            report["arrays"].append(entry)
            print(json.dumps(entry), flush=True)
    design = np.array([[row["bytes"], row["chunks"], row["requests"]] for row in host_rows], dtype=np.float64)
    seconds = np.array([row["seconds"] for row in host_rows])
    # Relative least squares: every pass weighs the same whatever its length.
    coefficients, *_ = np.linalg.lstsq(design / seconds[:, None], np.ones_like(seconds), rcond=None)
    report["host_fit"] = {"tau_seconds_per_byte": coefficients[0], "chunk_seconds": coefficients[1],
                          "request_seconds": coefficients[2], "rows": host_rows,
                          "max_relative_residual": float(np.max(np.abs(design @ coefficients / seconds - 1)))}
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(report["host_fit"] | {"rows": None}), flush=True)


if __name__ == "__main__":
    main()
