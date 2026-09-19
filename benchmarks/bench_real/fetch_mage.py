"""Extract the MAGE v1.0 expression, covariate and metadata members from the 59 GB Zenodo zip.

Only the zip's central directory and the wanted members are downloaded: one HTTP range request per member.
Each member's CRC32 is checked, and its sha256 is written to PROVENANCE.json.
"""
import hashlib
import io
import json
import pathlib
import struct
import sys
import urllib.request
import zipfile
import zlib

URL = "https://zenodo.org/records/10535719/files/MAGE.v1.0.data.zip?download=1"
ZIP_MD5 = "9b32d1e24aa883b3dc57359598420203"
PREFIX = "MAGE.v1.0.data/"
MEMBERS = (
    "README",
    "sample_library_info/README",
    "sample_library_info/sample.metadata.MAGE.v1.0.txt",
    "sample_library_info/sequencing_library.metadata.MAGE.v1.0.txt",
    "QTL_results/README",
    "QTL_results/eQTL_results/README",
    "QTL_results/eQTL_results/expression_filteredGenes.MAGE.v1.0.txt.gz",
    "QTL_results/eQTL_results/expression_quants/README",
    "QTL_results/eQTL_results/expression_quants/eQTL_covariates.tab.gz",
    "QTL_results/eQTL_results/expression_quants/inverse_normal_TMM.filtered.TSS.MAGE.v1.0.bed.gz",
    "QTL_results/eQTL_results/expression_quants/TMM.filtered.TSS.MAGE.v1.0.bed.gz",
    "QTL_results/eQTL_results/expression_quants/raw_pseudocounts.filtered.TSS.MAGE.v1.0.bed.gz",
    "QTL_results/eQTL_results/eQTL_summary.MAGE.v1.0.txt.gz",
    "dataset_comparison/README",
    "dataset_comparison/MAGE_Geuvadis_GTEx_AFGR.top_PCs.txt.gz",
)
# Local file header layout (PKWARE APPNOTE 4.3.7): 30 fixed bytes; name and extra-field lengths at offsets 26 and 28.
LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")


def byte_range(url: str, start: int, stop: int):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop - 1}"}))


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; zipfile uses it to read only the end record and directory."""

    def __init__(self, url: str):
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.url = head.url
        self.size = int(head.headers["Content-Length"])
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence] + offset
        return self.position

    def readinto(self, buffer):
        stop = min(self.position + len(buffer), self.size)
        if stop <= self.position:
            return 0
        data = byte_range(self.url, self.position, stop).read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def extract(remote: HttpRangeFile, info: zipfile.ZipInfo, target: pathlib.Path):
    header = LOCAL_HEADER.unpack(byte_range(remote.url, info.header_offset, info.header_offset + LOCAL_HEADER.size).read())
    data_start = info.header_offset + LOCAL_HEADER.size + header[9] + header[10]
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if info.compress_type == zipfile.ZIP_DEFLATED else None
    digest, crc = hashlib.sha256(), 0
    response = byte_range(remote.url, data_start, data_start + info.compress_size)
    with open(target, "wb") as sink:
        for chunk in iter(lambda: response.read(io.DEFAULT_BUFFER_SIZE), b""):
            plain = decompressor.decompress(chunk) if decompressor else chunk
            crc = zlib.crc32(plain, crc)
            digest.update(plain)
            sink.write(plain)
        if decompressor:
            tail = decompressor.flush()
            crc = zlib.crc32(tail, crc)
            digest.update(tail)
            sink.write(tail)
    if crc != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")
    return digest.hexdigest()


def main(out_dir: str):
    out = pathlib.Path(out_dir)
    remote = HttpRangeFile(URL)
    directory = zipfile.ZipFile(remote)
    provenance = {"source": URL, "zenodo_record": "10535719", "zip_md5": ZIP_MD5, "files": {}}
    for member in MEMBERS:
        info = directory.getinfo(PREFIX + member)
        target = out / member
        target.parent.mkdir(parents=True, exist_ok=True)
        provenance["files"][member] = {"bytes": info.file_size, "crc32": f"{info.CRC:08x}", "sha256": extract(remote, info, target)}
        print(member, info.file_size, flush=True)
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
