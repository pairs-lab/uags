"""Fetch ONLY the model state_dict of the official LSeg demo checkpoint (demo_e200.ckpt,
3.10 GB, sha256 f21aeb55..., HF isl-org/lang-seg) with HTTP range requests, skipping the
optimizer state. The .ckpt is a torch zip archive: read the central directory and data.pkl,
find which storage records the 'state_dict' references, download only those (CRC32 checked
against the official central directory), and torch.save the state dict.

Usage: python3 scripts/fetch_lseg_weights.py data/weights/lseg/demo_e200_state_dict.pt [--inspect]
(the path that usplat/extractors/lseg.py reads; about 1.7 GB of the 3.1 GB checkpoint)
"""
import io
import pickle
import struct
import sys
import time
import zipfile
import zlib
import collections

import requests
import torch

URL = "https://huggingface.co/isl-org/lang-seg/resolve/main/demo_e200.ckpt"
SIZE = 3095944292


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.sess = requests.Session()
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def get(self, a, n):
        """bytes [a, a+n)"""
        if n <= 0:
            return b""
        for attempt in range(5):
            try:
                r = self.sess.get(self.url, headers={"Range": f"bytes={a}-{a + n - 1}"},
                                  timeout=120)
                assert r.status_code == 206, r.status_code
                b = r.content
                assert len(b) == n, (len(b), n)
                self.fetched += n
                return b
            except Exception as e:  # retry
                print("retry", attempt, e, flush=True)
                time.sleep(2 + 3 * attempt)
        raise RuntimeError("range fetch failed")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        b = self.get(self.pos, n)
        self.pos += len(b)
        return b

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[:len(b)] = b
        return len(b)


class TensorRef:
    def __init__(self, key, dtype, offset, size, stride):
        self.key, self.dtype, self.offset, self.size, self.stride = key, dtype, offset, size, stride


class StorageRef:
    def __init__(self, key, dtype, numel):
        self.key, self.dtype, self.numel = key, dtype, numel


class Dummy:
    def __init__(self, *a, **k):
        self.a = a

    def __setstate__(self, s):
        self.state = s


def rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, *rest):
    return TensorRef(storage.key, storage.dtype, offset, tuple(size), tuple(stride))


class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor_v2
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda t, rg, hooks: t
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if mod == "torch" and name.endswith("Storage"):
            return name
        if mod in ("builtins", "copyreg", "_codecs"):
            return super().find_class(mod, name)
        if mod == "torch" and name in ("float32", "float16", "int64", "bfloat16"):
            return getattr(torch, name)
        return type(f"D_{mod}_{name}".replace(".", "_"), (Dummy,), {})

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, stype, key, loc, numel = pid
        dt = {"FloatStorage": torch.float32, "HalfStorage": torch.float16,
              "LongStorage": torch.int64, "IntStorage": torch.int32,
              "BoolStorage": torch.bool, "DoubleStorage": torch.float64}[stype] \
            if isinstance(stype, str) else stype
        return StorageRef(key, dt, numel)


def main():
    out = sys.argv[1]
    inspect = "--inspect" in sys.argv
    f = RangeFile(URL, SIZE)
    zf = zipfile.ZipFile(f)
    infos = {i.filename: i for i in zf.infolist()}
    prefix = zf.infolist()[0].filename.split("/")[0]
    print("entries", len(infos), "prefix", prefix, "fetched", f.fetched, flush=True)
    pkl = zf.read(f"{prefix}/data.pkl")
    print("data.pkl bytes", len(pkl), flush=True)
    obj = U(io.BytesIO(pkl)).load()
    print("top keys", list(obj.keys()))
    sd = obj["state_dict"]
    keys = {}
    for k, t in sd.items():
        keys.setdefault(t.key, t.dtype)
    need = sorted(keys, key=lambda k: infos[f"{prefix}/data/{k}"].header_offset)
    total = sum(infos[f"{prefix}/data/{k}"].file_size for k in need)
    print("state_dict tensors", len(sd), "storages", len(need), "bytes", total, flush=True)
    dts = collections.Counter(str(t.dtype) for t in sd.values())
    print("dtypes", dts)
    groups = collections.Counter(k.split(".")[1] if k.startswith("net.") else k.split(".")[0]
                                 for k in sd)
    print("groups", groups)
    for k in list(sd)[:5] + [k for k in sd if "clip_pretrained" in k][:3]:
        print(" ", k, sd[k].size, sd[k].dtype)
    if inspect:
        return
    # download the needed records, CRC-checked. Records are coalesced into large range
    # requests (gaps < 4 MB between needed records are fetched and dropped).
    raw = {}
    t0 = time.time()
    spans = [(k, infos[f"{prefix}/data/{k}"]) for k in need]
    groups, cur = [], []
    for k, zi in spans:
        assert zi.compress_type == zipfile.ZIP_STORED
        if cur and (zi.header_offset - (cur[-1][1].header_offset + cur[-1][1].file_size) > (4 << 20)
                    or zi.header_offset + zi.file_size - cur[0][1].header_offset > (96 << 20)):
            groups.append(cur)
            cur = []
        cur.append((k, zi))
    if cur:
        groups.append(cur)
    print("range groups", len(groups), flush=True)
    for g in groups:
        a0 = g[0][1].header_offset
        last = g[-1][1]
        a1 = min(last.header_offset + 30 + 1024 + 4096 + last.file_size, SIZE)  # generous tail
        blob = bytearray()
        CH = 64 << 20
        while len(blob) < a1 - a0:
            blob += f.get(a0 + len(blob), min(CH, a1 - a0 - len(blob)))
        for k, zi in g:
            o = zi.header_offset - a0
            assert blob[o:o + 4] == b"PK\x03\x04", k
            nlen, elen = struct.unpack("<HH", blob[o + 26:o + 30])
            st = o + 30 + nlen + elen
            buf = bytes(blob[st:st + zi.file_size])
            assert len(buf) == zi.file_size, (k, len(buf), zi.file_size)
            crc = zlib.crc32(buf) & 0xFFFFFFFF
            assert crc == zi.CRC, (k, crc, zi.CRC)
            raw[k] = buf
        print(f"  group of {len(g)} ok, fetched {f.fetched / 1e9:.2f} GB {time.time() - t0:.0f}s",
              flush=True)
    assert len(raw) == len(need)
    stor = {k: torch.frombuffer(bytearray(raw[k]), dtype=keys[k]) for k in need}
    out_sd = collections.OrderedDict()
    for name, t in sd.items():
        s = stor[t.key]
        out_sd[name] = torch.as_strided(s, t.size, t.stride, t.offset).clone()
    torch.save(out_sd, out)
    print("saved", out, "tensors", len(out_sd), "total fetched GB", f.fetched / 1e9)


if __name__ == "__main__":
    main()
