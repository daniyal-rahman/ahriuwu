"""Parse and generate RADS releasemanifest (RLSM) files.

Why: the repacked 4.20 client ships 5GB of assets under solutions/.../deploy but
no RADS project tree. Symlinking the assets into a fabricated project satisfies
project collection, but then RADS refuses every file with "not in the manifest" --
it uses releasemanifest as the index of files that are allowed to exist. So we
generate one that actually enumerates the asset tree.

Layout (verified by round-tripping lol_patcher's real manifest byte-for-byte):
  magic "RLSM", u32 version, u32 projectId, u32 releaseVersion
  u32 dirCount,  dirs[]  = (nameIdx, subdirStart, subdirCount, fileStart, fileCount)
  u32 fileCount, files[] = (nameIdx, version, md5[16], flags, sizeRaw, sizeComp, unk)
  u32 stringCount, u32 stringBlobLen, blob of NUL-terminated names
Directory 0 is the root; children are contiguous runs indexed by subdirStart.
"""
import struct, sys, os

DIR_FMT = "<IIIII"        # 20 bytes: nameIdx, subdirStart, subdirCount, fileStart, fileCount
FILE_FMT = "<II16sIIIQ"   # 44 bytes: nameIdx, version, md5, flags, sizeRaw, sizeComp, FILETIME
FILE_SIZE = 44


def parse(path):
    d = open(path, "rb").read()
    assert d[:4] == b"RLSM", "not an RLSM file"
    ver, proj, rel = struct.unpack("<III", d[4:16])
    off = 16
    ndir = struct.unpack("<I", d[off:off + 4])[0]; off += 4
    dirs = []
    for _ in range(ndir):
        dirs.append(struct.unpack(DIR_FMT, d[off:off + 20])); off += 20
    nfile = struct.unpack("<I", d[off:off + 4])[0]; off += 4
    files = []
    for _ in range(nfile):
        files.append(struct.unpack(FILE_FMT, d[off:off + FILE_SIZE])); off += FILE_SIZE
    nstr, blob_len = struct.unpack("<II", d[off:off + 8]); off += 8
    blob = d[off:off + blob_len]
    names = blob.split(b"\x00")[:nstr]
    return dict(version=ver, proj=proj, rel=rel, dirs=dirs, files=files,
                names=[n.decode("latin-1") for n in names], consumed=off + blob_len,
                total=len(d))


def build(root, proj_id, rel_version):
    """Walk `root` and emit an RLSM describing every file in it."""
    names, name_idx = [], {}

    def intern(s):
        if s not in name_idx:
            name_idx[s] = len(names)
            names.append(s)
        return name_idx[s]

    dirs, files = [], []
    # breadth-first so each directory's children occupy one contiguous run
    queue = [("", root)]
    dir_records = []
    while queue:
        rel_path, abs_path = queue.pop(0)
        try:
            entries = sorted(os.listdir(abs_path))
        except OSError:
            entries = []
        subs = [e for e in entries if os.path.isdir(os.path.join(abs_path, e))]
        fls = [e for e in entries if os.path.isfile(os.path.join(abs_path, e))]
        rec = dict(name=intern(os.path.basename(rel_path) if rel_path else ""),
                   subs=subs, files=fls, abs=abs_path)
        dir_records.append(rec)
        for s in subs:
            queue.append((os.path.join(rel_path, s), os.path.join(abs_path, s)))

    # assign contiguous child ranges in the same BFS order
    next_dir = 1
    for rec in dir_records:
        rec["subdir_start"] = next_dir
        next_dir += len(rec["subs"])
    for rec in dir_records:
        rec["file_start"] = len(files)
        for f in rec["files"]:
            try:
                sz = os.path.getsize(os.path.join(rec["abs"], f))
            except OSError:
                sz = 0
            # md5 left zeroed: RADS uses the manifest as an existence index here,
            # and zeroed digests are accepted for local (already-deployed) files.
            files.append((intern(f), rel_version, b"\x00" * 16, 0, sz, sz, 0))  # 7 fields incl. FILETIME
        dirs.append((rec["name"], rec["subdir_start"], len(rec["subs"]),
                     rec["file_start"], len(rec["files"])))

    blob = b"".join(n.encode("latin-1") + b"\x00" for n in names)
    out = [b"RLSM", struct.pack("<III", 0x10001, proj_id, rel_version),
           struct.pack("<I", len(dirs))]
    out += [struct.pack(DIR_FMT, *d) for d in dirs]
    out.append(struct.pack("<I", len(files)))
    out += [struct.pack(FILE_FMT, *f) for f in files]
    out.append(struct.pack("<II", len(names), len(blob)))
    out.append(blob)
    return b"".join(out), len(dirs), len(files)


if __name__ == "__main__":
    if sys.argv[1] == "parse":
        m = parse(sys.argv[2])
        print(f"proj={m['proj']} rel={m['rel']} dirs={len(m['dirs'])} "
              f"files={len(m['files'])} names={len(m['names'])}")
        print(f"consumed={m['consumed']} total={m['total']} "
              f"{'OK (exact)' if m['consumed'] == m['total'] else 'MISMATCH'}")
        print("sample names:", m["names"][:6])
    else:  # build <root> <projid> <relver> <out>
        data, nd, nf = build(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        open(sys.argv[5], "wb").write(data)
        print(f"wrote {sys.argv[5]}: {nd} dirs, {nf} files, {len(data)} bytes")
