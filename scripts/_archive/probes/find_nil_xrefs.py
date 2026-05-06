"""Static RE pass on /tmp/pe_dump_16.8.766/.

Goal: find the function(s) that write the click-dest alloc, by walking xrefs
to the "<nil" literal (which sits at offset +0x0C of every click-dest alloc).

Pipeline:
  1. Locate "<nil\\0" occurrences in rdata.bin → list of RVAs.
  2. For each occurrence's in-memory address, scan text.bin for
     `lea reg, [rip+disp32]` instructions whose displacement resolves to that
     address. Each hit = an xref site.
  3. For each xref, walk backward to function start (look for `48 89 5C 24`,
     `40 53`, `55`, `48 81 EC` prologue) to identify owning function RVAs.
  4. Also scan text.bin for triple-mirror writes: look for movss/movups
     instructions writing at offsets +0x308 and +0x374 from the same base
     register within a small window (e.g., 200 bytes).

Output: candidate writer-function RVAs printed + JSON file.
"""
import sys, os, json, struct, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PE_DIR = "/tmp/pe_dump_16.8.766"
OUT_JSON = "/tmp/nil_xref_candidates.json"

def main():
    meta = json.load(open(os.path.join(PE_DIR, "sections.json")))
    base = int(meta["module_base"], 16)
    sections = {s["name"]: s for s in meta["sections"]}
    text = open(os.path.join(PE_DIR, "text.bin"), "rb").read()
    rdata = open(os.path.join(PE_DIR, "rdata.bin"), "rb").read()
    text_rva = sections[".text"]["rva"]
    rdata_rva = sections[".rdata"]["rva"]
    print(f"module base = 0x{base:X}")
    print(f".text RVA=0x{text_rva:X} size=0x{len(text):X}")
    print(f".rdata RVA=0x{rdata_rva:X} size=0x{len(rdata):X}")

    # Step 1: find "<nil" sequences in rdata (the literal is "<nil>\0")
    needle = b"<nil"
    nil_offsets = []
    pos = 0
    while True:
        i = rdata.find(needle, pos)
        if i == -1: break
        nil_offsets.append(i)
        pos = i + 1
    print(f"\n'<nil\\0' occurrences in rdata: {len(nil_offsets)}")
    nil_addrs = []
    for off in nil_offsets:
        addr = base + rdata_rva + off
        nil_addrs.append(addr)
        print(f"  rdata+0x{off:X}  (RVA 0x{rdata_rva+off:X}, addr 0x{addr:X})")

    # Step 2: scan text.bin for `lea reg, [rip+disp32]` whose target ∈ nil_addrs
    # x64 lea encoding: REX.W (48) | 8D | mod/rm | disp32
    # We target the common forms 48 8D 0D, 48 8D 05, 48 8D 15, 48 8D 1D etc.
    # Also `mov reg, [rip+disp32]` which is: 48 8B ... — but for a string literal
    # lea is the canonical pattern.
    nil_set = set(nil_addrs)
    text_base_addr = base + text_rva
    xrefs = []   # list of (text_offset, abs_addr_of_lea, target_addr)
    # Sweep every byte position; check 4-byte disp32 against the next instruction's RIP
    # Instruction start at offset i, lea is 7 bytes total: 48 8D <modrm> <disp32>
    # RIP at end of instruction = text_base_addr + i + 7
    n = len(text)
    print(f"\nScanning .text for lea xrefs to <nil> string... ({n} bytes)")
    # Collect all 48 8D positions first
    for i in range(n - 7):
        if text[i] == 0x48 and text[i+1] == 0x8D:
            modrm = text[i+2]
            # Mod=00, R/M=101 means [RIP+disp32]; modrm bits: top 2 = mod, mid 3 = reg, low 3 = rm
            mod = modrm >> 6
            rm = modrm & 0x07
            if mod == 0 and rm == 0x05:
                disp32 = struct.unpack("<i", text[i+3:i+7])[0]
                rip = text_base_addr + i + 7
                target = rip + disp32
                if target in nil_set:
                    xrefs.append((i, text_base_addr + i, target))
    print(f"  found {len(xrefs)} lea xrefs to <nil>")
    for off, addr, tgt in xrefs[:50]:
        print(f"    .text+0x{off:08X}  (addr 0x{addr:X})  -> 0x{tgt:X}")

    # Step 3: walk backward to find function-start prologue for each xref
    PROLOGUES = [
        b"\x48\x89\x5C\x24",     # mov [rsp+offset], rbx
        b"\x48\x89\x6C\x24",     # mov [rsp+offset], rbp
        b"\x48\x89\x74\x24",     # mov [rsp+offset], rsi
        b"\x48\x89\x7C\x24",     # mov [rsp+offset], rdi
        b"\x40\x53",             # push rbx
        b"\x40\x55",             # push rbp
        b"\x40\x56",             # push rsi
        b"\x40\x57",             # push rdi
        b"\x48\x83\xEC",         # sub rsp, imm8
        b"\x48\x81\xEC",         # sub rsp, imm32
        b"\x55",                 # push rbp
    ]
    INT3_PAD = b"\xCC"           # function alignment padding
    def find_func_start(off, maxback=4096):
        # Walk back at most maxback bytes; look for INT3 padding followed by a prologue
        for d in range(8, maxback, 1):
            if off - d < 16: break
            # CC pad indicates prev function ended
            if text[off-d] == 0xCC and text[off-d+1] != 0xCC:
                cand = off - d + 1
                # Is the candidate byte a recognized prologue?
                for pro in PROLOGUES:
                    if text[cand:cand+len(pro)] == pro:
                        return cand
                # No recognized prologue but post-CC is still a function boundary
                return cand
        return None

    func_starts = {}  # func_rva -> [xref RVAs]
    for off, addr, tgt in xrefs:
        fs = find_func_start(off)
        if fs is None:
            continue
        rva = text_rva + fs
        func_starts.setdefault(rva, []).append(text_rva + off)
    print(f"\nDistinct functions referencing <nil>: {len(func_starts)}")
    for rva, sites in sorted(func_starts.items())[:50]:
        print(f"  func RVA 0x{rva:X}  ({len(sites)} xref{'s' if len(sites)>1 else ''})")

    # Step 4: triple-mirror write pattern scan
    # Look for movss [reg+0x308] and movss [reg+0x374] within a 256-byte window from same base register.
    # movss [reg+disp32], xmm has encoding: F3 0F 11 <modrm with disp32>
    # For mod=10 (disp32 form), modrm = 10|reg|rm where rm<>5 (else SIB).
    # For [reg+disp8], encoding uses mod=01 with disp8. But 0x308 = 776 doesn't fit signed disp8,
    # so it's mod=10 disp32. We search for the byte pattern "F3 0F 11 ?? 08 03 00 00" (disp=0x308)
    # AND "F3 0F 11 ?? 74 03 00 00" (disp=0x374), with same base register.
    print("\nScanning for triple-mirror write pattern (movss disp 0x308 + 0x374)...")
    re_308 = re.compile(rb"\xF3\x0F\x11.\x08\x03\x00\x00", re.DOTALL)
    re_374 = re.compile(rb"\xF3\x0F\x11.\x74\x03\x00\x00", re.DOTALL)
    pos308 = [m.start() for m in re_308.finditer(text)]
    pos374 = [m.start() for m in re_374.finditer(text)]
    print(f"  movss [reg+0x308] occurrences: {len(pos308)}")
    print(f"  movss [reg+0x374] occurrences: {len(pos374)}")

    # For each 308-write, see if there's a 374-write within 512 bytes after, with the same base register
    def base_reg(insn_off):
        # modrm at insn_off+3; bottom 3 bits = base register (rm)
        return text[insn_off+3] & 0x07
    pairs = []
    pos374_set = sorted(pos374)
    import bisect
    for o308 in pos308:
        i = bisect.bisect_left(pos374_set, o308)
        for k in range(i, min(i+10, len(pos374_set))):
            o374 = pos374_set[k]
            if o374 - o308 > 512: break
            if base_reg(o308) == base_reg(o374):
                pairs.append((o308, o374))
                break
    print(f"  triple-mirror-write pairs found: {len(pairs)}")
    pair_func_starts = {}
    for o308, o374 in pairs[:50]:
        fs = find_func_start(o308)
        rva = (text_rva + fs) if fs is not None else None
        rva_s = f"0x{rva:X}" if rva is not None else "?"
        print(f"  .text+0x{o308:X} + 0x{o374:X}  -> func RVA {rva_s}")
        if rva is not None:
            pair_func_starts.setdefault(rva, []).append(text_rva + o308)

    # Intersection: functions that BOTH reference <nil> AND have triple-mirror writes
    inter = set(func_starts.keys()) & set(pair_func_starts.keys())
    print(f"\n=== Functions with BOTH <nil> xref AND triple-mirror writes: {len(inter)} ===")
    for rva in sorted(inter):
        print(f"  func RVA 0x{rva:X}  (mod-base addr 0x{base+rva:X})")

    out = {
        "module_base": hex(base),
        "nil_string_addrs": [hex(a) for a in nil_addrs],
        "nil_xref_count": len(xrefs),
        "nil_xref_func_count": len(func_starts),
        "nil_xref_funcs": [{"rva": hex(rva), "xrefs": [hex(x) for x in sites]}
                           for rva, sites in sorted(func_starts.items())],
        "triple_mirror_pair_count": len(pairs),
        "triple_mirror_funcs": [{"rva": hex(rva), "sites": [hex(x) for x in sites]}
                                for rva, sites in sorted(pair_func_starts.items())],
        "intersection": [hex(rva) for rva in sorted(inter)],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_JSON}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
