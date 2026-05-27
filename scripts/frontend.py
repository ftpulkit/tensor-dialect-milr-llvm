#!/usr/bin/env python3
"""
frontend.py — High-level C++ → tensor_ext MLIR translator

Supported syntax (C++ style):
  // comments
  float A[4][8];                       → tensor_ext.alloc : !tensor_ext.array<4x8xf32>
  float x = A[i][j];                   → tensor_ext.load
  A[i][j] = value;                     → tensor_ext.store
  float B[2][4] = slice(A, {0,2}, {2,4});  → tensor_ext.slice
  float T[8][4] = transpose(A, {1,0});     → tensor_ext.transpose
  return T;                            → return
  void funcname(args) { ... }          → func.func
  float funcname(args) { ... }         → func.func with return type

Example input:
  float work(float v) {
    float A[4][8];
    A[0][0] = v;
    float T[8][4] = transpose(A, {1,0});
    return T;
  }
"""

import sys
import re
import textwrap

# ── types ────────────────────────────────────────────────────────────────────

def shape_to_mlir(dims: list[int], elem: str = "f32") -> str:
    return "x".join(str(d) for d in dims) + "x" + elem

def array_type(dims: list[int], elem: str = "f32") -> str:
    return f"!tensor_ext.array<{shape_to_mlir(dims, elem)}>"

def parse_array_decl(s: str):
    """Parse 'float name[d0][d1]...' → (name, [d0,d1,...])"""
    m = re.fullmatch(r"float\s+(\w+)\s*((?:\[\d+\])+)", s.strip())
    if not m:
        return None
    name = m.group(1)
    dims = list(map(int, re.findall(r"\[(\d+)\]", m.group(2))))
    return name, dims

def parse_indices(s: str) -> list[str]:
    """Parse '[i][j]...' → ['%i','%j',...]"""
    return [f"%{x.strip()}" for x in re.findall(r"\[([^\]]+)\]", s)]

# ── SSA name tracker ─────────────────────────────────────────────────────────

class Scope:
    def __init__(self):
        self.vars: dict[str, dict] = {}   # name → {type, dims, mlir_var}
        self.args: list[tuple] = []        # (mlir_name, mlir_type)
        self.body: list[str] = []
        self.ret_type: str | None = None
        self.fn_name: str = "main"
        self._tmp = 0

    def fresh(self, base: str) -> str:
        self._tmp += 1
        return f"%{base}_{self._tmp}"

    def mlir(self, name: str) -> str:
        if name in self.vars:
            return self.vars[name]["mlir_var"]
        return f"%{name}"

    def register(self, name: str, mlir_var: str, dims=None, elem="f32"):
        self.vars[name] = {"mlir_var": mlir_var, "dims": dims or [], "elem": elem}

    def dims_of(self, name: str) -> list[int]:
        return self.vars.get(name, {}).get("dims", [])

    def elem_of(self, name: str) -> str:
        return self.vars.get(name, {}).get("elem", "f32")

    def emit(self, line: str):
        self.body.append("    " + line)

# ── line-by-line translator ──────────────────────────────────────────────────

def translate_line(raw: str, scope: Scope) -> str | None:
    """Translate one statement. Returns None on success, error string on failure."""
    line = raw.strip().rstrip(";")

    # blank / comment
    if not line or line.startswith("//"):
        return None

    # ── return ───────────────────────────────────────────────────────────────
    m = re.fullmatch(r"return\s+(.*)", line)
    if m:
        val = m.group(1).strip()
        if val in scope.vars:
            v = scope.vars[val]
            mlir_var = v["mlir_var"]
            if v["dims"]:
                ret_t = array_type(v["dims"], v["elem"])
            else:
                ret_t = "f32"
            scope.ret_type = ret_t
            scope.emit(f"return {mlir_var} : {ret_t}")
        else:
            scope.emit("return")
        return None

    # ── void return ──────────────────────────────────────────────────────────
    if line == "return":
        scope.emit("return")
        return None

    # ── float A[d0][d1]; — alloc ─────────────────────────────────────────────
    r = parse_array_decl(line)
    if r and "=" not in line:
        name, dims = r
        res = f"%{name}"
        at = array_type(dims)
        scope.emit(f"{res} = tensor_ext.alloc : {at}")
        scope.register(name, res, dims)
        return None

    # ── float T[..] = transpose(A, {p0,p1,...}); ────────────────────────────
    m = re.fullmatch(r"float\s+(\w+)\s*((?:\[\d+\])*)\s*=\s*transpose\(\s*(\w+)\s*,\s*\{([^}]+)\}\s*\)", line)
    if m:
        dst, dim_str, src, perm_str = m.group(1), m.group(2), m.group(3), m.group(4)
        perm = [int(x.strip()) for x in perm_str.split(",")]
        src_dims = scope.dims_of(src)
        elem = scope.elem_of(src)
        if not src_dims:
            return f"unknown variable '{src}'"
        dst_dims = [src_dims[p] for p in perm]
        src_t = array_type(src_dims, elem)
        dst_t = array_type(dst_dims, elem)
        perm_mlir = ", ".join(str(p) for p in perm)
        res = f"%{dst}"
        scope.emit(f"{res} = tensor_ext.transpose {scope.mlir(src)} permutation = [{perm_mlir}]")
        scope.emit(f"        : {src_t} to {dst_t}")
        scope.register(dst, res, dst_dims, elem)
        return None

    # ── float S[..] = slice(A, {off0,off1}, {sz0,sz1}); ─────────────────────
    m = re.fullmatch(
        r"float\s+(\w+)\s*((?:\[\d+\])*)\s*=\s*slice\(\s*(\w+)\s*,\s*\{([^}]+)\}\s*,\s*\{([^}]+)\}\s*\)",
        line
    )
    if m:
        dst, _, src, off_str, sz_str = m.groups()
        offsets = [int(x.strip()) for x in off_str.split(",")]
        sizes   = [int(x.strip()) for x in sz_str.split(",")]
        src_dims = scope.dims_of(src)
        elem = scope.elem_of(src)
        if not src_dims:
            return f"unknown variable '{src}'"
        src_t = array_type(src_dims, elem)
        dst_t = array_type(sizes, elem)
        off_mlir = ", ".join(str(o) for o in offsets)
        sz_mlir  = ", ".join(str(s) for s in sizes)
        res = f"%{dst}"
        scope.emit(f"{res} = tensor_ext.slice {scope.mlir(src)}")
        scope.emit(f"        offsets = [{off_mlir}] sizes = [{sz_mlir}]")
        scope.emit(f"        : {src_t} to {dst_t}")
        scope.register(dst, res, sizes, elem)
        return None

    # ── float x = A[i][j]; — load ────────────────────────────────────────────
    m = re.fullmatch(r"float\s+(\w+)\s*=\s*(\w+)((?:\[[^\]]+\])+)", line)
    if m:
        dst, src, idx_str = m.groups()
        idxs = parse_indices(idx_str)
        src_t = array_type(scope.dims_of(src), scope.elem_of(src))
        res = f"%{dst}"
        scope.emit(f"{res} = tensor_ext.load {scope.mlir(src)}[{', '.join(idxs)}]")
        scope.emit(f"        : {src_t} -> f32")
        scope.register(dst, res)
        return None

    # ── A[i][j] = expr; — store ──────────────────────────────────────────────
    m = re.fullmatch(r"(\w+)((?:\[[^\]]+\])+)\s*=\s*(.*)", line)
    if m:
        tgt, idx_str, val_raw = m.groups()
        val_raw = val_raw.strip()
        idxs = parse_indices(idx_str)

        # Determine the SSA value to store
        if re.fullmatch(r"\d+(\.\d+)?f?", val_raw):
            # numeric literal → arith.constant
            fval = val_raw.rstrip("f")
            tmp = scope.fresh("c")
            scope.emit(f"{tmp} = arith.constant {fval} : f32")
            val_mlir = tmp
        elif val_raw in scope.vars:
            val_mlir = scope.mlir(val_raw)
        else:
            # treat as SSA name directly (e.g. a function argument)
            val_mlir = f"%{val_raw}"

        # collect index constants if needed
        idx_mlir = []
        for raw_idx in re.findall(r"\[([^\]]+)\]", idx_str):
            raw_idx = raw_idx.strip()
            if re.fullmatch(r"\d+", raw_idx):
                tmp = scope.fresh("idx")
                scope.emit(f"{tmp} = arith.constant {raw_idx} : index")
                idx_mlir.append(tmp)
            else:
                idx_mlir.append(f"%{raw_idx}")

        tgt_t = array_type(scope.dims_of(tgt), scope.elem_of(tgt))
        scope.emit(f"tensor_ext.store {val_mlir}, {scope.mlir(tgt)}[{', '.join(idx_mlir)}]")
        scope.emit(f"        : f32, {tgt_t}")
        return None

    # ── constant: float x = 3.14; ────────────────────────────────────────────
    m = re.fullmatch(r"float\s+(\w+)\s*=\s*(\d+(?:\.\d+)?)", line)
    if m:
        dst, val = m.groups()
        tmp = f"%{dst}"
        scope.emit(f"{tmp} = arith.constant {val} : f32")
        scope.register(dst, tmp)
        return None

    return f"unrecognised statement: `{raw.strip()}`"


# ── function signature parser ─────────────────────────────────────────────────

def parse_signature(sig: str, scope: Scope):
    """
    Parse:  [void|float] name([float varname, ...])
    Populates scope.fn_name, scope.args.
    """
    sig = sig.strip()
    m = re.match(r"(void|float)\s+(\w+)\s*\(([^)]*)\)", sig)
    if not m:
        raise ValueError(f"Cannot parse function signature: {sig!r}")
    ret_kw, name, params_str = m.groups()
    scope.fn_name = name

    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue
        pm = re.fullmatch(r"float\s+(\w+)", param)
        if pm:
            pname = pm.group(1)
            scope.args.append((f"%{pname}", "f32"))
            scope.register(pname, f"%{pname}")
        else:
            pm2 = re.fullmatch(r"int\s+(\w+)", param)
            if pm2:
                pname = pm2.group(1)
                scope.args.append((f"%{pname}", "index"))
                scope.register(pname, f"%{pname}")


# ── main translate function ───────────────────────────────────────────────────

def translate(source: str) -> str:
    """Translate full C++ source → tensor_ext MLIR string."""
    lines = source.splitlines()
    scope = Scope()
    errors = []

    # find function signature line
    sig_line = None
    body_lines = []
    in_body = False
    brace_depth = 0

    for raw in lines:
        stripped = raw.strip()
        if not in_body:
            m = re.match(r"(void|float)\s+\w+\s*\([^)]*\)\s*\{?", stripped)
            if m:
                sig_part = re.match(r"(void|float)\s+\w+\s*\([^)]*\)", stripped).group(0)
                parse_signature(sig_part, scope)
                in_body = True
                brace_depth = stripped.count("{") - stripped.count("}")
                continue
        else:
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                break
            # strip leading/trailing braces from single-brace lines
            clean = stripped.lstrip("{").rstrip("}")
            if clean.strip():
                body_lines.append(clean)

    if not in_body:
        # no function wrapper — treat whole input as body of @main
        scope.fn_name = "main"
        body_lines = [l.strip() for l in lines if l.strip()]

    for raw in body_lines:
        # skip blank / standalone braces
        s = raw.strip()
        if not s or s in ("{", "}"):
            continue
        # strip inline comments
        s = re.sub(r"\s*//.*", "", s).rstrip(";").strip()
        if not s:
            continue
        err = translate_line(s, scope)
        if err:
            errors.append(f"    // TRANSLATION ERROR: {err}")
            errors.append(f"    // original: {raw.strip()}")

    # build return type list
    if scope.ret_type:
        ret_types = f" -> {scope.ret_type}"
    else:
        ret_types = ""

    # build arg list
    arg_str = ", ".join(f"{n}: {t}" for n, t in scope.args)

    lines_out = []
    lines_out.append(f"func.func @{scope.fn_name}({arg_str}){ret_types} {{")
    lines_out.extend(scope.body)
    if errors:
        lines_out.extend(errors)
    # ensure there's a return
    if not any("return" in l for l in scope.body):
        lines_out.append("    return")
    lines_out.append("}")

    return "\n".join(lines_out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_help():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          tensor_ext Frontend  —  C++ → MLIR Translator        ║
╚═══════════════════════════════════════════════════════════════╝

Supported C++ constructs:
  float A[4][8];                     allocate a 4×8 f32 tensor
  float x = A[i][j];                 load element at [i][j]
  A[i][j] = value;                   store value at [i][j]
  A[i][j] = 3.14;                    store float literal
  float S[2][4] = slice(A, {0,2}, {2,4});   slice (offset, size per dim)
  float T[8][4] = transpose(A, {1,0});      transpose with permutation
  return T;                          return a tensor
  return;                            void return
  // comment                         ignored

Wrap in a function (optional):
  float myfunc(float v) {
    float A[4][8];
    ...
    return A;
  }

Without a function wrapper, code is placed in @main().
""")

def main():
    import os

    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        sys.exit(0)

    # read from stdin or file arg
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1]) as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    mlir = translate(source)
    print(mlir)

if __name__ == "__main__":
    main()
