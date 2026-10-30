"""Compare a recovered source file against the original .pyc, function by function.

Usage: python verify.py ORIGINAL.pyc RECOVERED.py [--detail NAME]
Needs Python 3.12 (the bytecode version of the original).

For each top-level function/class of the original module it reports:
  EXACT   - same bytecode, constants, names and variables (line numbers ignored)
  EQUIV   - same instruction stream after dropping NOPs and normalising jumps
  DIFF    - differs
  MISSING - not found in the recovered file (or the file does not compile)
"""
import dis, marshal, sys, types

def load_pyc(path):
    with open(path, 'rb') as f:
        data = f.read()
    return marshal.loads(data[16:])

def code_children(co):
    return [c for c in co.co_consts if isinstance(c, types.CodeType)]

def norm_consts(co):
    out = []
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            out.append(('<code>', c.co_name))
        else:
            out.append((type(c).__name__, repr(c)))
    return out

def same_exact(a, b):
    if a.co_code != b.co_code: return False
    for attr in ('co_names', 'co_varnames', 'co_freevars', 'co_cellvars', 'co_argcount',
                 'co_posonlyargcount', 'co_kwonlyargcount', 'co_flags', 'co_name'):
        if getattr(a, attr) != getattr(b, attr): return False
    if norm_consts(a) != norm_consts(b): return False
    ca, cb = code_children(a), code_children(b)
    return len(ca) == len(cb) and all(same_exact(x, y) for x, y in zip(ca, cb))

def instr_stream(co):
    ins = [i for i in dis.get_instructions(co) if i.opname not in ('NOP', 'CACHE')]
    off2idx = {i.offset: k for k, i in enumerate(ins)}
    out = []
    for i in ins:
        if i.opcode in dis.hasjrel or i.opcode in dis.hasjabs:
            tgt = i.argval
            # find next real instruction at or after target
            k = min((v for o, v in off2idx.items() if o >= tgt), default=-1)
            out.append((i.opname.replace('BACKWARD', 'FORWARD'), 'J', k))
        elif i.opname == 'LOAD_CONST' and isinstance(i.argval, types.CodeType):
            out.append((i.opname, i.argval.co_name))
        else:
            out.append((i.opname, repr(i.argval)))
    return out

def same_equiv(a, b):
    if instr_stream(a) != instr_stream(b): return False
    if norm_consts(a) != norm_consts(b): return False   # docstrings and unused constants too
    ca, cb = code_children(a), code_children(b)
    return len(ca) == len(cb) and all(same_equiv(x, y) for x, y in zip(ca, cb))

def top_defs(mod):
    return {c.co_name: c for c in code_children(mod) if c.co_name != '<lambda>'}

def main():
    orig = load_pyc(sys.argv[1])
    src = open(sys.argv[2]).read()
    detail = sys.argv[sys.argv.index('--detail') + 1] if '--detail' in sys.argv else None
    try:
        rec = marshal.loads(marshal.dumps(compile(src, 'crypto_quant_lab.py', 'exec')))  # same frozenset order as the .pyc
        rdefs = top_defs(rec)
    except SyntaxError as e:
        print('SYNTAX ERROR', e); rdefs = {}
    odefs = top_defs(orig)
    stats = {}
    for name, co in odefs.items():
        r = rdefs.get(name)
        if r is None: s = 'MISSING'
        elif same_exact(co, r): s = 'EXACT'
        elif same_equiv(co, r): s = 'EQUIV'
        else: s = 'DIFF'
        stats.setdefault(s, []).append(name)
        if detail is None or detail == name:
            print(f'{s:8s} {name}')
        if detail == name and r is not None:
            import difflib
            def flat(c, p=''):
                out = [f'{p}{x}' for x in instr_stream(c)]
                for ch in code_children(c): out += flat(ch, p + ch.co_name + '/')
                return out
            for l in difflib.unified_diff(flat(co), flat(r), 'orig', 'rec', lineterm='', n=2):
                print(l)
    mod_ok = same_equiv(orig, rec) if rdefs else False
    print('module body EQUIV' if mod_ok else 'module body DIFF')
    print({k: len(v) for k, v in stats.items()}, 'of', len(odefs))

if __name__ == '__main__':
    main()
