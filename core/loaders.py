# =============================================================================
# core/loaders.py — RanOptics backend data loaders
# Backends: Tao (Bmad), ELEGANT, MAD-X, xsuite
# =============================================================================

from __future__ import annotations
import math, os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

from core.utils import THIN_ELEMENT_THRESHOLD
def _read_elegant_beam_params(ele_file):
    res={'emit_x':0.0,'emit_y':0.0,'sigma_dp':0.0}
    try:
        with open(ele_file,'r') as f: content=f.read()
    except Exception: return res
    in_bb=False; bb=''
    for line in content.splitlines():
        s=line.strip()
        if '!' in s: s=s[:s.index('!')]
        if not s: continue
        if '&bunched_beam' in s.lower() or '&bunch' in s.lower(): in_bb=True
        if in_bb:
            bb+=' '+s
            if '&end' in s.lower(): break
    if not bb: return res
    def _ex(t,n):
        m=re.search(rf'{n}\s*=\s*([+-]?[\d.eE+-]+)',t,re.IGNORECASE)
        return float(m.group(1)) if m else 0.0
    res['emit_x']=_ex(bb,'emit_x'); res['emit_y']=_ex(bb,'emit_y')
    res['sigma_dp']=_ex(bb,'sigma_dp')
    return res

def _read_tao_beam_params(tao):
    res={'emit_x':0.0,'emit_y':0.0,'sigma_dp':0.0,
         'tune_a':None,'tune_b':None,'chroma_a':None,'chroma_b':None}
    try:
        for line in tao.cmd("show lattice -att emit_a -att emit_b -att sig_E END"):
            if line.startswith('#') or not line.strip(): continue
            parts=line.split()
            try:
                if len(parts)>=7:
                    res['emit_x']=float(parts[5]) if parts[5]!='---' else 0.0
                    res['emit_y']=float(parts[6]) if parts[6]!='---' else 0.0
                if len(parts)>=8:
                    res['sigma_dp']=float(parts[7]) if parts[7]!='---' else 0.0
            except (IndexError,ValueError): continue
    except Exception: pass
    # Tune from tune.a/b at last element
    try:
        result = tao.cmd('show lattice -att tune.a -att tune.b')
        for line in reversed(result):
            if line.startswith('#') or not line.strip(): continue
            parts = line.split()
            try:
                int(parts[0])  # must be element index
                if len(parts) >= 7:
                    v = parts[5]
                    if v != '---': res['tune_a'] = float(v)
                    v = parts[6]
                    if v != '---': res['tune_b'] = float(v)
                break  # only need last element
            except (IndexError, ValueError): continue
    except Exception: pass
    # Chromaticity from chrom.a and chrom.b (ring only — None for open lines)
    try:
        result = tao.cmd('show lattice -att chrom.a -att chrom.b')
        for line in reversed(result):
            if line.startswith('#') or not line.strip(): continue
            parts = line.split()
            try:
                int(parts[0])
                if len(parts) >= 7:
                    v = parts[5]
                    if v != '---': res['chroma_a'] = float(v)
                    v = parts[6]
                    if v != '---': res['chroma_b'] = float(v)
                break
            except (IndexError, ValueError): continue
    except Exception: pass
    return res

# ─── Tao init file parser ────────────────────────────────────────────────────
# _parse_tao_init(): reads n_universes and lattice file labels from .init file

def _parse_tao_init(init_file):
    """Parse a Tao init file and return (n_universes, {i: label}) .
    Reads n_universes and design_lattice(i)%file entries.
    Returns (1, {1: 'lattice'}) if no multi-universe block found.
    """
    try:
        with open(init_file, 'r') as f:
            content = f.read()
    except Exception:
        return 1, {1: 'u1'}

    n = 1
    m = re.search(r'n_universes\s*=\s*(\d+)', content, re.IGNORECASE)
    if m:
        n = int(m.group(1))

    labels = {}
    _pat = re.compile(
        r"design_lattice\s*\((\d+)\)\s*%\s*file\s*=\s*['\"]?([^\s'\"&,/]+)",
        re.IGNORECASE)
    for _m2 in _pat.finditer(content):
        idx  = int(_m2.group(1))
        path = _m2.group(2).strip()
        stem  = Path(path).stem
        label = stem.split('_')[0] if '_' in stem else stem
        labels[idx] = label

    # Fill any missing labels
    for i in range(1, n + 1):
        if i not in labels:
            labels[i] = f'u{i}'

    return n, labels

# ─── Tao backend ─────────────────────────────────────────────────────────────
# load_tao(): starts Tao, queries all universes, returns optics data dict
# _load_tao_universe(): queries one universe via show lattice

def _load_tao_universe(tao, uni_idx, log_fn=None, progress_fn=None):
    """Query one universe from an already-running Tao instance.
    Returns a single-universe data dict (same format as old load_tao return).
    """
    def L(m): (log_fn(m+'\n') if log_fn else print(m))
    def P(p,l): progress_fn(p,l) if progress_fn else None
    u     = f"-universe {uni_idx}"   # for show lattice
    u_at  = f"{uni_idx}@"            # for show element
    result = tao.cmd(
        f"show lattice {u} -all -att beta_a -att beta_b -att eta_x -att eta_y "
        "-att orbit_x -att orbit_y -att phi_a -att phi_b "
        "-att alpha_a -att alpha_b "
        "-att K1 -att K2 -att hkick -att vkick -att ref_tilt")
    s=[]; ba=[]; bb=[]; ex=[]; ey=[]; ox=[]; oy=[]; pa=[]; pb=[]; aa=[]; ab=[]; elems=[]
    for line in result:
        if 'Lord Elements:' in line: break
        if line.startswith('#') or not line.strip(): continue
        p=line.split()
        try:
            idx=int(p[0]); name=p[1]; key=p[2]
            s_end=float(p[3]); length=float(p[4]) if p[4]!='---' else 0.0
            s.append(s_end)
            ba.append(float(p[5])  if p[5] !='---' else 0.0)
            bb.append(float(p[6])  if p[6] !='---' else 0.0)
            ex.append(float(p[7])  if p[7] !='---' else 0.0)
            ey.append(float(p[8])  if p[8] !='---' else 0.0)
            ox.append(float(p[9])  if p[9] !='---' else 0.0)
            oy.append(float(p[10]) if p[10]!='---' else 0.0)
            pa.append(float(p[11]) if p[11]!='---' else 0.0)
            pb.append(float(p[12]) if p[12]!='---' else 0.0)
            aa.append(float(p[13]) if len(p)>13 and p[13]!='---' else 0.0)
            ab.append(float(p[14]) if len(p)>14 and p[14]!='---' else 0.0)
            def _f(i): return float(p[i]) if len(p)>i and p[i]!='---' else 0.0
            kl=key.lower()
            hk=_f(17); vk=_f(18)
            kick=hk if kl=='hkicker' else (vk if kl=='vkicker' else 0.0)
            elems.append({'name':name,'key':key,'index':idx,'s_start':s_end-length,
                'length':length,'angle':0.0,'k1':_f(15),'k2':_f(16),
                'hkick':hk,'vkick':vk,'kick':kick,'ref_tilt':_f(19)})
        except (IndexError,ValueError): continue
    # Bend angles
    for e in elems:
        if 'sbend' in e['key'].lower() and e['length']>0:
            for line in tao.cmd(f"show element {u_at}{e['index']}"):
                if 'ANGLE' in line and 'rad' in line:
                    try: e['angle']=float(line.split('=')[1].strip().split()[0])
                    except: pass
                    break
    # Cavity parameters
    for e in elems:
        kl=e['key'].lower()
        if ('rfcavity' in kl or 'lcavity' in kl) and e['length']>0:
            for line in tao.cmd(f"show element {u_at}{e['index']}"):
                lu=line.upper()
                if 'VOLTAGE' in lu and '=' in line:
                    try: e['voltage']=float(line.split('=')[1].strip().split()[0])
                    except: pass
                if 'RF_FREQUENCY' in lu and '=' in line:
                    try: e['frequency']=float(line.split('=')[1].strip().split()[0])
                    except: pass
    # Floor (survey) coordinates via pipe lat_list
    # Uses Bmad element attributes: x_position, y_position, z_position,
    # theta_position, phi_position — these are exit-end global coords.
    # x = horizontal, y = vertical, z = longitudinal (Bmad global frame).
    try:
        fp_result = tao.cmd(
            f"pipe lat_list {uni_idx}@0>>*|model "
            "ele.ix_ele,ele.x_position,ele.y_position,ele.z_position,"
            "ele.theta_position,ele.phi_position")
        # Output: one line per element, semicolon-separated:
        # ix_ele;x_position;y_position;z_position;theta_position;phi_position
        fp_map = {}  # ix_ele -> (x, y, z, theta, phi)
        for line in fp_result:
            line = line.strip()
            if not line: continue
            p = line.split(';')
            try:
                ei    = int(p[0])
                x_g   = float(p[1])
                y_g   = float(p[2])
                z_g   = float(p[3])
                theta = float(p[4])
                phi   = float(p[5])
                fp_map[ei] = (x_g, y_g, z_g, theta, phi)
            except (IndexError, ValueError):
                continue
        if fp_map:
            L(f"[tao] Floor plan loaded: {len(fp_map)} elements with survey coords")
            # Each fp_map entry gives EXIT coords of that element index.
            # Entry coords of element ei = exit coords of element ei-1 in fp_map.
            # Direct lookup: no accumulation, no dependency on elems walk order,
            # no phantom lines from gaps between universes or patch elements.
            for e in elems:
                ei = e['index']
                if ei in fp_map and (ei - 1) in fp_map:
                    x0, y0, z0, th0, ph0 = fp_map[ei - 1]
                    x1, y1, z1, th1, ph1 = fp_map[ei]
                    e['flr_z0'] = z0;  e['flr_z1'] = z1
                    e['flr_x0'] = x0;  e['flr_x1'] = x1
                    e['flr_y0'] = y0;  e['flr_y1'] = y1
                    e['flr_theta0'] = th0
                    e['flr_phi0']   = ph0
                    e['flr_theta1'] = th1  # exit theta for polygon bend sign
        else:
            L("[tao] No floor position data returned — using dead-reckoning")
    except Exception as _fp_err:
        L(f"[tao] Floor plan query failed ({_fp_err}) — using dead-reckoning")
    n=min(len(s),len(ba),len(bb),len(ex),len(ey),len(ox),len(oy),len(pa),len(pb))
    _pad0 = lambda a: np.array(a[:n]) if len(a)>=n else np.zeros(n)
    return dict(s=np.array(s[:n]),beta_a=np.array(ba[:n]),beta_b=np.array(bb[:n]),
        eta_x=np.array(ex[:n]),eta_y=np.array(ey[:n]),
        orbit_x=np.array(ox[:n]),orbit_y=np.array(oy[:n]),
        phi_a=np.array(pa[:n])/(2*np.pi),phi_b=np.array(pb[:n])/(2*np.pi),
        alpha_a=_pad0(aa),alpha_b=_pad0(ab),
        elements=elems,beam_params=_read_tao_beam_params(tao))

# ─── Expression evaluator engine ─────────────────────────────────────────────
# Handles arbitrary user-defined math expressions like (beta_a * beta_b) / alpha_a
# Supports: standard arrays, extra fetched attrs, global scalars, numpy as np

def _query_tao_attrs(tao, attrs, uni_idx=1, log_fn=None):
    """Query arbitrary per-element attributes from a running Tao instance.
    Sends a single show lattice command for all requested attributes.
    Returns dict of attr_name -> np.ndarray, plus 's' array.
    """
    def L(m):
        if log_fn: log_fn(m + '\n')

    if not attrs:
        return {'s': np.array([])}

    # Build show lattice command with all requested attributes at once
    attr_flags = ' '.join(f'-att {a}' for a in attrs)
    u   = f"-universe {uni_idx}"
    cmd = f"show lattice {u} -all {attr_flags}"
    L(f"[tao_expr] Querying: {cmd}")

    result = tao.cmd(cmd)
    s_arr  = []
    data   = {a: [] for a in attrs}

    _seen_errors = set()
    for line in result:
        if 'Lord Elements:' in line: break
        if line.startswith('#') or not line.strip(): continue
        # Deduplicate Tao error lines — log each unique error only once
        if 'ERROR' in line or 'error' in line.lower():
            _key = line.strip()
            if _key not in _seen_errors:
                _seen_errors.add(_key)
                L(f"[tao] {line.strip()}")
            continue
        parts = line.split()
        try: int(parts[0])
        except (ValueError, IndexError): continue
        try: s_arr.append(float(parts[3]))
        except (IndexError, ValueError): continue
        # Attributes start at column 5 in show lattice output
        for i, attr in enumerate(attrs):
            col = 5 + i
            try:
                val = float(parts[col]) if col < len(parts) and parts[col] != '---' else np.nan
            except (ValueError, IndexError):
                val = np.nan
            data[attr].append(val)

    n = len(s_arr)
    out = {'s': np.array(s_arr)}
    for attr in attrs:
        out[attr] = np.array(data[attr][:n])
    L(f"[tao_expr] Got {n} points for: {attrs}")
    return out

def _query_elegant_attrs(data, attrs, log_fn=None):
    """Extract per-element column arrays from already-loaded ELEGANT data dict.
    Handles both lux internal names (beta_a) and ELEGANT names (betax).
    Returns dict of attr_name -> np.ndarray.
    """
    # Map ELEGANT column names and lux internal names to data dict keys
    _KEY_MAP = {
        'beta_a':'beta_a',  'beta_b':'beta_b',
        'betax': 'beta_a',  'betay': 'beta_b',
        'alpha_a':'alpha_a','alpha_b':'alpha_b',
        'alphax':'alpha_a', 'alphay':'alpha_b',
        'eta_x': 'eta_x',   'eta_y': 'eta_y',
        'etax':  'eta_x',   'etay':  'eta_y',
        'etaxp': 'etaxp',   'etayp': 'etayp',
        'orbit_x':'orbit_x','orbit_y':'orbit_y',
        'phi_a': 'phi_a',   'phi_b': 'phi_b',
        'psix':  'phi_a',   'psiy':  'phi_b',
        'xAperture':'xAperture','yAperture':'yAperture',
        'dI1':'dI1','dI2':'dI2','dI3':'dI3','dI4':'dI4','dI5':'dI5',
    }
    result = {'s': data['s']}
    for attr in attrs:
        key = _KEY_MAP.get(attr, _KEY_MAP.get(attr.lower(), attr))
        if key in data:
            result[attr] = np.asarray(data[key], dtype=float)
        else:
            if log_fn: log_fn(f"[elegant_expr] Warning: '{attr}' not found\n")
            result[attr] = np.zeros_like(data['s'])
    return result

def _query_xsuite_attrs(tw, attrs, log_fn=None):
    """Extract per-element column arrays from an xsuite Twiss table object.
    Returns dict of attr_name -> np.ndarray.
    """
    result = {'s': np.array(tw.s, dtype=float)}
    for attr in attrs:
        try:
            result[attr] = np.array(getattr(tw, attr), dtype=float)
        except AttributeError:
            if log_fn: log_fn(f"[xsuite_expr] Warning: '{attr}' not found\n")
            result[attr] = np.zeros(len(tw.s))
    return result

def load_tao(init_file, log_fn=None, progress_fn=None, tao=None):
    def L(m): (log_fn(m+'\n') if log_fn else print(m))
    def P(p,l): progress_fn(p,l) if progress_fn else None
    from pytao import Tao
    P(5, 'Starting Tao...')
    if tao is None:
        tao = Tao(f"-init {init_file} -noplot")

    # Parse init file for universe count and labels
    n_uni, uni_labels = _parse_tao_init(init_file)
    L(f"[tao] {n_uni} universe(s) detected: { {i: uni_labels[i] for i in range(1, n_uni+1)} }")

    step_per_uni = 30 // max(n_uni, 1)
    universes = {}
    for i in range(1, n_uni + 1):
        label = uni_labels.get(i, f'u{i}')
        P(15 + (i-1)*step_per_uni, f'Querying universe {i} ({label})...')
        L(f"[tao] Loading universe {i}: {label}")
        universes[i] = _load_tao_universe(tao, i, log_fn=log_fn)

    P(40, 'All universes loaded.')

    # Single universe — return flat dict for backwards compatibility
    if n_uni == 1:
        data = universes[1]
        data['n_universes']   = 1
        data['universe_labels'] = uni_labels
        data['universes']     = universes
        return data

    # Multi-universe — return multi dict + flat dict of universe 1 for compat
    data = universes[1].copy()
    data['n_universes']     = n_uni
    data['universe_labels'] = uni_labels
    data['universes']       = universes
    return data

# ─── ELEGANT helpers ─────────────────────────────────────────────────────────
# _run_elegant(), _find_sdds(), _sdds_to_ascii(), _read_sdds_ascii()
# _read_mag(), _read_lte(), _key_from_type() — ELEGANT file I/O utilities

def _run_elegant(ele_file, log_fn=None):
    def L(m): (log_fn(m+'\n') if log_fn else print(m))
    p=Path(ele_file).resolve(); rd=p.parent
    L(f"[elegant] Running: elegant {p.name}  (cwd={rd})")
    r=subprocess.run(['elegant',p.name],cwd=str(rd),capture_output=True,text=True)
    if r.returncode!=0:
        L(r.stdout[-2000:]); L(r.stderr[-2000:])
        raise RuntimeError(f"elegant exited with code {r.returncode}")
    L("[elegant] Run complete.")
    return rd

def _find_sdds(run_dir, ext):
    _FIX={'.twi':"Add &twiss_output filename=\"%s.twi\" matched=1 &end to your .ele file.",
          '.cen':"Add centroid=\"%s.cen\" to &run_setup for orbit data.",
          '.mag':"Add magnets=\"%s.mag\" to &run_setup for layout styling.",
          '.flr':"Add &floor_coordinates filename=\"%s.flr\" &end for floor plan."}
    m=list(run_dir.glob(f'*{ext}'))
    if not m: raise FileNotFoundError(f"[elegant] Missing *{ext}\n{_FIX.get(ext,'')}")
    m.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    return m[0]

def _sdds_to_ascii(sdds_file):
    with open(sdds_file,'rb') as f: h=f.read(300).decode('ascii',errors='ignore')
    if not any(x in h.lower() for x in ('little-endian','big-endian','mode=binary')):
        return sdds_file,False
    fd,tmp=tempfile.mkstemp(suffix='.sdds_ascii'); os.close(fd)
    r=subprocess.run(['sddsconvert',sdds_file,tmp,'-ascii'],capture_output=True,text=True)
    if r.returncode!=0:
        os.unlink(tmp); raise RuntimeError(f"sddsconvert failed: {r.stderr.strip()}")
    return tmp,True

def _read_sdds_ascii(filepath, want_cols, read_params=False):
    with open(filepath,'r') as f: lines=f.readlines()
    col_names=[]; param_names=[]; ds=0
    for i,line in enumerate(lines):
        ls=line.lower()
        if '&parameter' in ls:
            m=re.search(r'name\s*=\s*([\w/]+)',line,re.IGNORECASE)
            if m: param_names.append(m.group(1))
        elif '&column' in ls:
            m=re.search(r'name\s*=\s*([\w/]+)',line,re.IGNORECASE)
            if m: col_names.append(m.group(1))
        elif '&data' in ls: ds=i+1; break
    # Parameter values appear right after &data, one per line
    params={}
    if read_params:
        for j,pname in enumerate(param_names):
            idx = ds + j
            if idx < len(lines):
                val = lines[idx].strip()
                try: params[pname] = float(val)
                except: params[pname] = val
    skip=ds+len(param_names)
    if skip<len(lines) and lines[skip].strip().isdigit(): skip+=1
    ci={n:i for i,n in enumerate(col_names) if n in set(want_cols)}
    data={c:[] for c in want_cols}
    for line in lines[skip:]:
        line=line.strip()
        if not line or line.startswith('!'): continue
        parts=line.split()
        if len(parts)<len(col_names): continue
        for col,idx in ci.items():
            v=parts[idx]
            try: data[col].append(float(v))
            except: data[col].append(v)
    if read_params:
        return data, params
    return data

def _read_sdds(sdds_file, want_cols, read_params=False):
    tmp,is_tmp=_sdds_to_ascii(str(sdds_file))
    try: return _read_sdds_ascii(tmp, want_cols, read_params=read_params)
    finally:
        if is_tmp: os.unlink(tmp)

def _read_sdds_all(sdds_file):
    """Read ALL numeric columns from an SDDS file. Returns dict col->list."""
    tmp, is_tmp = _sdds_to_ascii(str(sdds_file))
    try:
        with open(tmp, 'r') as f:
            lines = f.readlines()
    finally:
        if is_tmp: os.unlink(tmp)
    col_names = []; col_types = []; param_names = []; ds = 0
    for i, line in enumerate(lines):
        ls = line.lower()
        if '&parameter' in ls:
            m = re.search(r'name\s*=\s*([\w/]+)', line, re.IGNORECASE)
            if m: param_names.append(m.group(1))
        elif '&column' in ls:
            mn = re.search(r'name\s*=\s*([\w/]+)', line, re.IGNORECASE)
            mt = re.search(r'type\s*=\s*(\w+)', line, re.IGNORECASE)
            if mn:
                col_names.append(mn.group(1))
                col_types.append(mt.group(1).lower() if mt else 'double')
        elif '&data' in ls:
            ds = i + 1; break
    numeric = {'double', 'float', 'short', 'long', 'ulong', 'ulong64'}
    num_cols = [n for n, t in zip(col_names, col_types) if t in numeric]
    skip = ds + len(param_names)
    if skip < len(lines) and lines[skip].strip().isdigit(): skip += 1
    ci = {n: col_names.index(n) for n in num_cols}
    data = {c: [] for c in num_cols}
    for line in lines[skip:]:
        line = line.strip()
        if not line or line.startswith('!'): continue
        parts = line.split()
        if len(parts) < len(col_names): continue
        for col, idx in ci.items():
            try: data[col].append(float(parts[idx]))
            except: pass
    return data

def _read_mag(mag_file):
    with open(str(mag_file),'r') as f: lines=f.readlines()
    ds=0
    for i,line in enumerate(lines):
        if '&data' in line.lower(): ds=i+1; break
    names,types,sp,prof=[],[],[],[]
    for line in lines[ds:]:
        line=line.strip()
        if not line or line.startswith('!'): continue
        parts,iq,cur=[],False,''
        for ch in line:
            if ch=='"': iq=not iq
            elif ch in ' \t' and not iq:
                if cur: parts.append(cur); cur=''
            else: cur+=ch
        if cur: parts.append(cur)
        if len(parts)>=4:
            names.append(parts[0]); types.append(parts[1])
            sp.append(float(parts[2])); prof.append(float(parts[3]))
    return names,types,sp,prof

def _read_lte(lte_file):
    with open(str(lte_file),'r') as f: content=f.read()
    clean=[]
    for line in content.splitlines():
        if '!' in line: line=line[:line.index('!')]
        line=line.strip()
        if line: clean.append(line)
    text=' '.join(clean); elems={}
    for m in re.finditer(r'(\w+)\s*:\s*(\w+)\s*(?:,\s*)?([^:]*?)(?=\s+\w+\s*:|$)',text,re.DOTALL):
        name=m.group(1).strip().upper(); etype=m.group(2).strip().upper(); ps=m.group(3) or ''
        e={'type':etype,'K1':0.0,'K2':0.0,'ANGLE':0.0,'KICK':0.0,'HKICK':0.0,'VKICK':0.0,'VOLT':0.0,'FREQ':0.0,'TILT':0.0}
        for pm in re.finditer(r'(\w+)\s*=\s*([+-]?[\d.eE+-]+)',ps):
            pn=pm.group(1).upper()
            try: val=float(pm.group(2))
            except: continue
            if pn in e: e[pn]=val
        elems[name]=e
    return elems

def _key_from_type(etype):
    r=etype.upper()
    if r in ('CSBEND','SBEND','RBEN','RBEND','SBEN','CSRCSBEND','KSBEND'): return 'SBend'
    if r in ('KQUAD','QUAD','QUADRUPOLE','KQUSE'): return 'Quadrupole'
    if r in ('KSEXT','SEXT','SEXTUPOLE'): return 'Sextupole'
    if r in ('HKICK','EHKICK'): return 'Hkicker'
    if r in ('VKICK','EVKICK'): return 'Vkicker'
    if r in ('KICKER','EKICKER','HVCORRECTOR'): return 'Kicker'
    if r in ('MONI','MONITOR','HMON','VMON'): return 'Monitor'
    if r in ('MARK','MARKER'): return 'Marker'
    if r in ('RFCA','RFCW','RFDF','RFTMEZ0','RFTM110','MODRF','RFMODE'): return 'RFcavity'
    if r in ('LSRMDLTR',): return 'Lcavity'
    return etype.capitalize()

def load_elegant(ele_file, log_fn=None, progress_fn=None):
    def L(m): (log_fn(m+'\n') if log_fn else print(m))
    def P(p,l): progress_fn(p,l) if progress_fn else None
    ep=Path(ele_file).resolve()
    P(3, 'Running ELEGANT...'); run_dir=_run_elegant(ele_file,log_fn)
    lte_file=ep.parent/(ep.stem+'.lte')
    if not lte_file.exists():
        try:
            with open(str(ep),'r') as f:
                for line in f:
                    m=re.search(r'lattice\s*=\s*["\']?([^\s,"\'&]+)',line,re.IGNORECASE)
                    if m: lte_file=ep.parent/m.group(1).strip(); break
        except Exception: pass
    lte_data={}
    if lte_file.exists():
        L(f"[elegant] Reading lattice from {lte_file.name}")
        lte_data=_read_lte(str(lte_file))
    else: L("[elegant] Warning: no .lte file — K1/K2/angles will be zero.")

    try:
        twi_file=_find_sdds(run_dir,'.twi')
    except FileNotFoundError:
        raise SystemExit(
            f"[elegant] Cannot find .twi file in {run_dir}\n"
            "  → Add this to your .ele file:\n"
            "      &twiss_output filename=\"lattice.twi\" matched=1 &end\n"
            "  Then re-run ELEGANT before plotting."
        )
    twi_result=_read_sdds(str(twi_file),
        ['s','betax','betay','etax','etay','psix','psiy',
         'alphax','alphay','etaxp','etayp',
         'xAperture','yAperture',
         'dI1','dI2','dI3','dI4','dI5',
         'dbetax/dp','dbetay/dp','dalphax/dp','dalphay/dp',
         'ElementName','ElementType'], read_params=True)
    twi, twi_params = twi_result
    def _arr(d,*keys):
        for k in keys:
            if k in d and len(d[k]): return np.array(d[k],dtype=float)
        return np.array([])
    s=_arr(twi,'s'); ba=_arr(twi,'betax'); bb=_arr(twi,'betay')
    ex=_arr(twi,'etax'); ey=_arr(twi,'etay')
    phi_a=_arr(twi,'psix'); phi_b=_arr(twi,'psiy')
    al_a=_arr(twi,'alphax'); al_b=_arr(twi,'alphay')
    n=min(len(s),len(ba),len(bb),len(ex),len(ey),len(phi_a),len(phi_b))
    s=s[:n]; ba=ba[:n]; bb=bb[:n]; ex=ex[:n]; ey=ey[:n]
    phi_a=phi_a[:n]/(2*np.pi); phi_b=phi_b[:n]/(2*np.pi)
    _pad0e = lambda a: a[:n] if len(a)>=n else np.zeros(n)
    al_a=_pad0e(al_a); al_b=_pad0e(al_b)
    P(28, 'Reading orbit file...')
    try:
        cen = _read_sdds_all(str(_find_sdds(run_dir, '.cen')))
        ox=_arr(cen,'Cx'); oy=_arr(cen,'Cy')
        L(f'[elegant] Loaded centroid file ({len(cen)} columns).')
    except FileNotFoundError:
        L("[elegant] No .cen file found — orbit will be shown as zero.\n"
          "  → Add centroid=\"lattice.cen\" to &run_setup to enable orbit data.")
        cen={}; ox=np.zeros(n); oy=np.zeros(n)
    def _pad(a,l): return a[:l] if len(a)>=l else np.concatenate([a,np.zeros(l-len(a))])
    ox=_pad(ox,n); oy=_pad(oy,n)
    # Load sigma/beam-size file — read ALL numeric columns dynamically
    P(30, 'Reading sigma file...')
    try:
        sig = _read_sdds_all(str(_find_sdds(run_dir, '.sig')))
        L(f'[elegant] Loaded sigma file ({len(sig)} columns).')
    except FileNotFoundError:
        L("[elegant] No .sig file found — beam size data unavailable.\n"
          "  → Add sigma=\"%s.sig\" to &run_setup to enable sigma data.")
        sig = {}
    elements=[]
    P(32, 'Reading floor plan file...')
    try:
        flr=_read_sdds(str(_find_sdds(run_dir,'.flr')),
            ['s','ds','Z','X','Y','theta','phi','ElementName','ElementType'])
        fnames=flr.get('ElementName',[]); ftypes=flr.get('ElementType',[])
        fs=[float(v) for v in flr.get('s',[])]
        fds=[float(v) for v in flr.get('ds',[])]
        fZ=[float(v) for v in flr.get('Z',[])]; fX=[float(v) for v in flr.get('X',[])]
        fY=[float(v) for v in flr.get('Y',[])] if flr.get('Y') else []
        fth=[float(v) for v in flr.get('theta',[])]; fph=[float(v) for v in flr.get('phi',[])] if flr.get('phi') else []
        for i,name in enumerate(fnames):
            etype=str(ftypes[i]) if i<len(ftypes) else ''
            s_end=fs[i] if i<len(fs) else 0.0
            # Use ds from .flr for floor plan element thickness (stored as flr_ds)
            # but compute actual s_start from previous element s to avoid bar overlap.
            flr_ds = fds[i] if i<len(fds) else 0.0
            s_prev = fs[i-1] if i > 0 else 0.0
            ds_v = s_end - s_prev
            Z1=fZ[i] if i<len(fZ) else 0.0; X1=fX[i] if i<len(fX) else 0.0
            Y1=fY[i] if i<len(fY) else 0.0
            key=_key_from_type(etype); le=lte_data.get(str(name).upper(),{})
            Z0=fZ[i-1] if i>0 and i-1<len(fZ) else Z1
            X0=fX[i-1] if i>0 and i-1<len(fX) else X1
            Y0=fY[i-1] if i>0 and i-1<len(fY) else Y1
            # entry angle = exit angle of previous element (theta is always exit)
            th  = fth[i-1] if i>0 and i-1<len(fth) else 0.0
            phi = fph[i-1] if i>0 and i-1<len(fph) else 0.0
            elements.append({'name':str(name),'key':key,'index':i,
                's_start':s_prev,'length':ds_v,'flr_length':flr_ds,
                'angle':le.get('ANGLE',0.0),
                'k1':le.get('K1',0.0),'k2':le.get('K2',0.0),
                'hkick':le.get('HKICK',0.0),'vkick':le.get('VKICK',0.0),
                'kick':le.get('KICK',0.0),'ref_tilt':le.get('TILT',0.0),
                'voltage':le.get('VOLT',0.0),'frequency':le.get('FREQ',0.0),
                'flr_z0':Z0,'flr_x0':X0,'flr_y0':Y0,
                'flr_z1':Z1,'flr_x1':X1,'flr_y1':Y1,
                'flr_theta0':th,'flr_phi0':phi,
                'raw_angle':le.get('ANGLE',0.0)})
    except FileNotFoundError:
        L("[elegant] No .flr file found — floor plan will be unavailable.\n"
          "  → Add floor_output filename=\"lattice.flr\" &end to your .ele file.")
        tnames=twi.get('ElementName',[]); ttypes=twi.get('ElementType',[])
        for i in range(n):
            ds_v=float(s[i])-(float(s[i-1]) if i>0 else 0.0)
            etype=str(ttypes[i]) if i<len(ttypes) else ''
            key=_key_from_type(etype); le=lte_data.get(str(tnames[i]).upper() if i<len(tnames) else '',{})
            elements.append({'name':str(tnames[i]) if i<len(tnames) else f'e{i}',
                'key':key,'index':i,'s_start':float(s[i])-ds_v,'length':ds_v,
                'angle':le.get('ANGLE',0.0),'k1':le.get('K1',0.0),'k2':le.get('K2',0.0),
                'hkick':le.get('HKICK',0.0),'vkick':le.get('VKICK',0.0),
                'kick':le.get('KICK',0.0),'ref_tilt':le.get('TILT',0.0),
                'voltage':le.get('VOLT',0.0),'frequency':le.get('FREQ',0.0)})
    P(38, 'Reading magnet profile...')
    try:
        mn,_mt,_ms,mp=_read_mag(str(_find_sdds(run_dir,'.mag')))
        pm={n:v for n,v in zip(mn,mp)}
        for e in elements: e['profile']=pm.get(e['name'],0.0)
    except FileNotFoundError:
        L("[elegant] No .mag file found — using K1 as magnet profile.\n"
          "  → Add floor_output filename=\"lattice.mag\" &end for magnet profiles.")
        for e in elements: e['profile']=e.get('k1',0.0)
    bp=_read_elegant_beam_params(str(ep))
    if bp['emit_x']>0 or bp['emit_y']>0:
        L(f"[elegant] Beam params: emit_x={bp['emit_x']:.4e}, emit_y={bp['emit_y']:.4e}, sigma_dp={bp['sigma_dp']:.4e}")
    # Tune from last element phase advance (psix is in tune units = rad/2pi)
    import math as _math
    bp['tune_a'] = float(phi_a[-1]) if len(phi_a) else None
    bp['tune_b'] = float(phi_b[-1]) if len(phi_b) else None
    # Chromaticity from twi file scalar parameters
    try:
        bp['chroma_a'] = float(twi_params['dnux/dp']) if 'dnux/dp' in twi_params else None
        bp['chroma_b'] = float(twi_params['dnuy/dp']) if 'dnuy/dp' in twi_params else None
    except Exception: bp['chroma_a'] = bp['chroma_b'] = None
    # Store extra twi columns and scalar parameters directly in data dict
    def _twiarr(key):
        arr = twi.get(key, [])
        return np.array(arr[:len(s)], dtype=float) if arr else np.zeros_like(s)
    elegant_data = dict(
        s=s, beta_a=ba, beta_b=bb, eta_x=ex, eta_y=ey,
        orbit_x=ox, orbit_y=oy, phi_a=phi_a, phi_b=phi_b,
        alpha_a=al_a, alpha_b=al_b,
        etaxp=_twiarr('etaxp'), etayp=_twiarr('etayp'),
        xAperture=_twiarr('xAperture'), yAperture=_twiarr('yAperture'),
        dI1=_twiarr('dI1'), dI2=_twiarr('dI2'), dI3=_twiarr('dI3'),
        dI4=_twiarr('dI4'), dI5=_twiarr('dI5'),
        **{'dbetax/dp': _twiarr('dbetax/dp'),
           'dbetay/dp': _twiarr('dbetay/dp'),
           'dalphax/dp': _twiarr('dalphax/dp'),
           'dalphay/dp': _twiarr('dalphay/dp')},
        elements=elements, beam_params=bp,
    )
    # Bulk-load all numeric columns from .cen and .sig into elegant_data
    # so every column is available in expressions without hardcoding.
    _skip_cols = {'s', 'ElementName', 'ElementType', 'ElementOccurence',
                  'Pass', 'Step', 'SVNVersion', 'PassLength',
                  'Particles', 'pCentral', 'Charge'}
    for _src_dict in (cen, sig):
        for _col, _vals in _src_dict.items():
            if _col in _skip_cols or _col in elegant_data: continue
            _n2 = len(elegant_data['s'])
            _arr2 = (np.array(_vals[:_n2], dtype=float) if len(_vals) >= _n2
                     else np.concatenate([np.array(_vals, dtype=float),
                                          np.zeros(_n2 - len(_vals))]))
            elegant_data[_col] = _arr2
    # Store all scalar parameters from twi header into data dict
    elegant_data.update({k: v for k, v in twi_params.items()
                          if isinstance(v, (float, int))})
    return elegant_data

# ─── MAD-X backend ───────────────────────────────────────────────────────────
# load_madx(): reads twiss.tfs and optionally survey.tfs output files from MAD-X
# No cpymad required — user runs MAD-X, lux reads the output files.

def _read_tfs(filepath):
    """Parse a MAD-X TFS file.
    Returns (scalars_dict, col_names_list, data_dict).
    scalars_dict: header @ parameters (NAME -> value)
    col_names_list: ordered list of column names
    data_dict: col_name -> list of values (str or float)
    """
    scalars = {}
    col_names = []
    data = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip('\n')
        if not line.strip():
            continue

        if line.startswith('@'):
            # Scalar parameter: @ NAME %fmt value
            parts = line.split()
            if len(parts) >= 3:
                name = parts[1].upper()
                val_str = parts[-1].strip('"')
                try:
                    scalars[name] = float(val_str)
                except ValueError:
                    scalars[name] = val_str
            continue

        if line.startswith('*'):
            # Column names
            col_names = line[1:].split()
            for c in col_names:
                data[c] = []
            continue

        if line.startswith('$'):
            # Format line — skip
            continue

        # Data row
        if not col_names:
            continue

        # Parse row — strings are quoted, numbers are not
        tokens = []
        i = 0
        s = line.strip()
        while i < len(s):
            if s[i] == '"':
                j = s.index('"', i + 1)
                tokens.append(s[i+1:j])
                i = j + 1
            elif s[i] == ' ':
                i += 1
            else:
                j = i
                while j < len(s) and s[j] != ' ':
                    j += 1
                tokens.append(s[i:j])
                i = j

        for ci, col in enumerate(col_names):
            if ci < len(tokens):
                val = tokens[ci]
                try:
                    data[col].append(float(val))
                except ValueError:
                    data[col].append(val)
            else:
                data[col].append(None)

    return scalars, col_names, data


def _key_from_type_madx(keyword):
    """Map MAD-X KEYWORD to internal element key."""
    k = keyword.upper()
    if k in ('SBEND', 'RBEND', 'DIPEDGE'):           return 'sbend'
    if k in ('QUADRUPOLE',):                           return 'quadrupole'
    if k in ('SEXTUPOLE',):                            return 'sextupole'
    if k in ('OCTUPOLE',):                             return 'other'
    if k in ('HKICKER', 'VKICKER', 'KICKER',
             'TKICKER'):                               return 'kicker'
    if k in ('MONITOR', 'HMONITOR', 'VMONITOR',
             'INSTRUMENT', 'BPM'):                     return 'monitor'
    if k in ('MARKER',):                               return 'marker'
    if k in ('RFCAVITY',):                             return 'rfcavity'
    if k in ('LCAVITY',):                              return 'lcavity'
    if k in ('DRIFT', 'COLLIMATOR', 'RCOLLIMATOR',
             'ECOLLIMATOR', 'PLACEHOLDER',
             'MARKER', 'CHANGEREF', 'HACDIPOLE',
             'VACDIPOLE'):                             return 'drift'
    return 'other'


def load_madx(twiss_file, survey_file=None, log_fn=None, progress_fn=None):
    """Load optics from MAD-X TFS output files.

    Parameters
    ----------
    twiss_file  : path to twiss.tfs (required)
    survey_file : path to survey.tfs (optional — enables floor plan)
    """
    def L(m): (log_fn(m + '\n') if log_fn else print(m))
    def P(p, l): progress_fn(p, l) if progress_fn else None

    P(5, 'Reading MAD-X twiss file...')
    L(f"[madx] Reading twiss: {twiss_file}")
    scalars, col_names, twi = _read_tfs(twiss_file)

    def _arr(key):
        vals = twi.get(key.upper(), twi.get(key, []))
        return np.array([v if v is not None else 0.0 for v in vals], dtype=float)

    s    = _arr('S')
    L_   = _arr('L')
    ba   = _arr('BETX');  bb   = _arr('BETY')
    al_a = _arr('ALFX');  al_b = _arr('ALFY')
    ex   = _arr('DX');    ey   = _arr('DY')
    dpx  = _arr('DPX');   dpy  = _arr('DPY')
    mux  = _arr('MUX');   muy  = _arr('MUY')
    ox   = _arr('X');     oy   = _arr('Y')
    px_  = _arr('PX');    py_  = _arr('PY')
    n    = len(s)
    L(f"[madx] Twiss table: {n} elements, s_max={float(s[-1]):.3f} m")

    # Phase advance: MAD-X MUX/MUY are in tune units (cycles), consistent with lux convention
    pa = mux; pb = muy

    P(30, 'Building element list...')
    names    = twi.get('NAME',    twi.get('name',    [None]*n))
    keywords = twi.get('KEYWORD', twi.get('keyword', ['DRIFT']*n))
    k1l_arr  = _arr('K1L')
    k2l_arr  = _arr('K2L')
    ang_arr  = _arr('ANGLE')
    tilt_arr = _arr('TILT')
    hkick_arr = _arr('HKICK')
    vkick_arr = _arr('VKICK')

    elements = []
    for i in range(n):
        name    = str(names[i]) if names[i] is not None else f'e{i}'
        keyword = str(keywords[i]) if keywords[i] is not None else 'DRIFT'
        key     = _key_from_type_madx(keyword)
        length  = float(L_[i])
        s_end   = float(s[i])
        # If this element has the same s as the previous one it is a zero-length
        # element at the same position — force length=0 to avoid bar overlap
        if i > 0 and abs(s_end - float(s[i-1])) < 1e-9:
            length = 0.0
        s_start = s_end - length
        angle   = float(ang_arr[i])
        # K1L and K2L are integrated strengths — convert to k1, k2
        k1 = float(k1l_arr[i]) / length if length > 1e-6 else 0.0
        k2 = float(k2l_arr[i]) / length if length > 1e-6 else 0.0
        tilt   = float(tilt_arr[i])
        hkick  = float(hkick_arr[i])
        vkick  = float(vkick_arr[i])
        kick   = hkick if key == 'kicker' else 0.0

        elements.append({
            'name':      name,
            'key':       keyword.capitalize(),
            'index':     i,
            's_start':   s_start,
            'length':    length,
            'angle':     angle,
            'raw_angle': angle,
            'k1':        k1,
            'k2':        k2,
            'hkick':     hkick,
            'vkick':     vkick,
            'kick':      kick,
            'ref_tilt':  tilt,
            'voltage':   0.0,
            'frequency': 0.0,
            'profile':   k1,
        })

    # ── Survey (floor plan) ───────────────────────────────────────────────────
    if survey_file:
        P(50, 'Reading MAD-X survey file...')
        L(f"[madx] Reading survey: {survey_file}")
        try:
            _, _, sv = _read_tfs(survey_file)
            sv_names = sv.get('NAME', sv.get('name', []))
            sv_s     = [v if v is not None else 0.0 for v in sv.get('S', sv.get('s', []))]
            sv_X     = [v if v is not None else 0.0 for v in sv.get('X', sv.get('x', []))]
            sv_Y     = [v if v is not None else 0.0 for v in sv.get('Y', sv.get('y', []))]
            sv_Z     = [v if v is not None else 0.0 for v in sv.get('Z', sv.get('z', []))]
            sv_theta = [v if v is not None else 0.0 for v in sv.get('THETA', sv.get('theta', []))]
            sv_phi   = [v if v is not None else 0.0 for v in sv.get('PHI', sv.get('phi', []))]

            # Build positional index — twiss and survey have same elements in same order
            # Name-based lookup fails for duplicate names (e.g. QD_BATES appears many times)
            matched = 0
            for i, e in enumerate(elements):
                si = i
                if si >= len(sv_Z): break
                si0 = si - 1 if si > 0 else si
                e['flr_z0']    = float(sv_Z[si0])
                e['flr_z1']    = float(sv_Z[si])
                e['flr_x0']    = float(sv_X[si0])
                e['flr_x1']    = float(sv_X[si])
                e['flr_y0']    = float(sv_Y[si0])
                e['flr_y1']    = float(sv_Y[si])
                e['flr_theta0'] = float(sv_theta[si0])
                e['flr_phi0']   = float(sv_phi[si0])
                e['flr_theta1'] = float(sv_theta[si])
                matched += 1
            L(f"[madx] Survey matched {matched}/{len(elements)} elements.")
        except Exception as _sv_err:
            L(f"[madx] Survey load failed ({_sv_err}) — floor plan unavailable.")
    else:
        L("[madx] No survey file provided — floor plan will use dead-reckoning.")

    # ── Beam parameters from header scalars ───────────────────────────────────
    bp = {
        'emit_x':   0.0,
        'emit_y':   0.0,
        'sigma_dp': scalars.get('SIGE', 0.0),
        'tune_a':   scalars.get('Q1',   None),
        'tune_b':   scalars.get('Q2',   None),
        'chroma_a': scalars.get('DQ1',  None),
        'chroma_b': scalars.get('DQ2',  None),
    }
    # Store ALL TFS header scalars so they're available in expression panels
    for _sk, _sv in scalars.items():
        if isinstance(_sv, float) and _sk not in bp:
            bp[_sk.lower()] = _sv
            bp[_sk] = _sv

    P(80, 'MAD-X data loaded.')
    L(f"[madx] Done. {n} elements, s_max={float(s[-1]):.3f} m")

    _pad = lambda a, target: a[:target] if len(a) >= target else np.concatenate([a, np.zeros(target - len(a))])

    # Build the base return dict
    data = dict(
        s=s, beta_a=ba, beta_b=bb,
        eta_x=ex, eta_y=ey,
        orbit_x=_pad(ox, n), orbit_y=_pad(oy, n),
        phi_a=_pad(pa, n), phi_b=_pad(pb, n),
        alpha_a=_pad(al_a, n), alpha_b=_pad(al_b, n),
        elements=elements, beam_params=bp,
    )

    # Pass all remaining TFS columns through so expression panels can access them
    # (e.g. r11, r12, k1l, angle, etc.) — skip string/non-numeric columns
    _reserved = {'S', 'L', 'BETX', 'BETY', 'ALFX', 'ALFY', 'DX', 'DY',
                 'DPX', 'DPY', 'MUX', 'MUY', 'X', 'Y', 'PX', 'PY',
                 'NAME', 'KEYWORD', 'PARENT', 'TYPE'}
    for col in col_names:
        if col.upper() not in _reserved:
            try:
                arr = _arr(col)
                if len(arr) == n:
                    data[col.lower()] = arr
            except (ValueError, TypeError):
                pass  # skip string columns

    return data

# ─── xsuite backend ──────────────────────────────────────────────────────────
# load_xsuite(): loads JSON lattice, runs Twiss, builds element list from survey

def load_xsuite(json_file, log_fn=None, twiss_method='6d', line_name=None, progress_fn=None):
    """Load optics from an xsuite JSON lattice file.
    twiss_method: '6d' (standard) or '4d' (no RF / frozen longitudinal)
    line_name:    name of the line inside an Environment JSON (e.g. 'ring').
                  If None, auto-detected from the Environment or loaded as a plain Line.
    """
    def L(m): (log_fn(m + '\n') if log_fn else print(m))

    try:
        import xtrack as xt
    except ImportError:
        raise SystemExit(
            "[xsuite] xsuite is not installed.\n"
            "  → pip install xsuite"
        )

    def P(p,l): progress_fn(p,l) if progress_fn else None
    p = Path(json_file).resolve()
    L(f"[xsuite] Loading lattice from {p.name}")
    P(5, 'Loading JSON lattice...')

    # Try Environment first, then fall back to plain Line.from_json.
    # We separate the two steps so a bad line_name doesn't trigger the fallback.
    line = None
    env  = None
    try:
        env = xt.Environment.from_json(str(p))
        L(f"[xsuite] Loaded as Environment.")
    except Exception as e_env:
        L(f"[xsuite] Not an Environment JSON ({e_env}) — trying Line.from_json...")
        try:
            line = xt.Line.from_json(str(p))
            L("[xsuite] Loaded as plain Line.")
        except Exception as e_line:
            raise SystemExit(
                f"[xsuite] Could not load JSON as Environment or Line.\n"
                f"  Environment error: {e_env}\n"
                f"  Line error:        {e_line}"
            )

    if env is not None:
        # Pick the line from the Environment.
        # env.lines holds all named Line objects regardless of whether they
        # are rings or transfer lines — xsuite uses the same Line class for both.
        names = list(env.lines.keys()) if hasattr(env, 'lines') else []
        if not names:
            raise SystemExit("[xsuite] Environment has no lines.")

        if line_name and line_name in names:
            # User explicitly specified a line name
            chosen = line_name
        elif line_name:
            raise SystemExit(
                f"[xsuite] Line '{line_name}' not found in Environment.\n"
                f"  Available lines: {names}\n"
                f"  → Update the Line name field in the GUI."
            )
        else:
            # Auto-select: pick the line with the most elements (the full machine)
            # This works for both rings and transfer lines.
            def _line_len(n):
                try: return len(env[n].element_names)
                except Exception: return 0
            chosen = max(names, key=_line_len)

        L(f"[xsuite] Using line '{chosen}'  (available: {names})")
        line = env[chosen]

    # Attach a default particle ref if missing (needed for tracker)
    if line.particle_ref is None:
        L("[xsuite] No particle_ref found — defaulting to proton at 6500 GeV.")
        line.particle_ref = xt.Particles(p0c=6500e9, q0=1,
                                          mass0=xt.PROTON_MASS_EV)

    P(15, 'Building tracker (compiling)...')
    L("[xsuite] Building tracker...")
    line.build_tracker()

    # Try standard twiss, fall back to 4d.
    # Run each attempt in a sub-thread with a timeout so that a hanging
    # standard twiss (unstable longitudinal motion) does not block forever.
    method_arg = '4d' if str(twiss_method).strip().lower() == '4d' else None
    label      = '4d' if method_arg == '4d' else '6d (standard)'

    # Auto-detect twiss_init from JSON metadata — used for open/transfer lines
    # that have no periodic solution. Completely safe for rings (ignored if absent).
    _twiss_init = None
    try:
        import json as _json
        with open(str(p), 'r') as _jf:
            _raw = _json.load(_jf)
        _ti = _raw.get('metadata', {}).get('twiss_init', None)
        if _ti:
            _twiss_init = xt.TwissInit(
                betx  = _ti.get('betx',  1.0),
                alfx  = _ti.get('alfx',  0.0),
                bety  = _ti.get('bety',  1.0),
                alfy  = _ti.get('alfy',  0.0),
                dx    = _ti.get('dx',    0.0),
                dpx   = _ti.get('dpx',   0.0),
                dy    = _ti.get('dy',    0.0),
                dpy   = _ti.get('dpy',   0.0),
            )
            L(f"[xsuite] Found twiss_init in metadata — using as initial conditions (open line mode).")
    except Exception as _ti_err:
        L(f"[xsuite] Could not read twiss_init from metadata ({_ti_err}) — using periodic twiss.")

    P(25, f'Computing Twiss ({label})...')
    L(f"[xsuite] Computing Twiss ({label})...")
    try:
        if _twiss_init is not None:
            tw = line.twiss(init=_twiss_init, method=method_arg) if method_arg else line.twiss(init=_twiss_init)
        else:
            tw = line.twiss() if method_arg is None else line.twiss(method=method_arg)
        L(f"[xsuite] Twiss OK ({label}).")
        P(38, 'Twiss complete — reading elements...')
    except Exception as e:
        raise SystemExit(
            f"[xsuite] Twiss failed ({label} method).\n"
            f"  Error: {e}\n"
            + ("  → Try switching to 4d method in the GUI (xsuite Twiss toggle)."
               if method_arg is None else
               "  → Check your lattice definition.")
        )

    def _arr(name):
        try:
            v = getattr(tw, name, None)
            if v is None: return np.array([])
            return np.array(v, dtype=float)
        except Exception:
            return np.array([])

    s    = _arr('s')
    ba   = _arr('betx');  bb   = _arr('bety')
    al_a = _arr('alfx');  al_b = _arr('alfy')
    ex   = _arr('dx');    ey   = _arr('dy')
    mux  = _arr('mux');   muy  = _arr('muy')
    # Closed orbit — available in tw as x, y
    ox   = _arr('x');     oy   = _arr('y')
    n = len(s)
    L(f"[xsuite] Twiss table: {n} points, s_max={s[-1]:.3f} m")

    # Phase advance: xsuite mux/muy are in units of 2pi already
    # (they are fractional tune * 2pi, i.e. in radians / 2pi)
    # Keep consistent with Tao/ELEGANT normalisation (divide by 2pi done in load_tao)
    # mux from xsuite is already in tune units (cycles), so use directly
    pa = mux; pb = muy

    # ── Floor plan from survey + element table ───────────────────────────────
    elements = []
    try:
        # Element table: gives s positions (along beam path) and element types
        tab       = line.get_table()
        tab_names = list(tab.name)         if hasattr(tab, 'name')         else []
        tab_types = list(tab.element_type) if hasattr(tab, 'element_type') else []
        tab_s     = np.array(tab.s, dtype=float) if hasattr(tab, 's')     else np.array([])
        type_map  = {str(n): str(t) for n, t in zip(tab_names, tab_types)}
        s_end_map = {str(n): float(v) for n, v in zip(tab_names, tab_s)}

        # Survey: gives physical X/Z/Y coordinates for floor plan drawing
        # xsuite's survey_from_line calls np.bool() on arrays in some versions,
        # which fails both when np.bool is missing (NumPy>=1.24) and when
        # called on a multi-element array. Patch it to handle both cases.
        import numpy as _np_patch
        _had_bool = hasattr(_np_patch, 'bool')
        _orig_bool = getattr(_np_patch, 'bool', None)
        def _safe_bool(x):
            try:
                a = np.asarray(x)
                if a.ndim == 0: return bool(a.item())
                return bool(a.any())
            except Exception:
                return bool(x)
        _np_patch.bool = _safe_bool
        try:
            sv = line.survey()
        finally:
            if _had_bool and _orig_bool is not None:
                _np_patch.bool = _orig_bool
            else:
                try: del _np_patch.bool
                except Exception: pass
        sv_names = list(sv.name)               if hasattr(sv, 'name')   else []
        sv_X     = np.array(sv.X, dtype=float) if hasattr(sv, 'X')      else np.array([])
        sv_Z     = np.array(sv.Z, dtype=float) if hasattr(sv, 'Z')      else np.array([])
        sv_Y     = np.array(sv.Y, dtype=float) if hasattr(sv, 'Y')      else np.array([])
        sv_theta = np.array(sv.theta, dtype=float) if hasattr(sv, 'theta') else np.array([])
        # Build survey lookup by name (last occurrence wins for duplicates)
        sv_idx   = {str(nm): i for i, nm in enumerate(sv_names)}
        L(f"[xsuite] Element table: {len(tab_names)} elements, "
          f"Survey: {len(sv_names)} points.")

        # Build s_start for each element: use previous element's s_end
        # tab.s is the ENTRY s of each element, so:
        # s_start = tab_s[i], s_end = tab_s[i+1], length = tab_s[i+1] - tab_s[i]
        tab_s_list = list(tab_s)

        for i, tname in enumerate(tab_names):
            sname  = str(tname)
            etype  = type_map.get(sname, '')
            key    = _key_from_type_xs(etype)

            # s_start = tab_s[i], s_end = tab_s[i+1]
            s_start = float(tab_s_list[i]) if i < len(tab_s_list) else 0.0
            s_end   = float(tab_s_list[i+1]) if i + 1 < len(tab_s_list) else s_start
            length  = s_end - s_start

            # Skip pure drifts and markers — they clutter the layout bar
            if key in ('drift', 'other') and etype.lower() in ('drift', 'marker',
                        'multipole', ''):
                # still add to elements list for range filtering but mark as drift
                pass

            # Strengths from the Line object
            k1 = k2 = angle = 0.0
            try:
                el = line[sname]

                def _scalar(v):
                    """Safely extract a scalar from a value that might be an array."""
                    try:
                        f = float(v) if not hasattr(v, '__len__') else float(v.flat[0])
                        return f if np.isfinite(f) else 0.0
                    except Exception:
                        return 0.0

                # Thick element style (Quadrupole, Sextupole)
                if hasattr(el, 'k1'):
                    v = _scalar(el.k1)
                    if v: k1 = v
                if hasattr(el, 'k2'):
                    v = _scalar(el.k2)
                    if v: k2 = v
                # Thin multipole style (knl = integrated strengths)
                if hasattr(el, 'knl') and el.knl is not None:
                    try:
                        if len(el.knl) > 1:
                            v = _scalar(el.knl[1])
                            if v: k1 = k1 or v
                        if len(el.knl) > 2:
                            v = _scalar(el.knl[2])
                            if v: k2 = k2 or v
                    except Exception:
                        pass
                if hasattr(el, 'h'):
                    v = _scalar(el.h)
                    if v: angle = v * length
                # Prefer el.angle directly if available (Bend elements)
                if hasattr(el, 'angle'):
                    v = _scalar(el.angle)
                    if v: angle = -v
            except Exception:
                pass

            # Floor plan coordinates from survey (physical space)
            # tab_s[i] is entry of element i, so survey entry=sv[i], exit=sv[i+1]
            si  = i + 1 if i + 1 < len(sv_names) else i
            si0 = i
            X0  = float(sv_X[si0]) if 0 <= si0 < len(sv_X) else 0.0
            X1  = float(sv_X[si])  if 0 <= si  < len(sv_X) else 0.0
            Z0  = float(sv_Z[si0]) if 0 <= si0 < len(sv_Z) else 0.0
            Z1  = float(sv_Z[si])  if 0 <= si  < len(sv_Z) else 0.0
            Y0  = float(sv_Y[si0]) if 0 <= si0 < len(sv_Y) else 0.0
            Y1  = float(sv_Y[si])  if 0 <= si  < len(sv_Y) else 0.0
            th  = float(sv_theta[si0]) if 0 <= si0 < len(sv_theta) else 0.0

            elements.append({
                'name': sname, 'key': key, 'index': i,
                's_start': s_start, 'length': length,
                'angle': angle, 'k1': k1, 'k2': k2,
                'hkick': 0.0, 'vkick': 0.0, 'kick': 0.0,
                'ref_tilt': 0.0, 'voltage': 0.0, 'frequency': 0.0,
                'flr_z0': Z0, 'flr_x0': X0, 'flr_y0': Y0,
                'flr_z1': Z1, 'flr_x1': X1, 'flr_y1': Y1,
                'flr_theta0': th, 'flr_phi0': 0.0,
                'raw_angle': angle, 'profile': k1,
            })
        P(40, f'Built {len(elements)} elements.')
        L(f"[xsuite] Built {len(elements)} elements.")
    except Exception as e:
        import traceback as _tb
        L(f"[xsuite] Survey/floor plan failed:\n{_tb.format_exc()}\n— floor plan will be unavailable.")
        # Fallback: build elements from twiss table
        tw_names = list(getattr(tw, 'name', []))
        tw_types = list(getattr(tw, 'element_type', [''] * n))
        for i in range(n):
            sname = str(tw_names[i]) if i < len(tw_names) else f'e{i}'
            etype = str(tw_types[i]) if i < len(tw_types) else ''
            ds = float(s[i]) - (float(s[i-1]) if i > 0 else 0.0)
            elements.append({
                'name': sname, 'key': _key_from_type_xs(etype), 'index': i,
                's_start': float(s[i]) - ds, 'length': ds,
                'angle': 0.0, 'k1': 0.0, 'k2': 0.0,
                'hkick': 0.0, 'vkick': 0.0, 'kick': 0.0,
                'ref_tilt': 0.0, 'voltage': 0.0, 'frequency': 0.0,
                'profile': 0.0,
            })

    # Beam params from particle_ref if available
    bp = {'emit_x': 0.0, 'emit_y': 0.0, 'sigma_dp': 0.0,
          'tune_a': None, 'tune_b': None, 'chroma_a': None, 'chroma_b': None}
    try:
        pr = line.particle_ref
        if hasattr(pr, 'nemitt_x'): bp['emit_x'] = float(pr.nemitt_x[0])
        if hasattr(pr, 'nemitt_y'): bp['emit_y'] = float(pr.nemitt_y[0])
    except Exception:
        pass
    try:
        bp['tune_a']  = float(tw.qx)
        bp['tune_b']  = float(tw.qy)
        bp['chroma_a'] = float(tw.dqx) if hasattr(tw,'dqx') else None
        bp['chroma_b'] = float(tw.dqy) if hasattr(tw,'dqy') else None
        # Store all scalar twiss summary values into beam_params
        for _twk in dir(tw):
            if _twk.startswith('_'): continue
            try:
                _twv = getattr(tw, _twk)
                if isinstance(_twv, float) and _twk not in bp:
                    bp[_twk] = _twv
            except Exception: pass
    except Exception: pass

    def _pad(a, target_n):
        if len(a) >= target_n: return a[:target_n]
        return np.concatenate([a, np.zeros(target_n - len(a))])

    _skip = {'s', 'name', 'element_type', 'isthick', 'parent_name',
             'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy', 'dpx', 'dpy',
             'mux', 'muy', 'x', 'y', 'px', 'py'}

    base = dict(
        s=s, beta_a=ba, beta_b=bb,
        eta_x=ex, eta_y=ey,
        orbit_x=_pad(ox, n), orbit_y=_pad(oy, n),
        phi_a=_pad(pa, n), phi_b=_pad(pb, n),
        alpha_a=_pad(al_a, n), alpha_b=_pad(al_b, n),
        elements=elements, beam_params=bp,
        _tw=tw,   # keep twiss table for extra attr queries
        _line=line,  # keep line object for W-function ±δ twiss
    )

    # Pass all remaining numeric twiss columns through so expression panels
    # can access them directly (e.g. wx, wy, ddx, ddy, etc.)
    # xsuite TwissTable exposes columns via _col_names, keys(), or vars()
    _tw_col_names = (
        list(getattr(tw, '_col_names', None) or []) or
        (list(tw.keys()) if hasattr(tw, 'keys') else []) or
        [k for k in vars(tw) if not k.startswith('_')]
    )
    for col in _tw_col_names:
        if col in _skip:
            continue
        try:
            arr = np.array(getattr(tw, col), dtype=float)
            if len(arr) == n:
                base[col] = arr
        except (TypeError, ValueError, AttributeError):
            pass

    return base

def _key_from_type_xs(etype):
    """Map xsuite element_type string to internal key."""
    t = etype.lower()
    if 'bend'      in t: return 'sbend'
    if 'quadrupole' in t or t == 'quadrupole': return 'quadrupole'
    if 'sextupole' in t: return 'sextupole'
    if 'multipole' in t:
        return 'quadrupole'  # generic multipole — colour as quad
    if 'cavity'    in t: return 'rfcavity'
    if 'drift'     in t: return 'drift'
    if 'monitor'   in t: return 'monitor'
    if 'marker'    in t: return 'marker'
    if 'kicker'    in t or 'hkicker' in t or 'vkicker' in t: return 'kicker'
    return 'other'

# ─── Floor plan builders ─────────────────────────────────────────────────────
# _build_floor_plan(): X-Z plane using dead-reckoning or .flr coordinates
# _build_floor_plan_yz(): Y-Z plane for vertical bends
# _build_layout_bar(): element bar shown below optics panels

# ─── Tunnel wall ─────────────────────────────────────────────────────────────
# Reads a coordinate file with columns x_in, y_in, z_in, x_out, y_out, z_out
# and draws a shaded tunnel region on the floor plan views.
# Completely independent from lattice data — just its own coordinate set.
