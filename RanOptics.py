#!/usr/bin/env python3
# =============================================================================
# ranoptics — Accelerator Optics Plotter  (PySide6 edition)
# Version: 6.0
#
# Copyright (c) 2026 Randika Gamage
# Jefferson Lab (JLab), Newport News, VA
#
# Licensed under the MIT License. See LICENSE file for details.
#
# Backends: Tao (Bmad), ELEGANT, xsuite, MAD-X
# GUI:      PySide6
# Contact:  randika@jlab.org
# =============================================================================

"""
ranoptics.py
============
Self-contained GUI for accelerator optics plotting. PySide6 edition.
Backends: Tao (Bmad), ELEGANT, xsuite, MAD-X (TFS files).

Usage:
    python ranoptics.py

Requirements:
    pip install numpy plotly
    pip install PySide6            # GUI framework
    pip install pytao              # for Tao backend
    elegant + sddsconvert          # on PATH, for ELEGANT backend
    pip install xsuite             # for xsuite backend
    # MAD-X backend: no extra Python package needed —
    #   run MAD-X yourself and point lux at the twiss.tfs / survey.tfs output
    pip install kaleido            # optional: PNG/PDF export
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Engine Core
#  Contains: visual helpers, beam params, backends (Tao/ELEGANT/xsuite),
#  floor plan builders, optics panel builders, and plot_optics()
# ════════════════════════════════════════════════════════════════════════════

import os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

# ─── Visual helpers ──────────────────────────────────────────────────────────
# element_color(), element_thickness(), element_polygon(), element_oval()
# make_hover(), panel_title() — shared utilities for drawing elements on plots

def element_color(key):
    k = key.lower()
    if 'quadrupole' in k: return 'blue'
    if 'sbend'      in k: return 'red'
    if 'sextupole'  in k: return 'green'
    if k in ('hkicker','vkicker','kicker'): return 'orange'
    if 'monitor'    in k: return 'purple'
    if 'marker'     in k: return '#888888'   # gray — thin line only
    if 'rfcavity'   in k: return 'cyan'
    if 'lcavity'    in k: return 'cyan'
    return None

FULL_WIDTH_TYPES = ('sbend','quadrupole')
THIN_ELEMENT_THRESHOLD = 1e-3

def element_thickness(key, ft):
    return ft if any(t in key.lower() for t in FULL_WIDTH_TYPES) else ft/2.0

def element_polygon(x0, y0, theta0, L, angle, thickness):
    t = thickness/2.0
    if abs(angle) < 1e-6:
        dx,dy = np.cos(theta0)*L, np.sin(theta0)*L
        nx,ny = -np.sin(theta0)*t, np.cos(theta0)*t
        return ([x0+nx,x0+dx+nx,x0+dx-nx,x0-nx,x0+nx],
                [y0+ny,y0+dy+ny,y0+dy-ny,y0-ny,y0+ny])
    rho = L/angle; n_pts = max(30,int(abs(angle)*80))
    angs = np.linspace(0,angle,n_pts)
    cx=x0-rho*np.sin(theta0); cy=y0+rho*np.cos(theta0)
    ox=cx+(rho+t)*np.sin(theta0+angs); oy=cy-(rho+t)*np.cos(theta0+angs)
    ix=cx+(rho-t)*np.sin(theta0+angs); iy=cy-(rho-t)*np.cos(theta0+angs)
    return (np.concatenate([ox,ix[::-1],[ox[0]]]).tolist(),
            np.concatenate([oy,iy[::-1],[oy[0]]]).tolist())

def element_oval(x0, y0, theta0, L, thickness):
    a,b = L/2.0, thickness/2.0
    t = np.linspace(0,2*np.pi,49)
    u,v = a*np.cos(t), b*np.sin(t)
    cx=x0+(L/2)*np.cos(theta0); cy=y0+(L/2)*np.sin(theta0)
    return (cx+u*np.cos(theta0)-v*np.sin(theta0)).tolist(), \
           (cy+u*np.sin(theta0)+v*np.cos(theta0)).tolist()

def make_hover(elem):
    name=elem['name'].split('\\')[-1]; key=elem['key']; L=elem['length']
    k1=elem.get('k1',0.0); k2=elem.get('k2',0.0)
    angle=elem.get('angle',0.0)
    raw_angle=elem.get('raw_angle', angle)  # original angle before flip/remap
    s0=elem['s_start']; kc=key.lower()
    lines=[f'<b>{name}</b>',f'<i>{key}</i>',
           f'L = {L:.4f} m',
           f's_start = {s0:.4f} m',
           f's_end &nbsp;= {s0+L:.4f} m']
    if 'sbend' in kc:
        rt=elem.get('ref_tilt',0.0)
        bend_plane='Vertical' if abs(abs(rt)-np.pi/2)<0.01 else 'Horizontal'
        lines.append(f'Bend plane: {bend_plane}')
        lines.append(f'Angle = {np.degrees(raw_angle):.4f}°')
        if abs(raw_angle)>1e-9: lines.append(f'ρ = {abs(L/raw_angle):.4f} m')
    if 'quadrupole' in kc: lines.append(f'K1 = {k1:.6f} m⁻²')
    if 'sextupole'  in kc: lines.append(f'K2 = {k2:.6f} m⁻³')
    if kc=='kicker':
        lines+=[f'hkick={elem.get("hkick",0):.6f}',f'vkick={elem.get("vkick",0):.6f}']
    elif kc in ('hkicker','vkicker'):
        lines.append(f'kick={elem.get("kick",0):.6f}')
    if 'rfcavity' in kc or 'lcavity' in kc:
        v=elem.get('voltage',0.0); f=elem.get('frequency',0.0)
        if v: lines.append(f'V={v/1e6:.3f} MV' if abs(v)>=1e6 else f'V={v:.1f} V')
        if f: lines.append(f'f={f/1e9:.4f} GHz' if f>=1e9 else f'f={f/1e6:.4f} MHz')
    return '<br>'.join(lines)+'<extra></extra>'

def panel_title(p):
    return {'twiss':'Beta Functions & Dispersion','phase':'Phase Advance',
            'orbit':'Orbit','beamsize':'Beam Size',
            'beta':'Beta Functions','dispersion':'Dispersion','alpha':'Alpha Functions',
            'summary':'Lattice Summary','latdiff':'Lattice Diff',
            'bar':'Beamline','floor-xz':'Floor Plan — X vs Z',
            'floor-yz':'Floor Plan — Y vs Z'}.get(p,'')

# ─── Beam params ─────────────────────────────────────────────────────────────
# Read emittance, sigma_dp, tune, and chromaticity from each backend

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

    for line in result:
        if 'Lord Elements:' in line: break
        if line.startswith('#') or not line.strip(): continue
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

# ─── Tao dot-name pre-parser ─────────────────────────────────────────────────
# Handles Tao native dot notation (beta.a, chrom.deta.x, k.11b) transparently.
# User types Tao names with dots; we detect, fetch, and substitute internally.

import re as _re_expr

def _parse_tao_dotnames(expr):
    """Find all Tao-style dot-names in an expression.
    Matches word.word patterns NOT followed by '(' (excludes np.sqrt etc.)
    Returns list of unique dot-name strings.
    """
    pattern = _re_expr.compile(r'\b([a-zA-Z][\w]*(?:\.[\w]+)+)(?!\s*\()')
    found = []
    for m in pattern.finditer(expr):
        name = m.group(1)
        if name.startswith('np.') or name.startswith('math.'):
            continue
        if name not in found:
            found.append(name)
    return found

def _dotname_to_pyname(dotname):
    """Convert Tao dot-name to safe Python identifier.
    beta.a -> __tao_beta_a__
    chrom.deta.x -> __tao_chrom_deta_x__
    """
    safe = dotname.replace('.', '_')
    return f'__tao_{safe}__'

def _substitute_dotnames(expr, dotnames):
    """Replace Tao dot-names in expr with safe Python equivalents."""
    result = expr
    for name in sorted(dotnames, key=len, reverse=True):
        pyname = _dotname_to_pyname(name)
        result = _re_expr.sub(r'\b' + _re_expr.escape(name) + r'\b', pyname, result)
    return result

def _fetch_tao_dotnames(tao, dotnames, s_len, log_fn=None):
    """Fetch Tao dot-name arrays via show lattice -att.
    Returns dict mapping safe_pyname -> np.ndarray.
    """
    def L(m):
        if log_fn: log_fn(m + '\n')
    result = {}
    for name in dotnames:
        pyname = _dotname_to_pyname(name)
        try:
            attrs = _query_tao_attrs(tao, [name], log_fn=log_fn)
            arr = attrs.get(name, np.zeros(s_len))
            arr = np.where(np.isnan(arr), 0.0, arr)
            if len(arr) != s_len:
                arr = np.resize(arr, s_len)
            result[pyname] = arr
            L(f"[expr] Fetched {name} ({len(arr)} points)")
        except Exception as e:
            L(f"[expr] Warning: could not fetch \'{name}\': {e}")
            result[pyname] = np.zeros(s_len)
    return result

def _eval_expression_tao(expr_str, namespace, data, s_len, log_fn=None):
    """Evaluate expression with Tao dot-name auto-fetching.
    Detects beta.a style names, fetches from Tao, substitutes safe names, evals.
    """
    def L(m):
        if log_fn: log_fn(m + '\n')
    dotnames = _parse_tao_dotnames(expr_str)
    if dotnames:
        L(f"[expr] Auto-fetching: {dotnames}")
        tao = data.get('_tao')
        if tao:
            fetched = _fetch_tao_dotnames(tao, dotnames, s_len, log_fn)
            namespace.update(fetched)
        else:
            L("[expr] Warning: no live Tao instance for auto-fetch")
    safe_expr = _substitute_dotnames(expr_str, dotnames)
    try:
        result = eval(safe_expr, {"__builtins__": {}}, namespace)
        return np.asarray(result, dtype=float)
    except Exception as e:
        L(f"[expr] Error evaluating \'{expr_str}\': {e}")
        return None

def _build_expr_namespace(data, code, extra_attrs=None, log_fn=None, uni_idx=1):
    """Build the full numpy namespace for expression evaluation.

    Combines:
      - Standard optics arrays (beta_a, eta_x, orbit_x, etc.)
      - Per-backend native aliases (betax/betx etc.)
      - Element arrays (k1, k2, angle, length, rho)
      - Extra column attributes requested by the user
      - numpy as np

    Parameters
    ----------
    data        : loaded data dict from load_tao / load_elegant / load_xsuite
    code        : 'tao', 'elegant', or 'xsuite'
    extra_attrs : list of additional attribute name strings to fetch
    log_fn      : optional logging function

    Returns
    -------
    namespace dict ready for eval()
    s_array     : the s array to use for plotting x-axis
    """
    extra_attrs = extra_attrs or []

    # ── Standard arrays always available ─────────────────────────────────
    ns = {
        's':       data['s'],
        'beta_a':  data.get('beta_a',  np.zeros_like(data['s'])),
        'beta_b':  data.get('beta_b',  np.zeros_like(data['s'])),
        'alpha_a': data.get('alpha_a', np.zeros_like(data['s'])),
        'alpha_b': data.get('alpha_b', np.zeros_like(data['s'])),
        'eta_x':   data.get('eta_x',   np.zeros_like(data['s'])),
        'eta_y':   data.get('eta_y',   np.zeros_like(data['s'])),
        'orbit_x': data.get('orbit_x', np.zeros_like(data['s'])),
        'orbit_y': data.get('orbit_y', np.zeros_like(data['s'])),
        'phi_a':   data.get('phi_a',   np.zeros_like(data['s'])),
        'phi_b':   data.get('phi_b',   np.zeros_like(data['s'])),
        'np':      np,
    }
    # ── Per-element magnet arrays from elements list ─────────────────
    # k1, k2, angle, length are already normalized across all backends
    elements = data.get('elements', [])
    if elements:
        _n = len(ns['s'])
        # Build s_start -> index mapping to align element arrays with optics arrays
        def _elem_arr(key, default=0.0):
            vals = [e.get(key, default) for e in elements]
            arr  = np.array(vals, dtype=float)
            # Interpolate to match optics s-array length if needed
            if len(arr) != _n:
                elem_s = np.array([e.get('s_start', 0.0) + e.get('length', 0.0)
                                   for e in elements])
                arr = np.interp(ns['s'], elem_s, arr)
            return arr
        ns['k1']     = _elem_arr('k1')
        ns['k2']     = _elem_arr('k2')
        ns['angle']  = _elem_arr('angle')
        ns['length'] = _elem_arr('length')
        ns['rho']    = np.where(np.abs(ns['angle']) > 1e-9,
                                ns['length'] / np.where(np.abs(ns['angle']) > 1e-9,
                                                        ns['angle'], 1.0), 0.0)

    # Per-backend aliases so users can use native names for their code
    if code == 'elegant':
        # Bulk-load all numpy arrays from elegant_data into namespace
        # so every column (twi, cen, sig) is available by its native name.
        _slen = len(ns['s'])
        for _k, _v in data.items():
            if _k in ns or not isinstance(_v, np.ndarray): continue
            if len(_v) == _slen:
                ns[_k] = _v
        # Standard aliases so users can use expected ELEGANT names
        ns.update({
            'betax':  ns['beta_a'],  'betay':  ns['beta_b'],
            'alphax': ns['alpha_a'], 'alphay': ns['alpha_b'],
            'etax':   ns['eta_x'],   'etay':   ns['eta_y'],
            'psix':   ns['phi_a'],   'psiy':   ns['phi_b'],
        })
    elif code == 'xsuite':
        # xsuite users expect betx, bety, alfx, dx etc.
        ns.update({
            'betx':  ns['beta_a'],  'bety':  ns['beta_b'],
            'alfx':  ns['alpha_a'], 'alfy':  ns['alpha_b'],
            'dx':    ns['eta_x'],   'dy':    ns['eta_y'],
            'mux':   ns['phi_a'],   'muy':   ns['phi_b'],
            'x':     ns['orbit_x'], 'y':     ns['orbit_y'],
        })
    elif code == 'madx':
        # Bulk-load all numpy arrays from data dict so every TFS column
        # (k1l, k2l, angle, r11, r12, etc.) is available by its native name.
        _slen = len(ns['s'])
        for _k, _v in data.items():
            if _k in ns or not isinstance(_v, np.ndarray): continue
            if len(_v) == _slen:
                ns[_k] = _v
        # Standard aliases so users can use expected MAD-X names
        ns.update({
            'betx':  ns['beta_a'],  'bety':  ns['beta_b'],
            'alfx':  ns['alpha_a'], 'alfy':  ns['alpha_b'],
            'dx':    ns['eta_x'],   'dy':    ns['eta_y'],
            'mux':   ns['phi_a'],   'muy':   ns['phi_b'],
            'x':     ns['orbit_x'], 'y':     ns['orbit_y'],
        })
        # Tao users can also use common aliases
        ns.update({
            'betax':  ns['beta_a'],  'betay':  ns['beta_b'],
            'alphax': ns['alpha_a'], 'alphay': ns['alpha_b'],
            'etax':   ns['eta_x'],   'etay':   ns['eta_y'],
        })

    # ── Fetch extra per-element attributes ────────────────────────────────
    if extra_attrs:
        # Only fetch attrs not already in namespace
        unresolved = [a for a in extra_attrs if a not in ns]
        if unresolved:
            if code == 'tao':
                tao = data.get('_tao')
                if tao:
                    extra = _query_tao_attrs(tao, unresolved,
                                             uni_idx=uni_idx, log_fn=log_fn)
                    ns.update({k: v for k, v in extra.items() if k != 's'})
                else:
                    if log_fn: log_fn("[expr] Warning: no live Tao instance for extra attrs\n")
            elif code == 'elegant':
                extra = _query_elegant_attrs(data, unresolved, log_fn)
                ns.update({k: v for k, v in extra.items() if k != 's'})
            elif code == 'xsuite':
                # Try _tw first, fall back to reading columns from data dict
                tw = data.get('_tw')
                if tw:
                    extra = _query_xsuite_attrs(tw, unresolved, log_fn)
                else:
                    extra = {a: data.get(a, np.zeros_like(data['s']))
                             for a in unresolved}
                ns.update({k: v for k, v in extra.items() if k != 's'})

    return ns, data['s']

def _eval_expression(expr_str, namespace, log_fn=None):
    """Safely evaluate a numpy expression string against a namespace.

    Parameters
    ----------
    expr_str  : string like '(beta_a * beta_b) / alpha_a'
    namespace : dict from _build_expr_namespace()

    Returns
    -------
    np.ndarray result, or None if evaluation fails
    """
    def L(m):
        if log_fn: log_fn(m + '\n')
    try:
        result = eval(expr_str, {"__builtins__": {}}, namespace)
        return np.asarray(result, dtype=float)
    except Exception as e:
        L(f"[expr] Error evaluating '{expr_str}': {e}")
        return None

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
            ds_v=fds[i] if i<len(fds) else 0.0; s_end=fs[i] if i<len(fs) else 0.0
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
                's_start':s_end-ds_v,'length':ds_v,'angle':le.get('ANGLE',0.0),
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
                          if isinstance(v, float)})
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

def _read_tunnel_wall(filepath, log_fn=None):
    """Read tunnel wall coordinate file.

    Format: each row has 6 values (any delimiter — comma, tab, space):
        x_inner  y_inner  z_inner  x_outer  y_outer  z_outer

    Returns dict with keys:
        xi, yi, zi  — inner wall arrays
        xo, yo, zo  — outer wall arrays
    or None on failure.
    """
    import re as _re
    def _l(msg):
        if log_fn: log_fn(msg)
    xi,yi,zi,xo,yo,zo = [],[],[],[],[],[]
    short_lines = 0
    bad_floats  = 0
    try:
        with open(filepath, 'r') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'): continue
                # Split on any combination of comma, tab, or whitespace
                parts = _re.split(r'[,\t\s]+', line)
                parts = [p for p in parts if p]
                if len(parts) < 6:
                    if short_lines == 0:
                        _l(f"[tunnel] Line {lineno}: expected 6 columns, got {len(parts)}: {line!r}")
                    short_lines += 1
                    continue
                try:
                    xi.append(float(parts[0])); yi.append(float(parts[1])); zi.append(float(parts[2]))
                    xo.append(float(parts[3])); yo.append(float(parts[4])); zo.append(float(parts[5]))
                except ValueError as e:
                    if bad_floats == 0:
                        _l(f"[tunnel] Line {lineno}: float parse error ({e}): {line!r}")
                    bad_floats += 1
                    continue
    except Exception as e:
        _l(f"[tunnel] Error opening file: {e}")
        return None
    if short_lines:
        _l(f"[tunnel] {short_lines} line(s) skipped (< 6 columns)")
    if bad_floats:
        _l(f"[tunnel] {bad_floats} line(s) skipped (bad float values)")
    if len(xi) < 2:
        _l(f"[tunnel] Only {len(xi)} valid row(s) parsed — need at least 2. "
           f"Expected format: x_in y_in z_in x_out y_out z_out per line.")
        return None
    import numpy as np
    xi = np.array(xi); yi = np.array(yi); zi = np.array(zi)
    xo = np.array(xo); yo = np.array(yo); zo = np.array(zo)
    # Detect ring: if first and last points are close, it's a closed ring.
    # For rings, append the first point to close the dotted wall lines.
    # For open linacs, leave as-is — fill='toself' closes the polygon visually
    # without drawing a diagonal return line across the plot.
    dist = ((xi[0]-xi[-1])**2 + (zi[0]-zi[-1])**2) ** 0.5
    is_ring = dist < 1e-3
    if is_ring:
        xi = np.append(xi, xi[0]); yi = np.append(yi, yi[0]); zi = np.append(zi, zi[0])
        xo = np.append(xo, xo[0]); yo = np.append(yo, yo[0]); zo = np.append(zo, zo[0])
    return dict(xi=xi, yi=yi, zi=zi, xo=xo, yo=yo, zo=zo, is_ring=is_ring)

def _draw_tunnel_wall_xz(fig, wall, row=1, flip=False):
    """Draw tunnel wall shaded region on X-Z floor plan."""
    import plotly.graph_objects as go
    import numpy as np
    if wall is None: return

    sign = -1.0 if flip else 1.0
    zi = wall['zi']; xi = sign * wall['xi']
    zo = wall['zo']; xo = sign * wall['xo']

    # For fill='toself', Plotly closes the polygon automatically —
    # no need to append the start point (which would draw a diagonal on linacs).
    z_poly = list(zi) + list(zo[::-1])
    x_poly = list(xi) + list(xo[::-1])

    fig.add_trace(go.Scatter(
        x=z_poly, y=x_poly, fill='toself',
        fillcolor='rgba(120,120,140,0.15)',
        line=dict(color='rgba(0,0,0,0)', width=0),
        name='Tunnel', legendgroup='tunnel_xz',
        showlegend=False, hoverinfo='skip',
    ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=zi, y=xi, mode='lines',
        line=dict(color='rgba(150,150,170,0.7)', width=1, dash='dot'),
        name='Tunnel wall', legendgroup='tunnel_xz',
        showlegend=False, hoverinfo='skip',
    ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=zo, y=xo, mode='lines',
        line=dict(color='rgba(150,150,170,0.7)', width=1, dash='dot'),
        name='Tunnel wall', legendgroup='tunnel_xz',
        showlegend=False, hoverinfo='skip',
    ), row=row, col=1)

    all_x = np.concatenate([xi, xo])
    all_z = np.concatenate([zi, zo])
    x_pad = (all_x.max() - all_x.min()) * 0.05 + 0.1
    z_pad = (all_z.max() - all_z.min()) * 0.02 + 0.1
    return ([all_z.min()-z_pad, all_z.max()+z_pad],
            [all_x.min()-x_pad, all_x.max()+x_pad])

def _draw_tunnel_wall_yz(fig, wall, row=2, flip=False):
    """Draw tunnel wall shaded region on Y-Z floor plan."""
    import plotly.graph_objects as go
    import numpy as np
    if wall is None: return

    sign = -1.0 if flip else 1.0
    zi = wall['zi']; yi = sign * wall['yi']
    zo = wall['zo']; yo = sign * wall['yo']

    z_poly = list(zi) + list(zo[::-1])
    y_poly = list(yi) + list(yo[::-1])

    fig.add_trace(go.Scatter(
        x=z_poly, y=y_poly, fill='toself',
        fillcolor='rgba(120,120,140,0.15)',
        line=dict(color='rgba(0,0,0,0)', width=0),
        name='Tunnel', legendgroup='tunnel_yz',
        showlegend=False, hoverinfo='skip',
    ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=zi, y=yi, mode='lines',
        line=dict(color='rgba(150,150,170,0.7)', width=1, dash='dot'),
        name='Tunnel wall', legendgroup='tunnel_yz',
        showlegend=False, hoverinfo='skip',
    ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=zo, y=yo, mode='lines',
        line=dict(color='rgba(150,150,170,0.7)', width=1, dash='dot'),
        name='Tunnel wall', legendgroup='tunnel_yz',
        showlegend=False, hoverinfo='skip',
    ), row=row, col=1)

    all_y = np.concatenate([yi, yo])
    all_z = np.concatenate([zi, zo])
    y_pad = (all_y.max() - all_y.min()) * 0.15 + 0.1
    z_pad = (all_z.max() - all_z.min()) * 0.02 + 0.1
    return ([all_z.min()-z_pad, all_z.max()+z_pad],
            [all_y.min()-y_pad, all_y.max()+y_pad])

def _build_floor_plan(fig, elements, element_height, flip_bend,
                      row=1, legend_name='legend3', fp_legend_name='legend4',
                      show_fp_legend=True, beampipe_color='gray',
                      show_markers=False):
    # Use fp_legend_name for everything - one legend box per call
    legend_name = fp_legend_name
    import plotly.graph_objects as go
    import copy as _copy
    use_flr = any('flr_z0' in e for e in elements)
    # Filter out markers/monitors unless show_markers is True
    _MARKER_MONITOR_KEYS = {'marker','mark','monitor','hmon','vmon','instrument','bpm'}
    if not show_markers:
        elements = [e for e in elements
                    if e['key'].lower() not in _MARKER_MONITOR_KEYS]
    # For ELEGANT (.flr data): flip_bend mirrors ALL X coords (beampipe + polygons)
    # For Tao (dead-reckoning): flip_bend negates bend angle in polygon drawing
    if use_flr and flip_bend:
        flipped = []
        for e in elements:
            ec = _copy.copy(e)
            for k in ('flr_x0', 'flr_x1'):
                if k in ec: ec[k] = -ec[k]
            # Mirror X negates heading angle and arc curvature
            if 'flr_theta0' in ec:
                ec['flr_theta0'] = -ec['flr_theta0']
            if 'angle' in ec:
                if 'raw_angle' not in ec: ec['raw_angle'] = ec['angle']
                ec['angle'] = -ec['angle']
            flipped.append(ec)
        elements = flipped
    if use_flr:
        # Seed beampipe from the first element's entry coords — never assume (0,0)
        _first = next((e for e in elements if 'flr_z0' in e), None)
        pz=[_first['flr_z0']] if _first else [0.0]
        px=[_first['flr_x0']] if _first else [0.0]
        for e in elements:
            if 'flr_z1' not in e: continue
            z1,x1=e['flr_z1'],e['flr_x1']
            if z1!=pz[-1] or x1!=px[-1]: pz.append(z1); px.append(x1)
    else:
        pz,px=[0.0],[0.0]; xp,yp,tp=0.0,0.0,0.0
        for e in elements:
            L_=e['length']; ang=e.get('angle',0.0); key=e['key']
            if L_==0: continue
            if 'sbend' in key.lower() and abs(ang)>1e-6:
                iv=abs(abs(e.get('ref_tilt',0.0))-np.pi/2)<0.01
                ba=0.0 if iv else (-ang if flip_bend else ang)
                if abs(ba)>1e-6:
                    rho=L_/ba; aa=np.linspace(0,ba,max(30,int(abs(ba)*80)))
                    cx=xp-rho*np.sin(tp); cy=yp+rho*np.cos(tp)
                    az=cx+rho*np.sin(tp+aa); ax=cy-rho*np.cos(tp+aa)
                    pz.extend(az.tolist()); px.extend(ax.tolist())
                    xp,yp,tp=az[-1],ax[-1],tp+ba
                    continue
            xp+=L_*np.cos(tp); yp+=L_*np.sin(tp); pz.append(xp); px.append(yp)
    fig.add_trace(go.Scatter(x=pz,y=px,mode='lines',line=dict(color=beampipe_color,width=2),
        name='Beampipe',showlegend=False,hoverinfo='skip'),row=row,col=1)

    dr_x,dr_y,dr_t=0.0,0.0,0.0; la=set()
    _lt={'quadrupole':'Quadrupole','sbend':'Dipole','sextupole':'Sextupole',
         'kicker':'Kicker','hkicker':'Kicker','vkicker':'Kicker',
         'monitor':'Monitor','rfcavity':'RF Cavity','lcavity':'RF Cavity'}
    # Prefix legendgroup with legend_name so multiple calls on the same figure
    # (e.g. primary row + compare row) don't share legend entries
    _lg = legend_name  # shorthand for prefix
    for elem in elements:
        L_=elem['length']; ang=elem.get('angle',0.0); key=elem['key']
        color=element_color(key); rt=elem.get('ref_tilt',0.0)
        hover=make_hover(elem); th=element_thickness(key,element_height); kc=key.lower()
        ll=_lt.get(kc); sl=ll is not None and ll not in la
        if sl: la.add(ll)
        if L_==0:
            # Zero-length element — draw thin vertical line regardless of type
            _zc = color or '#888888'
            if use_flr and 'flr_z0' in elem:
                mx,my,theta = elem['flr_z0'],elem['flr_x0'],elem.get('flr_theta0',0.0)
            elif not use_flr:
                mx,my,theta = dr_x,dr_y,dr_t
            else:
                continue  # no floor coords available, skip silently
            ht=element_thickness(key,element_height)/2.0
            nx,ny=-np.sin(theta)*ht,np.cos(theta)*ht
            fig.add_trace(go.Scatter(x=[mx-nx,mx+nx],y=[my-ny,my+ny],mode='lines',
                line=dict(color=_zc,width=2),name=ll or key,legend=legend_name,
                hoverlabel=dict(bgcolor=_zc),
                legendgroup=f'{_lg}_{ll or key}',showlegend=False,hovertemplate=hover),row=row,col=1)
            continue
        if color is None:
            if not use_flr: dr_x+=L_*np.cos(dr_t); dr_y+=L_*np.sin(dr_t)
            continue
        if L_<THIN_ELEMENT_THRESHOLD:
            mx,my,theta=(elem['flr_z0'],elem['flr_x0'],elem.get('flr_theta0',0.0)) if use_flr else (dr_x,dr_y,dr_t)
            if not use_flr: dr_x+=L_*np.cos(dr_t); dr_y+=L_*np.sin(dr_t)
            ht=element_thickness(key,element_height)/2.0
            nx,ny=-np.sin(theta)*ht,np.cos(theta)*ht
            fig.add_trace(go.Scatter(x=[mx-nx,mx+nx],y=[my-ny,my+ny],mode='lines',
                line=dict(color=color,width=2),name=ll,legend=legend_name,
                hoverlabel=dict(bgcolor=color),
                legendgroup=f'{_lg}_{ll}',showlegend=False,hovertemplate=hover),row=row,col=1)
            continue
        if use_flr: x0,y0,theta=elem['flr_z0'],elem['flr_x0'],elem.get('flr_theta0',0.0)
        else:       x0,y0,theta=dr_x,dr_y,dr_t
        if 'rfcavity' in kc or 'lcavity' in kc:
            ov_x,ov_y=element_oval(x0,y0,theta,L_,th)
            hx=[x0,x0+L_*np.cos(theta)]; hy=[y0,y0+L_*np.sin(theta)]
            if not use_flr: dr_x,dr_y=hx[-1],hy[-1]
            fig.add_trace(go.Scatter(x=hx,y=hy,mode='lines',
                line=dict(color='rgba(0,0,0,0)',width=max(8,th*20)),
                hoverlabel=dict(bgcolor=color),
                legend=legend_name,legendgroup=f'{_lg}_h_{ll or "o"}',showlegend=False,hovertemplate=hover),row=row,col=1)
            fig.add_trace(go.Scatter(x=ov_x,y=ov_y,mode='lines',fill='toself',
                fillcolor=color,line=dict(color=color,width=0),name=ll,
                legend=legend_name,legendgroup=f'{_lg}_{ll}',showlegend=False,hoverinfo='skip'),row=row,col=1)
            continue
        ba=0.0
        if 'sbend' in kc and abs(ang)>1e-6:
            iv = abs(abs(rt) - np.pi/2) < 0.01
            if use_flr:
                if iv:
                    ba = 0.0
                else:
                    # Derive bend angle from survey geometry — theta1 - theta0
                    # gives the actual geometric sweep with the correct sign,
                    # independent of Bmad's ANGLE attribute sign convention.
                    _th1 = elem.get('flr_theta1', theta + ang)
                    ba = _th1 - theta
            else:
                ba = 0.0 if iv else (-ang if flip_bend else ang)
        px_,py_=element_polygon(x0,y0,theta,L_,ba,th)
        if abs(ba)>1e-6:
            rho=L_/ba; aa=np.linspace(0,ba,max(30,int(abs(ba)*80)))
            cx=x0-rho*np.sin(theta); cy=y0+rho*np.cos(theta)
            hx=(cx+rho*np.sin(theta+aa)).tolist(); hy=(cy-rho*np.cos(theta+aa)).tolist()
            if not use_flr: dr_x,dr_y,dr_t=hx[-1],hy[-1],dr_t+ba
        else:
            hx=[x0,x0+L_*np.cos(theta)]; hy=[y0,y0+L_*np.sin(theta)]
            if not use_flr: dr_x,dr_y=hx[-1],hy[-1]
        fig.add_trace(go.Scatter(x=hx,y=hy,mode='lines',
            line=dict(color='rgba(0,0,0,0)',width=max(8,th*20)),
            hoverlabel=dict(bgcolor=color),
            legend=legend_name,legendgroup=f'{_lg}_h_{ll or "o"}',showlegend=False,hovertemplate=hover),row=row,col=1)
        fig.add_trace(go.Scatter(x=px_,y=py_,mode='lines',fill='toself',
            fillcolor=color,line=dict(color=color,width=0),name=ll,
            legend=legend_name,legendgroup=f'{_lg}_{ll}',showlegend=False,hoverinfo='skip'),row=row,col=1)
    for label,col in [('Dipole','red'),('Quadrupole','blue'),('Sextupole','green'),
                      ('Kicker','orange'),('Monitor','purple'),('RF Cavity','cyan')]:
        if label not in la: continue
        fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',
            marker=dict(size=10,color=col,symbol='square'),name=label,
            legend=fp_legend_name,legendgroup=f'fp_icon_{label}',
            showlegend=show_fp_legend),row=row,col=1)

def _trap_band_yz(z0, y0, z1, y1, phi_entry, phi_exit, half_th):
    """Trapezoid polygon: perpendicular cuts at entry and exit tangents."""
    # Entry normal (perpendicular to entry tangent)
    en_z, en_y = -np.sin(phi_entry),  np.cos(phi_entry)
    # Exit normal
    xn_z, xn_y = -np.sin(phi_exit),   np.cos(phi_exit)
    zs = [z0 - en_z*half_th, z0 + en_z*half_th,
          z1 + xn_z*half_th, z1 - xn_z*half_th, z0 - en_z*half_th]
    ys = [y0 - en_y*half_th, y0 + en_y*half_th,
          y1 + xn_y*half_th, y1 - xn_y*half_th, y0 - en_y*half_th]
    return zs, ys

def _build_floor_plan_yz(fig, elements, element_height, flip_bend,
                         row=2, legend_name='legend5', fp_legend_name='legend6',
                         show_fp_legend=True, beampipe_color='gray',
                         show_markers=False):
    import plotly.graph_objects as go
    import copy

    use_flr = any('flr_y0' in e for e in elements)
    # Filter out markers/monitors unless show_markers is True
    _MARKER_MONITOR_KEYS = {'marker','mark','monitor','hmon','vmon','instrument','bpm'}
    if not show_markers:
        elements = [e for e in elements
                    if e['key'].lower() not in _MARKER_MONITOR_KEYS]

    # ── Beampipe: remap flr_y → flr_x and pass to shared builder ─────────────
    remapped = []
    for elem in elements:
        e = copy.copy(elem)
        if use_flr:
            e['flr_x0'] = elem.get('flr_y0', 0.0)
            e['flr_x1'] = elem.get('flr_y1', 0.0)
            dz = elem.get('flr_z1', 0.0) - elem.get('flr_z0', 0.0)
            dy_ = elem.get('flr_y1', 0.0) - elem.get('flr_y0', 0.0)
            chord = np.arctan2(dy_, dz) if (abs(dz)>1e-12 or abs(dy_)>1e-12) else 0.0
            e['flr_theta0'] = chord
        # Zero out all bend angles so _build_floor_plan draws every element
        # as a straight rectangle — we handle vertical bend polygons below.
        e['angle'] = 0.0
        e['ref_tilt'] = 0.0
        remapped.append(e)

    # Draw beampipe + all non-vertical-bend elements via shared routine
    _build_floor_plan(fig, remapped, element_height, flip_bend, row=row,
                      legend_name=legend_name, fp_legend_name=fp_legend_name,
                      show_fp_legend=show_fp_legend, beampipe_color=beampipe_color)

    if not use_flr:
        # ── Dead-reckoning YZ path (Tao without survey coords) ───────────────
        # Track beam in Y-Z plane. Vertical bends (ref_tilt≈±π/2) curve here;
        # horizontal bends are straight in this plane.
        pz, py = [0.0], [0.0]
        dr_z, dr_y, dr_t = 0.0, 0.0, 0.0   # position and heading angle in YZ

        for e in elements:
            L_   = e['length']; ang = e.get('angle', 0.0); key = e['key']
            rt   = e.get('ref_tilt', 0.0)
            iv   = abs(abs(rt) - np.pi/2) < 0.01   # vertical bend?
            if L_ == 0:
                continue
            if 'sbend' in key.lower() and abs(ang) > 1e-6 and iv:
                ba = -ang if flip_bend else ang
                if abs(ba) > 1e-6:
                    rho = L_ / ba
                    aa  = np.linspace(0, ba, max(30, int(abs(ba) * 80)))
                    cz  = dr_z - rho * np.sin(dr_t)
                    cy  = dr_y + rho * np.cos(dr_t)
                    az  = cz + rho * np.sin(dr_t + aa)
                    ay  = cy - rho * np.cos(dr_t + aa)
                    pz.extend(az.tolist()); py.extend(ay.tolist())
                    dr_z, dr_y, dr_t = az[-1], ay[-1], dr_t + ba
                    continue
            dr_z += L_ * np.cos(dr_t); dr_y += L_ * np.sin(dr_t)
            pz.append(dr_z); py.append(dr_y)

        fig.add_trace(go.Scatter(x=pz, y=py, mode='lines',
            line=dict(color='gray', width=2), name='Beampipe',
            showlegend=False, hoverinfo='skip'), row=row, col=1)

        # Draw element polygons in YZ
        dr_z, dr_y, dr_t = 0.0, 0.0, 0.0
        la = set()
        _lt = {'quadrupole':'Quadrupole','sbend':'Dipole','sextupole':'Sextupole',
               'kicker':'Kicker','hkicker':'Kicker','vkicker':'Kicker',
               'monitor':'Monitor','rfcavity':'RF Cavity','lcavity':'RF Cavity'}
        for elem in elements:
            L_   = elem['length']; ang = elem.get('angle', 0.0); key = elem['key']
            rt   = elem.get('ref_tilt', 0.0)
            iv   = abs(abs(rt) - np.pi/2) < 0.01
            color = element_color(key)
            hover = make_hover(elem)
            th    = element_thickness(key, element_height)
            kc    = key.lower()
            ll    = _lt.get(kc)
            sl    = ll is not None and ll not in la
            if sl: la.add(ll)

            if L_ == 0 or color is None:
                continue

            x0, y0, theta = dr_z, dr_y, dr_t
            ba = 0.0
            if 'sbend' in kc and abs(ang) > 1e-6:
                ba = (-ang if flip_bend else ang) if iv else 0.0

            px_, py_ = element_polygon(x0, y0, theta, L_, ba, th)
            if abs(ba) > 1e-6:
                rho = L_ / ba
                aa  = np.linspace(0, ba, max(30, int(abs(ba) * 80)))
                cz  = x0 - rho * np.sin(theta)
                cy  = y0 + rho * np.cos(theta)
                hx  = (cz + rho * np.sin(theta + aa)).tolist()
                hy  = (cy - rho * np.cos(theta + aa)).tolist()
                dr_z, dr_y, dr_t = hx[-1], hy[-1], dr_t + ba
            else:
                hx  = [x0, x0 + L_ * np.cos(theta)]
                hy  = [y0, y0 + L_ * np.sin(theta)]
                dr_z += L_ * np.cos(dr_t); dr_y += L_ * np.sin(dr_t)

            fig.add_trace(go.Scatter(x=hx, y=hy, mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=max(8, th * 20)),
                hoverlabel=dict(bgcolor=color),
                legend=legend_name, legendgroup='h_' + (ll or 'o'),
                showlegend=False, hovertemplate=hover), row=row, col=1)
            fig.add_trace(go.Scatter(x=px_, y=py_, mode='lines', fill='toself',
                fillcolor=color, line=dict(color=color, width=0),
                name=ll, legend=legend_name, legendgroup=ll,
                showlegend=sl, hoverinfo='skip'), row=row, col=1)

        for label, col in [('Dipole','red'),('Quadrupole','blue'),('Sextupole','green'),
                           ('Kicker','orange'),('Monitor','purple'),('RF Cavity','cyan')]:
            if label not in la: continue
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=col, symbol='square'), name=label,
                legend=fp_legend_name, legendgroup='fp_yz_' + label,
                showlegend=True), row=row, col=1)
        return

    # ── Vertical bend polygons: draw fresh arc-bands from flr coordinates ─────
    la = set()
    _lt = {'sbend': 'Dipole'}
    sign_y = -1.0 if flip_bend else 1.0

    for elem in elements:
        if 'sbend' not in elem['key'].lower():
            continue
        rt = elem.get('ref_tilt', 0.0)
        iv = abs(abs(rt) - np.pi/2) < 0.01
        if not iv:
            # Also detect via flr data
            dy_ = elem.get('flr_y1', 0.0) - elem.get('flr_y0', 0.0)
            dx_ = abs(elem.get('flr_x1', 0.0) - elem.get('flr_x0', 0.0))
            if not (abs(dy_) > 1e-6 and abs(dy_) > dx_ * 2.0):
                continue  # not a vertical bend

        z0 = elem.get('flr_z0', 0.0)
        z1 = elem.get('flr_z1', 0.0)
        y0 = sign_y * elem.get('flr_y0', 0.0)
        y1 = sign_y * elem.get('flr_y1', 0.0)
        phi0 = elem.get('flr_phi0', 0.0)
        # flip_bend mirrors Y so phi also negates
        phi_entry = -phi0 if flip_bend else phi0

        # Exit tangent: for symmetric bend, exit ≈ 2*chord_angle - entry
        dz_ = z1 - z0; dy__ = y1 - y0
        chord_ang = np.arctan2(dy__, dz_) if (abs(dz_)>1e-12 or abs(dy__)>1e-12) else phi_entry
        phi_exit = 2.0*chord_ang - phi_entry
        zs, ys = _trap_band_yz(z0, y0, z1, y1, phi_entry, phi_exit, element_height/2)
        if not zs:
            continue

        color = element_color(elem['key'])
        hover = make_hover(elem)
        ll = 'Dipole'
        sl = ll not in la
        if sl: la.add(ll)
        # Invisible centerline trace carries the hover tooltip
        zm = (z0+z1)/2; ym = (y0+y1)/2
        fig.add_trace(go.Scatter(
            x=[z0, zm, z1], y=[y0, ym, y1], mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=12),
            hoverlabel=dict(bgcolor=color),
            legend=legend_name, legendgroup='h_dip', showlegend=False,
            hovertemplate=hover), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=zs, y=ys, mode='lines', fill='toself',
            fillcolor=color, line=dict(color=color, width=0),
            name=ll, legend=legend_name, legendgroup=ll,
            showlegend=sl, hoverinfo='skip'), row=row, col=1)

def _build_summary_panel(fig, all_uni_data, plot_unis, uni_labels,
                          beam_params_primary, row=1):
    """Build a per-universe lattice summary table as a Plotly Table trace."""
    import plotly.graph_objects as go

    # Row labels
    row_labels = [
        'Length (m)',
        'Dipoles', 'Quadrupoles', 'Sextupoles',
        'Kickers', 'Monitors', 'RF Cavities',
        'Qₓ', 'Qᵧ', "Qₓ'", "Qᵧ'",
    ]

    # Build one column per universe
    col_headers = ['']
    col_values  = [row_labels]

    for _uid in plot_unis:
        _ud    = all_uni_data[_uid]
        _ulbl  = uni_labels.get(_uid, f'u{_uid}')
        _elems = _ud.get('elements', [])
        _bp    = _ud.get('beam_params', {})

        # Count elements by type
        counts = {}
        total_len = 0.0
        for e in _elems:
            kl = e['key'].lower()
            total_len = max(total_len, e['s_start'] + e['length'])
            if 'sbend'      in kl: counts['dip']  = counts.get('dip',  0) + 1
            elif 'quadrupole' in kl: counts['quad'] = counts.get('quad', 0) + 1
            elif 'sextupole'  in kl: counts['sext'] = counts.get('sext', 0) + 1
            elif kl in ('kicker','hkicker','vkicker'): counts['kick'] = counts.get('kick', 0) + 1
            elif 'monitor'    in kl: counts['mon']  = counts.get('mon',  0) + 1
            elif 'rfcavity'   in kl or 'lcavity' in kl: counts['rf'] = counts.get('rf', 0) + 1

        def _fmt(v, decimals=4):
            if v is None: return '—'
            try: return f'{float(v):.{decimals}f}'
            except: return '—'

        col_headers.append(_ulbl)
        col_values.append([
            f'{total_len:.3f}',
            str(counts.get('dip',  0)),
            str(counts.get('quad', 0)),
            str(counts.get('sext', 0)),
            str(counts.get('kick', 0)),
            str(counts.get('mon',  0)),
            str(counts.get('rf',   0)),
            _fmt(_bp.get('tune_a')),
            _fmt(_bp.get('tune_b')),
            _fmt(_bp.get('chroma_a'), 2),
            _fmt(_bp.get('chroma_b'), 2),
        ])

    fig.add_trace(go.Table(
        header=dict(
            values=col_headers,
            fill_color='#2a2a3d',
            font=dict(color='#f2f2f7', size=12),
            align='center',
            line_color='#4a4a6a',
        ),
        cells=dict(
            values=col_values,
            fill_color=[['#1e1e2e' if i % 2 == 0 else '#252535'
                         for i in range(len(row_labels))]] * len(col_headers),
            font=dict(color='#f2f2f7', size=11),
            align=['left'] + ['center'] * len(plot_unis),
            line_color='#4a4a6a',
        ),
    ), row=row, col=1)

def _build_latdiff_panel(fig, elems_a, elems_b, label_a, label_b, row=1):
    """Build a lattice diff panel comparing two element lists.
    Table 1: Strengths (L, k1, k2)
    Table 2: Entry positions (X, Y, Z)
    Table 3: Exit positions (X, Y, Z)
    Only compares physics magnets: dipoles, quadrupoles, sextupoles, RF cavities.
    """
    import plotly.graph_objects as go

    _PHYSICS_KEYS = {'sbend', 'quadrupole', 'sextupole', 'rfcavity', 'lcavity'}

    def _is_physics(e):
        return e['key'].lower() in _PHYSICS_KEYS

    def _fmt(v, d=4):
        if v is None: return '—'
        try: return f'{float(v):.{d}f}'
        except: return '—'

    def _dfmt(v, d=4):
        if v is None: return '—'
        try:
            f = float(v)
            return ('+' if f > 0 else '') + f'{f:.{d}f}'
        except: return '—'

    def _diff(a, b, d=4):
        if a is None or b is None: return '—'
        try: return _dfmt(float(b) - float(a), d)
        except: return '—'

    mag_a = [e for e in elems_a if _is_physics(e)]
    mag_b = [e for e in elems_b if _is_physics(e)]

    # Count check
    if len(mag_a) != len(mag_b):
        fig.add_trace(go.Table(
            header=dict(values=['Lattice Diff — Count Mismatch'],
                       fill_color='#8b0000', font=dict(color='white', size=13),
                       align='center'),
            cells=dict(
                values=[[
                    f'Cannot compare: {label_a} has {len(mag_a)} physics magnets, '
                    f'{label_b} has {len(mag_b)}.',
                    'Please reconcile element counts before running a lattice diff.'
                ]],
                fill_color='#1e1e2e', font=dict(color='#f2f2f7', size=12),
                align='center'),
        ), row=row, col=1)
        return

    # Build rows
    nums, names, types = [], [], []
    lA, lB, dl = [], [], []
    k1A, k1B, dk1 = [], [], []
    k2A, k2B, dk2 = [], [], []

    # Entry
    xeA, xeB, dxe = [], [], []
    yeA, yeB, dye = [], [], []
    zeA, zeB, dze = [], [], []

    # Exit
    xxA, xxB, dxx = [], [], []
    yxA, yxB, dyx = [], [], []
    zxA, zxB, dzx = [], [], []

    for i, (ea, eb) in enumerate(zip(mag_a, mag_b)):
        nums.append(i + 1)
        names.append(ea['name'])
        types.append(ea['key'].capitalize())

        la, lb = ea['length'], eb['length']
        lA.append(_fmt(la)); lB.append(_fmt(lb)); dl.append(_dfmt(lb - la))

        k1a, k1b = ea.get('k1', 0.0), eb.get('k1', 0.0)
        k1A.append(_fmt(k1a)); k1B.append(_fmt(k1b)); dk1.append(_dfmt(k1b - k1a))

        k2a, k2b = ea.get('k2', 0.0), eb.get('k2', 0.0)
        k2A.append(_fmt(k2a)); k2B.append(_fmt(k2b)); dk2.append(_dfmt(k2b - k2a))

        # Entry coords
        xa0 = ea.get('flr_x0'); xb0 = eb.get('flr_x0')
        ya0 = ea.get('flr_y0'); yb0 = eb.get('flr_y0')
        za0 = ea.get('flr_z0'); zb0 = eb.get('flr_z0')
        xeA.append(_fmt(xa0)); xeB.append(_fmt(xb0)); dxe.append(_diff(xa0, xb0))
        yeA.append(_fmt(ya0)); yeB.append(_fmt(yb0)); dye.append(_diff(ya0, yb0))
        zeA.append(_fmt(za0)); zeB.append(_fmt(zb0)); dze.append(_diff(za0, zb0))

        # Exit coords
        xa1 = ea.get('flr_x1'); xb1 = eb.get('flr_x1')
        ya1 = ea.get('flr_y1'); yb1 = eb.get('flr_y1')
        za1 = ea.get('flr_z1'); zb1 = eb.get('flr_z1')
        xxA.append(_fmt(xa1)); xxB.append(_fmt(xb1)); dxx.append(_diff(xa1, xb1))
        yxA.append(_fmt(ya1)); yxB.append(_fmt(yb1)); dyx.append(_diff(ya1, yb1))
        zxA.append(_fmt(za1)); zxB.append(_fmt(zb1)); dzx.append(_diff(za1, zb1))

    HDR  = '#2a2a3d'   # dark — strengths
    HDR2 = '#1e3050'   # light blue — entry
    HDR3 = '#1a3a3a'   # teal — exit
    CELL = '#1e1e2e'
    CELL2= '#252535'
    FONT_H = dict(color='#f2f2f7', size=11)
    FONT_C = dict(color='#f2f2f7', size=10)

    def _cell_colors(n):
        return [CELL if i % 2 == 0 else CELL2 for i in range(n)]

    cc = _cell_colors(len(nums))
    al_str = ['center', 'left', 'center']
    al_r9  = al_str + ['right'] * 9
    al_r12 = al_str + ['right'] * 9  # 3 coords × 3 cols each

    # ── Table 1: Strengths ─────────────────────────────────────────────────
    fig.add_trace(go.Table(
        header=dict(
            values=['#', 'Name', 'Type',
                    f'L ({label_a})', f'L ({label_b})', 'ΔL',
                    f'k1 ({label_a})', f'k1 ({label_b})', 'Δk1',
                    f'k2 ({label_a})', f'k2 ({label_b})', 'Δk2'],
            fill_color=HDR, font=FONT_H, align='center', line_color='#4a4a6a'),
        cells=dict(
            values=[nums, names, types,
                    lA, lB, dl,
                    k1A, k1B, dk1,
                    k2A, k2B, dk2],
            fill_color=[cc] * 12,
            font=FONT_C, align=al_r9, line_color='#4a4a6a'),
    ), row=row, col=1)

    # ── Table 2: Entry Positions ───────────────────────────────────────────
    fig.add_trace(go.Table(
        header=dict(
            values=['#', 'Name', 'Type',
                    f'X_entry ({label_a})', f'X_entry ({label_b})', 'ΔX',
                    f'Y_entry ({label_a})', f'Y_entry ({label_b})', 'ΔY',
                    f'Z_entry ({label_a})', f'Z_entry ({label_b})', 'ΔZ'],
            fill_color=HDR2, font=FONT_H, align='center', line_color='#4a4a6a'),
        cells=dict(
            values=[nums, names, types,
                    xeA, xeB, dxe,
                    yeA, yeB, dye,
                    zeA, zeB, dze],
            fill_color=[cc] * 12,
            font=FONT_C, align=al_r12, line_color='#4a4a6a'),
    ), row=row + 1, col=1)

    # ── Table 3: Exit Positions ────────────────────────────────────────────
    fig.add_trace(go.Table(
        header=dict(
            values=['#', 'Name', 'Type',
                    f'X_exit ({label_a})', f'X_exit ({label_b})', 'ΔX',
                    f'Y_exit ({label_a})', f'Y_exit ({label_b})', 'ΔY',
                    f'Z_exit ({label_a})', f'Z_exit ({label_b})', 'ΔZ'],
            fill_color=HDR3, font=FONT_H, align='center', line_color='#4a4a6a'),
        cells=dict(
            values=[nums, names, types,
                    xxA, xxB, dxx,
                    yxA, yxB, dyx,
                    zxA, zxB, dzx],
            fill_color=[cc] * 12,
            font=FONT_C, align=al_r12, line_color='#4a4a6a'),
    ), row=row + 2, col=1)


def _build_layout_bar(fig, elements, show_labels, row=4, show_markers=False):
    import plotly.graph_objects as go
    _MARKER_MONITOR_KEYS = {'marker','mark','monitor','hmon','vmon','instrument','bpm'}
    if not show_markers:
        elements = [e for e in elements if e['key'].lower() not in _MARKER_MONITOR_KEYS]
    for elem in elements:
        s0=elem['s_start']; L_=elem['length']; key=elem['key']
        color=element_color(key); hover=make_hover(elem); short=elem['name'].split('\\')[-1]
        kl=key.lower()
        if L_==0:
            # Zero-length element — thin vertical line
            _zc = color or '#888888'
            fig.add_shape(type='line',x0=s0,x1=s0,y0=-0.15,y1=0.15,
                line=dict(color=_zc,width=1.5),row=row,col=1)
            fig.add_trace(go.Scatter(x=[s0],y=[0.0],mode='markers',
                marker=dict(size=8,color=_zc,opacity=0),name=short,
                showlegend=False,hovertemplate=hover),row=row,col=1)
            continue
        if color is None: continue
        if L_<THIN_ELEMENT_THRESHOLD:
            fig.add_shape(type='line',x0=s0,x1=s0,y0=-0.1,y1=0.1,
                line=dict(color=color,width=1.5),row=row,col=1)
            fig.add_trace(go.Scatter(x=[s0],y=[0.0],mode='markers',
                marker=dict(size=8,color=color,opacity=0),name=short,
                showlegend=False,hovertemplate=hover),row=row,col=1)
            continue
        if 'rfcavity' in kl or 'lcavity' in kl:
            cx=s0+L_/2; a=L_/2; b=0.1; t=np.linspace(0,2*np.pi,37)
            fig.add_trace(go.Scatter(x=(cx+a*np.cos(t)).tolist(),y=(b*np.sin(t)).tolist(),
                mode='lines',fill='toself',fillcolor=color,line=dict(color='black',width=0.5),
                opacity=0.8,name=short,showlegend=False,hoverinfo='skip'),row=row,col=1)
            fig.add_trace(go.Scatter(x=[cx],y=[0.0],mode='markers',
                marker=dict(size=8,color=color,opacity=0),name=short,
                showlegend=False,hovertemplate=hover),row=row,col=1)
            continue
        if 'quadrupole' in kl:
            pol = elem.get('profile', 0.0) or elem.get('k1', 0.0)
            y0v=0.0 if pol>0 else (-0.2 if pol<0 else -0.1)
            y1v=0.2 if pol>0 else (0.0  if pol<0 else  0.1)
        elif 'sextupole' in kl:
            pol = elem.get('k2', 0.0)
            y0v=0.0 if pol>0 else (-0.2 if pol<0 else -0.1)
            y1v=0.2 if pol>0 else (0.0  if pol<0 else  0.1)
        else: y0v,y1v=-0.1,0.1
        fig.add_shape(type='rect',x0=s0,x1=s0+L_,y0=y0v,y1=y1v,
            line=dict(color='black',width=0.5),fillcolor=color,opacity=0.8,row=row,col=1)
        fig.add_trace(go.Scatter(x=[s0+L_/2],y=[y0v+(y1v-y0v)/2],mode='markers',
            marker=dict(size=8,color=color,opacity=0),name=short,
            showlegend=False,hovertemplate=hover),row=row,col=1)
    if show_labels:
        xref=fig.get_subplot(row,1).xaxis.plotly_name.replace('axis','')
        yref=fig.get_subplot(row,1).yaxis.plotly_name.replace('axis','')
        for elem in elements:
            if elem['key'].lower() not in ('quadrupole','sbend') or elem['length']==0: continue
            fig.add_annotation(x=elem['s_start']+elem['length']/2,y=0.15,
                text=elem['name'].split('\\')[-1],showarrow=False,textangle=-90,
                font=dict(size=7),xref=xref,yref=yref)

def _build_bar_annotations(fig, elements, pattern, row, annot_font_size=8):
    """Add wildcard annotations to the beamline bar panel."""
    import fnmatch
    if not pattern or not pattern.strip(): return
    patterns = [p.strip() for p in pattern.split(',') if p.strip()]
    xref = fig.get_subplot(row, 1).xaxis.plotly_name.replace('axis', '')
    yref = fig.get_subplot(row, 1).yaxis.plotly_name.replace('axis', '')
    annotated = set()
    for elem in elements:
        name = elem['name'].split('\\')[-1]
        if not any(fnmatch.fnmatch(name.upper(), p.upper()) for p in patterns):
            continue
        s_pos = elem['s_start'] + elem['length'] / 2.0
        key = round(s_pos, 6)
        if key in annotated: continue
        annotated.add(key)
        fig.add_annotation(
            x=s_pos, y=0.15,
            xref=xref, yref=yref,
            text=name, showarrow=False, textangle=-90,
            xanchor='center', yanchor='bottom',
            font=dict(size=annot_font_size, color='#a0a0c0'),
        )

def _dot_trace(fig, x, y, name, color, legend_name, legendgroup, row, col,
               hovertemplate='', secondary_y=False):
    """Add a line trace using a colored dot as the legend symbol."""
    import plotly.graph_objects as go
    # Dummy single-point trace for the legend dot
    fig.add_trace(go.Scatter(
        x=[None], y=[None], name=name, mode='markers',
        marker=dict(symbol='circle', size=10, color=color),
        legend=legend_name, legendgroup=legendgroup, showlegend=True,
        hoverinfo='skip'),
        row=row, col=col, **({'secondary_y': secondary_y} if secondary_y is not None else {}))
    # Real line trace — no legend entry
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode='lines',
        line=dict(color=color),
        legend=legend_name, legendgroup=legendgroup, showlegend=False,
        hovertemplate=hovertemplate),
        row=row, col=col, **({'secondary_y': secondary_y} if secondary_y is not None else {}))

# Dataset registry — maps (type, axis) → (data_getter, label, color, unit, fmt)
# data_getter receives the data_dict and returns a numpy array
_DS_COLORS = {
    ('beta',  'x'): 'blue',   ('beta',  'y'): 'red',
    ('disp',  'x'): 'green',  ('disp',  'y'): 'brown',
    ('alpha', 'x'): 'blue',   ('alpha', 'y'): 'red',
    ('orbit', 'x'): 'blue',   ('orbit', 'y'): 'red',
    ('phase', 'x'): 'blue',   ('phase', 'y'): 'red',
    ('beamsize','x'):'blue',  ('beamsize','y'):'red',
}
_DS_LABELS = {
    'beta': 'β (m)', 'disp': 'η (m)', 'alpha': 'α',
    'orbit': 'x, y (m)', 'phase': 'μ', 'beamsize': 'σ (mm)',
}
# Display names with unicode subscripts
_DS_NAMES = {
    ('beta','x'): 'β\u2093',  ('beta','y'): 'β\u1d67',
    ('disp','x'): 'η\u2093',  ('disp','y'): 'η\u1d67',
    ('alpha','x'): 'α\u2093', ('alpha','y'): 'α\u1d67',
    ('orbit','x'): 'x orbit', ('orbit','y'): 'y orbit',
    ('phase','x'): 'μ\u2093', ('phase','y'): 'μ\u1d67',
    ('beamsize','x'): 'σ\u2093', ('beamsize','y'): 'σ\u1d67',
}
# Units per dataset type (empty string = unitless)
_DS_UNITS = {
    'beta': 'm', 'disp': 'm', 'alpha': '',
    'orbit': 'm', 'phase': '', 'beamsize': 'mm',
}

def _axis_label_from_datasets(datasets):
    """Build a y-axis label with subscripts and units.
    Units are appended once per unique unit group, e.g.:
      βₓ (m)          — single dataset with units
      βₓ, βᵧ (m)      — same units
      βₓ (m), αₓ      — mixed units
      βₓ (m), αₓ, ηₓ (m)  — extreme case
    """
    parts = []
    for d, a in datasets:
        name = _DS_NAMES.get((d, a), f'{d}{a}')
        unit = _DS_UNITS.get(d, '')
        parts.append((name, unit))

    # Group consecutive entries by unit to avoid repeating (m)(m)(m)
    result = []
    i = 0
    while i < len(parts):
        name, unit = parts[i]
        # Collect all consecutive names with the same unit
        group_names = [name]
        j = i + 1
        while j < len(parts) and parts[j][1] == unit:
            group_names.append(parts[j][0])
            j += 1
        chunk = ', '.join(group_names)
        if unit:
            chunk += f' ({unit})'
        result.append(chunk)
        i = j
    return ', '.join(result)

def _get_dataset(dtype, axis, s, ba, bb, ex, ey, ox, oy, pa, pb,
                 al_a, al_b, beam_params):
    """Return (y_array, display_name, color, hover_fmt) for a dataset."""
    bp = beam_params or {}
    key = (dtype, axis)
    color = _DS_COLORS.get(key, 'gray')
    name  = _DS_NAMES.get(key, f'{dtype}ₓ')
    if dtype == 'beta':
        y = ba if axis == 'x' else bb
        fmt = f'{name}=%{{y:.3f}} m'
    elif dtype == 'disp':
        y = ex if axis == 'x' else ey
        fmt = f'{name}=%{{y:.4f}} m'
    elif dtype == 'alpha':
        y = al_a if axis == 'x' else al_b
        fmt = f'{name}=%{{y:.4f}}'
    elif dtype == 'orbit':
        y = ox if axis == 'x' else oy
        fmt = f'{name}=%{{y:.6f}} m'
    elif dtype == 'phase':
        y = pa if axis == 'x' else pb
        fmt = f'{name}=%{{y:.4f}}'
    elif dtype == 'beamsize':
        ex_v = bp.get('emit_x', 0.0); ey_v = bp.get('emit_y', 0.0)
        sdp  = bp.get('sigma_dp', 0.0)
        n    = bp.get('n_sigma', 1.0)
        if axis == 'x':
            y = n * np.sqrt(np.maximum(ex_v * ba + (ex * sdp)**2, 0)) * 1e3
        else:
            y = n * np.sqrt(np.maximum(ey_v * bb + (ey * sdp)**2, 0)) * 1e3
        n_lbl = f'{n:.4g}' if n != 1.0 else '1'
        fmt = f'{n_lbl}·{name}=%{{y:.4f}} mm'
    else:
        y = np.zeros_like(s)
        fmt = name
    return y, name, color, fmt

def _build_custom_panel(fig, spec, s, ba, bb, ex, ey, ox, oy, pa, pb,
                        al_a, al_b, beam_params, row, legend_name):
    """Build a user-composed panel from a custom spec dict.

    spec = {
        'name': 'My Panel',           # display name
        'y1':   [('beta','x'), ('beta','y')],   # list of (dtype, axis)
        'y2':   [('disp','x')],                 # optional, secondary axis
    }
    Returns (y1_label, y2_label or None)
    """
    # Distinct color palette — enough for up to 8 datasets across both axes
    _PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                '#9467bd', '#8c564b', '#e377c2', '#17becf']

    y1_ds = spec.get('y1', [])
    y2_ds = spec.get('y2', [])

    # Assign colors by global position so every dataset is visually distinct
    all_ds = list(y1_ds) + list(y2_ds)
    color_map = {(d, a): _PALETTE[i % len(_PALETTE)] for i, (d, a) in enumerate(all_ds)}

    for dtype, axis in y1_ds:
        y, name, _, fmt = _get_dataset(dtype, axis, s, ba, bb, ex, ey,
                                       ox, oy, pa, pb, al_a, al_b, beam_params)
        color = color_map[(dtype, axis)]
        _dot_trace(fig, s, y, name, color, legend_name, name, row, 1,
                   hovertemplate=f's=%{{x:.3f}} m<br>{fmt}<extra></extra>',
                   secondary_y=False)

    for dtype, axis in y2_ds:
        y, name, _, fmt = _get_dataset(dtype, axis, s, ba, bb, ex, ey,
                                       ox, oy, pa, pb, al_a, al_b, beam_params)
        color = color_map[(dtype, axis)]
        _dot_trace(fig, s, y, name, color, legend_name, name, row, 1,
                   hovertemplate=f's=%{{x:.3f}} m<br>{fmt}<extra></extra>',
                   secondary_y=True)

    y1_label = _axis_label_from_datasets(y1_ds) if y1_ds else ''
    y2_label = _axis_label_from_datasets(y2_ds) if y2_ds else None
    return y1_label, y2_label

# ─── Expression panel builder ────────────────────────────────────────────────
# Uses _build_expr_namespace() + _eval_expression() to evaluate user expressions
# and plot the results as a standard optics panel.

def _build_expr_panel(fig, spec, data, code, s, row, legend_name,
                      log_fn=None, uni_idx=1, palette=None):
    """Build a panel from a user-defined expression spec.

    spec = {
        'type':        'expr',
        'name':        'My Expression',
        'extra_attrs': ['k1', 'k2'],       # extra attrs to fetch (Y1+Y2 combined)
        'y1_expr':     'k1 * beta_a',      # expression for left Y axis
        'y2_expr':     'eta_x / np.sqrt(beta_a)',  # expression for right Y axis (optional)
        'y1_label':    'k1 * beta_a',      # axis label (defaults to expression string)
        'y2_label':    'eta_x / sqrt(beta_a)',
    }

    Returns (y1_label, y2_label or None)
    """
    # Use provided palette (for multi-universe) or default
    _DEFAULT_PALETTE = ['#0a84ff','#ff453a','#30d158','#ff9f0a',
                        '#5e5ce6','#64d2ff','#ff375f','#ffd60a']
    _pal = palette if palette else _DEFAULT_PALETTE

    extra_attrs = spec.get('extra_attrs', [])
    y1_expr_raw = spec.get('y1_expr', '').strip()
    y2_expr_raw = spec.get('y2_expr', '').strip()

    if not y1_expr_raw: return '', None

    # Split comma-separated expressions for multi-curve support.
    # 'beta_a, beta_b, eta_x' -> three curves on the same axis.
    y1_exprs = [e.strip() for e in y1_expr_raw.split(',') if e.strip()]
    y2_exprs = [e.strip() for e in y2_expr_raw.split(',') if e.strip()]

    # Build evaluation namespace with all available variables
    ns, s_plot = _build_expr_namespace(data, code,
                                        extra_attrs=extra_attrs,
                                        log_fn=log_fn, uni_idx=uni_idx)

    def L(m):
        if log_fn: log_fn(m + '\n')

    def _eval(expr):
        if code == 'tao':
            return _eval_expression_tao(expr, ns, data, len(s_plot), log_fn)
        return _eval_expression(expr, ns, log_fn)

    def _to_array(y, n):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if y.ndim == 0 or len(y) == 1: return np.full(n, float(y.flat[0]))
        return y if len(y) == n else np.resize(y, n)

    # User-specified label becomes the axis label.
    # When multiple curves, individual expressions are used as legend entries.
    y1_axis_label = (spec.get('y1_label') or '').strip() or (y1_exprs[0] if len(y1_exprs) == 1 else '')
    y2_axis_label_user = (spec.get('y2_label') or '').strip()

    # ── Plot Y1 curves ────────────────────────────────────────────────
    n = len(s_plot)
    for i, expr in enumerate(y1_exprs):
        y = _eval(expr)
        if y is None:
            L(f'[expr] Y1 expression failed: {expr}')
            continue
        y = _to_array(y, n)
        color = _pal[i % len(_pal)]
        # Legend entry: always the individual expression
        legend_entry = expr
        # Axis label: user label if set, else expression (single curve only)
        trace_label = y1_axis_label if len(y1_exprs) == 1 else legend_entry
        _dot_trace(fig, s_plot, y, trace_label, color, legend_name,
                   legend_entry, row, 1,
                   hovertemplate=f's=%{{x:.3f}} m<br>{legend_entry}=%{{y:.6g}}<extra></extra>',
                   secondary_y=False)

    # ── Plot Y2 curves ────────────────────────────────────────────────
    y2_label = None
    for i, expr in enumerate(y2_exprs):
        y = _eval(expr)
        if y is None:
            L(f'[expr] Y2 expression failed: {expr}')
            continue
        y = _to_array(y, n)
        # Y2 colors offset from Y1 to stay visually distinct
        color = _pal[(len(y1_exprs) + i) % len(_pal)]
        legend_entry = expr
        trace_label = (y2_axis_label_user if y2_axis_label_user and len(y2_exprs) == 1
                       else legend_entry)
        _dot_trace(fig, s_plot, y, trace_label, color, legend_name,
                   legend_entry, row, 1,
                   hovertemplate=f's=%{{x:.3f}} m<br>{legend_entry}=%{{y:.6g}}<extra></extra>',
                   secondary_y=True)
        if y2_label is None:
            y2_label = y2_axis_label_user or legend_entry

    y1_label = y1_axis_label or (y1_exprs[0] if y1_exprs else '')
    return y1_label, y2_label

def _build_twiss_panel(fig, s, ba, bb, ex, ey, row=2, legend_name='legend1'):
    import plotly.graph_objects as go
    for y,name,col,sec,fmt in [
        (ba,'βₓ','blue',False,'βₓ=%{y:.3f} m'),
        (bb,'βᵧ','red', False,'βᵧ=%{y:.3f} m'),
        (ex,'ηₓ','green',True,'ηₓ=%{y:.4f} m'),
        (ey,'ηᵧ','brown',True,'ηᵧ=%{y:.4f} m'),
    ]:
        _dot_trace(fig, s, y, name, col, legend_name, name, row, 1,
            hovertemplate=f's=%{{x:.3f}} m<br>{fmt}<extra></extra>', secondary_y=sec)

def _build_beta_panel(fig, s, ba, bb, row=2, legend_name='legend1'):
    import plotly.graph_objects as go
    for y,name,col in [(ba,'βₓ','blue'),(bb,'βᵧ','red')]:
        _dot_trace(fig, s, y, name, col, legend_name, name, row, 1,
            hovertemplate=f's=%{{x:.3f}} m<br>{name}=%{{y:.3f}} m<extra></extra>')

def _build_dispersion_panel(fig, s, ex, ey, row=2, legend_name='legend1'):
    import plotly.graph_objects as go
    for y,name,col in [(ex,'ηₓ','green'),(ey,'ηᵧ','brown')]:
        _dot_trace(fig, s, y, name, col, legend_name, name, row, 1,
            hovertemplate=f's=%{{x:.3f}} m<br>{name}=%{{y:.4f}} m<extra></extra>')

def _build_alpha_panel(fig, s, al_a, al_b, row=2, legend_name='legend1'):
    import plotly.graph_objects as go
    for y,name,col in [(al_a,'αₓ','blue'),(al_b,'αᵧ','red')]:
        _dot_trace(fig, s, y, name, col, legend_name, name, row, 1,
            hovertemplate=f's=%{{x:.3f}} m<br>{name}=%{{y:.4f}}<extra></extra>')

def _build_panel3(fig, panel3, s, ba, bb, ex, ey, ox, oy, pa, pb,
                  row=3, legend_name='legend2', row3_secondary=False, beam_params=None):
    import plotly.graph_objects as go
    bp=beam_params or {}
    # Expression panel spec
    if isinstance(panel3, dict) and panel3.get('type') == 'expr':
        # Expression panels need live Tao — handled in plot_optics directly
        return '', None
    # Custom panel spec dict
    if isinstance(panel3, dict):
        al_a = bp.get('alpha_a', np.zeros_like(s))
        al_b = bp.get('alpha_b', np.zeros_like(s))
        return _build_custom_panel(fig, panel3, s, ba, bb, ex, ey, ox, oy,
                                   pa, pb, al_a, al_b, bp, row, legend_name)
    if panel3=='phase':
        for y,n,c in [(pa,'μₓ','blue'),(pb,'μᵧ','red')]:
            _dot_trace(fig, s, y, n, c, legend_name, n, row, 1,
                hovertemplate=f's=%{{x:.3f}} m<br>{n}=%{{y:.4f}}<extra></extra>')
    elif panel3=='orbit':
        for y,n,c in [(ox,'x orbit','blue'),(oy,'y orbit','red')]:
            _dot_trace(fig, s, y, n, c, legend_name, n, row, 1,
                hovertemplate=f's=%{{x:.3f}} m<br>{n}=%{{y:.6f}} m<extra></extra>')
    elif panel3=='beamsize':
        ex_v=bp.get('emit_x',0.0); ey_v=bp.get('emit_y',0.0); sdp=bp.get('sigma_dp',0.0)
        n_sig=bp.get('n_sigma',1.0)
        sx=n_sig*np.sqrt(np.maximum(ex_v*ba+(ex*sdp)**2,0))*1e3
        sy=n_sig*np.sqrt(np.maximum(ey_v*bb+(ey*sdp)**2,0))*1e3
        n_lbl = f'{n_sig:.4g}·' if n_sig != 1.0 else ''
        for y,n,c in [(sx,f'{n_lbl}σₓ','blue'),(sy,f'{n_lbl}σᵧ','red')]:
            _dot_trace(fig, s, y, n, c, legend_name, n, row, 1,
                hovertemplate=f's=%{{x:.3f}} m<br>{n}=%{{y:.4f}} mm<extra></extra>')
    elif panel3=='twiss':
        _build_twiss_panel(fig,s,ba,bb,ex,ey,row=row,legend_name=legend_name)
    elif panel3=='beta':
        _build_beta_panel(fig,s,ba,bb,row=row,legend_name=legend_name)
    elif panel3=='dispersion':
        _build_dispersion_panel(fig,s,ex,ey,row=row,legend_name=legend_name)
    elif panel3=='alpha':
        al_a=beam_params.get('alpha_a',np.zeros_like(s)) if bp else np.zeros_like(s)
        al_b=beam_params.get('alpha_b',np.zeros_like(s)) if bp else np.zeros_like(s)
        _build_alpha_panel(fig,s,al_a,al_b,row=row,legend_name=legend_name)

def _build_panel3_uni(fig, panel3, s, ba, bb, ex, ey, ox, oy, pa, pb,
                      al_a, al_b, beam_params, row, legend_name,
                      uni_label='u1', palette=None, uni_idx=0):
    """Like _build_panel3 but tags every trace with the universe label
    and uses palette colors for visual distinction."""
    import plotly.graph_objects as go
    if palette is None:
        palette = ['#1f77b4','#d62728','#2ca02c','#ff7f0e']
    bp = beam_params or {}

    def _tagged(name): return f'{name} ({uni_label})'

    def _utrace(fig, s, y, name, color, sec=False):
        tagged = _tagged(name)
        _dot_trace(fig, s, y, tagged, color, legend_name, tagged, row, 1,
                   hovertemplate=f's=%{{x:.3f}} m<br>{tagged}=%{{y:.6g}}<extra></extra>',
                   secondary_y=sec)

    if isinstance(panel3, dict):
        # Custom panel — assign palette colors by position
        y1_ds = panel3.get('y1', [])
        y2_ds = panel3.get('y2', [])
        all_ds = list(y1_ds) + list(y2_ds)
        for ci, (dtype, axis) in enumerate(y1_ds):
            y, name, _, fmt = _get_dataset(dtype, axis, s, ba, bb, ex, ey,
                                           ox, oy, pa, pb, al_a, al_b, bp)
            color = palette[ci % len(palette)]
            _utrace(fig, s, y, name, color, sec=False)
        for ci, (dtype, axis) in enumerate(y2_ds):
            y, name, _, fmt = _get_dataset(dtype, axis, s, ba, bb, ex, ey,
                                           ox, oy, pa, pb, al_a, al_b, bp)
            color = palette[(len(y1_ds)+ci) % len(palette)]
            _utrace(fig, s, y, name, color, sec=True)
        return

    c0, c1, c2, c3 = palette[0], palette[1], palette[2], palette[3]

    if panel3 == 'twiss':
        for y, name, color, sec in [
            (ba, 'βₓ', c0, False), (bb, 'βᵧ', c1, False),
            (ex, 'ηₓ', c2, True),  (ey, 'ηᵧ', c3, True),
        ]:
            _utrace(fig, s, y, name, color, sec=sec)
    elif panel3 == 'beta':
        for y, name, color in [(ba,'βₓ',c0),(bb,'βᵧ',c1)]:
            _utrace(fig, s, y, name, color)
    elif panel3 == 'dispersion':
        for y, name, color in [(ex,'ηₓ',c0),(ey,'ηᵧ',c1)]:
            _utrace(fig, s, y, name, color)
    elif panel3 == 'alpha':
        for y, name, color in [(al_a,'αₓ',c0),(al_b,'αᵧ',c1)]:
            _utrace(fig, s, y, name, color)
    elif panel3 == 'phase':
        for y, name, color in [(pa,'μₓ',c0),(pb,'μᵧ',c1)]:
            _utrace(fig, s, y, name, color)
    elif panel3 == 'orbit':
        for y, name, color in [(ox,'x orbit',c0),(oy,'y orbit',c1)]:
            _utrace(fig, s, y, name, color)
    elif panel3 == 'beamsize':
        ex_v=bp.get('emit_x',0.0); ey_v=bp.get('emit_y',0.0)
        sdp=bp.get('sigma_dp',0.0); n_sig=bp.get('n_sigma',1.0)
        sx=n_sig*np.sqrt(np.maximum(ex_v*ba+(ex*sdp)**2,0))*1e3
        sy=n_sig*np.sqrt(np.maximum(ey_v*bb+(ey*sdp)**2,0))*1e3
        n_lbl = f'{n_sig:.4g}·' if n_sig != 1.0 else ''
        for y, name, color in [(sx,f'{n_lbl}σₓ',c0),(sy,f'{n_lbl}σᵧ',c1)]:
            _utrace(fig, s, y, name, color)

# ─── Single-file loader helper ───────────────────────────────────────────────

def _load_one(input_file, code, log_fn=None, progress_fn=None,
              xsuite_twiss='4d', xsuite_line=None, universes=None,
              madx_survey=None):
    """Load one lattice file and return (data, tao_instance).

    data has the standard keys: s, beta_a, beta_b, eta_x, eta_y,
    alpha_a, alpha_b, orbit_x, orbit_y, phi_a, phi_b, elements,
    beam_params, and optionally 'universes' / 'universe_labels'.

    tao_instance is the live Tao object (for expr panels) or None.
    """
    code = code.lower()
    tao_instance = None
    if code == 'tao':
        from pytao import Tao
        tao_instance = Tao(f'-init {input_file} -noplot')
        data = load_tao(input_file, log_fn, progress_fn=progress_fn,
                        tao=tao_instance)
        data['_tao'] = tao_instance
    elif code == 'elegant':
        data = load_elegant(input_file, log_fn, progress_fn=progress_fn)
    elif code == 'xsuite':
        data = load_xsuite(input_file, log_fn, twiss_method=xsuite_twiss,
                           line_name=xsuite_line, progress_fn=progress_fn)
    elif code == 'madx':
        data = load_madx(input_file, survey_file=madx_survey,
                         log_fn=log_fn, progress_fn=progress_fn)
    else:
        raise ValueError(f"Unknown code '{code}'.")

    # Resolve universe structure
    all_uni = data.get('universes', {1: data})
    uni_labels = data.get('universe_labels', {1: 'u1'})
    if universes:
        plot_unis = [u for u in universes if u in all_uni]
    else:
        plot_unis = list(all_uni.keys())

    return data, tao_instance, all_uni, uni_labels, plot_unis

# ─── Main plot_optics ─────────────────────────────────────────────────────────

# ─── Available data inspector ─────────────────────────────────────────────────
# Queries the loaded lattice and returns a categorized dict of available
# attributes and scalars. Used by the "Show available data" button in the GUI.

def _inspect_available_data(input_file, code, log_fn=None,
                             xsuite_twiss='4d', xsuite_line=None,
                             madx_survey=None):
    """Load the lattice and return categorized available attributes.

    Returns
    -------
    dict with keys:
        'standard'  : list of (name, description) always available
        'extra'     : list of (name, description) extra fetchable attrs
        'scalars'   : list of (name, value, description) global params
        'error'     : str or None
    """
    def L(m):
        if log_fn: log_fn(m + '\n')

    result = {'standard': [], 'extra': [], 'scalars': [], 'error': None}

    # Standard arrays — shown with native names for each backend
    # All names work in expressions (aliases are set up in _build_expr_namespace)
    _STANDARD = {
        'tao': [
            ('s',        'Longitudinal position (m)'),
            ('beta.a',   'Normal-mode beta function a (m)'),
            ('beta.b',   'Normal-mode beta function b (m)'),
            ('alpha.a',  'Normal-mode alpha function a'),
            ('alpha.b',  'Normal-mode alpha function b'),
            ('eta.x',    'Horizontal dispersion (m)'),
            ('eta.y',    'Vertical dispersion (m)'),
            ('orbit.x',  'Horizontal closed orbit (m)'),
            ('orbit.y',  'Vertical closed orbit (m)'),
            ('phase.a',  'Horizontal phase advance (rad)'),
            ('phase.b',  'Vertical phase advance (rad)'),
            ('k1',       'Quadrupole strength (m⁻²)'),
            ('k2',       'Sextupole strength (m⁻³)'),
            ('angle',    'Bend angle (rad)'),
            ('length',   'Element length (m)'),
            ('rho',      'Bend radius (m)'),
        ],
        'elegant': [
            ('s',        'Longitudinal position (m)'),
            ('betax',    'Horizontal beta function (m)'),
            ('betay',    'Vertical beta function (m)'),
            ('alphax',   'Horizontal alpha function'),
            ('alphay',   'Vertical alpha function'),
            ('etax',     'Horizontal dispersion (m)'),
            ('etay',     'Vertical dispersion (m)'),
            ('etaxp',    'Slope of horizontal dispersion'),
            ('etayp',    'Slope of vertical dispersion'),
            ('psix',     'Horizontal phase advance (rad)'),
            ('psiy',     'Vertical phase advance (rad)'),
            ('k1',       'Quadrupole strength (m⁻²)'),
            ('k2',       'Sextupole strength (m⁻³)'),
            ('angle',    'Bend angle (rad)'),
            ('length',   'Element length (m)'),
            ('rho',      'Bend radius (m)'),
        ],
        'xsuite': [
            ('s',        'Longitudinal position (m)'),
            ('betx',     'Horizontal beta function (m)'),
            ('bety',     'Vertical beta function (m)'),
            ('alfx',     'Horizontal alpha function'),
            ('alfy',     'Vertical alpha function'),
            ('dx',       'Horizontal dispersion (m)'),
            ('dy',       'Vertical dispersion (m)'),
            ('mux',      'Horizontal phase advance (tune units)'),
            ('muy',      'Vertical phase advance (tune units)'),
            ('x',        'Horizontal closed orbit (m)'),
            ('y',        'Vertical closed orbit (m)'),
            ('k1',       'Quadrupole strength (m⁻²)'),
            ('k2',       'Sextupole strength (m⁻³)'),
            ('angle',    'Bend angle (rad)'),
            ('length',   'Element length (m)'),
            ('rho',      'Bend radius (m)'),
        ],
    }
    result['standard'] = _STANDARD.get(code, _STANDARD['tao'])

    try:
        if code == 'tao':
            # ── Tao: do not spin up a Tao instance here — the TaoDataBrowser
            # already handles all attribute browsing. Just return the standard
            # list with common extra attributes; no file loading needed.
            result['extra'] = [
                ('k1',           'Quadrupole strength (m⁻²)'),
                ('k2',           'Sextupole strength (m⁻³)'),
                ('k3',           'Octupole strength (m⁻⁴)'),
                ('angle',        'Bend angle (rad)'),
                ('rho',          'Bend radius (m)'),
                ('e_tot',        'Total energy (eV)'),
                ('p0c',          'Reference momentum (eV/c)'),
                ('ref_tilt',     'Element tilt (rad)'),
                ('x_offset',     'Horizontal misalignment (m)'),
                ('y_offset',     'Vertical misalignment (m)'),
                ('voltage',      'RF voltage (V)'),
                ('rf_frequency', 'RF frequency (Hz)'),
                ('emit_a',       'Horizontal emittance (m·rad)'),
                ('emit_b',       'Vertical emittance (m·rad)'),
                ('sig_E',        'Energy spread'),
                ('sigma_x',      'Horizontal beam size (m)'),
                ('sigma_y',      'Vertical beam size (m)'),
            ]

        elif code == 'elegant':
            # ── ELEGANT: static extra attributes (no re-run needed) ────────
            result['extra'] = [
                ('etaxp',      "Slope of horizontal dispersion"),
                ('etayp',      "Slope of vertical dispersion"),
                ('xAperture',  "Effective horizontal aperture (m)"),
                ('yAperture',  "Effective vertical aperture (m)"),
                ('dI1',        "Per-element radiation integral I1 (m)"),
                ('dI2',        "Per-element radiation integral I2 (1/m)"),
                ('dI3',        "Per-element radiation integral I3 (1/m²)"),
                ('dI4',        "Per-element radiation integral I4 (1/m)"),
                ('dI5',        "Per-element radiation integral I5 (1/m²)"),
            ]

        elif code == 'xsuite':
            # ── xsuite: load and inspect twiss table columns ──────────────
            data = load_xsuite(input_file, log_fn=log_fn,
                               twiss_method=xsuite_twiss,
                               line_name=xsuite_line)
            tw_cols = [k for k in data.keys()
                       if k not in ('s', 'elements', 'beam_params', '_tao', '_tw')
                       and isinstance(data.get(k), np.ndarray)]
            result['extra'] = [(c, "xsuite twiss column") for c in sorted(tw_cols)]

        elif code == 'madx':
            # ── MAD-X: read actual columns from the TFS files ─────────────
            if not input_file or not Path(input_file).exists():
                result['error'] = f"Twiss file not found: {input_file}"
            else:
                try:
                    scalars, twi_cols, _ = _read_tfs(input_file)
                    # Exclude bookkeeping columns that aren't plottable quantities
                    _skip = {'NAME', 'KEYWORD', 'PARENT', 'TYPE', 'ORIGIN', 'COMMENTS'}
                    result['extra'] = [
                        (c.lower(), f"twiss column  ({c})")
                        for c in twi_cols if c.upper() not in _skip
                    ]
                    # Scalars from twiss header
                    result['scalars'] = [
                        (k.lower(), v, "twiss header scalar")
                        for k, v in scalars.items()
                    ]
                except Exception as _te:
                    result['error'] = f"Could not read twiss TFS: {_te}"

            # Survey file columns (separate section shown in browser)
            if madx_survey and Path(madx_survey).exists():
                try:
                    _, sv_cols, _ = _read_tfs(madx_survey)
                    _skip_sv = {'NAME', 'KEYWORD', 'PARENT', 'TYPE'}
                    result['survey_cols'] = [
                        (c.lower(), f"survey column  ({c})")
                        for c in sv_cols if c.upper() not in _skip_sv
                    ]
                except Exception as _se:
                    result['survey_cols'] = []
                    L(f"[inspector] Survey read failed: {_se}")
            else:
                result['survey_cols'] = []

    except Exception as e:
        import traceback
        result['error'] = traceback.format_exc()
        L(f"[inspector] Error: {e}")

    return result


def _build_panel_annotations(fig, elements, pattern, row,
                              annot_font_size=8):
    """Add rotated element-name annotations to a data panel.

    Places each label at the maximum y value of all plotted traces at
    that s position, so annotations follow the data rather than sitting
    at a fixed position.

    Parameters
    ----------
    fig             : plotly Figure
    elements        : list of element dicts
    pattern         : fnmatch wildcard string, comma-separated
    row             : subplot row number
    annot_font_size : font size for annotation labels
    """
    import fnmatch
    import numpy as np
    if not pattern or not pattern.strip():
        return

    patterns = [p.strip() for p in pattern.split(',') if p.strip()]
    if not patterns:
        return

    # Collect all x/y data from traces on this row
    # Build a combined array: for each s_pos, find max y across all traces
    row_traces = [t for t in fig.data
                  if hasattr(t, 'xaxis') and
                  t.xaxis == fig.get_subplot(row, 1).xaxis.plotly_name.replace('xaxis', 'x').replace('x', 'xaxis').replace('xaxisaxis','xaxis')]

    # Simpler: collect all (x_arr, y_arr) pairs from traces in this subplot
    # Match by checking xaxis attribute against expected axis for this row
    expected_xaxis = fig.get_subplot(row, 1).xaxis.plotly_name  # e.g. 'xaxis3'
    expected_x_ref = expected_xaxis.replace('xaxis', 'x')       # e.g. 'x3'
    # xaxis attr on traces is stored as 'x', 'x2', 'x3' etc.
    trace_pairs = []
    for t in fig.data:
        t_xaxis = getattr(t, 'xaxis', 'x') or 'x'
        if t_xaxis == expected_x_ref:
            x_data = getattr(t, 'x', None)
            y_data = getattr(t, 'y', None)
            if x_data is not None and y_data is not None and len(x_data) == len(y_data):
                try:
                    trace_pairs.append((np.array(x_data, dtype=float),
                                        np.array(y_data, dtype=float)))
                except (TypeError, ValueError):
                    pass

    def _max_y_at_s(s_pos):
        """Find max y value across all traces at the nearest s position."""
        if not trace_pairs:
            return None
        best = None
        for xs, ys in trace_pairs:
            if len(xs) == 0: continue
            idx = int(np.argmin(np.abs(xs - s_pos)))
            val = float(ys[idx])
            if np.isfinite(val):
                best = val if best is None else max(best, val)
        return best

    xax = fig.get_subplot(row, 1).xaxis.plotly_name.replace('xaxis', 'x')
    yax = fig.get_subplot(row, 1).yaxis.plotly_name.replace('yaxis', 'y')

    annotated = set()
    for elem in elements:
        name = elem['name'].split('\\')[-1]
        matched = any(fnmatch.fnmatch(name.upper(), p.upper()) for p in patterns)
        if not matched:
            continue
        s_pos = elem['s_start'] + elem['length'] / 2.0
        key = round(s_pos, 6)
        if key in annotated:
            continue
        annotated.add(key)

        y_val = _max_y_at_s(s_pos)
        if y_val is not None:
            # Place label at the data value, in data coordinates
            fig.add_annotation(
                x=s_pos, y=y_val,
                xref=xax, yref=yax,
                text=name,
                showarrow=False,
                textangle=-90,
                xanchor='center', yanchor='bottom',
                font=dict(size=annot_font_size, color='#a0a0c0'),
            )
        else:
            # Fallback to top of panel if no trace data found
            fig.add_annotation(
                x=s_pos, y=1.0,
                xref=xax, yref=f'{yax} domain',
                text=name,
                showarrow=False,
                textangle=-90,
                xanchor='center', yanchor='top',
                font=dict(size=annot_font_size, color='#a0a0c0'),
            )


def _build_tune_annotation(fig, beam_params, row=1):
    """Add a tune/chromaticity info box as an annotation on the given row."""
    import plotly.graph_objects as go
    bp = beam_params or {}
    qa = bp.get('tune_a'); qb = bp.get('tune_b')
    ca = bp.get('chroma_a'); cb = bp.get('chroma_b')
    if qa is None and qb is None: return
    lines = []
    if qa is not None: lines.append(f"Qₓ = {qa:.4f}")
    if qb is not None: lines.append(f"Qᵧ = {qb:.4f}")
    if ca is not None: lines.append(f"Qₓ’ = {ca:.2f}")
    if cb is not None: lines.append(f"Qᵧ’ = {cb:.2f}")
    if not lines: return
    text = "<br>".join(lines)
    # plotly_name is e.g. 'xaxis', 'xaxis2' — strip 'axis' to get 'x', 'x2'
    xref = fig.get_subplot(row, 1).xaxis.plotly_name.replace('xaxis', 'x')
    yref = fig.get_subplot(row, 1).yaxis.plotly_name.replace('yaxis', 'y')
    fig.add_annotation(
        text=text, xref=f"{xref} domain", yref=f"{yref} domain",
        x=0.01, y=0.98, xanchor="left", yanchor="top",
        showarrow=False, align="left",
        bgcolor="rgba(30,30,46,0.8)", bordercolor="#4a4a6a",
        borderwidth=1, borderpad=6,
        font=dict(size=12, color="#f2f2f7", family="monospace"),
    )

def _parse_fp_range(rng_str):
    """Parse a range string like '-0.5:0.5' into [float, float] or None."""
    if not rng_str: return None
    try:
        parts = str(rng_str).split(':')
        if len(parts) != 2: return None
        return [float(parts[0].strip()), float(parts[1].strip())]
    except (ValueError, AttributeError):
        return None

def plot_optics(
    input_file, code='tao', output_file='optics.html',
    show_element_labels=True, show=False,
    save_png=False, save_pdf=False, dpi=300,
    save_csv=False, csv_base='lattice',
    flip_bend=False, element_height_xz=None, element_height_yz=None,
    fp_xz_range=None, fp_yz_range=None,
    panels=None, layout='panels', srange=None,
    emit_x=None, emit_y=None, sigma_dp=None, n_sigma=1.0,
    title=None, dark_mode=False, log_fn=None,
    aspect_ratio=None, legend_inside=True,
    xsuite_twiss='4d', xsuite_line=None,
    universes=None,
    madx_survey=None,
    uni_label_overrides=None,  # {universe_index: 'custom label'} to override auto-detected labels
    show_tune=False,
    tunnel_wall_file=None,
    show_tunnel=False,
    show_floor=True,   # show floor plan row in panels layout
    color_beampipes=False,  # if True, each universe beampipe gets a distinct calm color
    show_markers=False,      # if True, show markers/monitors in floor plan
    show_markers_bar=False,  # if True, show markers/monitors in beamline bar
    show_xz=True,     # show X vs Z floor plan (floor layout)
    show_yz=True,     # show Y vs Z floor plan (floor layout)
    show_titles=True, # show subplot titles
    panel_spacing=80,  # vertical spacing between panels in pixels
    panel_heights=None,  # dict: panel_index -> height_px, overrides default per-panel height
    panel_annotations=None,  # dict: panel_index -> wildcard pattern, e.g. {0: 'IPM*', 2: 'QF*,QD*'}
    font_sizes=None,  # dict with keys: 'axis_label', 'tick', 'title', 'annot', 'legend'
                      # e.g. {'axis_label': 12, 'tick': 10, 'title': 13, 'annot': 8, 'legend': 10}
    progress_fn=None,
    # ── Multi-file comparison ─────────────────────────────────────────────────
    compare=None,        # list of {'file':..., 'code':..., 'label':...}
    compare_mode='overlay',  # 'overlay', 'separate', 'difference', 'difference%'
    normalize_s=False,   # if True, plot s/s_max so all files share [0,1]
):
    """
    panels : list of panel types to include, in display order.
             Choices: 'twiss', 'orbit', 'phase', 'beamsize'
             Default: ['twiss']
    element_height_yz : fraction of Y span used for element height in Y-Z plane.
                        Default 0.05 (5%). Ignored for X-Z plane.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def _log(m): (log_fn(m+'\n') if log_fn else print(m))

    def _prog(pct, label):
        if progress_fn:
            progress_fn(pct, label)
        _log(f"[{int(pct):3d}%] {label}")

    # Per-universe color palettes — each universe gets its own distinct set
    _UNI_PALETTES = [
        ['#1f77b4','#d62728','#2ca02c','#ff7f0e'],  # u1: blue/red/green/orange
        ['#9467bd','#e377c2','#17becf','#bcbd22'],  # u2: purple/pink/cyan/yellow
        ['#8c564b','#7f7f7f','#aec7e8','#ffbb78'],  # u3: brown/grey/lt-blue/lt-orange
        ['#98df8a','#ff9896','#c5b0d5','#c49c94'],  # u4: pastel set
    ]

    # Calm beampipe colors — distinct from element colors (red/blue/green/orange/purple/cyan)
    _BEAMPIPE_COLORS = [
        '#5d8aa8',  # steel blue
        '#c19a6b',  # camel/tan
        '#7b9e87',  # sage green
        '#9e7bb5',  # lavender
        '#c47c5a',  # muted terracotta
        '#4e8098',  # teal blue
        '#b5835a',  # warm brown
        '#7a9e9f',  # muted cyan-teal
    ]

    code = code.lower()
    # Normalise layout aliases: 'all' and 'optics' → 'panels'
    layout = layout.lower()
    if layout in ('all', 'optics'): layout = 'panels'
    # Load tunnel wall coordinates if provided
    _tunnel = None
    if show_tunnel and tunnel_wall_file:
        _tunnel = _read_tunnel_wall(tunnel_wall_file, log_fn=_log)
        if _tunnel is None:
            _log(f"[tunnel] Could not load tunnel wall file: {tunnel_wall_file}")
        else:
            _log(f"[tunnel] Loaded {len(_tunnel['zi'])} wall points")

    _prog(2, 'Loading lattice...')
    data, _tao_instance, _all_uni_data, _uni_labels, _plot_unis = _load_one(
        input_file, code, log_fn=log_fn, progress_fn=progress_fn,
        xsuite_twiss=xsuite_twiss, xsuite_line=xsuite_line,
        universes=universes, madx_survey=madx_survey)
    # Apply user-defined label overrides
    if uni_label_overrides:
        for uid, lbl in uni_label_overrides.items():
            if lbl and lbl.strip():
                _uni_labels[uid] = lbl.strip()
    _multi = len(_plot_unis) > 1

    # ── Load compare files ────────────────────────────────────────────────────
    # Each compare entry becomes a dataset alongside the primary.
    # In overlay mode they are merged into _all_uni_data (reusing the
    # multi-universe machinery). In separate mode they are kept in _cmp_datasets.
    _cmp_datasets = []   # list of dicts: {label, data, all_uni, plot_unis}
    if compare:
        for ci, centry in enumerate(compare):
            cfile  = centry.get('file', '')
            ccode  = centry.get('code', 'tao')
            clabel = centry.get('label') or Path(cfile).stem
            _log(f"[compare] Loading {clabel} ({ccode}) ← {cfile}")
            try:
                cdata, _, call_uni, clabels, cplot_unis = _load_one(
                    cfile, ccode, log_fn=log_fn,
                    xsuite_twiss=centry.get('xsuite_twiss', '4d'),
                    xsuite_line=centry.get('xsuite_line'),
                    universes=centry.get('universes'))
                _cmp_datasets.append({
                    'label':     clabel,
                    'data':      cdata,
                    'all_uni':   call_uni,
                    'plot_unis': cplot_unis,
                    'code':      ccode,
                })
            except Exception as e:
                _log(f"[compare] ERROR loading {clabel}: {e}")

        # In overlay mode: merge compare datasets into _all_uni_data using
        # synthetic integer keys beyond the primary universe range.
        if compare_mode == 'overlay' and _cmp_datasets:
            next_key = max(_plot_unis) + 1
            for cd in _cmp_datasets:
                # Use only the first (primary) universe of each compare file
                cprimary = cd['plot_unis'][0]
                cpdata   = cd['all_uni'][cprimary]
                _all_uni_data[next_key] = cpdata
                _uni_labels[next_key]   = cd['label']
                _plot_unis.append(next_key)
                next_key += 1
            _multi = True

    _prog(40, 'Lattice loaded — building plot...')

    # ── normalize_s helper ────────────────────────────────────────────────────
    def _norm_s(s_arr):
        """Scale s to [0, 1] relative to its own max, for cross-file comparison."""
        smax = float(s_arr[-1]) if len(s_arr) else 1.0
        return s_arr / smax if smax > 0 else s_arr

    if normalize_s:
        for _uid, _ud in _all_uni_data.items():
            _ud['s'] = _norm_s(_ud['s'])
        for cd in _cmp_datasets:
            for _uid, _ud in cd['all_uni'].items():
                _ud['s'] = _norm_s(_ud['s'])

    # Use universe 1 (or first selected) as primary for elements/floor plan
    _primary = _plot_unis[0]
    _pdata   = _all_uni_data[_primary]

    s  = _pdata['s'];      ba = _pdata['beta_a']; bb = _pdata['beta_b']
    ex = _pdata['eta_x'];  ey = _pdata['eta_y']
    al_a = _pdata.get('alpha_a', np.zeros_like(s))
    al_b = _pdata.get('alpha_b', np.zeros_like(s))
    ox = _pdata['orbit_x']; oy = _pdata['orbit_y']
    pa = _pdata['phi_a'];  pb = _pdata['phi_b']
    elements = _pdata['elements']
    bp_raw   = _pdata.get('beam_params', {})
    beam_params = {
        'emit_x':   emit_x   if emit_x   is not None else bp_raw.get('emit_x',   0.0),
        'emit_y':   emit_y   if emit_y   is not None else bp_raw.get('emit_y',   0.0),
        'sigma_dp': sigma_dp if sigma_dp is not None else bp_raw.get('sigma_dp', 0.0),
        'n_sigma':  float(n_sigma) if n_sigma is not None else 1.0,
        'tune_a':   bp_raw.get('tune_a'),
        'tune_b':   bp_raw.get('tune_b'),
        'chroma_a': bp_raw.get('chroma_a'),
        'chroma_b': bp_raw.get('chroma_b'),
    }

    # ── Default panels ────────────────────────────────────────────────────────
    if not panels:
        panels = ['twiss']

    # ── Range filter ─────────────────────────────────────────────────────────
    if srange:
        pts = srange.split(':')
        if len(pts) != 2:
            raise ValueError(f"Invalid range '{srange}'. Use START:END.")
        def _res(tok):
            try: return float(tok)
            except ValueError: pass
            tu = tok.upper()
            for e in elements:
                if e['name'].upper() == tu: return e['s_start']
            raise ValueError(f"Element '{tok}' not found.")
        s_lo = _res(pts[0].strip()); s_hi = _res(pts[1].strip())
        if s_lo > s_hi: s_lo, s_hi = s_hi, s_lo
        _log(f"[range] {s_lo:.4f} → {s_hi:.4f} m")
        mask = (s >= s_lo) & (s <= s_hi)
        s  = s[mask];  ba = ba[mask]; bb = bb[mask]
        ex = ex[mask]; ey = ey[mask]
        ox = ox[mask]; oy = oy[mask]; pa = pa[mask]; pb = pb[mask]
        elements = [e for e in elements
                    if (e['s_start'] + e['length']) >= s_lo and e['s_start'] <= s_hi]

    layout = layout.lower()

    # ── Theme ─────────────────────────────────────────────────────────────────
    if dark_mode:
        _th = dict(paper_bgcolor='#1e1e1e', plot_bgcolor='#2d2d2d',
                   font_color='#e0e0e0',    gridcolor='#444444',
                   zerolinecolor='#555555')
    else:
        _th = dict(paper_bgcolor='white',   plot_bgcolor='white',
                   font_color='#333333',    gridcolor='#e5e5e5',
                   zerolinecolor='#aaaaaa')

    def _apply(fig):
        fig.update_layout(paper_bgcolor=_th['paper_bgcolor'],
                          plot_bgcolor =_th['plot_bgcolor'],
                          font_color   =_th['font_color'])
        fig.update_xaxes(showgrid=True, gridcolor=_th['gridcolor'],
                         gridwidth=1,   zerolinecolor=_th['zerolinecolor'],
                         zerolinewidth=1)
        fig.update_yaxes(showgrid=True, gridcolor=_th['gridcolor'],
                         gridwidth=1,   zerolinecolor=_th['zerolinecolor'],
                         zerolinewidth=1)
        if title:
            fig.update_layout(title=dict(text=title, x=0.5, xanchor='center',
                                         font=dict(size=16)))

    _ytitle_l = {'twiss':'Beta (m)', 'phase':'Phase Advance (2π)',
                 'orbit':'Orbit (m)', 'beamsize':'Beam Size (mm)',
                 'beta':'Beta (m)', 'dispersion':'Dispersion (m)', 'alpha':'Alpha',
}
    _ytitle_r = {'twiss':'Dispersion (m)', 'phase':'', 'orbit':'', 'beamsize':'',
                 'beta':'', 'dispersion':'', 'alpha':''}

    # ── Floor plan height helper — defined here so both layout branches can use it
    _primary_xz_height = max(1.0 * (element_height_xz if element_height_xz is not None else 0.05), 0.001)
    _primary_yz_height = _primary_xz_height  # updated by each branch after primary is built

    def _floor_heights(celems, cpdata):
        """Compute compare floor plan element heights/ranges matching the primary."""
        cuse_flr_y = any('flr_y0' in e for e in celems)
        cxz_ratio  = element_height_xz if element_height_xz is not None else 0.05
        cyz_ratio  = element_height_yz if element_height_yz is not None else 0.05
        csign      = -1.0 if flip_bend else 1.0
        if cuse_flr_y:
            cy_data = ([csign * e.get('flr_y0', 0.0) for e in celems] +
                       [csign * e.get('flr_y1', 0.0) for e in celems])
            cy_min, cy_max = min(cy_data), max(cy_data)
        else:
            cy_min, cy_max = 0.0, 0.0
        cy_data_span = cy_max - cy_min
        _cyz_rng_p = _parse_fp_range(fp_yz_range)
        _cxz_rng_p = _parse_fp_range(fp_xz_range)
        if _cyz_rng_p:
            cyz_half  = abs(_cyz_rng_p[1] - _cyz_rng_p[0]) / 2.0
            cy_center = (_cyz_rng_p[0] + _cyz_rng_p[1]) / 2.0
        elif cy_data_span < 0.01:
            cs_max    = max((e['s_start'] + e['length']) for e in celems) if celems else 1.0
            cyz_half  = cs_max * 0.02
            cy_center = 0.0
        else:
            cyz_half  = (cy_data_span / 2.0) * 1.2
            cy_center = (cy_min + cy_max) / 2.0
        # Reuse primary element heights for visual consistency across backends
        cxz_height = _primary_xz_height
        cyz_height = _primary_yz_height
        chalf_range = cyz_half + cyz_height
        cyz_range   = _cyz_rng_p if _cyz_rng_p else [cy_center - chalf_range,
                                                       cy_center + chalf_range]
        return cxz_height, cyz_height, _cxz_rng_p, cyz_range

    # ═══════════════════════════════════════════════════════════════════════════
    # LAYOUT: floor — two floor plans only
    # ═══════════════════════════════════════════════════════════════════════════
    if layout == 'floor':
        use_flr_y = any('flr_y0' in e for e in elements)
        if use_flr_y:
            y_vals = ([e.get('flr_y0', 0.0) for e in elements] +
                      [e.get('flr_y1', 0.0) for e in elements])
            y_span = max(y_vals) - min(y_vals) if y_vals else 0.0
        else:
            z_vals = [e.get('flr_z1', e['s_start'] + e['length'])
                      for e in elements if e['length'] > 0]
            y_span = (max(z_vals) - min(z_vals)) * 0.01 if z_vals else 1.0

        xz_ratio = element_height_xz if element_height_xz is not None else 0.05
        yz_ratio = element_height_yz if element_height_yz is not None else 0.05

        sign = -1.0 if flip_bend else 1.0

        # ── Y-Z axis span ─────────────────────────────────────────────────────
        if use_flr_y:
            y_data = ([sign * e.get('flr_y0', 0.0) for e in elements] +
                      [sign * e.get('flr_y1', 0.0) for e in elements])
            y_min_fp, y_max_fp = min(y_data), max(y_data)
        else:
            y_min_fp, y_max_fp = 0.0, 0.0
        y_data_span = y_max_fp - y_min_fp

        # Use user-specified range if provided, otherwise auto
        _yz_rng_parsed = _parse_fp_range(fp_yz_range)
        _xz_rng_parsed = _parse_fp_range(fp_xz_range)

        if _yz_rng_parsed:
            yz_axis_span = abs(_yz_rng_parsed[1] - _yz_rng_parsed[0])
            yz_half = yz_axis_span / 2.0
            y_center = (_yz_rng_parsed[0] + _yz_rng_parsed[1]) / 2.0
        elif y_data_span < 0.01:
            s_max = max((e['s_start'] + e['length']) for e in elements) if elements else 1.0
            yz_half = s_max * 0.02
            yz_axis_span = yz_half * 2.0
            y_center = 0.0
        else:
            yz_half = (y_data_span / 2.0) * 1.2
            yz_axis_span = yz_half * 2.0
            y_center = (y_min_fp + y_max_fp) / 2.0
        yz_height = max(yz_axis_span * yz_ratio, 0.001)

        # ── X-Z axis span ─────────────────────────────────────────────────────
        if _xz_rng_parsed:
            xz_axis_span = abs(_xz_rng_parsed[1] - _xz_rng_parsed[0])
        elif use_flr_y:
            x_vals = ([sign * e.get('flr_x0', 0.0) for e in elements] +
                      [sign * e.get('flr_x1', 0.0) for e in elements])
            x_data_span = max(x_vals) - min(x_vals) if x_vals else 0.0
            xz_axis_span = x_data_span * 1.2 if x_data_span > 0.01 else yz_axis_span
        else:
            xz_axis_span = yz_axis_span
        xz_height = max(xz_axis_span * xz_ratio, 0.001)
        _primary_xz_height = xz_height
        _primary_yz_height = yz_height

        # Build subplots based on which planes are enabled
        _n_floor_rows = int(show_xz) + int(show_yz)
        if _n_floor_rows == 0:
            fig = make_subplots(rows=1, cols=1)
        elif _n_floor_rows == 1:
            _title = 'Floor Plan — X vs Z' if show_xz else 'Floor Plan — Y vs Z'
            fig = make_subplots(rows=1, cols=1, shared_xaxes=False,
                subplot_titles=((_title,) if show_titles else ('',)),
                specs=[[{'secondary_y': False}]])
        else:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                row_heights=[0.5, 0.5], vertical_spacing=0.08,
                subplot_titles=(('Floor Plan — X vs Z', 'Floor Plan — Y vs Z') if show_titles else ('', '')),
                specs=[[{'secondary_y': False}], [{'secondary_y': False}]])

        _xz_row = 1 if show_xz else None
        _yz_row = (2 if show_xz else 1) if show_yz else None

        _prog(50, 'Building floor plan...')
        # Overlay all selected universes on the same floor plan
        for _ui, _uid in enumerate(_plot_unis):
            _ud     = _all_uni_data[_uid]
            _uelems = _ud['elements']
            _fp_leg = 'legend' if _ui == 0 else f'legend{_ui * 2 + 1}'
            _el_leg = 'legend2' if _ui == 0 else f'legend{_ui * 2 + 2}'
            if _xz_row is not None:
                _build_floor_plan(fig, _uelems, xz_height, flip_bend, row=_xz_row,
                                  legend_name=_el_leg, fp_legend_name=_fp_leg,
                                  show_fp_legend=(_ui == 0),
                                  beampipe_color=_BEAMPIPE_COLORS[_ui % len(_BEAMPIPE_COLORS)] if color_beampipes else 'gray',
                                  show_markers=show_markers)
            if _yz_row is not None:
                _build_floor_plan_yz(fig, _uelems, yz_height, flip_bend, row=_yz_row,
                                     legend_name=_el_leg, fp_legend_name=_fp_leg,
                                     show_fp_legend=False,
                                     beampipe_color=_BEAMPIPE_COLORS[_ui % len(_BEAMPIPE_COLORS)] if color_beampipes else 'gray',
                                     show_markers=show_markers)
        flr_lkw = dict(height=900, hovermode='closest',
            xaxis=dict(domain=[0.0, 0.95]))
        if _n_floor_rows == 2:
            flr_lkw['xaxis2'] = dict(domain=[0.0, 0.95])
        # Declare one legend pair per universe
        for _ui in range(len(_plot_unis)):
            _fp_leg = 'legend' if _ui == 0 else f'legend{_ui * 2 + 1}'
            _el_leg = 'legend2' if _ui == 0 else f'legend{_ui * 2 + 2}'
            y_pos = 1.0 - _ui * 0.25
            flr_lkw[_fp_leg] = dict(x=1.02, y=y_pos, xanchor='left', yanchor='top')
            flr_lkw[_el_leg] = dict(x=1.02, y=y_pos, xanchor='left', yanchor='top')
        fig.update_layout(**flr_lkw)
        if _xz_row is not None:
            fig.update_xaxes(title_text='Z (m)', row=_xz_row, col=1)
            fig.update_yaxes(title_text='X (m)', row=_xz_row, col=1)
        if _yz_row is not None:
            fig.update_xaxes(title_text='Z (m)', row=_yz_row, col=1)
            fig.update_yaxes(title_text='Y (m)', row=_yz_row, col=1)
        # ── Axis ranges: tunnel takes priority, then user input, then auto ────
        _yz_rng = _yz_rng_parsed
        _xz_rng = _xz_rng_parsed
        if _tunnel is not None:
            if _xz_row is not None:
                _txz_z, _txz_x = _draw_tunnel_wall_xz(fig, _tunnel, row=_xz_row, flip=flip_bend)
                fig.update_xaxes(range=_txz_z, row=_xz_row, col=1)
                fig.update_yaxes(range=_xz_rng if _xz_rng else _txz_x, row=_xz_row, col=1)
            if _yz_row is not None:
                _tyz_z, _tyz_y = _draw_tunnel_wall_yz(fig, _tunnel, row=_yz_row, flip=flip_bend)
                fig.update_xaxes(range=_tyz_z, row=_yz_row, col=1)
                fig.update_yaxes(range=_yz_rng if _yz_rng else _tyz_y, row=_yz_row, col=1)
        else:
            y_center = (y_min_fp + y_max_fp) / 2.0
            half_range = yz_half + yz_height
            if _yz_row is not None:
                fig.update_yaxes(
                    range=_yz_rng if _yz_rng else [y_center - half_range, y_center + half_range],
                    row=_yz_row, col=1)
            if _xz_row is not None and _xz_rng:
                fig.update_yaxes(range=_xz_rng, row=_xz_row, col=1)

        # ── Floor layout: separate compare figures ────────────────────────────
        if compare_mode == 'separate' and _cmp_datasets:
            for ci, cd in enumerate(_cmp_datasets):
                clabel   = cd['label']
                cprimary = cd['plot_unis'][0]
                cpdata   = cd['all_uni'][cprimary]
                celems   = cpdata['elements']
                cxz_h, cyz_h, cxz_rng, cyz_rng = _floor_heights(celems, cpdata)

                cfig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                    row_heights=[0.5, 0.5], vertical_spacing=0.08,
                    subplot_titles=((f'Floor Plan (X-Z) — {clabel}', f'Floor Plan (Y-Z) — {clabel}') if show_titles else ('', '')),
                    specs=[[{'secondary_y': False}], [{'secondary_y': False}]])
                _build_floor_plan(cfig, celems, cxz_h, flip_bend, row=1,
                                  legend_name='legend2', fp_legend_name='legend1')
                _build_floor_plan_yz(cfig, celems, cyz_h, flip_bend, row=2,
                                     legend_name='legend2', fp_legend_name='legend1')
                cfig.update_layout(height=900, hovermode='closest',
                    legend=dict(x=1.02, y=1.0, xanchor='left'),
                    legend2=dict(x=1.02, y=0.55, xanchor='left'),
                    xaxis=dict(domain=[0.0, 0.95]),
                    xaxis2=dict(domain=[0.0, 0.95]))
                cfig.update_xaxes(title_text='Z (m)', row=1, col=1)
                cfig.update_xaxes(title_text='Z (m)', row=2, col=1)
                cfig.update_yaxes(title_text='X (m)', row=1, col=1,
                                  **({'range': cxz_rng} if cxz_rng else {}))
                cfig.update_yaxes(title_text='Y (m)', row=2, col=1, range=cyz_rng)
                if not hasattr(fig, '_compare_figs'): fig._compare_figs = []
                fig._compare_figs.append((clabel, cfig))

        # ── Floor layout: overlay compare traces onto primary figure ──────────
        elif compare_mode == 'overlay' and _cmp_datasets:
            for ci, cd in enumerate(_cmp_datasets):
                clabel   = cd['label']
                cprimary = cd['plot_unis'][0]
                cpdata   = cd['all_uni'][cprimary]
                celems   = cpdata['elements']
                cxz_h, cyz_h, _, _ = _floor_heights(celems, cpdata)
                _build_floor_plan(fig, celems, cxz_h, flip_bend, row=1,
                                  legend_name=f'legend{(ci+1)*2}',
                                  fp_legend_name=f'legend{(ci+1)*2+1}')
                _build_floor_plan_yz(fig, celems, cyz_h, flip_bend, row=2,
                                     legend_name=f'legend{(ci+1)*2}',
                                     fp_legend_name=f'legend{(ci+1)*2+1}')

    # ═══════════════════════════════════════════════════════════════════════════
    # LAYOUT: all / optics — dynamic panels
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        # ── Reorder panels: floor-xz/floor-yz first, bar last ─────────────────
        _floor_panels = [p for p in panels if p in ('floor-xz', 'floor-yz')]
        _data_panels  = [p for p in panels if p not in ('floor-xz', 'floor-yz', 'bar')]
        _bar_panels   = [p for p in panels if p == 'bar']
        panels_ordered = _floor_panels + _data_panels + _bar_panels

        # Warn if user had floor panels out of order
        if panels != panels_ordered and _floor_panels:
            _log("[info] Floor plan panels moved to top automatically.")

        panels = panels_ordered

        # ── Keep backward compat: show_floor still works if no floor panels ───
        # If show_floor is True and no floor panels added, inject floor-xz
        if show_floor and not _floor_panels:
            panels = ['floor-xz'] + panels
            _floor_panels = ['floor-xz']

        include_floor = bool(_floor_panels)
        include_bar   = any(p == 'bar' for p in panels)
        n_panels      = len(panels)

        # ── Default panel heights (px) ─────────────────────────────────────────
        _DEFAULT_H = {
            'floor-xz': 220, 'floor-yz': 220,
            'bar': 80,
            'latdiff': 260,  # per table × 3
            'summary': 260,
        }
        _DATA_H = 280  # default for data panels

        def _panel_px(p, idx):
            _raw_spec = p if isinstance(p, str) else p.get('spec', '')
            spec = _raw_spec if isinstance(_raw_spec, str) else _raw_spec.get('type', 'custom')
            if panel_heights and spec in panel_heights:
                return int(panel_heights[spec])
            if isinstance(p, str):
                return _DEFAULT_H.get(p, _DATA_H)
            return _DATA_H

        # ── Build row list with heights ────────────────────────────────────────
        # latdiff = 3 rows, others = 1
        row_list = []  # list of (panel, row_height_px)
        for idx, p in enumerate(panels):
            h = _panel_px(p, idx)
            if p == 'latdiff':
                row_list.extend([(p, h), (p, h), (p, h)])
            else:
                row_list.append((p, h))

        n_rows  = len(row_list)
        # Panel spacing in pixels — adds to total height, doesn't steal from panels
        _spacing_px = max(20, int(float(panel_spacing))) if panel_spacing else 80
        total_h = sum(r[1] for r in row_list) + max(0, (n_rows - 1)) * _spacing_px

        # Normalized row heights: panel px / total_h (spacing handled by v_spacing)
        v_spacing = _spacing_px / total_h if total_h > 0 else 0.08
        row_heights_norm = [r[1] / total_h for r in row_list]

        # ── Subplot titles and specs ───────────────────────────────────────────
        titles, specs = [], []
        for p, _ in row_list:
            if p == 'latdiff' and row_list.index((p, _)) == next(
                    i for i,(rp,_) in enumerate(row_list) if rp == 'latdiff'):
                titles.append(panel_title(p))
                specs.append([{'type': 'table'}])
            elif p == 'latdiff':
                # 2nd and 3rd rows of latdiff
                titles.append('')
                specs.append([{'type': 'table'}])
            elif p in ('floor-xz', 'floor-yz', 'bar'):
                titles.append(panel_title(p))
                specs.append([{'secondary_y': False}])
            elif p == 'summary':
                titles.append(panel_title(p))
                specs.append([{'type': 'table'}])
            else:
                titles.append(p.get('name', 'Custom') if isinstance(p, dict) else panel_title(p))
                specs.append([{'secondary_y': (p == 'twiss') or
                    (isinstance(p, dict) and bool(p.get('y2'))) or
                    (isinstance(p, dict) and p.get('type') == 'expr' and
                     bool(p.get('y2_expr', '').strip()))}])

        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=False,
            row_heights=row_heights_norm, vertical_spacing=v_spacing,
            subplot_titles=titles if show_titles else [''] * len(titles), specs=specs,
        )
        fig.update_layout(height=max(total_h, 400))

        current_row = 1

        # ── Floor plan rows ────────────────────────────────────────────────────
        _primary_xz_height = 0.05
        _primary_yz_height = 0.05

        if 'floor-xz' in panels:
            _prog(50, 'Building floor plan (X-Z)...')
            sign = -1.0 if flip_bend else 1.0
            _xz_ratio = element_height_xz if element_height_xz is not None else 0.05
            use_flr_y = any('flr_y0' in e for e in elements)
            if use_flr_y:
                x_vals = ([sign * e.get('flr_x0', 0.0) for e in elements] +
                          [sign * e.get('flr_x1', 0.0) for e in elements])
                x_data_span = max(x_vals) - min(x_vals) if x_vals else 1.0
                xz_axis_span = x_data_span * 1.2 if x_data_span > 0.01 else 1.0
                _xz_h = max(xz_axis_span * _xz_ratio, 0.001)
            else:
                _xz_rng_val = _parse_fp_range(fp_xz_range)
                if _xz_rng_val:
                    _xz_h = (_xz_rng_val[1] - _xz_rng_val[0]) * _xz_ratio
                else:
                    _xz_h = 1.0 * _xz_ratio
                _xz_h = max(_xz_h, 0.001)
            _primary_xz_height = _xz_h
            _xz_floor_row = current_row
            for _ui, _uid in enumerate(_plot_unis):
                _ud = _all_uni_data[_uid]
                _uelems = _ud['elements']
                _fp_leg = 'legend91' if _ui == 0 else f'legend{90 + _ui * 2 + 1}'
                _el_leg = 'legend90' if _ui == 0 else f'legend{90 + _ui * 2}'
                _build_floor_plan(fig, _uelems, _xz_h, flip_bend,
                                  row=current_row,
                                  legend_name=_el_leg, fp_legend_name=_fp_leg,
                                  show_fp_legend=(_ui == 0),
                                  beampipe_color=_BEAMPIPE_COLORS[_ui % len(_BEAMPIPE_COLORS)] if color_beampipes else 'gray',
                                  show_markers=show_markers)
            fig.update_xaxes(title_text='Z (m)', row=current_row, col=1)
            _xz_rng = _parse_fp_range(fp_xz_range)
            if _tunnel is not None:
                _, _txz_x = _draw_tunnel_wall_xz(fig, _tunnel, row=current_row, flip=flip_bend)
                fig.update_yaxes(title_text='X (m)', row=current_row, col=1,
                                 range=_xz_rng if _xz_rng else _txz_x)
            else:
                fig.update_yaxes(title_text='X (m)', row=current_row, col=1,
                                 **({'range': _xz_rng} if _xz_rng else {}))
            current_row += 1

        if 'floor-yz' in panels:
            _prog(52, 'Building floor plan (Y-Z)...')
            _yz_ratio = element_height_yz if element_height_yz is not None else 0.05
            use_flr_y2 = any('flr_y0' in e for e in elements)
            if use_flr_y2:
                y_vals = [e.get('flr_y0', 0.0) for e in elements] + [e.get('flr_y1', 0.0) for e in elements]
                y_data_span = max(y_vals) - min(y_vals) if y_vals else 0.0
                y_min_fp = min(y_vals) if y_vals else 0.0
                y_max_fp = max(y_vals) if y_vals else 0.0
                y_center = (y_min_fp + y_max_fp) / 2.0
                # Use display range for element height — pad to avoid invisible elements
                _yz_rng_val = _parse_fp_range(fp_yz_range)
                if _yz_rng_val:
                    yz_display_span = _yz_rng_val[1] - _yz_rng_val[0]
                elif y_data_span > 0.01:
                    yz_display_span = y_data_span * 1.4
                else:
                    # Very flat — use XZ scale as reference for display
                    yz_display_span = max(_primary_xz_height * 20, 0.002)
                yz_half = yz_display_span / 2.0
            else:
                _yz_rng_val = _parse_fp_range(fp_yz_range)
                yz_display_span = (_yz_rng_val[1] - _yz_rng_val[0]) if _yz_rng_val else max(_primary_xz_height * 20, 1.0)
                y_center = 0.0; yz_half = yz_display_span / 2.0
            _yz_h = max(yz_display_span * _yz_ratio, 0.001)
            _primary_yz_height = _yz_h
            for _ui, _uid in enumerate(_plot_unis):
                _ud = _all_uni_data[_uid]
                _uelems = _ud['elements']
                _fp_leg = 'legend91' if _ui == 0 else f'legend{90 + _ui * 2 + 1}'
                _el_leg = 'legend90' if _ui == 0 else f'legend{90 + _ui * 2}'
                _build_floor_plan_yz(fig, _uelems, _yz_h, flip_bend,
                                     row=current_row,
                                     legend_name=_el_leg, fp_legend_name=_fp_leg,
                                     show_fp_legend=(_ui == 0 and 'floor-xz' not in panels),
                                     beampipe_color=_BEAMPIPE_COLORS[_ui % len(_BEAMPIPE_COLORS)] if color_beampipes else 'gray',
                                     show_markers=show_markers)
            fig.update_xaxes(title_text='Z (m)', row=current_row, col=1)
            _yz_rng = _parse_fp_range(fp_yz_range)
            if _tunnel is not None:
                _, _tyz_y = _draw_tunnel_wall_yz(fig, _tunnel, row=current_row, flip=flip_bend)
                fig.update_yaxes(title_text='Y (m)', row=current_row, col=1,
                                 range=_yz_rng if _yz_rng else _tyz_y)
            else:
                fig.update_yaxes(title_text='Y (m)', row=current_row, col=1,
                                 range=_yz_rng if _yz_rng else [y_center - yz_half - _yz_h, y_center + yz_half + _yz_h])
            current_row += 1

        # Data panels
        first_s_row = current_row
        _tune_annotated = False  # only annotate once on first data panel
        for i, p in enumerate(panels):
            _prog(55 + int(30 * i / max(len(panels), 1)), f'Building panel: {p}...')
            legend_n      = f'legend{i+1}'
            has_secondary = (p == 'twiss') or\
                (isinstance(p, dict) and bool(p.get('y2'))) or\
                (isinstance(p, dict) and p.get('type') == 'expr' and bool(p.get('y2_expr', '').strip()))
            bp_full = {**beam_params, 'alpha_a':al_a, 'alpha_b':al_b}

            # ── Skip floor panels — already rendered above ─────────────────
            if p in ('floor-xz', 'floor-yz'):
                continue

            # ── Beamline bar panel ────────────────────────────────────────
            if p == 'bar':
                _build_layout_bar(fig, elements, show_element_labels, row=current_row,
                                  show_markers=show_markers_bar)
                # Wildcard annotations on bar
                _bar_annot = (panel_annotations or {}).get(i, '')
                if not _bar_annot and isinstance(p, dict):
                    _bar_annot = p.get('annot_pattern', '').strip()
                if _bar_annot:
                    _build_bar_annotations(fig, elements, _bar_annot, row=current_row,
                                           annot_font_size=int((font_sizes or {}).get('annot', 8)))
                ref = f'x{first_s_row}' if first_s_row > 1 else 'x'
                fig.update_xaxes(matches=ref, row=current_row, col=1)
                fig.update_xaxes(title_text='s (m)' if not normalize_s else 's/s_max',
                                 row=current_row, col=1)
                fig.update_yaxes(title_text='', showticklabels=False,
                                 range=[-0.4, 0.4], row=current_row, col=1)
                current_row += 1
                continue

            # ── Summary panel ─────────────────────────────────────────────
            if p == 'summary':
                _build_summary_panel(fig, _all_uni_data, _plot_unis, _uni_labels,
                                     beam_params, row=current_row)
                current_row += 1
                continue

            # ── Lattice diff panel ─────────────────────────────────────────
            if p == 'latdiff':
                if _cmp_datasets:
                    _primary_uid = _plot_unis[0]
                    _elems_a = _all_uni_data[_primary_uid]['elements']
                    _label_a = _uni_labels.get(_primary_uid, 'Primary')
                    _cd = _cmp_datasets[0]
                    _cprimary = _cd['plot_unis'][0]
                    _elems_b = _cd['all_uni'][_cprimary]['elements']
                    _label_b = _cd['label']
                    _build_latdiff_panel(fig, _elems_a, _elems_b, _label_a, _label_b,
                                         row=current_row)
                    current_row += 3  # three table rows: strengths, entry, exit
                else:
                    current_row += 3
                continue

            # ── Expression panel: query live from backend ─────────────────
            if isinstance(p, dict) and p.get('type') == 'expr':
                if _multi:
                    # ── Multi-universe: evaluate expression for each universe ──
                    y1_lbl = p.get('y1_label', p.get('y1_expr', ''))
                    y2_lbl = p.get('y2_label', p.get('y2_expr', '')) or None
                    for _ui, _uid in enumerate(_plot_unis):
                        _ulbl = _uni_labels.get(_uid, f'u{_uid}')
                        _ud   = _all_uni_data[_uid]
                        _pal  = _UNI_PALETTES[_ui % len(_UNI_PALETTES)]
                        # Tag the legend name with universe label
                        _p_tagged = dict(p)
                        _p_tagged['y1_label'] = f"{p.get('y1_label', p.get('y1_expr', ''))} ({_ulbl})"
                        if p.get('y2_expr', '').strip():
                            _p_tagged['y2_label'] = f"{p.get('y2_label', p.get('y2_expr', ''))} ({_ulbl})"
                        result = _build_expr_panel(
                            fig, _p_tagged, _ud, code, _ud['s'],
                            row=current_row, legend_name=legend_n,
                            log_fn=log_fn, uni_idx=_uid, palette=_pal)
                else:
                    # ── Single universe ───────────────────────────────────────
                    result = _build_expr_panel(
                        fig, p, _pdata, code, s,
                        row=current_row, legend_name=legend_n,
                        log_fn=log_fn, uni_idx=_plot_unis[0])
                    y1_lbl = result[0] if result else ''
                    y2_lbl = result[1] if result else None
                fig.update_yaxes(title_text=y1_lbl, row=current_row, col=1, secondary_y=False)
                if y2_lbl:
                    fig.update_yaxes(title_text=y2_lbl, row=current_row, col=1, secondary_y=True)
                if not _tune_annotated and not _multi and show_tune:
                    _build_tune_annotation(fig, beam_params, row=current_row)
                    _tune_annotated = True
                current_row += 1
                continue
            if _multi:
                # ── Multi-universe: use _build_panel3_uni for ALL universes ──
                result = None
                for _ui, _uid in enumerate(_plot_unis):
                    _ulbl = _uni_labels.get(_uid, f'u{_uid}')
                    _ud   = _all_uni_data[_uid]
                    _pal  = _UNI_PALETTES[_ui % len(_UNI_PALETTES)]
                    _us = _ud['s']; _uba = _ud['beta_a']; _ubb = _ud['beta_b']
                    _uex = _ud['eta_x']; _uey = _ud['eta_y']
                    _uox = _ud['orbit_x']; _uoy = _ud['orbit_y']
                    _upa = _ud['phi_a']; _upb = _ud['phi_b']
                    _ual_a = _ud.get('alpha_a', np.zeros_like(_us))
                    _ual_b = _ud.get('alpha_b', np.zeros_like(_us))
                    _ubp = {**beam_params, 'alpha_a': _ual_a, 'alpha_b': _ual_b}
                    _build_panel3_uni(fig, p,
                        _us, _uba, _ubb, _uex, _uey,
                        _uox, _uoy, _upa, _upb,
                        _ual_a, _ual_b, _ubp,
                        row=current_row, legend_name=legend_n,
                        uni_label=_ulbl, palette=_pal,
                        uni_idx=_ui)
            else:
                # ── Single universe: original path ────────────────────────────
                result = _build_panel3(fig, p, s, ba, bb, ex, ey, ox, oy, pa, pb,
                              row=current_row, legend_name=legend_n,
                              row3_secondary=has_secondary,
                              beam_params=bp_full)

            # y-axis labels
            if isinstance(p, dict):
                y1_lbl = result[0] if result else ''
                y2_lbl = result[1] if result else None
            else:
                y1_lbl = p.get('name', '') if isinstance(p, dict) else _ytitle_l.get(p, '')
                y2_lbl = _ytitle_r.get(p, '') if p == 'twiss' else None
            fig.update_yaxes(title_text=y1_lbl,
                             row=current_row, col=1, secondary_y=False)
            if has_secondary:
                if p == 'twiss':
                    # ── Twiss preset: align beta/dispersion gridlines at zero ──
                    nice = [1, 2, 2.5, 5, 10]
                    # Compute true max across ALL plotted universes
                    _all_ba = [_all_uni_data[_uid]['beta_a'] for _uid in _plot_unis]
                    _all_bb = [_all_uni_data[_uid]['beta_b'] for _uid in _plot_unis]
                    _all_ex = [_all_uni_data[_uid]['eta_x']  for _uid in _plot_unis]
                    _all_ey = [_all_uni_data[_uid]['eta_y']  for _uid in _plot_unis]
                    _beta_all = np.concatenate(_all_ba + _all_bb)
                    _disp_all = np.concatenate(_all_ex + _all_ey)
                    beta_max = float(np.nanmax(_beta_all)) * 1.1 if len(_beta_all) else 1.0
                    raw_beta_dt = beta_max / 5
                    mag = 10 ** np.floor(np.log10(raw_beta_dt))
                    beta_dt = mag * min(nice, key=lambda x: abs(x - raw_beta_dt / mag))
                    beta_range_max = np.ceil(beta_max / beta_dt) * beta_dt
                    n_above = int(round(beta_range_max / beta_dt))
                    d_min = float(np.nanmin(_disp_all)) * 1.1 if len(_disp_all) else 0.0
                    d_max = float(np.nanmax(_disp_all)) * 1.1 if len(_disp_all) else 1.0
                    raw_disp_dt = max(d_max, abs(d_min)) / max(n_above, 1)
                    mag2 = 10 ** np.floor(np.log10(raw_disp_dt)) if raw_disp_dt > 0 else 1.0
                    disp_dt = mag2 * min(nice, key=lambda x: abs(x - raw_disp_dt / mag2))
                    n_disp_above = int(np.ceil(d_max / disp_dt)) if d_max > 0 else 0
                    n_disp_below = int(np.ceil(abs(d_min) / disp_dt)) if d_min < 0 else 0
                    n_above = max(n_above, n_disp_above)
                    beta_range_max = n_above * beta_dt
                    disp_range_max = n_disp_above * disp_dt
                    disp_range_min = -n_disp_below * disp_dt
                    fig.update_yaxes(title_text=y1_lbl,
                                     row=current_row, col=1, secondary_y=False,
                                     range=[0, beta_range_max], dtick=beta_dt,
                                     showgrid=True)
                    fig.update_yaxes(title_text=y2_lbl or _ytitle_r.get(p, ''),
                                     row=current_row, col=1, secondary_y=True,
                                     range=[disp_range_min, disp_range_max],
                                     dtick=disp_dt, showgrid=True,
                                     gridcolor='rgba(100,200,100,0.3)',
                                     griddash='dash')
                else:
                    # ── Custom panel: auto-scale each axis independently ───────
                    fig.update_yaxes(title_text=y1_lbl,
                                     row=current_row, col=1, secondary_y=False,
                                     autorange=True, showgrid=True)
                    fig.update_yaxes(title_text=y2_lbl or '',
                                     row=current_row, col=1, secondary_y=True,
                                     autorange=True, showgrid=True,
                                     gridcolor='rgba(100,200,100,0.3)',
                                     griddash='dash')
            # Link all data panel x-axes together (since shared_xaxes=False)
            ref = f'x{first_s_row}' if first_s_row > 1 else 'x'
            if current_row != first_s_row:
                fig.update_xaxes(matches=ref, row=current_row, col=1)
            # Element annotations for this panel
            _annot_pat = (panel_annotations or {}).get(i, '')
            if not _annot_pat and isinstance(p, dict):
                _annot_pat = p.get('annot_pattern', '').strip()
            if _annot_pat:
                _build_panel_annotations(fig, elements, _annot_pat, row=current_row,
                                         annot_font_size=int((font_sizes or {}).get('annot', 8)))
            # Add tune/chroma annotation to first data panel
            if show_tune and not _tune_annotated and not _multi:
                _build_tune_annotation(fig, beam_params, row=current_row)
                _tune_annotated = True
            current_row += 1

        # If no bar panel in list, put s-axis label on last data panel
        if not include_bar:
            last_data_row = current_row - 1
            fig.update_xaxes(title_text='s (m)' if not normalize_s else 's/s_max',
                             row=last_data_row, col=1)

        # Figure-level layout — height already set above from panel_heights
        fig_h = total_h
        fig_w = None
        if aspect_ratio:
            try:
                parts = str(aspect_ratio).split(':')
                aw, ah = float(parts[0]), float(parts[1])
                fig_w = int(fig_h * aw / ah)
            except Exception:
                pass
        lkw = dict(height=fig_h, hovermode='closest')
        if fig_w: lkw['width'] = fig_w

        # Compute exact vertical midpoint of each row in normalized [0,1] coords.
        row_tops = []
        cursor = 1.0
        for h in row_heights_norm:
            row_tops.append(cursor)
            cursor -= h + v_spacing

        row_mids = [row_tops[i] - row_heights_norm[i] / 2 for i in range(len(row_heights_norm))]

        # Legend positioning — inside (top-right of subplot) or outside (right margin)
        LEGEND_OFFSET = 0.01
        if legend_inside:
            LEGEND_X  = 0.98
            LEGEND_XA = 'right'
            LEGEND_BG = 'rgba(0,0,0,0)'
            LEGEND_BC = '#1a3d1a'
            x_domain  = [0.0, 1.0]
        else:
            LEGEND_X  = 1.02
            LEGEND_XA = 'left'
            LEGEND_BG = 'rgba(0,0,0,0)'
            LEGEND_BC = '#1a3d1a'
            x_domain  = [0.0, 0.95]

        def _lgd(y_top):
            return dict(
                itemsizing='constant',
                bgcolor=LEGEND_BG,
                bordercolor=LEGEND_BC,
                borderwidth=1,
                x=LEGEND_X, xanchor=LEGEND_XA,
                y=y_top - LEGEND_OFFSET, yanchor='top')

        row_idx = 0
        if include_floor:
            # One legend pair per universe — for floor-xz and floor-yz rows
            n_floor_rows = len(_floor_panels)
            for _fi in range(n_floor_rows):
                for _ui in range(len(_plot_unis)):
                    _fp_leg = 'legend91' if _ui == 0 else f'legend{90 + _ui * 2 + 1}'
                    _el_leg = 'legend90' if _ui == 0 else f'legend{90 + _ui * 2}'
                    if row_idx < len(row_tops):
                        lkw[_fp_leg] = _lgd(row_tops[row_idx])
                        lkw[_el_leg] = _lgd(row_tops[row_idx])
                row_idx += 1

        for i, p in enumerate(panels):
            if p in ('floor-xz', 'floor-yz'): continue
            if row_idx < len(row_tops):
                lkw[f'legend{i+1}'] = _lgd(row_tops[row_idx])
            row_idx += 1

        for r in range(1, n_rows + 1):
            ax = f'xaxis{r}' if r > 1 else 'xaxis'
            lkw[ax] = dict(domain=x_domain)
        fig.update_layout(**lkw)

        # ── Separate mode: one mini-figure per panel slot, interleaved ──────────
        # Order: [floor group] [panel0 group] [panel1 group] ... [bar group]
        # Each group = primary row + one row per compare file
        if compare_mode == 'separate' and _cmp_datasets:
            _log(f"[compare] Building interleaved panels for {len(_cmp_datasets)} file(s)...")

            # Pre-extract compare data for all files
            _csep = []
            for ci, cd in enumerate(_cmp_datasets):
                cprimary = cd['plot_unis'][0]
                cpdata   = cd['all_uni'][cprimary]
                cs_c  = cpdata['s']
                if normalize_s and float(cs_c[-1]) > 0:
                    cs_c = cs_c / float(cs_c[-1])
                cbp_raw = cpdata.get('beam_params', {})
                _csep.append({
                    'label':  cd['label'],
                    'code':   cd['code'],
                    'pdata':  cpdata,
                    's':      cs_c,
                    'ba':     cpdata['beta_a'],   'bb': cpdata['beta_b'],
                    'ex':     cpdata['eta_x'],    'ey': cpdata['eta_y'],
                    'al_a':   cpdata.get('alpha_a', np.zeros_like(cs_c)),
                    'al_b':   cpdata.get('alpha_b', np.zeros_like(cs_c)),
                    'ox':     cpdata['orbit_x'],  'oy': cpdata['orbit_y'],
                    'pa':     cpdata['phi_a'],    'pb': cpdata['phi_b'],
                    'elems':  cpdata['elements'],
                    'bp': {
                        'emit_x':   emit_x   if emit_x   is not None else cbp_raw.get('emit_x',   0.0),
                        'emit_y':   emit_y   if emit_y   is not None else cbp_raw.get('emit_y',   0.0),
                        'sigma_dp': sigma_dp if sigma_dp is not None else cbp_raw.get('sigma_dp', 0.0),
                        'n_sigma':  float(n_sigma) if n_sigma is not None else 1.0,
                        'alpha_a':  cpdata.get('alpha_a', np.zeros_like(cs_c)),
                        'alpha_b':  cpdata.get('alpha_b', np.zeros_like(cs_c)),
                    },
                    'pal':    _UNI_PALETTES[ci % len(_UNI_PALETTES)],
                })

            def _make_group_fig(n_group_rows, titles, specs, heights, lkw_extra=None):
                """Build a small make_subplots figure for one panel group."""
                gv = 0.06
                gfig = make_subplots(
                    rows=n_group_rows, cols=1, shared_xaxes=True,
                    row_heights=heights, vertical_spacing=gv,
                    subplot_titles=titles if show_titles else [''] * len(titles), specs=specs)
                gh = 120 + n_group_rows * 200
                glkw = dict(height=gh, hovermode='closest',
                            xaxis=dict(domain=[0.0, x_domain[1]]))
                for ri in range(2, n_group_rows + 1):
                    glkw[f'xaxis{ri}'] = dict(domain=[0.0, x_domain[1]])
                if lkw_extra:
                    glkw.update(lkw_extra)
                gfig.update_layout(**glkw)
                return gfig

            n_grp = 1 + len(_csep)  # primary + compare files
            grp_h = [1.0 / n_grp] * n_grp

            # ── Floor plan group ──────────────────────────────────────────────
            if include_floor:
                fp_titles = [''] * n_grp  # no per-row titles — legend identifies each row
                fp_specs  = [[{'secondary_y': False}]] * n_grp
                # Valid Plotly legend names: 'legend' (first), 'legend2', 'legend3', ...
                # Each row gets two legend slots: fp_icons and element traces
                # Row 0 (primary):  fp_legend='legend',  legend='legend2'
                # Row 1 (compare1): fp_legend='legend3', legend='legend4'
                # Row 2 (compare2): fp_legend='legend5', legend='legend6'  etc.
                fp_lkw = {}
                for fi in range(n_grp):
                    y_pos = 1.0 - fi / n_grp
                    fp_leg_name = 'legend' if fi == 0 else f'legend{fi*2+1}'
                    el_leg_name = f'legend{fi*2+2}'
                    fp_lkw[fp_leg_name] = dict(x=0.87, xanchor='left', y=y_pos,
                        yanchor='top', itemsizing='constant', bgcolor='rgba(0,0,0,0)')
                    fp_lkw[el_leg_name] = dict(x=0.87, xanchor='left', y=y_pos,
                        yanchor='top', itemsizing='constant', bgcolor='rgba(0,0,0,0)')
                fp_fig = _make_group_fig(n_grp, fp_titles, fp_specs, grp_h, fp_lkw)
                # Primary floor row — fp icons go to 'legend', traces to 'legend2'
                _build_floor_plan(fp_fig, elements, _primary_xz_height, flip_bend,
                                  row=1, legend_name='legend2', fp_legend_name='legend')
                fp_fig.update_xaxes(title_text='Z (m)', row=1, col=1)
                fp_fig.update_yaxes(title_text='X (m)', row=1, col=1)
                # Compare floor rows — each gets its own pair of legend slots
                for ri, c in enumerate(_csep, start=2):
                    cxz_h, _, cxz_rng, _ = _floor_heights(c['elems'], c['pdata'])
                    fp_leg_name = f'legend{(ri-1)*2+1}'
                    el_leg_name = f'legend{(ri-1)*2+2}'
                    _build_floor_plan(fp_fig, c['elems'], cxz_h, flip_bend,
                                      row=ri,
                                      legend_name=el_leg_name,
                                      fp_legend_name=fp_leg_name)
                    fp_fig.update_xaxes(title_text='Z (m)', row=ri, col=1)
                    fp_fig.update_yaxes(title_text='X (m)', row=ri, col=1,
                                        **({'range': cxz_rng} if cxz_rng else {}))
                fp_fig.update_layout(title=dict(text='Floor Plan', x=0.5, xanchor='center'))
                if not hasattr(fig, '_compare_figs'): fig._compare_figs = []
                fig._compare_figs.append(('__floor__', fp_fig))

            # ── Data panel groups ─────────────────────────────────────────────
            for pi, p in enumerate(panels):
                ptitle  = p.get('name', 'Custom') if isinstance(p, dict) else panel_title(p)
                has_sec = (p == 'twiss') or \
                    (isinstance(p, dict) and bool(p.get('y2'))) or \
                    (isinstance(p, dict) and p.get('type') == 'expr' and bool(p.get('y2_expr','').strip()))
                pg_titles = [''] * n_grp  # no per-row titles
                pg_specs  = [[{'secondary_y': has_sec}]] * n_grp
                pg_lkw    = {}
                for li in range(n_grp):
                    leg_name = 'legend' if li == 0 else f'legend{li+1}'
                    pg_lkw[leg_name] = dict(
                        x=LEGEND_X, xanchor=LEGEND_XA,
                        y=1.0 - li / n_grp, yanchor='top',
                        itemsizing='constant', bgcolor='rgba(0,0,0,0)')
                pg_fig = _make_group_fig(n_grp, pg_titles, pg_specs, grp_h, pg_lkw)
                pg_fig.update_layout(title=dict(text=ptitle, x=0.5, xanchor='center'))
                # Primary panel — use 'legend' (bare) for first legend
                _build_panel3_uni(pg_fig, p, s, ba, bb, ex, ey,
                                  ox, oy, pa, pb, al_a, al_b, bp_full,
                                  row=1, legend_name='legend',
                                  uni_label='primary', palette=_UNI_PALETTES[0])
                pg_fig.update_yaxes(title_text=_ytitle_l.get(p,'') if isinstance(p,str) else p.get('name',''),
                                    row=1, col=1)
                # Compare panels
                for ri, c in enumerate(_csep, start=2):
                    cleg = f'legend{ri}'
                    if isinstance(p, dict) and p.get('type') == 'expr':
                        _build_expr_panel(pg_fig, p, c['pdata'], c['code'], c['s'],
                                          row=ri, legend_name=cleg, log_fn=log_fn)
                    else:
                        _build_panel3_uni(pg_fig, p, c['s'], c['ba'], c['bb'],
                                          c['ex'], c['ey'], c['ox'], c['oy'],
                                          c['pa'], c['pb'], c['al_a'], c['al_b'], c['bp'],
                                          row=ri, legend_name=cleg,
                                          uni_label=c['label'], palette=c['pal'])
                    pg_fig.update_yaxes(
                        title_text=_ytitle_l.get(p,'') if isinstance(p,str) else p.get('name',''),
                        row=ri, col=1)
                # s-axis label on last row
                pg_fig.update_xaxes(title_text='s (m)' if not normalize_s else 's/s_max',
                                    row=n_grp, col=1)
                if not hasattr(fig, '_compare_figs'): fig._compare_figs = []
                fig._compare_figs.append((f'__panel_{pi}__', pg_fig))

            # ── Beamline bar group ────────────────────────────────────────────
            if include_bar:
                bar_titles = [''] * n_grp  # no per-row titles
                bar_specs  = [[{'secondary_y': False}]] * n_grp
                bar_h_vals = [1.0 / n_grp] * n_grp
                bar_fig    = _make_group_fig(n_grp, bar_titles, bar_specs, bar_h_vals)
                bar_fig.update_layout(height=80 + n_grp * 100,
                                      title=dict(text='Beamline', x=0.5, xanchor='center'))
                _build_layout_bar(bar_fig, elements, show_element_labels, row=1,
                                  show_markers=show_markers_bar)
                bar_fig.update_xaxes(title_text='s (m)' if not normalize_s else 's/s_max',
                                     row=1, col=1)
                bar_fig.update_yaxes(title_text='', showticklabels=False,
                                     range=[-0.4, 0.4], row=1, col=1)
                for ri, c in enumerate(_csep, start=2):
                    _build_layout_bar(bar_fig, c['elems'], show_element_labels, row=ri,
                                      show_markers=show_markers_bar)
                    bar_fig.update_yaxes(title_text='', showticklabels=False,
                                         range=[-0.4, 0.4], row=ri, col=1)
                if not hasattr(fig, '_compare_figs'): fig._compare_figs = []
                fig._compare_figs.append(('__bar__', bar_fig))

        # ── Difference mode: one panel per optics quantity ───────────────────
        if compare_mode in ('difference', 'difference%') and _cmp_datasets:
            _log(f"[compare] Building difference panels for {len(_cmp_datasets)} file(s)...")
            is_pct = compare_mode == 'difference%'

            # Quantities to difference: (array_key, label, unit)
            _DIFF_QUANTITIES = [
                ('beta_a',  'Δβₓ',   'm'   if not is_pct else '%'),
                ('beta_b',  'Δβᵧ',   'm'   if not is_pct else '%'),
                ('eta_x',   'Δηₓ',   'm'   if not is_pct else '%'),
                ('eta_y',   'Δηᵧ',   'm'   if not is_pct else '%'),
                ('alpha_a', 'Δαₓ',   ''    if not is_pct else '%'),
                ('alpha_b', 'Δαᵧ',   ''    if not is_pct else '%'),
                ('orbit_x', 'Δx',    'm'   if not is_pct else '%'),
                ('orbit_y', 'Δy',    'm'   if not is_pct else '%'),
                ('phi_a',   'Δμₓ',   ''    if not is_pct else '%'),
                ('phi_b',   'Δμᵧ',   ''    if not is_pct else '%'),
            ]

            nd = len(_DIFF_QUANTITIES)
            dv_spacing = 0.05
            dn_gaps    = nd - 1
            dpanel_h   = (1.0 - dv_spacing * dn_gaps) / nd
            d_row_heights = [dpanel_h] * nd
            d_titles = []
            for key, lbl, unit in _DIFF_QUANTITIES:
                suffix = ' (%)' if is_pct else (f' ({unit})' if unit else '')
                d_titles.append(f'{lbl}{suffix}')

            dfig = make_subplots(
                rows=nd, cols=1, shared_xaxes=True,
                row_heights=d_row_heights, vertical_spacing=dv_spacing,
                subplot_titles=d_titles if show_titles else [''] * len(d_titles),
                specs=[[{'secondary_y': False}]] * nd,
            )

            _DIFF_PALETTE = ['#0a84ff','#ff453a','#30d158','#ff9f0a',
                             '#5e5ce6','#64d2ff','#ff375f','#ffd60a']

            for ci, cd in enumerate(_cmp_datasets):
                clabel   = cd['label']
                cprimary = cd['plot_unis'][0]
                cpdata   = cd['all_uni'][cprimary]
                cs_cmp   = cpdata['s']
                # Normalize s if requested
                if normalize_s:
                    s_plot  = s / float(s[-1]) if float(s[-1]) > 0 else s
                    sc_plot = cs_cmp / float(cs_cmp[-1]) if float(cs_cmp[-1]) > 0 else cs_cmp
                else:
                    s_plot  = s
                    sc_plot = cs_cmp

                color = _DIFF_PALETTE[ci % len(_DIFF_PALETTE)]
                dleg  = f'legend{ci+1}'

                for row_i, (key, lbl, unit) in enumerate(_DIFF_QUANTITIES, start=1):
                    p_arr = _pdata.get(key, np.zeros_like(s))
                    c_arr = cpdata.get(key, np.zeros_like(cs_cmp))
                    # Interpolate compare onto primary s-grid
                    c_interp = np.interp(s_plot, sc_plot, c_arr)
                    if is_pct:
                        with np.errstate(invalid='ignore', divide='ignore'):
                            diff = np.where(np.abs(p_arr) > 1e-12,
                                            (p_arr - c_interp) / np.abs(p_arr) * 100.0,
                                            np.nan)
                    else:
                        diff = p_arr - c_interp

                    trace_name = f'{lbl} ({clabel})'
                    dfig.add_trace(go.Scatter(
                        x=s_plot, y=diff, mode='lines',
                        name=trace_name, legendgroup=trace_name,
                        line=dict(color=color, width=1.5),
                        hovertemplate=f's=%{{x:.3f}} m<br>{lbl}=%{{y:.6g}}<extra>{clabel}</extra>',
                        legend=dleg,
                    ), row=row_i, col=1)

                    # Zero reference line
                    dfig.add_hline(y=0, line=dict(color='gray', width=0.8, dash='dot'),
                                   row=row_i, col=1)

                    if row_i == nd:
                        dfig.update_xaxes(
                            title_text='s (m)' if not normalize_s else 's/s_max',
                            row=row_i, col=1)
                    dfig.update_yaxes(
                        title_text=f'{lbl} ({unit})' if unit else lbl,
                        row=row_i, col=1)

                # Legend positioning
                dfig_lkw = {}
                dfig_lkw[f'legend{ci+1}'] = dict(
                    x=1.02, xanchor='left', y=1.0, yanchor='top',
                    itemsizing='constant', bgcolor='rgba(0,0,0,0)')
                dfig.update_layout(**dfig_lkw)

            # Height
            dfig_h = 200 + nd * 180
            dfig.update_layout(height=dfig_h, hovermode='closest',
                               title=dict(
                                   text=f'Optics Differences{"  (%)" if is_pct else ""}'
                                        + (f' — {title}' if title else ''),
                                   x=0.5, xanchor='center'))

            if not hasattr(fig, '_compare_figs'):
                fig._compare_figs = []
            fig._compare_figs.append(('Differences', dfig))

    # ── Apply theme + save ────────────────────────────────────────────────────
    _prog(88, 'Applying theme...')
    _apply(fig)

    # Apply global font sizes if specified
    if font_sizes:
        _fs = font_sizes
        ax_lbl  = _fs.get('axis_label', None)
        tick_sz = _fs.get('tick',       None)
        ttl_sz  = _fs.get('title',      None)
        leg_sz  = _fs.get('legend',     None)
        if ax_lbl:
            fig.update_xaxes(title_font=dict(size=ax_lbl))
            fig.update_yaxes(title_font=dict(size=ax_lbl))
        if tick_sz:
            fig.update_xaxes(tickfont=dict(size=tick_sz))
            fig.update_yaxes(tickfont=dict(size=tick_sz))
        if ttl_sz and show_titles:
            fig.update_layout(
                title_font=dict(size=ttl_sz),
                annotations=[dict(a, font=dict(size=ttl_sz))
                              if a.get('text','') and not a.get('showarrow', True)
                                 and a.get('xref','') == 'paper'
                              else a
                              for a in fig.to_dict().get('layout', {}).get('annotations', [])])
        if leg_sz:
            fig.update_layout(legend=dict(font=dict(size=leg_sz)))

    # Apply theme to compare sub-figures too
    if hasattr(fig, '_compare_figs'):
        for _, cfig in fig._compare_figs:
            _apply(cfig)

    _prog(93, 'Writing HTML...')
    if hasattr(fig, '_compare_figs') and fig._compare_figs:
        import plotly.io as pio
        # In separate mode with interleaved groups, skip the primary fig —
        # each group already contains the primary row.
        if compare_mode == 'separate' and layout != 'floor':
            html_parts = []
            first = True
            for clabel, cfig in fig._compare_figs:
                html_parts.append(pio.to_html(cfig, full_html=False,
                                              include_plotlyjs='cdn' if first else False))
                first = False
        else:
            html_parts = [pio.to_html(fig, full_html=False, include_plotlyjs='cdn')]
            for clabel, cfig in fig._compare_figs:
                html_parts.append(pio.to_html(cfig, full_html=False, include_plotlyjs=False))
        combined = (
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{title or "RanOptics — Optics Comparison"}</title>'
            '<style>body{{margin:0;padding:8px;background:#fff;}}'
            '.ran-sep{{border-top:2px solid #ccc;margin:16px 0;}}</style>'
            '</head><body>'
            + '<div class="ran-sep"></div>'.join(html_parts)
            + '</body></html>'
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined)
    else:
        fig.write_html(output_file)
    _prog(100, 'Done')
    _log(f"✓ Saved HTML → {output_file}")
    base = output_file.rsplit('.', 1)[0]
    if save_png:
        import time; fig.write_image('/tmp/warmup.png', width=100, height=100)
        time.sleep(1); pf = base + '.png'
        fig.write_image(pf, width=1600, height=1100, scale=dpi/96)
        time.sleep(1); _log(f"✓ Saved PNG  → {pf}")
    if save_pdf:
        import time; pf = base + '.pdf'
        fig.write_image(pf, width=1600, height=1100)
        time.sleep(1); _log(f"✓ Saved PDF  → {pf}")

    if save_csv:
        import csv as _csv, os as _os
        _csv_dir = _os.path.dirname(output_file) or '.'
        _base = (csv_base or 'lattice').strip()

        # Panel name -> short slug
        def _slug(p):
            if isinstance(p, dict): return p.get('name', 'custom').lower().replace(' ', '_')
            return {'twiss':'twiss','beta':'beta','dispersion':'dispersion',
                    'alpha':'alpha','orbit':'orbit','phase':'phase',
                    'beamsize':'beamsize','summary':'summary','latdiff':'latdiff'}.get(p, str(p))

        # Columns per panel type
        _PANEL_COLS = {
            'twiss':      ['s','beta_a','beta_b','eta_x','eta_y','alpha_a','alpha_b'],
            'beta':       ['s','beta_a','beta_b'],
            'dispersion': ['s','eta_x','eta_y'],
            'alpha':      ['s','alpha_a','alpha_b'],
            'orbit':      ['s','orbit_x','orbit_y'],
            'phase':      ['s','phi_a','phi_b'],
            'beamsize':   ['s','beta_a','beta_b','eta_x','eta_y'],
        }
        _COL_LABELS = {
            's':'s(m)','beta_a':'betx(m)','beta_b':'bety(m)',
            'eta_x':'etax(m)','eta_y':'etay(m)',
            'alpha_a':'alfx','alpha_b':'alfy',
            'orbit_x':'x(m)','orbit_y':'y(m)',
            'phi_a':'mux','phi_b':'muy',
        }

        for p in panels:
            slug = _slug(p)
            if p == 'bar': continue  # no tabular data

            # ── latdiff: write 3 CSVs ──────────────────────────────────
            if p == 'latdiff' and _cmp_datasets:
                _puid  = _plot_unis[0]
                _ea    = _all_uni_data[_puid]['elements']
                _la    = _uni_labels.get(_puid, 'primary')
                _cd    = _cmp_datasets[0]
                _eb    = _cd['all_uni'][_cd['plot_unis'][0]]['elements']
                _lb    = _cd['label']
                _PHY   = {'sbend','quadrupole','sextupole','rfcavity','lcavity'}
                _ma    = [e for e in _ea if e['key'].lower() in _PHY]
                _mb    = [e for e in _eb if e['key'].lower() in _PHY]
                if len(_ma) == len(_mb):
                    # Strengths
                    _fp = _os.path.join(_csv_dir, f"{_base}-latdiff-strengths.csv")
                    with open(_fp, 'w', newline='') as f:
                        w = _csv.writer(f)
                        w.writerow(['#','name','type',
                                    f'L_{_la}',f'L_{_lb}','dL',
                                    f'k1_{_la}',f'k1_{_lb}','dk1',
                                    f'k2_{_la}',f'k2_{_lb}','dk2'])
                        for i,(ea,eb) in enumerate(zip(_ma,_mb)):
                            w.writerow([i+1, ea['name'], ea['key'],
                                        f"{ea['length']:.6f}", f"{eb['length']:.6f}",
                                        f"{eb['length']-ea['length']:.6f}",
                                        f"{ea.get('k1',0):.6f}", f"{eb.get('k1',0):.6f}",
                                        f"{eb.get('k1',0)-ea.get('k1',0):.6f}",
                                        f"{ea.get('k2',0):.6f}", f"{eb.get('k2',0):.6f}",
                                        f"{eb.get('k2',0)-ea.get('k2',0):.6f}"])
                    _log(f"✓ CSV → {_fp}")
                    # Entry/Exit positions
                    for suffix, k0, k1 in [('entry','flr_','0'), ('exit','flr_','1')]:
                        _fp = _os.path.join(_csv_dir, f"{_base}-latdiff-{suffix}.csv")
                        with open(_fp, 'w', newline='') as f:
                            w = _csv.writer(f)
                            w.writerow(['#','name','type',
                                        f'X_{suffix}_{_la}',f'X_{suffix}_{_lb}',f'dX_{suffix}',
                                        f'Y_{suffix}_{_la}',f'Y_{suffix}_{_lb}',f'dY_{suffix}',
                                        f'Z_{suffix}_{_la}',f'Z_{suffix}_{_lb}',f'dZ_{suffix}'])
                            for i,(ea,eb) in enumerate(zip(_ma,_mb)):
                                n = k1
                                xa,xb = ea.get(f'flr_x{n}'), eb.get(f'flr_x{n}')
                                ya,yb = ea.get(f'flr_y{n}'), eb.get(f'flr_y{n}')
                                za,zb = ea.get(f'flr_z{n}'), eb.get(f'flr_z{n}')
                                fmt = lambda v: f'{v:.6f}' if v is not None else ''
                                dfmt = lambda a,b: f'{b-a:.6f}' if a is not None and b is not None else ''
                                w.writerow([i+1, ea['name'], ea['key'],
                                            fmt(xa),fmt(xb),dfmt(xa,xb),
                                            fmt(ya),fmt(yb),dfmt(ya,yb),
                                            fmt(za),fmt(zb),dfmt(za,zb)])
                        _log(f"✓ CSV → {_fp}")
                continue

            # ── summary: write one CSV per universe ───────────────────
            if p == 'summary':
                for _uid in _plot_unis:
                    _ulbl = _uni_labels.get(_uid, f'u{_uid}')
                    _ud   = _all_uni_data[_uid]
                    _bp   = _ud.get('beam_params', {})
                    _fp   = _os.path.join(_csv_dir, f"{_base}-summary-{_ulbl}.csv")
                    with open(_fp, 'w', newline='') as f:
                        w = _csv.writer(f)
                        w.writerow(['quantity','value'])
                        for k,v in _bp.items():
                            if v is not None: w.writerow([k, f'{v:.6f}' if isinstance(v,float) else v])
                    _log(f"✓ CSV → {_fp}")
                continue

            # ── data panels: twiss, orbit, dispersion etc ─────────────
            slug = _slug(p)
            cols = _PANEL_COLS.get(slug, None)
            if cols is None: continue

            for _uid in _plot_unis:
                _ulbl = _uni_labels.get(_uid, f'u{_uid}')
                _ud   = _all_uni_data[_uid]
                suffix = f'-{_ulbl}' if _multi else ''
                _fp = _os.path.join(_csv_dir, f"{_base}-{slug}{suffix}.csv")
                with open(_fp, 'w', newline='') as f:
                    w = _csv.writer(f)
                    w.writerow([_COL_LABELS.get(c, c) for c in cols])
                    _arr = lambda k: _ud.get(k, np.array([]))
                    n_pts = len(_arr('s'))
                    for i in range(n_pts):
                        w.writerow([f"{_arr(c)[i]:.6e}" if i < len(_arr(c)) else '' for c in cols])
                _log(f"✓ CSV → {_fp}")
    if show: fig.show()
    return fig

# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PySide6 GUI
# ════════════════════════════════════════════════════════════════════════════

import threading
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QLineEdit, QCheckBox, QComboBox, QTabWidget, QScrollArea, QPlainTextEdit,
    QTextEdit, QProgressBar, QMenuBar, QMenu, QFileDialog, QMessageBox,
    QInputDialog, QHBoxLayout, QVBoxLayout, QGridLayout, QButtonGroup,
    QRadioButton, QSizePolicy, QSplitter, QDialog, QDialogButtonBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize, QObject
from PySide6.QtGui import (
    QFont, QColor, QPalette, QAction, QPainter, QPen, QBrush, QFontDatabase,
    QTextCharFormat,
)

# ── RanOptics Color Palette ───────────────────────────────────────────────────
# To retheme: edit only this block. All UI colors derive from these constants.
# Background family (dark teal, deepest to lightest)
BG       = "#2C5446"   # Main background — deep dark teal
MANTLE   = "#234038"   # Deeper background (input fields, log area)
CRUST    = "#1a2f28"   # Darkest (menubar, status bar)
# Surface family (card/panel surfaces)
PANEL    = "#3D6B5C"   # Card / panel surface
SURFACE2 = "#4A7D6C"   # Hover state
BORDER   = "#5A8A78"   # Borders and dividers
# Text family
FG       = "#EEF5F2"   # Primary text (light on dark bg)
FG_DIM   = "#A8C4BC"   # Dimmed / hint text
FG_LBL   = "#8AB0A6"   # Label text
# Accents
ACCENT   = "#FDA769"   # Peach — UI highlights, borders, active states
RAN_CLR  = "#00e676"   # Bright green — RanOptics "Ran" logo
ERROR    = "#d62828"   # Bright red   — RanOptics "Optics" + symbol (DO NOT CHANGE)
ACCENT2  = "#FDA769"   # Peach / orange accent
WARN     = "#FEC868"   # Warm yellow
SUCCESS  = "#00e676"   # Bright green — success state / Open Plot button
TEAL     = "#FEC868"   # Info (reuse warm yellow)
PEACH    = "#FDA769"   # Peach (reuse accent2)

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT_MAIN  = QFont(); FONT_MAIN.setPointSize(11)
FONT_BOLD  = QFont(); FONT_BOLD.setPointSize(11);  FONT_BOLD.setBold(True)
FONT_SMALL = QFont(); FONT_SMALL.setPointSize(11)
FONT_MONO  = QFont("Monospace"); FONT_MONO.setPointSize(10)
FONT_HDR   = QFont("Monospace"); FONT_HDR.setPointSize(16);  FONT_HDR.setBold(True)
FONT_SEC   = QFont(); FONT_SEC.setPointSize(11);   FONT_SEC.setBold(True)

# ── Stylesheet helpers — Catppuccin Mocha ────────────────────────────────────
_ENTRY_SS = f"""
    QLineEdit {{
        background: {MANTLE}; border: 1px solid {BORDER};
        border-radius: 8px; color: {FG}; padding: 4px 10px;
        selection-background-color: {ACCENT}; selection-color: {CRUST};
    }}
    QLineEdit:focus {{
        border-color: {ACCENT};
        border-left: 3px solid {ACCENT};
        background: {BG};
    }}
    QLineEdit[readOnly="true"] {{ color: {FG_DIM}; background: {PANEL}; }}
"""
_COMBO_SS = f"""
    QComboBox {{
        background: {MANTLE}; border: 1px solid {BORDER};
        border-radius: 8px; color: {FG}; padding: 4px 10px;
    }}
    QComboBox:focus {{ border-color: {ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox::down-arrow {{ width: 0; height: 0; }}
    QComboBox QAbstractItemView {{
        background: {PANEL}; color: {FG}; border: 1px solid {BORDER};
        border-radius: 6px; padding: 2px;
        selection-background-color: {ACCENT}; selection-color: {CRUST};
        outline: none;
    }}
"""
_BTN_SS = f"""
    QPushButton {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 8px; color: {ACCENT}; padding: 4px 10px;
        font-weight: 500;
    }}
    QPushButton:hover  {{
        background: {SURFACE2}; border-color: {ACCENT}; color: {ACCENT};
    }}
    QPushButton:pressed {{ background: {BORDER}; }}
    QPushButton:disabled {{ color: {FG_DIM}; border-color: {BORDER}; background: {PANEL}; }}
"""
_CHK_SS = f"""
    QCheckBox {{ color: {FG}; spacing: 7px; }}
    QCheckBox::indicator {{
        width: 15px; height: 15px; border-radius: 4px;
        border: 1px solid {SURFACE2}; background: {MANTLE};
    }}
    QCheckBox::indicator:unchecked:hover {{ border-color: {ACCENT}; }}
    QCheckBox::indicator:checked {{
        background: {ACCENT}; border-color: {ACCENT};
        image: none;
    }}
"""
_RB_SS = f"""
    QRadioButton {{ color: {FG}; spacing: 7px; }}
    QRadioButton::indicator {{
        width: 14px; height: 14px; border-radius: 7px;
        border: 1px solid {SURFACE2}; background: {MANTLE};
    }}
    QRadioButton::indicator:checked {{
        background: {ACCENT}; border-color: {ACCENT};
        border-width: 3px;
    }}
"""
_TAB_SS = f"""
    QTabWidget::pane {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 10px; top: -1px;
    }}
    QTabBar::tab {{
        background: {MANTLE}; color: {FG_LBL}; padding: 7px 20px;
        border: 1px solid {BORDER}; border-bottom: none; margin-right: 3px;
        border-top-left-radius: 8px; border-top-right-radius: 8px;
        font-weight: 500;
    }}
    QTabBar::tab:selected {{
        background: {PANEL}; color: {ACCENT};
        border-bottom-color: {PANEL};
    }}
    QTabBar::tab:hover:!selected {{ background: {SURFACE2}; color: {FG}; }}
"""
_SCROLL_SS = f"""
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: {MANTLE}; width: 6px; margin: 0; border-radius: 3px;
    }}
    QScrollBar::handle:vertical {{
        background: {SURFACE2}; border-radius: 3px; min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""

# ── GUI helper functions ──────────────────────────────────────────────────────

def _make_scroll_widget(parent=None):
    """Returns (scroll_area, inner_widget, vbox_layout)."""
    sa = QScrollArea(parent)
    sa.setWidgetResizable(True)
    sa.setStyleSheet(_SCROLL_SS)
    inner = QWidget()
    inner.setStyleSheet(f"background: transparent;")
    vbox = QVBoxLayout(inner)
    vbox.setContentsMargins(0, 4, 0, 8)
    vbox.setSpacing(0)
    sa.setWidget(inner)
    return sa, inner, vbox

def _sec(layout, title):
    """Section header: pill label + horizontal rule."""
    w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(8, 8, 8, 2); h.setSpacing(8)
    lbl = QLabel(f"  {title.upper()}  "); lbl.setFont(FONT_SEC)
    lbl.setStyleSheet(f"""
        color: {CRUST}; background: {ACCENT2};
        border-radius: 4px; padding: 1px 4px;
    """)
    line = QFrame(); line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {BORDER}; background: {BORDER};")
    h.addWidget(lbl); h.addWidget(line, 1)
    layout.addWidget(w)

def _card(layout):
    """Transparent container widget. Returns (widget, vbox)."""
    w = QWidget(); w.setStyleSheet("background: transparent;")
    v = QVBoxLayout(w); v.setContentsMargins(0, 2, 0, 2); v.setSpacing(0)
    layout.addWidget(w)
    return w, v

def _row(layout):
    """Horizontal row. Returns QHBoxLayout added to *layout*."""
    w = QWidget(); w.setStyleSheet("background: transparent;")
    h = QHBoxLayout(w); h.setContentsMargins(8, 2, 8, 2); h.setSpacing(6)
    h.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.addWidget(w)
    return h

def _lbl(layout, text, width=160):
    lbl = QLabel(text); lbl.setFont(FONT_MAIN)
    lbl.setStyleSheet(f"color: {FG_LBL}; background: transparent;")
    lbl.setFixedWidth(width); lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.addWidget(lbl)

def _ent(layout, width=200, placeholder=""):
    e = QLineEdit(); e.setFont(FONT_MONO)
    e.setPlaceholderText(placeholder); e.setFixedWidth(width)
    e.setStyleSheet(_ENTRY_SS)
    layout.addWidget(e); return e

def _btn(layout, text, cmd, width=80, color=ACCENT):
    b = QPushButton(text); b.setFont(FONT_MAIN); b.setFixedWidth(width)
    b.clicked.connect(cmd)
    b.setStyleSheet(f"""
        QPushButton {{
            background: {PANEL}; border: 1px solid {BORDER};
            border-radius: 8px; color: {color}; padding: 4px 8px;
        }}
        QPushButton:hover {{ background: {SURFACE2}; border-color: {color}; }}
        QPushButton:pressed {{ background: {BORDER}; }}
        QPushButton:disabled {{ color: {FG_DIM}; border-color: {BORDER}; background: {PANEL}; }}
    """)
    layout.addWidget(b); return b

def _chk(layout, text):
    c = QCheckBox(text); c.setFont(FONT_MAIN); c.setStyleSheet(_CHK_SS)
    layout.addWidget(c); return c

def _dd(layout, items, width=120):
    cb = QComboBox(); cb.setFont(FONT_MAIN); cb.addItems(items)
    cb.setFixedWidth(width); cb.setStyleSheet(_COMBO_SS)
    layout.addWidget(cb); return cb

def _hint(layout, text):
    lbl = QLabel(text); lbl.setFont(FONT_SMALL)
    lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
    layout.addWidget(lbl)

def _help(layout, text):
    """Dimmed description line."""
    lbl = QLabel(text); lbl.setFont(FONT_SMALL); lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 0 12px 4px 12px;")
    layout.addWidget(lbl)

def _rb(layout, text, group, val, cmd=None):
    r = QRadioButton(text); r.setFont(FONT_MAIN); r.setStyleSheet(_RB_SS)
    r.setProperty("value", val)
    if cmd: r.clicked.connect(cmd)
    group.addButton(r); layout.addWidget(r); return r

def _clf(line):
    lo = line.lower()
    if any(w in lo for w in ("error", "traceback", "exception", "failed", "✗")): return "error"
    if any(w in lo for w in ("warning", "warn")): return "warn"
    if any(w in lo for w in ("saved", "done", "complete", "✓")): return "ok"
    return "info"

# ── Worker thread ─────────────────────────────────────────────────────────────

class _WorkerThread(QThread):
    log_signal      = Signal(str, str)   # (text, tag)
    progress_signal = Signal(int, str)   # (pct, label)
    done_signal     = Signal(str)        # output_path
    error_signal    = Signal(str)        # traceback

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        self._fn()

# ── FODO logo widget ──────────────────────────────────────────────────────────

# ── Tao data browser categories ───────────────────────────────────────────────
_TAO_DATA_CATEGORIES = {
    "Twiss — Normal Mode": [
        ("alpha.a",    "Normal-mode alpha function"),
        ("alpha.b",    "Normal-mode alpha function"),
        ("beta.a",     "Normal-mode beta function (m)"),
        ("beta.b",     "Normal-mode beta function (m)"),
        ("beta.c",     "Normal-mode beta function (m)"),
        ("gamma.a",    "Normal-mode gamma function (1/m)"),
        ("gamma.b",    "Normal-mode gamma function (1/m)"),
        ("emit.a",     "Normal-mode emittance (m·rad)"),
        ("emit.b",     "Normal-mode emittance (m·rad)"),
        ("emit.c",     "Normal-mode emittance (m·rad)"),
        ("eta.a",      "Normal-mode dispersion (m)"),
        ("eta.b",      "Normal-mode dispersion (m)"),
        ("etap.a",     "Normal-mode momentum dispersion"),
        ("etap.b",     "Normal-mode momentum dispersion"),
    ],
    "Dispersion": [
        ("eta.x",      "Horizontal dispersion (m)"),
        ("eta.y",      "Vertical dispersion (m)"),
        ("etap.x",     "Horizontal momentum dispersion"),
        ("etap.y",     "Vertical momentum dispersion"),
        ("deta_ds.a",  "Dispersion derivative a-mode"),
        ("deta_ds.b",  "Dispersion derivative b-mode"),
        ("deta_ds.x",  "Horizontal dispersion derivative"),
        ("deta_ds.y",  "Vertical dispersion derivative"),
    ],
    "Orbit": [
        ("orbit.x",    "Horizontal phase-space orbit (m)"),
        ("orbit.y",    "Vertical phase-space orbit (m)"),
        ("orbit.z",    "Longitudinal phase-space orbit (m)"),
        ("orbit.px",   "Horizontal canonical momentum"),
        ("orbit.py",   "Vertical canonical momentum"),
        ("orbit.pz",   "Longitudinal canonical momentum"),
    ],
    "Phase Advance": [
        ("phi.a",      "Phase advance a-mode (2π units)"),
        ("phi.b",      "Phase advance b-mode (2π units)"),
    ],
    "Courant-Snyder": [
        ("alpha_a",    "Courant-Snyder alpha x"),
        ("alpha_b",    "Courant-Snyder alpha y"),
        ("beta_a",     "Courant-Snyder beta x (m)"),
        ("beta_b",     "Courant-Snyder beta y (m)"),
        ("gamma_a",    "Courant-Snyder gamma x (1/m)"),
        ("gamma_b",    "Courant-Snyder gamma y (1/m)"),
    ],
    "Beam Size / Emittance": [
        ("beam_energy",     "Beam energy (eV)"),
        ("e_tot",           "Total energy (eV)"),
        ("p0c",             "Reference momentum × c (eV)"),
        ("s",               "Longitudinal position (m)"),
        ("ref_time",        "Reference time (s)"),
    ],
    "W Function / Chromatic": [
        ("chrom.w.a",       "W function a-mode"),
        ("chrom.w.b",       "W function b-mode"),
        ("chrom.dw.a",      "dW/ds a-mode"),
        ("chrom.dw.b",      "dW/ds b-mode"),
    ],
    "Element Attributes": [
        ("k1",         "Quadrupole strength (1/m²)"),
        ("k2",         "Sextupole strength (1/m³)"),
        ("k3",         "Octupole strength (1/m⁴)"),
        ("angle",      "Bend angle (rad)"),
        ("l",          "Element length (m)"),
        ("tilt",       "Tilt angle (rad)"),
        ("x_offset",   "Horizontal offset (m)"),
        ("y_offset",   "Vertical offset (m)"),
        ("voltage",    "RF voltage (V)"),
        ("phi0",       "RF phase (rad)"),
    ],
}

# ── ELEGANT data browser constants ────────────────────────────────────────────
_ELEGANT_TWI_COLUMNS = [
    ("s",           "Longitudinal position (m)"),
    ("betax",       "Horizontal beta function (m)"),
    ("betay",       "Vertical beta function (m)"),
    ("alphax",      "Horizontal alpha function"),
    ("alphay",      "Vertical alpha function"),
    ("etax",        "Horizontal dispersion (m)"),
    ("etay",        "Vertical dispersion (m)"),
    ("etaxp",       "Horizontal dispersion prime"),
    ("etayp",       "Vertical dispersion prime"),
    ("psix",        "Horizontal phase advance (rad)"),
    ("psiy",        "Vertical phase advance (rad)"),
    ("x",           "Centroid x (m)"),
    ("y",           "Centroid y (m)"),
    ("xp",          "Centroid x' (rad)"),
    ("yp",          "Centroid y' (rad)"),
    ("Sx",          "RMS beam size x (m)"),
    ("Sy",          "RMS beam size y (m)"),
]

_ELEGANT_CEN_COLUMNS = [
    ("s",    "Longitudinal position (m)"),
    ("x",    "Centroid x (m)"),
    ("xp",   "Centroid x' (rad)"),
    ("y",    "Centroid y (m)"),
    ("yp",   "Centroid y' (rad)"),
    ("t",    "Time (s)"),
    ("p",    "Momentum deviation"),
]

_ELEGANT_SIG_COLUMNS = [
    ("s",    "Longitudinal position (m)"),
    ("Sx",   "RMS x (m)"),
    ("Sy",   "RMS y (m)"),
    ("Ss",   "RMS s (m)"),
    ("Sxp",  "RMS x' (rad)"),
    ("Syp",  "RMS y' (rad)"),
    ("Sdelta", "RMS δp/p"),
    ("ex",   "Emittance x (m·rad)"),
    ("ey",   "Emittance y (m·rad)"),
]

_ELEGANT_TWI_SCALARS = [
    ("pCentral",   "Central momentum (m_e c)"),
    ("Ex",         "Horizontal emittance (m·rad)"),
    ("Ey",         "Vertical emittance (m·rad)"),
    ("Sdelta0",    "Energy spread δp/p"),
    ("nux",        "Horizontal tune"),
    ("nuy",        "Vertical tune"),
    ("dnux/dp",    "Horizontal chromaticity"),
    ("dnuy/dp",    "Vertical chromaticity"),
]


class _FodoLogo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(108, 64)

    def paintEvent(self, event):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = 108, 64
        sx = W / 128.0; sy = H / 88.0

        def lx(x): return (x - 28) * sx
        def ly(y): return H - (y - 40) * sy

        from PySide6.QtCore import QPointF
        # betax curve
        pts_x = [QPointF(lx(32 + i / 80 * 104), ly(67 - 20 * math.cos(math.pi * 2 * i / 80)))
                 for i in range(81)]
        pen = QPen(QColor(RAN_CLR), 2); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        for i in range(len(pts_x) - 1): p.drawLine(pts_x[i], pts_x[i + 1])
        # betay curve
        pts_y = [QPointF(lx(32 + i / 80 * 104), ly(67 + 20 * math.cos(math.pi * 2 * i / 80)))
                 for i in range(81)]
        pen2 = QPen(QColor(ERROR), 2); pen2.setDashPattern([4, 2]); pen2.setCapStyle(Qt.RoundCap)
        p.setPen(pen2)
        for i in range(len(pts_y) - 1): p.drawLine(pts_y[i], pts_y[i + 1])
        # Element bar
        ey = ly(106); eh = max(3, int(9 * sy))
        from PySide6.QtCore import QRectF
        for ex_, ew, col, lbl_text in [
            (30, 18, RAN_CLR, 'F'), (48, 30, None, None),
            (78, 18, ERROR, 'D'), (96, 30, None, None), (126, 18, RAN_CLR, 'F')
        ]:
            x1 = lx(ex_); x2 = lx(ex_ + ew)
            if col is None:
                p.setPen(QPen(QColor('#aaaaaa'), 1.2))
                my = ey - eh / 2
                p.drawLine(QPointF(x1, my), QPointF(x2, my))
            else:
                p.fillRect(int(x1), int(ey - eh), int(x2 - x1), int(eh), QColor(col))
                if lbl_text:
                    f = QFont("Monospace"); f.setPointSize(6); f.setBold(True)
                    p.setFont(f); p.setPen(QPen(QColor("white")))
                    p.drawText(QRectF(x1, ey - eh, x2 - x1, eh), Qt.AlignCenter, lbl_text)

# ── Main GUI class ────────────────────────────────────────────────────────────

class LuxV4GUI(QMainWindow):
    # Signals for cross-thread communication — emitting a Signal is always thread-safe
    _sig_log      = Signal(str, str)   # (text, tag)
    _sig_progress = Signal(int, str)   # (pct, label)
    _sig_done     = Signal(str)        # output_path
    _sig_failed   = Signal(str)        # traceback text
    _sig_finally  = Signal()           # always fires at end of run

    _PRESET_PANELS = [
        ('Twiss & Dispersion', 'twiss'), ('Beta Functions', 'beta'),
        ('Dispersion', 'dispersion'), ('Alpha Functions', 'alpha'),
        ('Orbit', 'orbit'), ('Phase Advance', 'phase'), ('Beam Size', 'beamsize'),
        ('Lattice Summary', 'summary'),
        ('Lattice Diff',    'latdiff'),
        ('Beamline Bar',    'bar'),
        ('Floor Plan X-Z',  'floor-xz'),
        ('Floor Plan Y-Z',  'floor-yz'),
    ]
    _RECENT_FILE = Path.home() / ".ranoptics_recent.json"
    _PRESET_FILE = Path.home() / ".ranoptics_presets.json"
    _MAX_RECENT  = 8

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RanOptics — Accelerator Optics Plotter")
        self.resize(1280, 960); self.setMinimumSize(1000, 800)
        self.setStyleSheet(f"""
            QMainWindow {{ background: {BG}; }}
            QWidget {{ background: {BG}; }}
            QToolTip {{
                background: {PANEL}; color: {FG};
                border: 1px solid {BORDER}; border-radius: 6px;
                padding: 4px 8px;
            }}
        """)

        self._last_output = None
        self._worker      = None
        self._uni_checks  = {}   # {i: QCheckBox}
        self._uni_label_edits = {}  # {i: QLineEdit}
        self._uni_n       = 1
        self._panels      = [{'name': 'Floor Plan X-Z',      'spec': 'floor-xz'},
                              {'name': 'Twiss & Dispersion', 'spec': 'twiss'},
                              {'name': 'Beamline Bar',       'spec': 'bar'}]
        self._panel_rows  = []

        central = QWidget()
        self.setCentralWidget(central)
        self._root_layout = QVBoxLayout(central)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        self._build_menubar()
        self._build_header()
        self._build_form()
        self._build_run_bar()
        self._build_log()
        self._build_statusbar()

        # Wire cross-thread signals to GUI slots
        self._sig_log.connect(self._log)
        self._sig_progress.connect(self._set_progress)
        self._sig_done.connect(self._on_run_done)
        self._sig_failed.connect(self._on_run_failed)
        self._sig_finally.connect(self._on_run_finally)

        self._refresh_recent_menu()
        self._refresh_preset_menu()

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menubar(self):
        mb = self.menuBar()
        mb.setStyleSheet(f"""
            QMenuBar {{
                background: {CRUST}; color: {FG_LBL};
                border-bottom: 1px solid {BORDER};
            }}
            QMenuBar::item {{ padding: 4px 10px; border-radius: 4px; }}
            QMenuBar::item:selected {{ background: {SURFACE2}; color: {FG}; }}
            QMenu {{
                background: {MANTLE}; color: {FG};
                border: 1px solid {BORDER}; border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{ padding: 5px 20px; border-radius: 4px; }}
            QMenu::item:selected {{ background: {PANEL}; color: {ACCENT}; }}
            QMenu::separator {{ background: {BORDER}; height: 1px; margin: 4px 8px; }}
        """)

        # File menu
        fm = mb.addMenu("File")
        fm.addAction(QAction("Browse Input…",    self, triggered=self._browse_input))
        fm.addAction(QAction("Save Output As…",  self, triggered=self._browse_output))
        fm.addSeparator()
        self._recent_menu = fm.addMenu("Recent Files")
        fm.addSeparator()
        fm.addAction(QAction("Export CSV…",      self, triggered=self._export_csv))
        fm.addAction(QAction("Copy Output Path", self, triggered=self._copy_path))

        # Presets menu
        pm = mb.addMenu("Presets")
        pm.addAction(QAction("Save Current as Preset…", self, triggered=self._preset_save_dialog))
        pm.addSeparator()
        self._preset_menu = pm.addMenu("Load Preset")
        pm.addAction(QAction("Delete a preset…", self, triggered=self._preset_delete_dialog))

        # Run menu
        rm = mb.addMenu("Run")
        rm.addAction(QAction("▶ Run",       self, triggered=self._run))
        rm.addAction(QAction("🔍 Dry Run",  self, triggered=self._dry_run))
        rm.addAction(QAction("■ Cancel",    self, triggered=self._cancel))

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        h = QWidget(); h.setFixedHeight(64)
        h.setStyleSheet(f"background: {MANTLE}; border-bottom: 2px solid {BORDER};")
        row = QHBoxLayout(h); row.setContentsMargins(16, 0, 20, 0); row.setSpacing(8)

        row.addWidget(_FodoLogo())

        txt = QWidget(); txt.setStyleSheet(f"background: transparent;")
        tv  = QVBoxLayout(txt); tv.setContentsMargins(6, 8, 0, 8); tv.setSpacing(2)
        name_row = QWidget(); name_row.setStyleSheet("background: transparent;")
        nr = QHBoxLayout(name_row); nr.setContentsMargins(0,0,0,0); nr.setSpacing(4)
        ran = QLabel("Ran"); ran.setFont(FONT_HDR); ran.setStyleSheet(f"color: {RAN_CLR}; background: transparent; letter-spacing: 2px;")
        opt = QLabel("Optics"); opt.setFont(FONT_HDR); opt.setStyleSheet(f"color: {ERROR}; background: transparent; letter-spacing: 2px;")
        nr.addWidget(ran); nr.addWidget(opt); nr.addStretch()
        tv.addWidget(name_row)
        sub = QLabel("Accelerator Optics Plotter"); sub.setFont(FONT_SMALL)
        sub.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
        tv.addWidget(sub)
        row.addWidget(txt)
        row.addStretch()

        rf = QWidget(); rf.setStyleSheet("background: transparent;")
        rv = QVBoxLayout(rf); rv.setContentsMargins(0,0,0,0); rv.setSpacing(2)
        for t in ("Author: Randika Gamage (randika@jlab.org)", "Support: It worked on my machine."):
            l = QLabel(t); l.setFont(FONT_SMALL); l.setAlignment(Qt.AlignLeft)
            l.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
            rv.addWidget(l)
        row.addWidget(rf)

        self._root_layout.addWidget(h)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = QWidget(); sb.setFixedHeight(28)
        sb.setStyleSheet(f"background: {MANTLE}; border-top: 1px solid {BORDER};")
        row = QHBoxLayout(sb); row.setContentsMargins(12, 0, 8, 0); row.setSpacing(8)
        self._status_lbl = QLabel("Idle"); self._status_lbl.setFont(FONT_SMALL)
        self._status_lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
        row.addWidget(self._status_lbl); row.addStretch()
        self._progress = QProgressBar(); self._progress.setFixedWidth(180)
        self._progress.setFixedHeight(6); self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {CRUST}; border-radius: 3px; border: none;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT}, stop:1 {TEAL});
                border-radius: 3px;
            }}
        """)
        self._pct_lbl = QLabel(""); self._pct_lbl.setFont(FONT_BOLD)
        self._pct_lbl.setFixedWidth(40)
        self._pct_lbl.setStyleSheet(f"color: {ACCENT}; background: transparent;")
        row.addWidget(self._progress); row.addWidget(self._pct_lbl)
        self._root_layout.addWidget(sb)

    def _set_status(self, text):
        self._status_lbl.setText(text)

    # ── Run bar ───────────────────────────────────────────────────────────────

    def _build_run_bar(self):
        bar = QWidget(); bar.setFixedHeight(52)
        bar.setStyleSheet(f"background: {MANTLE}; border-top: 1px solid {BORDER};")
        row = QHBoxLayout(bar); row.setContentsMargins(12, 6, 12, 6); row.setSpacing(6)

        self.run_btn = QPushButton("▶  Run"); self.run_btn.setFont(FONT_BOLD)
        self.run_btn.setFixedSize(100, 36); self.run_btn.clicked.connect(self._run)
        self.run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT}; border-radius: 8px;
                color: {CRUST}; font-weight: bold; border: none;
            }}
            QPushButton:hover {{ background: {TEAL}; color: {CRUST}; }}
            QPushButton:disabled {{ background: {BORDER}; color: {FG_DIM}; border: none; }}
        """)
        row.addWidget(self.run_btn)

        def _action_btn(text, cmd, color, width=100):
            b = QPushButton(text); b.setFont(FONT_BOLD)
            b.setFixedSize(width, 36); b.clicked.connect(cmd)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {PANEL}; border: 1px solid {color};
                    border-radius: 8px; color: {color}; font-weight: 500;
                }}
                QPushButton:hover {{ background: {color}; color: {CRUST}; }}
                QPushButton:disabled {{ color: {FG_DIM}; border-color: {BORDER}; background: {PANEL}; }}
            """)
            row.addWidget(b); return b

        self.stop_btn    = _action_btn("■  Cancel",    self._cancel,   ERROR)
        self.open_btn    = _action_btn("🌐  Open Plot", self._open_plot, SUCCESS, 130)
        self.dryrun_btn  = _action_btn("🔍  Dry Run",   self._dry_run,  ACCENT2, 115)
        self.stop_btn.setEnabled(False); self.open_btn.setEnabled(False)

        self.csv_btn = QPushButton("💾  Export CSV"); self.csv_btn.setFont(FONT_MAIN)
        self.csv_btn.setFixedSize(130, 36); self.csv_btn.clicked.connect(self._export_csv)
        self.csv_btn.setStyleSheet(f"""
            QPushButton {{
                background: {MANTLE}; border: 1px solid {BORDER};
                border-radius: 8px; color: {FG_LBL};
            }}
            QPushButton:hover {{ background: {SURFACE2}; color: {FG}; border-color: {ACCENT2}; }}
        """)
        row.addWidget(self.csv_btn)
        row.addStretch()

        clr = QPushButton("⊗  Clear log"); clr.setFont(FONT_MAIN)
        clr.setFixedSize(115, 36); clr.clicked.connect(self._clear_log)
        clr.setStyleSheet(self.csv_btn.styleSheet())
        row.addWidget(clr)

        self._root_layout.addWidget(bar)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _build_log(self):
        lf = QWidget(); lf.setStyleSheet(f"background: {BG};")
        lv = QVBoxLayout(lf); lv.setContentsMargins(12, 4, 12, 4); lv.setSpacing(2)
        hdr = QLabel("OUTPUT LOG"); hdr.setFont(FONT_SEC)
        hdr.setStyleSheet(f"color: {ACCENT2}; background: transparent;")
        lv.addWidget(hdr)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(FONT_MONO); self.log.setFixedHeight(140)
        self.log.setStyleSheet(f"""
            QTextEdit {{
                background: {MANTLE}; color: {FG};
                border: 1px solid {BORDER}; border-radius: 8px;
                padding: 6px; selection-background-color: {ACCENT};
            }}
        """)
        lv.addWidget(self.log)
        self._root_layout.addWidget(lf)
        self._log("Ready. Configure options above and click ▶ Run.\n", "dim")

    # ── Form (tabs) ───────────────────────────────────────────────────────────

    def _build_form(self):
        outer = QWidget(); outer.setStyleSheet(f"background: {BG};")
        outer_h = QHBoxLayout(outer); outer_h.setContentsMargins(8, 4, 8, 0); outer_h.setSpacing(8)

        self._tab_l = QTabWidget(); self._tab_l.setStyleSheet(_TAB_SS); self._tab_l.setFont(FONT_SEC)
        self._tab_r = QTabWidget(); self._tab_r.setStyleSheet(_TAB_SS); self._tab_r.setFont(FONT_SEC)

        for name in ("Input", "Beam Settings"):   self._tab_l.addTab(QWidget(), name)
        for name in ("Panels", "Visual", "Export"): self._tab_r.addTab(QWidget(), name)

        # Wrap each tab's QWidget in a scroll area
        def _scroll_tab(tab_widget, idx):
            w = tab_widget.widget(idx)
            sa, inner, vbox = _make_scroll_widget()
            vbox.addStretch()
            layout = QVBoxLayout(w); layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(sa)
            return vbox

        self._input_layout  = _scroll_tab(self._tab_l, 0)
        self._beam_layout   = _scroll_tab(self._tab_l, 1)
        self._panels_layout = _scroll_tab(self._tab_r, 0)
        self._visual_layout = _scroll_tab(self._tab_r, 1)
        self._export_layout = _scroll_tab(self._tab_r, 2)

        outer_h.addWidget(self._tab_l, 1)
        outer_h.addWidget(self._tab_r, 1)
        self._root_layout.addWidget(outer, 1)

        # Remove trailing stretch from each, build content, re-add stretch
        for lay in (self._input_layout, self._beam_layout,
                    self._panels_layout, self._visual_layout, self._export_layout):
            lay.takeAt(lay.count() - 1)  # remove placeholder stretch

        self._build_input_section(self._input_layout)
        self._build_beam_section(self._beam_layout)
        self._build_panels_section(self._panels_layout)
        self._build_visual_section(self._visual_layout)
        self._build_export_section(self._export_layout)

        for lay in (self._input_layout, self._beam_layout,
                    self._panels_layout, self._visual_layout, self._export_layout):
            lay.addStretch(1)

    # ── Input section ─────────────────────────────────────────────────────────

    def _build_input_section(self, layout):
        r = _row(layout); _lbl(r, "Input file  *")
        self.w_input = _ent(r, width=220, placeholder="tao.init / run.ele / lattice.json")
        self.w_input.textChanged.connect(lambda t: self._on_input_change(t.strip()))
        _btn(r, "Browse", self._browse_input, width=70)
        _help(layout, "Auto-detected from extension: .init=Tao, .ele=ELEGANT, .json=xsuite.")

        r = _row(layout); _lbl(r, "Code backend")
        self.w_code = _dd(r, ["tao", "elegant", "xsuite", "madx"], width=110)
        self.w_code.currentTextChanged.connect(lambda _: (self._update_xsuite_rows(), self._update_madx_rows()))
        _help(layout, "Auto-detected from file extension. Override here if needed.")

        # xsuite extra rows (hidden initially)
        self._xsuite_widget = QWidget(); self._xsuite_widget.setStyleSheet("background: transparent;")
        xv = QVBoxLayout(self._xsuite_widget); xv.setContentsMargins(0, 0, 0, 0); xv.setSpacing(0)
        rx = _row(xv); _lbl(rx, "xsuite Twiss")
        self.w_xsuite_twiss = _dd(rx, ["4d", "6d"], width=80); _hint(rx, "4d = no RF")
        _help(xv, "4d: no RF (default). 6d: full longitudinal, requires RF cavities.")
        rl = _row(xv); _lbl(rl, "Line name")
        self.w_xsuite_line = _ent(rl, width=160, placeholder="e.g. ring  (auto-detect)")
        _help(xv, "Line name in xsuite JSON. Leave blank to auto-detect.")
        layout.addWidget(self._xsuite_widget)
        self._xsuite_widget.hide()

        # MAD-X extra row — survey file (hidden until madx selected)
        self._madx_widget = QWidget(); self._madx_widget.setStyleSheet("background: transparent;")
        mv = QVBoxLayout(self._madx_widget); mv.setContentsMargins(0, 0, 0, 0); mv.setSpacing(0)
        rm = _row(mv); _lbl(rm, "Survey file (.tfs)")
        self.w_madx_survey = _ent(rm, width=200, placeholder="optional — for floor plan")
        _btn(rm, "Browse", self._browse_madx_survey, width=70)
        _help(mv, "MAD-X SURVEY output. Leave blank to use dead-reckoning for floor plan.")
        layout.addWidget(self._madx_widget)
        self._madx_widget.hide()

        # Universe selector (hidden initially)
        self._uni_widget = QWidget(); self._uni_widget.setStyleSheet("background: transparent;")
        self._uni_vbox = QVBoxLayout(self._uni_widget); self._uni_vbox.setContentsMargins(0, 0, 0, 0)
        self._uni_row_h = QHBoxLayout(); self._uni_row_h.setContentsMargins(8, 2, 8, 2)
        lbl_u = QLabel("Universes"); lbl_u.setFont(FONT_MAIN)
        lbl_u.setStyleSheet(f"color: {FG_LBL}; background: transparent;"); lbl_u.setFixedWidth(160)
        self._uni_row_h.addWidget(lbl_u)
        self._uni_checks_widget = QWidget(); self._uni_checks_widget.setStyleSheet("background: transparent;")
        self._uni_checks_h = QHBoxLayout(self._uni_checks_widget)
        self._uni_checks_h.setContentsMargins(0, 0, 0, 0); self._uni_checks_h.setSpacing(8)
        self._uni_row_h.addWidget(self._uni_checks_widget)
        w_uni_row = QWidget(); w_uni_row.setStyleSheet("background: transparent;")
        w_uni_row.setLayout(self._uni_row_h)
        self._uni_vbox.addWidget(w_uni_row)
        _help(self._uni_vbox, "Uncheck universes to exclude from the plot.")
        layout.addWidget(self._uni_widget)
        self._uni_widget.hide()

        r = _row(layout); _lbl(r, "Output HTML")
        self.w_output = _ent(r, width=180); self.w_output.setText("optics.html")
        _btn(r, "Save as", self._browse_output, width=70)
        _help(layout, "Output HTML file. Open in any browser.")

        _sec(layout, "Plot Settings")
        r = _row(layout); _lbl(r, "Layout mode")
        self.w_layout = _dd(r, ["panels", "floor"], width=110); _hint(r, "panels · floor")
        r = _row(layout); _lbl(r, "Range  START:END")
        self.w_range = _ent(r, width=220, placeholder="QUA01:QUA06  or  3.0:19.0")
        _help(layout, "Sub-range: element names (QUA01:QUA06) or s positions (3.0:19.0).")

        _sec(layout, "Tunnel Wall")
        r = _row(layout); _lbl(r, "Tunnel wall file")
        self.w_tunnel_file = _ent(r, width=200, placeholder="path/to/tunnel.dat")
        _btn(r, "Browse", self._browse_tunnel, width=70)
        _help(layout, "Overlay tunnel on floor plan. Format: x_in y_in z_in x_out y_out z_out.")

        _sec(layout, "Compare Files")
        _help(layout, "Load additional files to overlay or compare against the primary.")

        # List of compare file rows
        self._compare_files = []
        self._compare_list_w = QWidget(); self._compare_list_w.setStyleSheet("background: transparent;")
        self._compare_list_v = QVBoxLayout(self._compare_list_w)
        self._compare_list_v.setContentsMargins(8, 2, 8, 2); self._compare_list_v.setSpacing(3)
        layout.addWidget(self._compare_list_w)

        # Add button
        add_row = _row(layout)
        _btn(add_row, "+ Add file", self._add_compare_file, width=90, color=ACCENT2)

        # Mode + normalize
        r = _row(layout); _lbl(r, "Compare mode")
        self.w_compare_mode = _dd(r, ["Overlay", "Separate", "Difference", "Difference (%)"], width=140)
        r = _row(layout)
        self.w_normalize_s = _chk(r, "Normalize s (0→1)")
        _hint(r, "aligns lattices of different lengths")

    # ── Beam section ──────────────────────────────────────────────────────────

    def _build_beam_section(self, layout):
        _sec(layout, "Beam Size Parameters")

        r = _row(layout); _lbl(r, "Emittance type")
        self.w_emit_geo = QPushButton("Geometric"); self.w_emit_geo.setCheckable(True)
        self.w_emit_geo.setChecked(True)
        self.w_emit_norm = QPushButton("Normalized"); self.w_emit_norm.setCheckable(True)
        for b in (self.w_emit_geo, self.w_emit_norm):
            b.setFont(FONT_MAIN); b.setFixedSize(110, 30)
            b.setStyleSheet(f"""
                QPushButton {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; color: {FG}; }}
                QPushButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
                QPushButton:hover   {{ background: {BORDER}; }}
            """)
            r.addWidget(b)
        self.w_emit_geo.clicked.connect(lambda: (self.w_emit_norm.setChecked(False), self._update_emit_ui()))
        self.w_emit_norm.clicked.connect(lambda: (self.w_emit_geo.setChecked(False), self._update_emit_ui()))

        r = _row(layout); _lbl(r, "Emit-x  [m·rad]")
        self.w_emitx = _ent(r, width=140, placeholder="e.g.  1e-9")
        r = _row(layout); _lbl(r, "Emit-y  [m·rad]")
        self.w_emity = _ent(r, width=140, placeholder="e.g.  1e-9")
        r = _row(layout); _lbl(r, "σ_dp  (δp/p)")
        self.w_sigmadp = _ent(r, width=140, placeholder="e.g.  1e-3")
        r = _row(layout); _lbl(r, "n·σ  (beam size)")
        self.w_nsigma = _ent(r, width=60); self.w_nsigma.setText("1")

        # Normalized emittance extras (hidden until "Normalized" selected)
        self._norm_widget = QWidget(); self._norm_widget.setStyleSheet("background: transparent;")
        nv = QVBoxLayout(self._norm_widget); nv.setContentsMargins(0, 0, 0, 0); nv.setSpacing(0)

        r2 = _row(nv); _lbl(r2, "Particle")
        self.w_particle = _dd(r2, ["Electron", "Proton", "Muon", "Custom"], width=120)
        self.w_particle.currentTextChanged.connect(lambda _: self._update_emit_ui())
        _help(nv, "Auto-set for Electron/Proton/Muon. Choose Custom to enter manually.")

        r2 = _row(nv); _lbl(r2, "Beam energy [MeV]")
        self.w_energy = _ent(r2, width=140, placeholder="e.g.  100")
        self.w_energy.textChanged.connect(lambda _: self._update_betagamma())
        _help(nv, "Total energy in MeV. Used to compute βγ.")

        self._mass_widget = QWidget(); self._mass_widget.setStyleSheet("background: transparent;")
        mv = QVBoxLayout(self._mass_widget); mv.setContentsMargins(0, 0, 0, 0); mv.setSpacing(0)
        rm2 = _row(mv); _lbl(rm2, "Rest mass [MeV/c²]")
        self.w_mass = _ent(rm2, width=140, placeholder="e.g.  938.3"); self.w_mass.setText("0.511")
        self.w_mass.textChanged.connect(lambda _: self._update_betagamma())
        nv.addWidget(self._mass_widget)

        r2 = _row(nv); _lbl(r2, "βγ  (computed)")
        self.w_betagamma = QLabel("—"); self.w_betagamma.setFont(FONT_MONO)
        self.w_betagamma.setStyleSheet(f"color: {SUCCESS}; background: transparent;")
        r2.addWidget(self.w_betagamma)

        layout.addWidget(self._norm_widget)
        self._norm_widget.hide()
        self._update_emit_ui()

    # ── Panels section ────────────────────────────────────────────────────────

    def _build_panels_section(self, layout):
        self._panels_layout_ref = layout   # save for overlay swap

        lbl = QLabel("Add panels, click name to rename:"); lbl.setFont(FONT_MAIN)
        lbl.setStyleSheet(f"color: {FG_LBL}; background: transparent; padding: 4px 12px 2px 12px;")
        layout.addWidget(lbl)
        _help(layout, "Stacked plots below the floor plan. ▲▼ to reorder, click name to rename.")

        self._panel_frame_widget = QWidget(); self._panel_frame_widget.setStyleSheet("background: transparent;")
        self._panel_frame_vbox = QVBoxLayout(self._panel_frame_widget)
        self._panel_frame_vbox.setContentsMargins(8, 4, 8, 4); self._panel_frame_vbox.setSpacing(2)
        layout.addWidget(self._panel_frame_widget)
        self._render_panel_list()

        lbl2 = QLabel("Add preset:"); lbl2.setFont(FONT_MAIN)
        lbl2.setStyleSheet(f"color: {FG_LBL}; background: transparent; padding: 4px 12px 2px 12px;")
        layout.addWidget(lbl2)

        btn_grid_w = QWidget(); btn_grid_w.setStyleSheet("background: transparent;")
        btn_grid = QGridLayout(btn_grid_w); btn_grid.setContentsMargins(8, 0, 8, 4); btn_grid.setSpacing(4)
        # Preset panels
        for i, (name, key) in enumerate(self._PRESET_PANELS):
            b = QPushButton(name); b.setFont(FONT_MAIN); b.setFixedSize(185, 30)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {MANTLE}; border: 1px solid {BORDER};
                    border-radius: 8px; color: {FG_LBL};
                }}
                QPushButton:hover {{ background: {SURFACE2}; color: {ACCENT}; border-color: {ACCENT}; }}
            """)
            b.clicked.connect(lambda _=False, n=name, k=key: self._add_preset_panel(n, k))
            btn_grid.addWidget(b, i // 2, i % 2)
        # Custom and Expression panel buttons in the same grid
        n_presets = len(self._PRESET_PANELS)
        for j, (text, cmd) in enumerate([
            ("+ Custom panel...",     self._add_custom_panel_dialog),
            ("+ Expression panel...", self._add_expr_panel_dialog),
        ]):
            idx = n_presets + j
            b = QPushButton(text); b.setFont(FONT_MAIN); b.setFixedSize(185, 30)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {MANTLE}; border: 1px solid {ACCENT2};
                    border-radius: 8px; color: {ACCENT2};
                }}
                QPushButton:hover {{ background: {ACCENT2}; color: {CRUST}; }}
            """)
            b.clicked.connect(cmd)
            btn_grid.addWidget(b, idx // 2, idx % 2)
        layout.addWidget(btn_grid_w)

        _sec(layout, "Panel Options")
        r = _row(layout)
        self.w_show_tune = _chk(r, "Show tune & chromaticity")
        _help(layout, "Annotates Qₓ, Qᵧ, Qₓ', Qᵧ' on the first panel.")
        r = _row(layout)
        self.w_show_titles = _chk(r, "Show panel titles"); self.w_show_titles.setChecked(True)
        r = _row(layout); _lbl(r, "Panel spacing (px)")
        self.w_panel_spacing = _ent(r, width=60, placeholder="80"); self.w_panel_spacing.setText("80")
        _hint(r, "pixels between panels")

    # ── Visual section ────────────────────────────────────────────────────────

    def _build_visual_section(self, layout):
        # ── General ──────────────────────────────────────────────────────────
        r = _row(layout); _lbl(r, "Plot title")
        self.w_title = _ent(r, width=200, placeholder="optional")
        r = _row(layout); _lbl(r, "Aspect ratio  W:H")
        self.w_aspect = _ent(r, width=80, placeholder="e.g.  1:2")

        # ── Floor Plan ───────────────────────────────────────────────────────
        _sec(layout, "Floor Plan")
        r = _row(layout); _lbl(r, "X-Z elem ratio")
        self.w_elem_h = _ent(r, width=60); self.w_elem_h.setText("0.05")
        _hint(r, "fraction of axis span")
        r = _row(layout); _lbl(r, "Y-Z elem ratio")
        self.w_elem_h_yz = _ent(r, width=60); _hint(r, "blank = same as X-Z")
        r = _row(layout); _lbl(r, "XZ Y-range")
        self.w_fp_xz_range = _ent(r, width=100, placeholder="-0.5:0.5"); _hint(r, "blank = auto")
        r = _row(layout); _lbl(r, "YZ Y-range")
        self.w_fp_yz_range = _ent(r, width=100, placeholder="-1:1"); _hint(r, "blank = auto")
        r = _row(layout)
        self.w_show_xz = _chk(r, "Show X-Z"); self.w_show_xz.setChecked(True)
        self.w_show_yz = _chk(r, "Show Y-Z"); self.w_show_yz.setChecked(True)
        _hint(r, "(floor mode only)")

        # ── Display ──────────────────────────────────────────────────────────
        _sec(layout, "Display")
        _disp_w = QWidget(); _disp_w.setStyleSheet("background: transparent;")
        _disp_g = QGridLayout(_disp_w)
        _disp_g.setContentsMargins(8, 2, 8, 4); _disp_g.setSpacing(4)
        _disp_g.setColumnStretch(0, 1); _disp_g.setColumnStretch(1, 1); _disp_g.setColumnStretch(2, 1)
        _chk_items = [
            ("No labels",          lambda c: setattr(self, 'w_no_labels',        c)),
            ("Flip bends",         lambda c: setattr(self, 'w_flip_bend',         c)),
            ("Dark mode",          lambda c: setattr(self, 'w_dark',              c)),
            ("Color beampipes",    lambda c: setattr(self, 'w_color_beampipes',   c)),
            ("Show tunnel",        lambda c: setattr(self, 'w_show_tunnel',       c)),
            ("Legend inside",      lambda c: setattr(self, 'w_legend_inside',     c)),
            ("Markers in floor",   lambda c: setattr(self, 'w_show_markers',      c)),
            ("Markers in bar",     lambda c: setattr(self, 'w_show_markers_bar',  c)),
        ]
        for i, (lbl_txt, setter) in enumerate(_chk_items):
            cb = QCheckBox(lbl_txt); cb.setFont(FONT_MAIN); cb.setStyleSheet(_CHK_SS)
            setter(cb)
            _disp_g.addWidget(cb, i // 3, i % 3)
        layout.addWidget(_disp_w)

        # ── Font Sizes ───────────────────────────────────────────────────────
        _sec(layout, "Font Sizes")
        _fs_w = QWidget(); _fs_w.setStyleSheet("background: transparent;")
        _fs_g = QGridLayout(_fs_w)
        _fs_g.setContentsMargins(8, 2, 8, 4); _fs_g.setSpacing(6)
        _fs_g.setColumnStretch(0, 1); _fs_g.setColumnStretch(1, 1); _fs_g.setColumnStretch(2, 1)
        for col, (lbl_txt, placeholder, attr) in enumerate([
                ("Axis labels",  "12", 'w_fs_axis'),
                ("Tick labels",  "10", 'w_fs_tick'),
                ("Panel titles", "13", 'w_fs_title'),
                ("Annotations",  "8",  'w_fs_annot'),
                ("Legend",       "10", 'w_fs_legend'),
        ]):
            r_idx = col // 3; c_idx = col % 3
            row_w = QWidget(); row_w.setStyleSheet("background: transparent;")
            row_h = QHBoxLayout(row_w); row_h.setContentsMargins(0, 0, 0, 0); row_h.setSpacing(4)
            l = QLabel(lbl_txt); l.setFont(FONT_SMALL)
            l.setStyleSheet(f"color: {FG_LBL}; background: transparent;")
            e = QLineEdit(); e.setFont(FONT_MONO); e.setFixedWidth(42); e.setFixedHeight(24)
            e.setPlaceholderText(placeholder); e.setStyleSheet(_ENTRY_SS)
            row_h.addWidget(l); row_h.addWidget(e); row_h.addStretch()
            setattr(self, attr, e)
            _fs_g.addWidget(row_w, r_idx, c_idx)
        layout.addWidget(_fs_w)
        _help(layout, "Leave blank to use Plotly defaults.")

    # ── Export section ────────────────────────────────────────────────────────

    def _build_export_section(self, layout):
        r = _row(layout)
        self.w_png = _chk(r, "Save PNG"); self.w_pdf = _chk(r, "Save PDF")
        _hint(r, "requires: pip install kaleido")
        _help(layout, "Requires: pip install kaleido")
        r = _row(layout); _lbl(r, "DPI (PNG)")
        self.w_dpi = _ent(r, width=60); self.w_dpi.setText("300")
        _help(layout, "PNG resolution. 300 DPI = publication quality.")

        _sec(layout, "CSV Export")
        r = _row(layout); _lbl(r, "CSV base name")
        self.w_csv_base = _ent(r, width=160, placeholder="lattice")
        self.w_csv_base.setText("lattice")
        _help(layout, "e.g. 'ltr' → ltr-twiss.csv, ltr-orbit.csv, ...")

    # ── Panel list rendering ──────────────────────────────────────────────────

    def _render_panel_list(self):
        for w in self._panel_rows:
            w.setParent(None); w.deleteLater()
        self._panel_rows = []
        if not hasattr(self, '_panel_height_edits'):
            self._panel_height_edits = {}
        n = len(self._panels)
        _DEFAULT_H = {'floor-xz': 220, 'floor-yz': 220, 'bar': 80,
                      'latdiff': 260, 'summary': 260}
        for pos, panel in enumerate(self._panels):
            row_w = QWidget(); row_w.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(row_w); rh.setContentsMargins(0, 1, 0, 1); rh.setSpacing(4)

            name_btn = QPushButton(panel['name']); name_btn.setFont(FONT_MAIN)
            name_btn.setFixedWidth(155); name_btn.setFixedHeight(26)
            name_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; color: {FG}; text-align: left; padding: 2px 6px; }}
                QPushButton:hover {{ background: {PANEL}; border-radius: 4px; }}
            """)
            name_btn.clicked.connect(lambda _=False, p=pos: self._rename_panel(p))
            rh.addWidget(name_btn)

            # Height field — keyed by spec string for reliable lookup
            spec = panel.get('spec', '')
            spec_key = spec if isinstance(spec, str) else spec.get('type', 'custom')
            default_h = str(_DEFAULT_H.get(spec_key, 280))
            prev_val = self._panel_height_edits.get(spec_key, default_h)
            h_edit = QLineEdit(prev_val); h_edit.setFixedWidth(55); h_edit.setFixedHeight(28)
            h_edit.setFont(FONT_MAIN); h_edit.setStyleSheet(_ENTRY_SS)
            h_edit.setToolTip("Panel height in pixels")
            self._panel_height_edits[spec_key] = prev_val
            h_edit.textChanged.connect(lambda v, s=spec_key: self._panel_height_edits.update({s: v}))
            rh.addWidget(h_edit)

            for sym, cmd, col in [
                ("▲", lambda _=False, p=pos: self._move_panel(p, -1), ACCENT),
                ("▼", lambda _=False, p=pos: self._move_panel(p, +1), ACCENT),
                ("✕", lambda _=False, p=pos: self._remove_panel(p),   ERROR),
            ]:
                disabled = (sym == "▲" and pos == 0) or (sym == "▼" and pos == n - 1)
                b = QPushButton(sym); b.setFixedSize(26, 26); b.setFont(FONT_SMALL)
                b.setEnabled(not disabled)
                b.setStyleSheet(f"""
                    QPushButton {{ background: {BORDER}; border-radius: 4px; color: {FG}; border: none; }}
                    QPushButton:hover:enabled {{ background: {col}; color: white; }}
                    QPushButton:disabled {{ color: {FG_DIM}; }}
                """)
                b.clicked.connect(cmd)
                rh.addWidget(b)
            # Annotation toggle — not available for floor plan panels
            _no_annot_panels = {'floor-xz', 'floor-yz'}
            _spec_str = spec if isinstance(spec, str) else spec.get('type', '')
            if _spec_str not in _no_annot_panels:
                if not hasattr(self, '_panel_annot_checks'):
                    self._panel_annot_checks = {}
                if not hasattr(self, '_panel_annot_edits'):
                    self._panel_annot_edits = {}
                annot_on = bool(panel.get('annot_pattern', ''))
                annot_btn = QPushButton('✎'); annot_btn.setFixedSize(26, 26)
                annot_btn.setFont(FONT_SMALL); annot_btn.setCheckable(True)
                annot_btn.setChecked(annot_on)
                annot_btn.setToolTip('Annotate elements (wildcard pattern)')
                annot_btn.setStyleSheet(f"""
                    QPushButton {{ background: {BORDER}; border-radius: 4px; color: {FG}; border: none; }}
                    QPushButton:checked {{ background: {ACCENT2}; color: white; }}
                    QPushButton:hover {{ background: {ACCENT2}; color: white; }}
                """)
                rh.addWidget(annot_btn)

                # Pattern field — visible only when toggled on
                annot_edit = QLineEdit(panel.get('annot_pattern', ''))
                annot_edit.setFixedWidth(90); annot_edit.setFixedHeight(22)
                annot_edit.setFont(FONT_MAIN); annot_edit.setStyleSheet(_ENTRY_SS)
                annot_edit.setPlaceholderText('e.g. IPM*')
                annot_edit.setVisible(annot_on)
                rh.addWidget(annot_edit)

                def _on_annot_toggle(checked, p=pos, ae=annot_edit):
                    ae.setVisible(checked)
                    self._panels[p]['annot_pattern'] = ae.text().strip() if checked else ''
                def _on_annot_text(text, p=pos, ab=annot_btn):
                    if ab.isChecked():
                        self._panels[p]['annot_pattern'] = text.strip()
                annot_btn.toggled.connect(_on_annot_toggle)
                annot_edit.textChanged.connect(_on_annot_text)

                self._panel_annot_checks[pos] = annot_btn
                self._panel_annot_edits[pos]  = annot_edit

            rh.addStretch()
            self._panel_frame_vbox.addWidget(row_w)
            self._panel_rows.append(row_w)

    def _move_panel(self, pos, d):
        new = pos + d
        if 0 <= new < len(self._panels):
            self._panels[pos], self._panels[new] = self._panels[new], self._panels[pos]
        self._render_panel_list()

    def _remove_panel(self, pos):
        if len(self._panels) > 1:
            self._panels.pop(pos)
        else:
            QMessageBox.warning(self, "Cannot Remove", "At least one panel must remain.")
        self._render_panel_list()

    def _rename_panel(self, pos):
        name, ok = QInputDialog.getText(self, "Rename Panel", "Panel name:",
                                         text=self._panels[pos]['name'])
        if ok and name.strip():
            self._panels[pos]['name'] = name.strip()
            self._render_panel_list()

    def _get_panels(self):
        return [p['spec'] for p in self._panels] if self._panels else ['twiss']

    def _get_panel_annotations(self):
        """Return {panel_index: pattern} for panels with annot_pattern set."""
        result = {}
        for i, p in enumerate(self._panels):
            pat = p.get('annot_pattern', '').strip()
            if pat:
                result[i] = pat
        return result or None

    def _get_panel_heights(self):
        if not hasattr(self, '_panel_height_edits') or not self._panel_height_edits:
            return None
        result = {}
        for spec, val in self._panel_height_edits.items():
            try:
                result[spec] = int(val)
            except (ValueError, TypeError):
                pass
        return result if result else None

    def _add_preset_panel(self, name, key):
        existing = [p['spec'] for p in self._panels if isinstance(p['spec'], str)]
        if key in existing:
            r = QMessageBox.question(self, "Duplicate Panel",
                                     f"'{name}' already in list. Add again?")
            if r != QMessageBox.Yes: return
        self._panels.append({'name': name, 'spec': key})
        self._render_panel_list()

    def _add_expr_panel_dialog(self):
        self._push_overlay(lambda container, done:
            ExprPanelOverlay(container, done,
                             code=self.w_code.currentText(),
                             input_file=self.w_input.text().strip(),
                             xsuite_twiss=self.w_xsuite_twiss.currentText(),
                             xsuite_line=self.w_xsuite_line.text().strip(),
                             madx_survey=self.w_madx_survey.text().strip() or None))

    def _add_custom_panel_dialog(self):
        self._push_overlay(lambda container, done:
            CustomPanelOverlay(container, done))

    def _push_overlay(self, builder_fn):
        """Replace panels tab content with an overlay widget."""
        tab_widget = self._tab_r.widget(0)   # "Panels" tab
        # Hide existing layout widget
        old_sa = tab_widget.layout().itemAt(0).widget()
        old_sa.hide()

        overlay_w = QWidget(); overlay_w.setStyleSheet(f"background: {BG};")
        overlay_v = QVBoxLayout(overlay_w); overlay_v.setContentsMargins(0, 0, 0, 0)
        tab_widget.layout().addWidget(overlay_w)

        def _on_done(result):
            overlay_w.hide(); overlay_w.setParent(None); overlay_w.deleteLater()
            old_sa.show()
            if result:
                self._panels.append({'name': result['name'], 'spec': result})
                self._render_panel_list()

        builder_fn(overlay_v, _on_done)

    # ── Compare file management ───────────────────────────────────────────────

    def _add_compare_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select compare file", "",
            "All supported (*.init *.ele *.json);;All files (*.*)")
        if not f: return
        ext = Path(f).suffix.lower()
        code = {'.init': 'tao', '.ele': 'elegant', '.json': 'xsuite'}.get(ext, 'tao')
        entry = {'file': f, 'code': code, 'label': Path(f).stem,
                 'uni_n': 1, 'uni_labels': {}, 'uni_checks': {}}
        if ext == '.init':
            try:
                n, labels = _parse_tao_init(f)
                entry['uni_n'] = n
                entry['uni_labels'] = labels
            except Exception:
                pass
        self._compare_files.append(entry)
        self._render_compare_list()

    def _render_compare_list(self):
        # Clear existing rows
        while self._compare_list_v.count():
            item = self._compare_list_v.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        for i, entry in enumerate(self._compare_files):
            row_w = QWidget(); row_w.setStyleSheet("background: transparent;")
            row_h = QHBoxLayout(row_w); row_h.setContentsMargins(0, 0, 0, 0); row_h.setSpacing(4)

            # Label (editable)
            lbl_e = QLineEdit(entry['label']); lbl_e.setFont(FONT_MONO)
            lbl_e.setFixedWidth(100); lbl_e.setStyleSheet(_ENTRY_SS)
            lbl_e.setToolTip("Display label")
            lbl_e.textChanged.connect(lambda t, idx=i: self._compare_files.__setitem__(
                idx, {**self._compare_files[idx], 'label': t}))
            row_h.addWidget(lbl_e)

            # File path (truncated display)
            path_lbl = QLabel(Path(entry['file']).name); path_lbl.setFont(FONT_SMALL)
            path_lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
            path_lbl.setToolTip(entry['file'])
            row_h.addWidget(path_lbl)

            # Code badge
            code_dd = QComboBox(); code_dd.setFont(FONT_SMALL)
            code_dd.addItems(["tao", "elegant", "xsuite"])
            code_dd.setCurrentText(entry['code']); code_dd.setFixedWidth(82)
            code_dd.setStyleSheet(_COMBO_SS)
            code_dd.currentTextChanged.connect(lambda t, idx=i: self._compare_files.__setitem__(
                idx, {**self._compare_files[idx], 'code': t}))
            row_h.addWidget(code_dd)

            # Remove button
            rm = QPushButton("✕"); rm.setFixedSize(22, 22); rm.setFont(FONT_SMALL)
            rm.setStyleSheet(f"QPushButton {{ background: {BORDER}; color: {FG_DIM}; border: none; border-radius: 3px; }}"
                             f"QPushButton:hover {{ background: {ERROR}; color: white; }}")
            rm.clicked.connect(lambda _=False, idx=i: self._remove_compare_file(idx))
            row_h.addWidget(rm)
            row_h.addStretch()
            self._compare_list_v.addWidget(row_w)

            # Universe checkboxes for multi-universe Tao files
            n = entry.get('uni_n', 1)
            if n > 1:
                uni_row_w = QWidget(); uni_row_w.setStyleSheet("background: transparent;")
                uni_row_h = QHBoxLayout(uni_row_w)
                uni_row_h.setContentsMargins(12, 0, 0, 2); uni_row_h.setSpacing(6)
                ul = QLabel("Universes:"); ul.setFont(FONT_SMALL)
                ul.setStyleSheet(f"color: {FG_LBL}; background: transparent;")
                uni_row_h.addWidget(ul)
                uni_checks = {}
                labels = entry.get('uni_labels', {})
                for ui in range(1, n + 1):
                    cb = QCheckBox(f"u{ui}:{labels.get(ui, f'u{ui}')}")
                    cb.setChecked(True); cb.setFont(FONT_SMALL); cb.setStyleSheet(_CHK_SS)
                    uni_checks[ui] = cb
                    uni_row_h.addWidget(cb)
                uni_row_h.addStretch()
                self._compare_list_v.addWidget(uni_row_w)
                self._compare_files[i]['uni_checks'] = uni_checks

    def _remove_compare_file(self, idx):
        if 0 <= idx < len(self._compare_files):
            self._compare_files.pop(idx)
            self._render_compare_list()

    def _get_compare_list(self):
        result = []
        for e in self._compare_files:
            entry = {'file': e['file'], 'code': e['code'], 'label': e['label']}
            uni_checks = e.get('uni_checks', {})
            if uni_checks:
                sel = [i for i, cb in uni_checks.items() if cb.isChecked()]
                if sel: entry['universes'] = sel
            result.append(entry)
        return result or None

    # ── Auto-detect / reactive UI ─────────────────────────────────────────────

    def _on_input_change(self, path):
        self._autodetect_code(path)
        if path.endswith('.init'):
            self._update_universe_selector(path)

    def _autodetect_code(self, path):
        ext = Path(path).suffix.lower()
        if ext == '.init':    self.w_code.setCurrentText('tao')
        elif ext == '.ele':   self.w_code.setCurrentText('elegant')
        elif ext == '.json':  self.w_code.setCurrentText('xsuite')
        elif ext == '.tfs':   self.w_code.setCurrentText('madx')
        self._update_xsuite_rows()
        self._update_madx_rows()

    def _update_xsuite_rows(self):
        if not hasattr(self, '_xsuite_widget'): return
        if self.w_code.currentText() == 'xsuite':
            self._xsuite_widget.show()
        else:
            self._xsuite_widget.hide()

    def _update_madx_rows(self):
        if not hasattr(self, '_madx_widget'): return
        if self.w_code.currentText() == 'madx':
            self._madx_widget.show()
        else:
            self._madx_widget.hide()

    def _update_universe_selector(self, path):
        # Clear existing checkboxes and label edits
        while self._uni_checks_h.count():
            item = self._uni_checks_h.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._uni_checks = {}
        self._uni_label_edits = {}
        try:
            n, labels = _parse_tao_init(path)
        except Exception:
            self._uni_widget.hide(); return
        self._uni_n = n
        if n <= 1:
            self._uni_widget.hide(); return
        for i in range(1, n + 1):
            lbl = labels.get(i, f'u{i}')
            # Container for checkbox + label edit
            cell = QWidget(); cell.setStyleSheet("background: transparent;")
            cell_h = QHBoxLayout(cell); cell_h.setContentsMargins(0, 0, 0, 0); cell_h.setSpacing(4)
            cb = QCheckBox(f"u{i}"); cb.setChecked(True); cb.setFont(FONT_MAIN)
            cb.setStyleSheet(_CHK_SS); self._uni_checks[i] = cb
            cell_h.addWidget(cb)
            le = QLineEdit(lbl); le.setFixedWidth(100); le.setFont(FONT_MAIN)
            le.setStyleSheet(_ENTRY_SS)
            self._uni_label_edits[i] = le
            cell_h.addWidget(le)
            self._uni_checks_h.addWidget(cell)
        self._uni_checks_h.addStretch()
        self._uni_widget.show()

    def _get_selected_universes(self):
        if not self._uni_checks or self._uni_n <= 1: return None
        sel = [i for i, cb in self._uni_checks.items() if cb.isChecked()]
        return sel if sel else None

    def _get_uni_label_overrides(self):
        if not self._uni_label_edits: return None
        return {i: le.text().strip() for i, le in self._uni_label_edits.items()
                if le.text().strip()}

    _PARTICLE_MASS = {"Electron": 0.511, "Proton": 938.272, "Muon": 105.658}

    def _update_emit_ui(self):
        is_norm = self.w_emit_norm.isChecked()
        self._norm_widget.setVisible(is_norm)
        if is_norm and self.w_particle.currentText() == "Custom":
            self._mass_widget.show()
        else:
            self._mass_widget.hide()
        self._update_betagamma()

    def _update_betagamma(self):
        if not hasattr(self, 'w_emit_norm'): return
        if not self.w_emit_norm.isChecked():
            self.w_betagamma.setText("—"); return
        try:
            import math
            E = float(self.w_energy.text().strip())
            m = self._PARTICLE_MASS.get(self.w_particle.currentText())
            if m is None: m = float(self.w_mass.text().strip())
            self.w_betagamma.setText(f"{math.sqrt((E/m)**2 - 1):.4f}")
        except Exception:
            self.w_betagamma.setText("—")

    def _get_font_sizes(self):
        """Collect font size overrides. Returns None if all blank."""
        def _iv(w):
            t = w.text().strip()
            try: return int(t) if t else None
            except: return None
        d = {}
        v = _iv(self.w_fs_axis);   d['axis_label'] = v if v else None
        v = _iv(self.w_fs_tick);   d['tick']       = v if v else None
        v = _iv(self.w_fs_title);  d['title']      = v if v else None
        v = _iv(self.w_fs_annot);  d['annot']      = v if v else None
        v = _iv(self.w_fs_legend); d['legend']     = v if v else None
        d = {k: v for k, v in d.items() if v is not None}
        return d if d else None

    def _get_geometric_emittances(self):
        def _p(widget):
            t = widget.text().strip()
            if not t: return None
            try: return float(t)
            except: return None
        ex = _p(self.w_emitx); ey = _p(self.w_emity)
        if self.w_emit_norm.isChecked():
            try:
                bg = float(self.w_betagamma.text())
                if ex is not None: ex /= bg
                if ey is not None: ey /= bg
            except: pass
        return ex, ey

    # ── File dialogs ──────────────────────────────────────────────────────────

    def _browse_input(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select input file", "",
            "All supported (*.init *.ele *.json *.tfs);;Tao init (*.init);;ELEGANT ele (*.ele);;xsuite JSON (*.json);;MAD-X TFS (*.tfs);;All files (*.*)")
        if f: self.w_input.setText(f)

    def _browse_madx_survey(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select MAD-X survey file", "",
            "TFS files (*.tfs);;All files (*.*)")
        if f: self.w_madx_survey.setText(f)

    def _browse_tunnel(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select tunnel wall file", "",
            "Data files (*.dat *.txt *.csv);;All files (*.*)")
        if f: self.w_tunnel_file.setText(f)

    def _browse_output(self):
        f, _ = QFileDialog.getSaveFileName(
            self, "Save output HTML", "optics.html",
            "HTML files (*.html);;All files (*.*)")
        if f: self.w_output.setText(f)

    # ── Collect kwargs ────────────────────────────────────────────────────────

    def _collect_kwargs(self):
        inp = self.w_input.text().strip()
        if not inp or inp == "tao.init / run.ele / lattice.json":
            raise ValueError("Please select an input file.")

        def _fn(widget):
            t = widget.text().strip()
            if not t: return None
            try: return float(t)
            except: raise ValueError(f"Invalid number: '{t}'")

        rng = self.w_range.text().strip() or None
        ttl = self.w_title.text().strip() or None
        fp_xz = self.w_fp_xz_range.text().strip() or None
        fp_yz = self.w_fp_yz_range.text().strip() or None
        xsl   = self.w_xsuite_line.text().strip() or None
        ex, ey = self._get_geometric_emittances()

        return dict(
            input_file=inp, code=self.w_code.currentText(),
            output_file=self.w_output.text().strip() or "optics.html",
            show_element_labels=not self.w_no_labels.isChecked(), show=False,
            save_png=self.w_png.isChecked(), save_pdf=self.w_pdf.isChecked(),
            csv_base=self.w_csv_base.text().strip() or 'lattice',
            dpi=int(self.w_dpi.text().strip() or "300"),
            flip_bend=self.w_flip_bend.isChecked(),
            element_height_xz=float(self.w_elem_h.text().strip() or "0.05"),
            element_height_yz=_fn(self.w_elem_h_yz),
            fp_xz_range=fp_xz, fp_yz_range=fp_yz,
            panels=self._get_panels(), layout=self.w_layout.currentText(), srange=rng,
            panel_annotations=self._get_panel_annotations(),
            font_sizes=self._get_font_sizes(),
            panel_heights=self._get_panel_heights(),
            emit_x=ex, emit_y=ey, sigma_dp=_fn(self.w_sigmadp),
            n_sigma=_fn(self.w_nsigma) or 1.0, title=ttl,
            dark_mode=self.w_dark.isChecked(), aspect_ratio=self.w_aspect.text().strip() or None,
            legend_inside=self.w_legend_inside.isChecked(),
            xsuite_twiss=self.w_xsuite_twiss.currentText(), xsuite_line=xsl,
            universes=self._get_selected_universes(),
            uni_label_overrides=self._get_uni_label_overrides(),
            madx_survey=self.w_madx_survey.text().strip() or None,
            show_tune=self.w_show_tune.isChecked(),
            show_tunnel=self.w_show_tunnel.isChecked(),
            tunnel_wall_file=self.w_tunnel_file.text().strip() or None,
            show_markers=self.w_show_markers.isChecked(),
            show_markers_bar=self.w_show_markers_bar.isChecked(),
            show_floor=False,  # floor plan now handled as panel type
            color_beampipes=self.w_color_beampipes.isChecked(),
            show_xz=self.w_show_xz.isChecked(),
            show_yz=self.w_show_yz.isChecked(),
            show_titles=self.w_show_titles.isChecked(),
            panel_spacing=float(self.w_panel_spacing.text().strip() or '80'),
            compare=self._get_compare_list(),
            compare_mode=self.w_compare_mode.currentText().lower().replace(' ', '').replace('(%)', '%'),
            normalize_s=self.w_normalize_s.isChecked(),
        )

    # ── Run / cancel ──────────────────────────────────────────────────────────

    def _run(self):
        try: kwargs = self._collect_kwargs()
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Configuration Error", str(e)); return

        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.open_btn.setEnabled(False); self.dryrun_btn.setEnabled(False)
        self.open_btn.setStyleSheet("")  # reset green highlight from previous run
        self._set_status("Running…"); self._progress.setValue(0); self._pct_lbl.setText("0%")
        self._log("\n" + "─" * 60 + "\n", "dim")
        self._log(f"▶ code={kwargs['code']}  layout={kwargs['layout']}  panels={kwargs['panels']}\n", "info")
        self._log("─" * 60 + "\n", "dim")

        self._cancelled = False

        def _worker():
            try:
                kwargs['log_fn']      = lambda m: self._sig_log.emit(m, _clf(m))
                kwargs['progress_fn'] = lambda p, l: self._sig_progress.emit(int(p), l or "")
                plot_optics(**kwargs)
                out = str(Path(kwargs['output_file']).resolve())
                self._last_output = out
                self._sig_log.emit(f"\n✓ Done — {out}\n", "ok")
                self._sig_progress.emit(100, "Done ✓")
                self._sig_done.emit(kwargs['input_file'])
            except Exception:
                import traceback; tb = traceback.format_exc()
                self._sig_log.emit(f"\n✗ Error:\n{tb}\n", "error")
                self._sig_progress.emit(0, "Failed")
                self._sig_failed.emit("")
            finally:
                self._sig_finally.emit()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def _cancel(self):
        self._log("\n[Cancel requested.]\n", "warn")
        self._set_status("Cancelling…")

    def _open_plot(self):
        import webbrowser
        if self._last_output and Path(self._last_output).exists():
            webbrowser.open(f"file://{self._last_output}")
        else:
            QMessageBox.warning(self, "Open Plot", "Output file not found. Run first.")

    def _set_progress(self, pct, label=""):
        self._progress.setValue(int(pct))
        self._pct_lbl.setText(f"{int(pct)}%" if pct > 0 else "")
        if label: self._set_status(label)

    def _on_run_done(self, input_file):
        self.open_btn.setEnabled(True)
        self.open_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SUCCESS}; border: 1px solid {SUCCESS};
                border-radius: 8px; color: {CRUST}; font-weight: bold;
            }}
            QPushButton:hover {{ background: {SUCCESS}; color: {CRUST}; opacity: 0.9; }}
        """)
        self._save_recent(input_file)

    def _on_run_failed(self, _unused):
        pass   # status already set via _sig_progress

    def _on_run_finally(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.dryrun_btn.setEnabled(True)
        self._progress.setValue(0)
        self._pct_lbl.setText("")

    def _progress_safe(self, pct, label=None):
        """Thread-safe: emit signal, Qt delivers it to main thread."""
        self._sig_progress.emit(int(pct), label or "")

    def _dry_run(self):
        try: kwargs = self._collect_kwargs()
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Configuration Error", str(e)); return

        self.run_btn.setEnabled(False); self.dryrun_btn.setEnabled(False)
        self._set_status("Inspecting…")
        self._log("\n🔍 Dry run — loading lattice only…\n", "info")

        def _worker():
            try:
                code = kwargs['code']; inp = kwargs['input_file']
                log = lambda m: self._sig_log.emit(m, "info")
                if code == 'tao':     data = load_tao(inp, log_fn=log)
                elif code == 'xsuite': data = load_xsuite(inp, log_fn=log)
                else:                  data = load_elegant(inp, log_fn=log)
                s = data['s']; elems = data['elements']
                msg = f"\n✓ {len(elems)} elements, s = {float(s[0]):.3f} → {float(s[-1]):.3f} m\n"
                self._sig_log.emit(msg, "ok")
                self._sig_progress.emit(0, "Inspection done ✓")
            except Exception:
                import traceback; tb = traceback.format_exc()
                self._sig_log.emit(f"\n✗ Error:\n{tb}\n", "error")
                self._sig_progress.emit(0, "Failed")
            finally:
                self._sig_finally.emit()

        threading.Thread(target=_worker, daemon=True).start()

    def _export_csv(self):
        try: kwargs = self._collect_kwargs()
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Configuration Error", str(e)); return
        # Use the base name from the field, trigger via save_csv flag
        kwargs['save_csv'] = True
        kwargs['csv_base'] = self.w_csv_base.text().strip() or 'lattice'
        self._set_status("Exporting CSV…")
        def _worker():
            try:
                plot_optics(**kwargs)
                self._sig_progress.emit(0, "CSV exported ✓")
            except Exception:
                import traceback; tb = traceback.format_exc()
                self._sig_log.emit(f"\n✗ CSV error:\n{tb}\n", "error")
        threading.Thread(target=_worker, daemon=True).start()

    def _copy_path(self):
        if self._last_output:
            QApplication.clipboard().setText(self._last_output)
            self._set_status("Path copied ✓")
        else:
            QMessageBox.warning(self, "Copy Path", "No output yet. Run first.")

    # ── Log ───────────────────────────────────────────────────────────────────

    _LOG_COLORS = {"ok": SUCCESS, "warn": WARN, "error": ERROR, "dim": FG_DIM, "info": FG}

    def _log(self, text, tag="info"):
        color = self._LOG_COLORS.get(tag, FG)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFont(FONT_MONO)
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _log_safe(self, text, tag="info"):
        """Thread-safe: emit signal, Qt delivers it to main thread."""
        self._sig_log.emit(text, tag)

    def _clear_log(self):
        self.log.clear()
        self._log("Log cleared.\n", "dim")

    # ── Recent files ──────────────────────────────────────────────────────────

    def _load_recent(self):
        try:
            import json
            data = json.loads(self._RECENT_FILE.read_text())
            return [p for p in data if Path(p).exists()]
        except: return []

    def _save_recent(self, path):
        import json
        recent = [p for p in self._load_recent() if p != path]
        recent.insert(0, path)
        try: self._RECENT_FILE.write_text(json.dumps(recent[:self._MAX_RECENT]))
        except: pass
        self._refresh_recent_menu()

    def _refresh_recent_menu(self):
        if not hasattr(self, '_recent_menu'): return
        self._recent_menu.clear()
        recent = self._load_recent()
        if not recent:
            a = QAction("(no recent files)", self); a.setEnabled(False)
            self._recent_menu.addAction(a); return
        for p in recent:
            label = Path(p).name + "  —  " + str(Path(p).parent)
            act = QAction(label, self)
            act.triggered.connect(lambda _=False, f=p: self.w_input.setText(f))
            self._recent_menu.addAction(act)

    # ── Presets ───────────────────────────────────────────────────────────────

    def _load_presets(self):
        try:
            import json; return json.loads(self._PRESET_FILE.read_text())
        except: return {}

    def _save_presets(self, presets):
        import json
        try: self._PRESET_FILE.write_text(json.dumps(presets, indent=2))
        except Exception as e: QMessageBox.critical(self, "Preset Error", str(e))

    def _collect_preset(self):
        return {
            'code':       self.w_code.currentText(),
            'output':     self.w_output.text(),
            'layout':     self.w_layout.currentText(),
            'range':      self.w_range.text(),
            'panels':     [{'name': p['name'], 'spec': p['spec'], 'annot_pattern': p.get('annot_pattern','')} for p in self._panels],
            'emit_type':  'normalized' if self.w_emit_norm.isChecked() else 'geometric',
            'emitx':      self.w_emitx.text(),     'emity':    self.w_emity.text(),
            'sigmadp':    self.w_sigmadp.text(),   'nsigma':   self.w_nsigma.text(),
            'particle':   self.w_particle.currentText(), 'energy': self.w_energy.text(),
            'title':      self.w_title.text(),     'elem_h':   self.w_elem_h.text(),
            'elem_h_yz':  self.w_elem_h_yz.text(), 'fp_xz_range': self.w_fp_xz_range.text(),
            'fp_yz_range': self.w_fp_yz_range.text(),
            'no_labels':  self.w_no_labels.isChecked(), 'flip_bend': self.w_flip_bend.isChecked(),
            'dark_mode':  self.w_dark.isChecked(), 'png': self.w_png.isChecked(),
            'pdf':        self.w_pdf.isChecked(),  'dpi': self.w_dpi.text(),
            'csv_base': self.w_csv_base.text().strip(),
            'aspect':     self.w_aspect.text(),    'legend_inside': self.w_legend_inside.isChecked(),
            'compare_files': list(self._compare_files),
            'compare_mode':  self.w_compare_mode.currentText(),
            'normalize_s':   self.w_normalize_s.isChecked(),
            'color_beampipes': self.w_color_beampipes.isChecked(),
            'show_markers':     self.w_show_markers.isChecked(),
            'show_markers_bar': self.w_show_markers_bar.isChecked(),
            'fs_axis':  self.w_fs_axis.text(),  'fs_tick':  self.w_fs_tick.text(),
            'fs_title': self.w_fs_title.text(), 'fs_annot': self.w_fs_annot.text(),
            'fs_legend':self.w_fs_legend.text(),
            'show_xz':         self.w_show_xz.isChecked(),
            'show_yz':         self.w_show_yz.isChecked(),
            'show_titles':     self.w_show_titles.isChecked(),
            'panel_spacing':   self.w_panel_spacing.text().strip(),
            'madx_survey':     self.w_madx_survey.text().strip(),
        }

    def _apply_preset(self, data):
        def _st(widget, key):
            if key in data and hasattr(widget, 'setText'): widget.setText(str(data[key]))
        def _sc(widget, key):
            if key in data and hasattr(widget, 'setChecked'): widget.setChecked(bool(data[key]))
        def _sct(widget, key):
            if key in data and hasattr(widget, 'setCurrentText'): widget.setCurrentText(str(data[key]))

        _sct(self.w_code,     'code');     _st(self.w_output, 'output')
        _sct(self.w_layout,   'layout');   _st(self.w_range,  'range')
        _st(self.w_emitx,  'emitx');       _st(self.w_emity,  'emity')
        _st(self.w_sigmadp,'sigmadp');     _st(self.w_nsigma, 'nsigma')
        _sct(self.w_particle,'particle');  _st(self.w_energy, 'energy')
        _st(self.w_title,  'title');       _st(self.w_elem_h, 'elem_h')
        _st(self.w_elem_h_yz,'elem_h_yz'); _st(self.w_fp_xz_range,'fp_xz_range')
        _st(self.w_fp_yz_range,'fp_yz_range')
        _sc(self.w_no_labels,'no_labels'); _sc(self.w_flip_bend,'flip_bend')
        _sc(self.w_dark,'dark_mode');      _sc(self.w_png,'png');  _sc(self.w_pdf,'pdf')
        _st(self.w_dpi,'dpi');             _st(self.w_aspect,'aspect')
        _st(self.w_csv_base, 'csv_base')
        _sc(self.w_legend_inside,'legend_inside')
        _sct(self.w_compare_mode, 'compare_mode')
        _sc(self.w_normalize_s, 'normalize_s')
        _sc(self.w_color_beampipes, 'color_beampipes')
        _sc(self.w_show_markers,     'show_markers')
        _sc(self.w_show_markers_bar, 'show_markers_bar')
        _st(self.w_fs_axis,  'fs_axis');  _st(self.w_fs_tick,  'fs_tick')
        _st(self.w_fs_title, 'fs_title'); _st(self.w_fs_annot, 'fs_annot')
        _st(self.w_fs_legend,'fs_legend')
        _sc(self.w_show_xz, 'show_xz')
        _sc(self.w_show_yz, 'show_yz')
        _sc(self.w_show_titles, 'show_titles')
        _st(self.w_panel_spacing, 'panel_spacing')
        if 'madx_survey' in data: self.w_madx_survey.setText(str(data.get('madx_survey', '')))

        if 'emit_type' in data:
            is_norm = str(data['emit_type']).lower() == 'normalized'
            self.w_emit_norm.setChecked(is_norm)
            self.w_emit_geo.setChecked(not is_norm)

        if 'panels' in data:
            loaded = [{'name': p.get('name', 'Panel'), 'spec': p.get('spec', 'twiss'),
                       'annot_pattern': p.get('annot_pattern', '')}
                      for p in data['panels']]
            if loaded: self._panels = loaded; self._render_panel_list()

        if 'compare_files' in data:
            self._compare_files = list(data['compare_files'])
            self._render_compare_list()

        self._update_emit_ui()

    def _preset_save_dialog(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name: return
        presets = self._load_presets(); presets[name] = self._collect_preset()
        self._save_presets(presets); self._refresh_preset_menu()
        self._log(f"[preset] Saved '{name}'\n", "ok")

    def _preset_delete_dialog(self):
        presets = self._load_presets()
        if not presets:
            QMessageBox.information(self, "Delete Preset", "No saved presets."); return
        name, ok = QInputDialog.getText(self, "Delete Preset",
            "Preset to delete:\n" + ", ".join(presets.keys()))
        if ok and name and name in presets:
            del presets[name]; self._save_presets(presets); self._refresh_preset_menu()
            self._log(f"[preset] Deleted '{name}'\n", "warn")

    def _refresh_preset_menu(self):
        if not hasattr(self, '_preset_menu'): return
        self._preset_menu.clear()
        presets = self._load_presets()
        if not presets:
            a = QAction("(no saved presets)", self); a.setEnabled(False)
            self._preset_menu.addAction(a); return
        for name in presets:
            act = QAction(name, self)
            act.triggered.connect(lambda _=False, n=name:
                self._apply_preset(self._load_presets().get(n, {})))
            self._preset_menu.addAction(act)
        self._preset_menu.addSeparator()
        self._preset_menu.addAction(QAction("Delete a preset…", self,
                                            triggered=self._preset_delete_dialog))

# ═══════════════════════════════════════════════════════════════════════════════
#  Overlay compositors — CustomPanelOverlay, ExprPanelOverlay
# ═══════════════════════════════════════════════════════════════════════════════

class CustomPanelOverlay:
    """Custom panel composer built inside the Panels tab."""
    DTYPES = [
        ('Beta', 'beta'), ('Dispersion', 'disp'), ('Alpha', 'alpha'),
        ('Orbit', 'orbit'), ('Phase Advance', 'phase'), ('Beam Size', 'beamsize'),
    ]

    def __init__(self, layout, on_done):
        self._on_done = on_done
        self._y1_rows = []; self._y2_rows = []

        # Title row
        title_w = QWidget(); title_w.setStyleSheet("background: transparent;")
        title_h = QHBoxLayout(title_w); title_h.setContentsMargins(12, 12, 8, 4)
        lbl = QLabel("CUSTOM PANEL"); lbl.setFont(FONT_SEC)
        lbl.setStyleSheet(f"color: {ACCENT2}; background: transparent;")
        title_h.addWidget(lbl); title_h.addStretch()
        add_b = QPushButton("Add"); add_b.setFont(FONT_BOLD); add_b.setFixedSize(80, 28)
        add_b.setStyleSheet(f"QPushButton {{ background: {ACCENT}; border-radius: 6px; color: white; }} QPushButton:hover {{ background: #3a9fff; }}")
        add_b.clicked.connect(self._ok)
        can_b = QPushButton("Cancel"); can_b.setFont(FONT_MAIN); can_b.setFixedSize(90, 28)
        can_b.setStyleSheet(f"QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; color: {FG_DIM}; }} QPushButton:hover {{ background: {BORDER}; }}")
        can_b.clicked.connect(lambda: on_done(None))
        title_h.addWidget(add_b); title_h.addWidget(can_b)
        layout.addWidget(title_w)

        # Scroll area
        sa, inner_w, inner_v = _make_scroll_widget()
        inner_w.setStyleSheet(f"background: #28282a;")
        layout.addWidget(sa)
        self._inner_v = inner_v

        # Panel name
        lbl2 = QLabel("Panel name"); lbl2.setFont(FONT_MAIN)
        lbl2.setStyleSheet(f"color: {FG_LBL}; background: transparent; padding: 4px 4px 2px 4px;")
        inner_v.addWidget(lbl2)
        self.e_name = QLineEdit("Custom Panel"); self.e_name.setFont(FONT_MONO)
        self.e_name.setFixedWidth(300); self.e_name.setStyleSheet(_ENTRY_SS)
        inner_v.addWidget(self.e_name)

        # Y1
        lbl_y1 = QLabel("Y1 AXIS  (left) — required"); lbl_y1.setFont(FONT_SEC)
        lbl_y1.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 6px 4px 2px 4px;")
        inner_v.addWidget(lbl_y1)
        self._y1_box_w = QWidget(); self._y1_box_w.setStyleSheet(f"background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px;")
        self._y1_box_v = QVBoxLayout(self._y1_box_w); self._y1_box_v.setContentsMargins(4, 4, 4, 4)
        inner_v.addWidget(self._y1_box_w)
        self._add_y1_row()
        add_y1 = QPushButton("+ add dataset to Y1"); add_y1.setFont(FONT_MAIN)
        add_y1.setStyleSheet(f"QPushButton {{ background: transparent; color: {ACCENT2}; border: none; text-align: left; padding: 2px 4px; }} QPushButton:hover {{ color: white; }}")
        add_y1.clicked.connect(self._add_y1_row)
        inner_v.addWidget(add_y1)

        # Y2
        lbl_y2 = QLabel("Y2 AXIS  (right) — optional"); lbl_y2.setFont(FONT_SEC)
        lbl_y2.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 6px 4px 2px 4px;")
        inner_v.addWidget(lbl_y2)
        self._y2_box_w = QWidget(); self._y2_box_w.setStyleSheet(f"background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px;")
        self._y2_box_v = QVBoxLayout(self._y2_box_w); self._y2_box_v.setContentsMargins(4, 4, 4, 4)
        inner_v.addWidget(self._y2_box_w)
        self._y2_box_w.hide()
        add_y2 = QPushButton("+ add dataset to Y2"); add_y2.setFont(FONT_MAIN)
        add_y2.setStyleSheet(add_y1.styleSheet())
        add_y2.clicked.connect(self._add_y2_row)
        inner_v.addWidget(add_y2)
        inner_v.addStretch()

    def _make_row(self, box_layout):
        outer = QWidget(); outer.setStyleSheet("background: transparent;")
        ov = QVBoxLayout(outer); ov.setContentsMargins(8, 6, 8, 6); ov.setSpacing(6)
        # Enable + X/Y checkboxes
        top = QWidget(); top.setStyleSheet("background: transparent;")
        th = QHBoxLayout(top); th.setContentsMargins(0, 0, 0, 0); th.setSpacing(16)
        en = QCheckBox("Enable"); en.setChecked(True); en.setFont(FONT_MAIN); en.setStyleSheet(_CHK_SS)
        xc = QCheckBox("X"); xc.setChecked(True); xc.setFont(FONT_MAIN); xc.setStyleSheet(_CHK_SS)
        yc = QCheckBox("Y"); yc.setChecked(True); yc.setFont(FONT_MAIN); yc.setStyleSheet(_CHK_SS)
        th.addWidget(en); th.addWidget(xc); th.addWidget(yc); th.addStretch()
        ov.addWidget(top)
        # Radio buttons for dtype (2 rows × 3)
        rg = QWidget(); rg.setStyleSheet("background: transparent;")
        rgg = QGridLayout(rg); rgg.setContentsMargins(0, 0, 0, 0); rgg.setSpacing(4)
        bg = QButtonGroup(rg)
        for i, (lbl_text, key) in enumerate(self.DTYPES):
            rb = QRadioButton(lbl_text); rb.setFont(FONT_MAIN); rb.setStyleSheet(_RB_SS)
            rb.setProperty("value", key)
            if i == 0: rb.setChecked(True)
            bg.addButton(rb); rgg.addWidget(rb, i // 3, i % 3)
        ov.addWidget(rg)
        box_layout.addWidget(outer)

        def _get():
            if not en.isChecked(): return []
            key = next((b.property("value") for b in bg.buttons() if b.isChecked()), 'beta')
            r = []
            if xc.isChecked(): r.append((key, 'x'))
            if yc.isChecked(): r.append((key, 'y'))
            return r
        return _get

    def _add_y1_row(self):
        if len(self._y1_rows) >= 4: return
        self._y1_rows.append(self._make_row(self._y1_box_v))

    def _add_y2_row(self):
        if len(self._y2_rows) >= 4: return
        if len(self._y2_rows) == 0: self._y2_box_w.show()
        self._y2_rows.append(self._make_row(self._y2_box_v))

    def _ok(self):
        y1 = [ds for fn in self._y1_rows for ds in fn()]
        if not y1:
            QMessageBox.warning(None, "Y1 Required", "Please enable at least one Y1 dataset.")
            return
        y2 = [ds for fn in self._y2_rows for ds in fn()]
        pname = self.e_name.text().strip() or "Custom Panel"
        self._on_done({'name': pname, 'y1': y1, 'y2': y2})

class ExprPanelOverlay:
    """Composer for expression-based panels — in-tab, no Toplevel."""

    def __init__(self, layout, on_done, code='tao',
                 input_file='', xsuite_twiss='4d', xsuite_line=None,
                 madx_survey=None):
        self._on_done      = on_done
        self._code         = code
        self._input_file   = input_file
        self._xsuite_twiss = xsuite_twiss
        self._xsuite_line  = xsuite_line
        self._madx_survey  = madx_survey
        self._last_entry   = None
        self._tao_browser      = None
        self._elegant_browser  = None
        self._madx_browser     = None
        self._xsuite_browser   = None

        # Title row
        title_w = QWidget(); title_w.setStyleSheet("background: transparent;")
        title_h = QHBoxLayout(title_w); title_h.setContentsMargins(12, 12, 8, 4)
        lbl = QLabel("EXPRESSION PANEL"); lbl.setFont(FONT_SEC)
        lbl.setStyleSheet(f"color: {ACCENT2}; background: transparent;")
        title_h.addWidget(lbl); title_h.addStretch()
        add_b = QPushButton("Add"); add_b.setFont(FONT_BOLD); add_b.setFixedSize(80, 28)
        add_b.setStyleSheet(f"QPushButton {{ background: {ACCENT}; border-radius: 6px; color: white; }} QPushButton:hover {{ background: #3a9fff; }}")
        add_b.clicked.connect(self._ok)
        can_b = QPushButton("Cancel"); can_b.setFont(FONT_MAIN); can_b.setFixedSize(90, 28)
        can_b.setStyleSheet(f"QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; color: {FG_DIM}; }} QPushButton:hover {{ background: {BORDER}; }}")
        can_b.clicked.connect(lambda: on_done(None))
        title_h.addWidget(add_b); title_h.addWidget(can_b)
        layout.addWidget(title_w)

        sa, inner_w, inner_v = _make_scroll_widget()
        inner_w.setStyleSheet(f"background: #28282a;")
        layout.addWidget(sa)
        self._inner_v = inner_v
        self._inner_w = inner_w

        # Panel name
        lbl2 = QLabel("Panel name"); lbl2.setFont(FONT_MAIN)
        lbl2.setStyleSheet(f"color: {FG_LBL}; background: transparent; padding: 4px 4px 2px 4px;")
        inner_v.addWidget(lbl2)
        self.e_name = QLineEdit("Expression Panel"); self.e_name.setFont(FONT_MONO)
        self.e_name.setFixedWidth(380); self.e_name.setStyleSheet(_ENTRY_SS)
        inner_v.addWidget(self.e_name)

        # Hint
        hint = {
            'tao':     "Namespace: s, beta_a/b, alpha_a/b, eta_x/y, orbit_x/y, phi_a/b + any Tao attribute + global scalars",
            'elegant': "Namespace: s, beta_a/b, alpha_a/b, eta_x/y, orbit_x/y, phi_a/b + twi columns + scalar parameters",
            'xsuite':  "Namespace: s, beta_a/b, alpha_a/b, eta_x/y, orbit_x/y, phi_a/b + twiss columns + global scalars",
        }.get(code, "")
        hl = QLabel(hint); hl.setFont(FONT_SMALL); hl.setWordWrap(True)
        hl.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 2px 4px 4px 4px;")
        inner_v.addWidget(hl)

        # Buttons row
        btn_row_w = QWidget(); btn_row_w.setStyleSheet("background: transparent;")
        btn_row_h = QHBoxLayout(btn_row_w); btn_row_h.setContentsMargins(4, 0, 4, 8); btn_row_h.setSpacing(6)
        _b = lambda text, cmd, color: (lambda b: (
            b.setFont(FONT_MAIN), b.setFixedSize(len(text)*8+20, 30),
            b.setStyleSheet(f"QPushButton {{ background: {BG}; border: 1px solid {color}; border-radius: 6px; color: {color}; }} QPushButton:hover {{ background: {color}; color: white; }}"),
            b.clicked.connect(cmd),
            btn_row_h.addWidget(b)
        ))(QPushButton(text))

        if code == 'tao':
            _b("Browse Tao data ⌕",      self._open_tao_browser,      SUCCESS)
        elif code == 'elegant':
            _b("Show available data ↗",  self._show_data_popup,       ACCENT2)
            _b("Browse ELEGANT data ⌕",  self._open_elegant_browser,  SUCCESS)
        elif code == 'madx':
            _b("Show available data ↗",  self._show_data_popup,       ACCENT2)
            _b("Browse MAD-X data ⌕",    self._open_madx_browser,     SUCCESS)
        elif code == 'xsuite':
            _b("Show available data ↗",  self._show_data_popup,       ACCENT2)
            _b("Browse xsuite data ⌕",   self._open_xsuite_browser,   SUCCESS)
        btn_row_h.addStretch()
        inner_v.addWidget(btn_row_w)

        # Extra attrs (non-tao only)
        if code != 'tao':
            lbl_ex = QLabel("EXTRA ATTRIBUTES TO FETCH"); lbl_ex.setFont(FONT_SEC)
            lbl_ex.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 4px 4px 2px 4px;")
            inner_v.addWidget(lbl_ex)
            _help(inner_v, "Extra attributes to fetch, e.g.: k1, k2, angle")
            self.e_extra = QLineEdit(); self.e_extra.setFont(FONT_MONO)
            self.e_extra.setPlaceholderText("e.g.  k1, k2  (leave blank if not needed)")
            self.e_extra.setFixedWidth(380); self.e_extra.setStyleSheet(_ENTRY_SS)
            inner_v.addWidget(self.e_extra)
        else:
            self.e_extra = None
            note = QLabel("Tao: attributes are auto-fetched from dot-names in your expression")
            note.setFont(FONT_SMALL); note.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 2px 4px 8px 4px;")
            inner_v.addWidget(note)

        # Y1
        lbl_y1 = QLabel("Y1 EXPRESSION  (left axis) — required"); lbl_y1.setFont(FONT_SEC)
        lbl_y1.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 4px 4px 2px 4px;")
        inner_v.addWidget(lbl_y1)
        _help(inner_v, "e.g.: k1 * beta_a  or  np.sqrt(beta_a * beta_b)")
        self.e_y1 = QLineEdit(); self.e_y1.setFont(FONT_MONO)
        self.e_y1.setPlaceholderText("e.g.  k1 * beta_a  or  s1, s2, s3  (comma-separated)")
        self.e_y1.setFixedWidth(380); self.e_y1.setStyleSheet(_ENTRY_SS)
        inner_v.addWidget(self.e_y1)
        r1_w = QWidget(); r1_w.setStyleSheet("background: transparent;")
        r1_h = QHBoxLayout(r1_w); r1_h.setContentsMargins(4, 0, 4, 8)
        lbl_y1l = QLabel("Y1 label:"); lbl_y1l.setFont(FONT_MAIN); lbl_y1l.setFixedWidth(80)
        lbl_y1l.setStyleSheet(f"color: {FG_LBL}; background: transparent;")
        self.e_y1_label = QLineEdit(); self.e_y1_label.setFont(FONT_MONO)
        self.e_y1_label.setPlaceholderText("axis label (defaults to expression)")
        self.e_y1_label.setFixedWidth(280); self.e_y1_label.setStyleSheet(_ENTRY_SS)
        r1_h.addWidget(lbl_y1l); r1_h.addWidget(self.e_y1_label); r1_h.addStretch()
        inner_v.addWidget(r1_w)

        # Y2
        lbl_y2 = QLabel("Y2 EXPRESSION  (right axis) — optional"); lbl_y2.setFont(FONT_SEC)
        lbl_y2.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 4px 4px 2px 4px;")
        inner_v.addWidget(lbl_y2)
        _help(inner_v, "e.g.: eta_x / np.sqrt(beta_a)  — leave blank for none.")
        self.e_y2 = QLineEdit(); self.e_y2.setFont(FONT_MONO)
        self.e_y2.setPlaceholderText("e.g.  eta_x / np.sqrt(beta_a)  (leave blank for none)")
        self.e_y2.setFixedWidth(380); self.e_y2.setStyleSheet(_ENTRY_SS)
        inner_v.addWidget(self.e_y2)
        r2_w = QWidget(); r2_w.setStyleSheet("background: transparent;")
        r2_h = QHBoxLayout(r2_w); r2_h.setContentsMargins(4, 0, 4, 8)
        lbl_y2l = QLabel("Y2 label:"); lbl_y2l.setFont(FONT_MAIN); lbl_y2l.setFixedWidth(80)
        lbl_y2l.setStyleSheet(f"color: {FG_LBL}; background: transparent;")
        self.e_y2_label = QLineEdit(); self.e_y2_label.setFont(FONT_MONO)
        self.e_y2_label.setPlaceholderText("axis label (defaults to expression)")
        self.e_y2_label.setFixedWidth(280); self.e_y2_label.setStyleSheet(_ENTRY_SS)
        r2_h.addWidget(lbl_y2l); r2_h.addWidget(self.e_y2_label); r2_h.addStretch()
        inner_v.addWidget(r2_w)
        inner_v.addStretch()

        # Focus tracking for insert-at-cursor — use focusChanged signal (safe on C++ objects)
        _tracked = [x for x in (self.e_extra, self.e_y1, self.e_y2) if x is not None]
        def _on_focus_changed(old_w, new_w):
            if new_w in _tracked:
                self._on_entry_focus(new_w)
        QApplication.instance().focusChanged.connect(_on_focus_changed)
        self._focus_conn = _on_focus_changed   # keep reference alive
        self._last_entry = self.e_y1

    def _on_entry_focus(self, entry):
        self._last_entry = entry
        if self._tao_browser and self._tao_browser.is_open():
            self._tao_browser.set_target(entry)

    def _open_tao_browser(self):
        if self._tao_browser and self._tao_browser.is_open():
            self._tao_browser.set_target(self._last_entry)
            self._tao_browser.lift(); return
        parent_win = self._inner_w.window()
        self._tao_browser = TaoDataBrowser(parent_win, target_entry=self._last_entry)

    def _open_elegant_browser(self):
        if self._elegant_browser and self._elegant_browser.is_open():
            self._elegant_browser.set_target(self._last_entry)
            self._elegant_browser.lift(); return
        parent_win = self._inner_w.window()
        # Resolve to absolute path so _parse_ele_outputs finds files relative to .ele location
        ele_abs = str(Path(self._input_file).resolve()) if self._input_file else ''
        self._elegant_browser = ElegantDataBrowser(
            parent_win, ele_file=ele_abs, target_entry=self._last_entry)

    def _open_madx_browser(self):
        if self._madx_browser and self._madx_browser.is_open():
            self._madx_browser.set_target(self._last_entry)
            self._madx_browser.lift(); return
        parent_win = self._inner_w.window()
        twiss_abs   = str(Path(self._input_file).resolve()) if self._input_file else ''
        survey_abs  = str(Path(self._madx_survey).resolve()) if self._madx_survey else None
        self._madx_browser = MadxDataBrowser(
            parent_win, twiss_file=twiss_abs, survey_file=survey_abs,
            target_entry=self._last_entry)

    def _open_xsuite_browser(self):
        if self._xsuite_browser and self._xsuite_browser.is_open():
            self._xsuite_browser.set_target(self._last_entry)
            self._xsuite_browser.lift(); return
        parent_win = self._inner_w.window()
        file_abs = str(Path(self._input_file).resolve()) if self._input_file else ''
        self._xsuite_browser = XsuiteDataBrowser(
            parent_win, json_file=file_abs,
            xsuite_twiss=self._xsuite_twiss, xsuite_line=self._xsuite_line,
            target_entry=self._last_entry)

    def _show_data_popup(self):
        if not self._input_file:
            QMessageBox.warning(None, "No Input File",
                "Please select an input file in the Input tab first."); return
        loading = QLabel("Loading lattice data…"); loading.setFont(FONT_MAIN)
        loading.setStyleSheet(f"color: {WARN}; background: transparent; padding: 2px 4px;")
        self._inner_v.insertWidget(self._inner_v.count() - 1, loading)

        class _Bridge(QObject):
            done = Signal(object)

        bridge = _Bridge()
        bridge.done.connect(lambda result: (loading.deleteLater(), self._open_data_popup(result)))

        input_abs  = str(Path(self._input_file).resolve()) if self._input_file else ''
        survey_abs = str(Path(self._madx_survey).resolve()) if getattr(self, '_madx_survey', None) else None

        def _worker():
            try:
                result = _inspect_available_data(
                    input_abs, self._code,
                    xsuite_twiss=getattr(self, '_xsuite_twiss', '4d'),
                    xsuite_line=getattr(self, '_xsuite_line', None),
                    madx_survey=survey_abs)
            except Exception as e:
                result = {'standard': [], 'extra': [], 'scalars': [],
                          'error': str(e)}
            bridge.done.emit(result)

        threading.Thread(target=_worker, daemon=True).start()

    def _open_data_popup(self, result):
        if result.get('error'):
            QMessageBox.critical(None, "Lattice Load Error",
                f"Could not load lattice:\n\n{result['error'][:500]}"); return

        win = QDialog()
        win.setWindowTitle("Available Data")
        win.resize(600, 700)
        win.setStyleSheet(f"background: {BG}; color: {FG};")
        dv = QVBoxLayout(win); dv.setContentsMargins(16, 12, 16, 8)

        hdr = QLabel("AVAILABLE DATA"); hdr.setFont(FONT_SEC)
        hdr.setStyleSheet(f"color: {ACCENT2};")
        dv.addWidget(hdr)
        sub = QLabel("Click any name to insert into the focused field.")
        sub.setFont(FONT_SMALL); sub.setStyleSheet(f"color: {FG_DIM};")
        dv.addWidget(sub)

        sa, inner_w, inner_v = _make_scroll_widget()
        dv.addWidget(sa)

        def _section(title):
            sl = QLabel(title); sl.setFont(FONT_SEC)
            sl.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 8px 2px 8px;")
            inner_v.addWidget(sl)
            line = QFrame(); line.setFrameShape(QFrame.HLine)
            line.setStyleSheet(f"background: {BORDER};")
            inner_v.addWidget(line)

        def _insert(name):
            entry = self._last_entry
            if entry is None: return
            cur = entry.text().strip()
            entry.setText((cur + ", " + name) if cur else name)

        def _item(name, desc, val=None):
            rw = QWidget(); rw.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(8)
            nb = QPushButton(name); nb.setFont(FONT_MONO); nb.setFixedHeight(24)
            nb.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
            nb.clicked.connect(lambda _=False, n=name: _insert(n))
            rh.addWidget(nb)
            if val is not None:
                vl = QLabel(str(val)); vl.setFont(FONT_MONO)
                vl.setStyleSheet(f"color: {SUCCESS}; background: transparent;")
                rh.addWidget(vl)
            if desc:
                dl = QLabel(desc); dl.setFont(FONT_SMALL)
                dl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
                rh.addWidget(dl)
            rh.addStretch(); inner_v.addWidget(rw)

        _section("Standard Arrays  (always available)")
        for name, desc in result.get('standard', []):
            _item(name, desc)

        if result.get('extra'):
            _section("Extra Attributes  (add to \"Extra attributes\" field)")
            for item in result['extra']:
                name = item[0]; desc = item[1] if len(item) > 1 else ''
                _item(name, desc)

        inner_v.addStretch()
        close_b = QPushButton("Close"); close_b.setFont(FONT_MAIN); close_b.setFixedWidth(80)
        close_b.setStyleSheet(_BTN_SS); close_b.clicked.connect(win.close)
        dv.addWidget(close_b, alignment=Qt.AlignCenter)
        win.exec()

    def _parse(self, text):
        return [a.strip() for a in text.replace(',', ' ').split() if a.strip()]

    def _ok(self):
        y1_expr = self.e_y1.text().strip()
        if not y1_expr:
            QMessageBox.warning(None, "Y1 Required", "Please enter a Y1 expression."); return
        y2_expr = self.e_y2.text().strip()
        extra   = self._parse(self.e_extra.text()) if self.e_extra else []
        y1_label = self.e_y1_label.text().strip() or None
        y2_label = self.e_y2_label.text().strip() or None
        name     = self.e_name.text().strip() or "Expression Panel"
        self._on_done({
            'name': name, 'type': 'expr',
            'extra_attrs': extra,
            'y1_expr': y1_expr, 'y2_expr': y2_expr,
            'y1_label': y1_label, 'y2_label': y2_label,
        })

# ═══════════════════════════════════════════════════════════════════════════════
#  Data browsers — TaoDataBrowser, ElegantDataBrowser
# ═══════════════════════════════════════════════════════════════════════════════

def _build_browser_window(title, width=440, height=720, parent=None):
    """Create a styled floating QDialog-style QWidget browser window."""
    win = QWidget(None, Qt.Window)
    win.setWindowTitle(title)
    win.resize(width, height)
    win.setStyleSheet(f"background: {BG}; color: {FG};")
    if parent:
        pg = parent.geometry()
        win.move(pg.right() + 10, pg.top())
    return win

def _make_browser_layout(win, title_text, subtitle_text=""):
    """Build header + search bar + scroll area for browser windows.
    Returns (main_vbox, inner_vbox, search_entry, all_items_list)."""
    vbox = QVBoxLayout(win); vbox.setContentsMargins(0, 0, 0, 0); vbox.setSpacing(0)

    # Header
    hdr = QWidget(); hdr.setFixedHeight(44)
    hdr.setStyleSheet(f"background: {PANEL};")
    hh = QHBoxLayout(hdr); hh.setContentsMargins(12, 0, 12, 0)
    tl = QLabel(title_text); tl.setFont(FONT_SEC); tl.setStyleSheet(f"color: {ACCENT2};")
    hh.addWidget(tl)
    if subtitle_text:
        sl = QLabel(subtitle_text); sl.setFont(FONT_SMALL); sl.setStyleSheet(f"color: {FG_DIM};")
        hh.addWidget(sl)
    hh.addStretch()
    vbox.addWidget(hdr)

    # Search bar
    sf = QWidget(); sf.setStyleSheet(f"background: {BG};"); sf.setFixedHeight(40)
    sh = QHBoxLayout(sf); sh.setContentsMargins(8, 4, 8, 4); sh.setSpacing(4)
    fl = QLabel("Filter:"); fl.setFont(FONT_MAIN); fl.setStyleSheet(f"color: {FG_LBL};")
    sh.addWidget(fl)
    search_e = QLineEdit(); search_e.setFont(FONT_MONO); search_e.setFixedWidth(200)
    search_e.setStyleSheet(_ENTRY_SS)
    clr_b = QPushButton("✕"); clr_b.setFont(FONT_MAIN); clr_b.setFixedSize(28, 24)
    clr_b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {FG_DIM}; border: none; }} QPushButton:hover {{ color: {FG}; }}")
    clr_b.clicked.connect(lambda: search_e.clear())
    sh.addWidget(search_e); sh.addWidget(clr_b); sh.addStretch()
    vbox.addWidget(sf)

    # Scroll area
    sa, inner_w, inner_v = _make_scroll_widget()
    vbox.addWidget(sa, 1)

    all_items = []   # list of (row_widget, name, desc)

    def _filter_fn():
        q = search_e.text().lower()
        for row_w, name, desc in all_items:
            row_w.setVisible(not q or q in name.lower() or q in desc.lower())

    search_e.textChanged.connect(lambda _: _filter_fn())
    return vbox, inner_v, search_e, all_items

class TaoDataBrowser:
    """Floating window showing all Tao lat data types from Table 6.2."""

    def __init__(self, parent_window, target_entry=None):
        self._target = target_entry
        self._win    = None
        self._all_items = []
        self._build(parent_window)

    def _build(self, parent):
        self._win = _build_browser_window(
            "Tao Data Types — Table 6.2", 440, 720, parent)
        vbox, inner_v, search_e, self._all_items = _make_browser_layout(
            self._win, "Tao Data Types", "(lat source only)  •  click to insert")
        self._inner_v = inner_v
        self._build_content()
        # Footer
        foot = QLabel("Tao native names — auto-fetched when used in expressions")
        foot.setFont(FONT_SMALL); foot.setAlignment(Qt.AlignCenter)
        foot.setStyleSheet(f"color: {FG_DIM}; padding: 4px;")
        vbox.addWidget(foot)
        self._win.show()

    def _build_content(self):
        for cat, items in _TAO_DATA_CATEGORIES.items():
            # Category header
            cl = QLabel(cat); cl.setFont(FONT_SEC)
            cl.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 4px 2px 4px;")
            self._inner_v.addWidget(cl)
            line = QFrame(); line.setFrameShape(QFrame.HLine)
            line.setStyleSheet(f"background: {BORDER};")
            self._inner_v.addWidget(line)
            for name, desc in items:
                rw = self._make_item(name, desc)
                self._inner_v.addWidget(rw)
                self._all_items.append((rw, name, desc))
        self._inner_v.addStretch()

    def _make_item(self, name, desc):
        rw = QWidget(); rw.setStyleSheet("background: transparent;")
        rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(6)
        b = QPushButton(name); b.setFont(FONT_MONO); b.setFixedHeight(24)
        b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
        b.clicked.connect(lambda _=False, n=name: self._insert(n))
        dl = QLabel(desc); dl.setFont(FONT_SMALL); dl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
        rh.addWidget(b); rh.addWidget(dl); rh.addStretch()
        return rw

    def _insert(self, name):
        if self._target is None: return
        cur = self._target.text().strip()
        self._target.setText((cur + ", " + name) if cur else name)

    def set_target(self, entry_widget): self._target = entry_widget

    def lift(self):
        if self._win: self._win.raise_(); self._win.activateWindow()

    def is_open(self):
        return bool(self._win and self._win.isVisible())

def _parse_ele_outputs(ele_file):
    """Parse an ELEGANT .ele file and return output file paths.

    Returns dict with keys 'twi', 'cen', 'sig' — values are absolute
    paths or None if not configured in the .ele file.
    """
    result = {'twi': None, 'cen': None, 'sig': None}
    try:
        ele_dir = str(Path(ele_file).parent)
        with open(ele_file, 'r') as f:
            content = f.read()
        # Strip comments
        lines = []
        for line in content.splitlines():
            if '!' in line: line = line[:line.index('!')]
            lines.append(line)
        text = ' '.join(lines)

        # Helper: find filename= value in a namelist block
        def _find_param(block_name, param):
            pat = rf'&{block_name}.*?{param}\s*=\s*"([^"]+)"'
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m: return m.group(1)
            pat2 = rf'&{block_name}.*?{param}\s*=\s*(\S+)'
            m2 = re.search(pat2, text, re.IGNORECASE | re.DOTALL)
            if m2: return m2.group(1).rstrip(',')
            return None

        # run_setup provides rootname — used to expand %s
        # If not explicitly set, ELEGANT defaults to the .ele file stem
        rootname = _find_param('run_setup', 'rootname') or Path(ele_file).stem
        def _expand(p):
            if p is None: return None
            p = p.replace('%s', rootname)
            if not os.path.isabs(p):
                p = os.path.join(ele_dir, p)
            # ELEGANT appends .twi/.cen/.sig automatically if no extension
            return p

        # twiss_output filename
        twi = _find_param('twiss_output', 'filename')
        if twi:
            p = _expand(twi)
            result['twi'] = p if '.' in Path(p).name else p + '.twi'

        # centroid from run_setup
        cen = _find_param('run_setup', 'centroid')
        if cen:
            p = _expand(cen)
            result['cen'] = p if '.' in Path(p).name else p + '.cen'

        # sigma from run_setup
        sig = _find_param('run_setup', 'sigma')
        if sig:
            p = _expand(sig)
            result['sig'] = p if '.' in Path(p).name else p + '.sig'

    except Exception as _pe:
        result['_parse_error'] = str(_pe)
    return result


def _sddsquery(filepath):
    """Query an SDDS file using sddsquery.
    Returns (columns, parameters) where each is a list of (name, description) tuples.
    Returns ([], []) if sddsquery is not available or the file can't be read.

    sddsquery output format (fixed columns):
        NAME  UNITS  SYMBOL  FORMAT  TYPE  FIELD_LENGTH  DESCRIPTION
    The DESCRIPTION is the last whitespace-separated token(s) after FIELD_LENGTH.
    Columns and parameters are in separate sections separated by a blank line.
    """
    def _parse_section(lines):
        """Parse one section (columns or parameters) of sddsquery output.
        Skips the header line (NAME UNITS SYMBOL ...) and blank lines.
        Returns list of (name, description) tuples.
        """
        items = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip header line
            if stripped.startswith('NAME') and 'DESCRIPTION' in stripped:
                continue
            # Skip the units/format sub-header (second header line with just dashes or LENGTH)
            if stripped == 'LENGTH' or stripped.startswith('---'):
                continue
            # Split on whitespace — name is first token, description is last token(s)
            # Format: name  units  symbol  format  type  field_len  description
            # Description may be NULL or a multi-word string
            parts = stripped.split()
            if len(parts) < 1:
                continue
            name = parts[0]
            # Description: everything after the 6th field (field_length), or last token
            # If fewer than 7 fields, no description
            if len(parts) >= 7:
                desc = ' '.join(parts[6:])
                if desc == 'NULL':
                    desc = ''
            else:
                desc = ''
            items.append((name, desc))
        return items

    columns = []; parameters = []
    try:
        r = subprocess.run(['sddsquery', filepath],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError("sddsquery failed")

        # Split output into columns section and parameters section
        text = r.stdout
        col_section = []; par_section = []
        in_cols = False; in_params = False

        for line in text.splitlines():
            low = line.lower()
            if 'columns of data' in low or 'column of data' in low:
                in_cols = True; in_params = False; continue
            if 'parameters' in low and ('parameter' in low):
                in_params = True; in_cols = False; continue
            if in_cols:
                col_section.append(line)
            elif in_params:
                par_section.append(line)

        columns   = _parse_section(col_section)
        parameters = _parse_section(par_section)

    except Exception:
        # Fall back to -columnList / -parameterList for just names
        try:
            r = subprocess.run(['sddsquery', '-columnList', filepath],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                columns = [(l.strip(), '') for l in r.stdout.splitlines() if l.strip()]
        except Exception:
            pass
        try:
            r = subprocess.run(['sddsquery', '-parameterList', filepath],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                parameters = [(l.strip(), '') for l in r.stdout.splitlines() if l.strip()]
        except Exception:
            pass

    return columns, parameters


class ElegantDataBrowser:
    """Floating window showing available ELEGANT output data."""

    def __init__(self, parent_window, ele_file, target_entry=None):
        self._target   = target_entry
        self._win      = None
        self._all_items = []
        self._ele_file = ele_file
        self._outputs  = _parse_ele_outputs(ele_file) if ele_file else {}
        self._build(parent_window)

    def _build(self, parent):
        self._win = _build_browser_window(
            "ELEGANT Data Browser", 460, 740, parent)
        vbox, inner_v, search_e, self._all_items = _make_browser_layout(
            self._win, "ELEGANT Data", "click to insert")
        self._inner_v = inner_v
        self._build_content()
        foot = QLabel("Columns are plottable vs s  •  Scalars are lattice-wide values")
        foot.setFont(FONT_SMALL); foot.setAlignment(Qt.AlignCenter)
        foot.setStyleSheet(f"color: {FG_DIM}; padding: 4px;")
        vbox.addWidget(foot)
        self._win.show()

    def _section(self, title, subtitle=""):
        lbl = QLabel(title + (f"  {subtitle}" if subtitle else ""))
        lbl.setFont(FONT_SEC)
        lbl.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 4px 2px 4px;")
        self._inner_v.addWidget(lbl)
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {BORDER};")
        self._inner_v.addWidget(line)

    def _unavailable_note(self, msg):
        lbl = QLabel(msg); lbl.setFont(FONT_SMALL)
        lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 2px 16px 6px 16px;")
        self._inner_v.addWidget(lbl)

    def _add_items(self, items):
        for name, desc in items:
            rw = QWidget(); rw.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(6)
            b = QPushButton(name); b.setFont(FONT_MONO); b.setFixedHeight(24)
            b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
            b.clicked.connect(lambda _=False, n=name: self._insert(n))
            dl = QLabel(desc); dl.setFont(FONT_SMALL); dl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
            rh.addWidget(b); rh.addWidget(dl); rh.addStretch()
            self._inner_v.addWidget(rw)
            self._all_items.append((rw, name, desc))

    def _build_content(self):
        # Show parse error or resolved .ele path for diagnostics
        if self._outputs.get('_parse_error'):
            self._unavailable_note(f"⚠  Parse error: {self._outputs['_parse_error']}")
        elif self._ele_file:
            self._unavailable_note(f"📄  {self._ele_file}")

        # Skip column — always present in SDDS, not a plottable quantity
        _skip_cols = {'ElementName', 'ElementType', 'ElementOccurence',
                      'ElementGroup', 'pCentral'}

        def _add_sdds_section(title, filepath, kind):
            """Add a section for one SDDS output file."""
            fname = Path(filepath).name if filepath else ''
            self._section(title, f"({fname})" if fname else "(not configured)")
            if not filepath:
                if kind == 'twi':
                    self._unavailable_note("⚠  No &twiss_output filename= found in .ele")
                elif kind == 'cen':
                    self._unavailable_note('⚠  Add centroid="%s.cen" to &run_setup to enable')
                elif kind == 'sig':
                    self._unavailable_note('⚠  Add sigma="%s.sig" to &run_setup to enable')
                return
            if not Path(filepath).exists():
                self._unavailable_note(f"⚠  File not found — run ELEGANT first\n    ({filepath})")
                return
            cols, params = _sddsquery(filepath)
            if not cols and not params:
                # sddsquery not available or failed — fall back to hardcoded list
                fallback = {
                    'twi': _ELEGANT_TWI_COLUMNS,
                    'cen': _ELEGANT_CEN_COLUMNS,
                    'sig': _ELEGANT_SIG_COLUMNS,
                }.get(kind, [])
                if fallback:
                    self._add_items(fallback)
                    self._unavailable_note("(sddsquery not found — showing default column list)")
                else:
                    self._unavailable_note("⚠  sddsquery not found — cannot read column names")
                return
            # Show columns
            col_items = [(name, desc) for name, desc in cols if name not in _skip_cols]
            if col_items:
                self._add_items(col_items)
            # Show parameters as a sub-section
            if params:
                self._section(f"  {title} — Parameters", "(lattice-wide scalars)")
                param_items = [(name, desc) for name, desc in params if name not in _skip_cols]
                if param_items:
                    self._add_items(param_items)

        _add_sdds_section("Twiss Columns",            self._outputs.get('twi'), 'twi')
        _add_sdds_section("Centroid Columns",         self._outputs.get('cen'), 'cen')
        _add_sdds_section("Sigma / Beam Size Columns", self._outputs.get('sig'), 'sig')

        self._inner_v.addStretch()

    def _insert(self, name):
        if self._target is None: return
        cur = self._target.text().strip()
        self._target.setText((cur + ", " + name) if cur else name)

    def set_target(self, entry_widget): self._target = entry_widget

    def lift(self):
        if self._win: self._win.raise_(); self._win.activateWindow()

    def is_open(self):
        return bool(self._win and self._win.isVisible())

    def destroy(self):
        if self._win: self._win.close()


class MadxDataBrowser:
    """Floating browser showing available columns from MAD-X TFS files."""

    def __init__(self, parent_window, twiss_file='', survey_file=None,
                 target_entry=None):
        self._target      = target_entry
        self._win         = None
        self._all_items   = []
        self._twiss_file  = twiss_file
        self._survey_file = survey_file
        self._build(parent_window)

    def _build(self, parent):
        self._win = _build_browser_window(
            "MAD-X Data Browser", 480, 740, parent)
        vbox, inner_v, search_e, self._all_items = _make_browser_layout(
            self._win, "MAD-X TFS Data", "click to insert")
        self._inner_v = inner_v
        self._build_content()
        foot = QLabel("Column names from your TFS files  •  click to insert into expression")
        foot.setFont(FONT_SMALL); foot.setAlignment(Qt.AlignCenter)
        foot.setStyleSheet(f"color: {FG_DIM}; padding: 4px;")
        vbox.addWidget(foot)
        self._win.show()

    def _section(self, title, subtitle=""):
        lbl = QLabel(title + (f"  {subtitle}" if subtitle else ""))
        lbl.setFont(FONT_SEC)
        lbl.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 4px 2px 4px;")
        self._inner_v.addWidget(lbl)
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {BORDER};")
        self._inner_v.addWidget(line)

    def _note(self, msg):
        lbl = QLabel(msg); lbl.setFont(FONT_SMALL)
        lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 2px 16px 6px 16px;")
        self._inner_v.addWidget(lbl)

    def _add_items(self, items):
        for name, desc in items:
            rw = QWidget(); rw.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(6)
            b = QPushButton(name); b.setFont(FONT_MONO); b.setFixedHeight(24)
            b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
            b.clicked.connect(lambda _=False, n=name: self._insert(n))
            dl = QLabel(desc); dl.setFont(FONT_SMALL)
            dl.setStyleSheet(f"color: {FG_DIM}; background: transparent;")
            rh.addWidget(b); rh.addWidget(dl); rh.addStretch()
            self._inner_v.addWidget(rw)
            self._all_items.append((rw, name, desc))

    def _build_content(self):
        _skip = {'NAME', 'KEYWORD', 'PARENT', 'TYPE', 'ORIGIN', 'COMMENTS'}

        # ── Twiss file ──────────────────────────────────────────────────────
        twi_name = Path(self._twiss_file).name if self._twiss_file else ''
        self._section("Twiss Columns", f"({twi_name})" if twi_name else "(no file)")
        if self._twiss_file and Path(self._twiss_file).exists():
            try:
                scalars, col_names, _ = _read_tfs(self._twiss_file)
                cols = [(c.lower(), f"twiss  ({c})") for c in col_names
                        if c.upper() not in _skip]
                if cols:
                    self._add_items(cols)
                else:
                    self._note("⚠  No data columns found in twiss file")
                # Scalars
                if scalars:
                    self._section("Twiss Scalars", "(header @ parameters)")
                    sc_items = [(k.lower(), f"{v}") for k, v in scalars.items()]
                    self._add_items(sc_items)
            except Exception as e:
                self._note(f"⚠  Could not read twiss file: {e}")
        elif self._twiss_file:
            self._note("⚠  Twiss file not found")
        else:
            self._note("⚠  No twiss file loaded")

        # ── Survey file ─────────────────────────────────────────────────────
        sv_name = Path(self._survey_file).name if self._survey_file else ''
        self._section("Survey Columns", f"({sv_name})" if sv_name else "(not loaded)")
        if self._survey_file and Path(self._survey_file).exists():
            try:
                _, sv_cols, _ = _read_tfs(self._survey_file)
                _skip_sv = {'NAME', 'KEYWORD', 'PARENT', 'TYPE'}
                sv_items = [(c.lower(), f"survey  ({c})") for c in sv_cols
                            if c.upper() not in _skip_sv]
                if sv_items:
                    self._add_items(sv_items)
                else:
                    self._note("⚠  No data columns found in survey file")
            except Exception as e:
                self._note(f"⚠  Could not read survey file: {e}")
        elif self._survey_file:
            self._note("⚠  Survey file not found")
        else:
            self._note("No survey file loaded  —  floor plan uses dead-reckoning")

        self._inner_v.addStretch()

    def _insert(self, name):
        if self._target is None: return
        cur = self._target.text().strip()
        self._target.setText((cur + ", " + name) if cur else name)

    def set_target(self, entry_widget): self._target = entry_widget

    def lift(self):
        if self._win: self._win.raise_(); self._win.activateWindow()

    def is_open(self):
        return bool(self._win and self._win.isVisible())

    def destroy(self):
        if self._win: self._win.close()


class XsuiteDataBrowser:
    """Floating browser showing available columns from an xsuite twiss table."""

    def __init__(self, parent_window, json_file='', xsuite_twiss='4d',
                 xsuite_line=None, target_entry=None):
        self._target       = target_entry
        self._win          = None
        self._all_items    = []
        self._json_file    = json_file
        self._xsuite_twiss = xsuite_twiss
        self._xsuite_line  = xsuite_line
        self._build(parent_window)

    def _build(self, parent):
        self._win = _build_browser_window(
            "xsuite Data Browser", 480, 740, parent)
        vbox, inner_v, search_e, self._all_items = _make_browser_layout(
            self._win, "xsuite Twiss Data", "click to insert")
        self._inner_v = inner_v

        # Loading label — replaced once data arrives
        self._loading_lbl = QLabel("Loading xsuite twiss table…")
        self._loading_lbl.setFont(FONT_MAIN)
        self._loading_lbl.setStyleSheet(f"color: {WARN}; background: transparent; padding: 8px;")
        self._inner_v.addWidget(self._loading_lbl)

        foot = QLabel("Column names from xsuite twiss  •  click to insert into expression")
        foot.setFont(FONT_SMALL); foot.setAlignment(Qt.AlignCenter)
        foot.setStyleSheet(f"color: {FG_DIM}; padding: 4px;")
        vbox.addWidget(foot)
        self._win.show()

        # Load in background thread
        class _Bridge(QObject):
            done = Signal(object)
        self._bridge = _Bridge()
        self._bridge.done.connect(self._on_loaded)
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            data = load_xsuite(self._json_file, log_fn=None,
                               twiss_method=self._xsuite_twiss,
                               line_name=self._xsuite_line)
            _skip = {'s', 'elements', 'beam_params', '_tao', '_tw', 'name',
                     'element_type', 'isthick', 'parent_name'}
            cols = sorted([k for k in data.keys()
                           if k not in _skip
                           and isinstance(data.get(k), np.ndarray)])
            self._bridge.done.emit({'cols': cols, 'error': None})
        except Exception as e:
            self._bridge.done.emit({'cols': [], 'error': str(e)})

    def _on_loaded(self, result):
        if not (self._win and self._win.isVisible()):
            return
        self._loading_lbl.deleteLater()
        if result['error']:
            lbl = QLabel(f"⚠  Could not load xsuite file:\n{result['error'][:300]}")
            lbl.setFont(FONT_SMALL)
            lbl.setStyleSheet(f"color: {ERROR}; background: transparent; padding: 8px;")
            lbl.setWordWrap(True)
            self._inner_v.insertWidget(0, lbl)
            return
        cols = result['cols']
        if not cols:
            lbl = QLabel("⚠  No twiss columns found in file")
            lbl.setFont(FONT_SMALL)
            lbl.setStyleSheet(f"color: {FG_DIM}; background: transparent; padding: 8px;")
            self._inner_v.insertWidget(0, lbl)
            return
        # Add section header + items
        sec = QLabel("Twiss Columns"); sec.setFont(FONT_SEC)
        sec.setStyleSheet(f"color: {ACCENT2}; background: transparent; padding: 8px 4px 2px 4px;")
        self._inner_v.insertWidget(0, sec)
        for i, col in enumerate(cols):
            rw = QWidget(); rw.setStyleSheet("background: transparent;")
            rh = QHBoxLayout(rw); rh.setContentsMargins(8, 1, 8, 1); rh.setSpacing(6)
            b = QPushButton(col); b.setFont(FONT_MONO); b.setFixedHeight(24)
            b.setStyleSheet(f"QPushButton {{ background: {PANEL}; color: {ACCENT}; border: none; border-radius: 3px; padding: 1px 6px; }} QPushButton:hover {{ background: {ACCENT}; color: white; }}")
            b.clicked.connect(lambda _=False, n=col: self._insert(n))
            rh.addWidget(b); rh.addStretch()
            self._inner_v.insertWidget(i + 1, rw)
            self._all_items.append((rw, col, ''))

    def _insert(self, name):
        if self._target is None: return
        cur = self._target.text().strip()
        self._target.setText((cur + ", " + name) if cur else name)

    def set_target(self, entry_widget): self._target = entry_widget

    def lift(self):
        if self._win: self._win.raise_(); self._win.activateWindow()

    def is_open(self):
        return bool(self._win and self._win.isVisible())

    def destroy(self):
        if self._win: self._win.close()


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(BG))
    palette.setColor(QPalette.WindowText,      QColor(FG))
    palette.setColor(QPalette.Base,            QColor(PANEL))
    palette.setColor(QPalette.AlternateBase,   QColor(BG))
    palette.setColor(QPalette.Text,            QColor(FG))
    palette.setColor(QPalette.Button,          QColor(PANEL))
    palette.setColor(QPalette.ButtonText,      QColor(FG))
    palette.setColor(QPalette.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor(FG))
    palette.setColor(QPalette.PlaceholderText, QColor(FG_DIM))
    app.setPalette(palette)

    win = LuxV4GUI()
    win.show()
    sys.exit(app.exec())
