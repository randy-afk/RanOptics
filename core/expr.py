# =============================================================================
# core/expr.py — RanOptics expression evaluator engine
# Handles arbitrary user-defined math expressions across all backends
# =============================================================================

from __future__ import annotations
import math, re as _re_expr
import numpy as np

from core.loaders import _query_tao_attrs, _query_elegant_attrs, _query_xsuite_attrs
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

