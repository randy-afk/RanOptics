# =============================================================================
# core/engine.py — RanOptics plot_optics engine
# =============================================================================

from __future__ import annotations
import fnmatch, os, re, time, traceback
from pathlib import Path
import numpy as np

from core.utils import _parse_fp_range, panel_title
from core.loaders import (
    load_tao, load_elegant, load_madx, load_xsuite,
    _parse_tao_init, _load_tao_universe,
)
from core.panels import (
    _build_floor_plan, _build_floor_plan_yz,
    _build_layout_bar, _build_bar_annotations, _build_summary_panel,
    _build_latdiff_panel, _build_twiss_panel, _build_beta_panel,
    _build_dispersion_panel, _build_alpha_panel,
    _build_panel3, _build_panel3_uni,
    _build_custom_panel, _build_expr_panel,
    _build_panel_annotations, _build_tune_annotation,
    _make_elem_name_array, _read_tunnel_wall,
)
from core.expr import _build_expr_namespace
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
    bar_lite=False,          # if True, use floor-plan two-trace method for beamline bar (faster on large lattices)
    show_xz=True,     # show X vs Z floor plan (floor layout)
    show_yz=True,     # show Y vs Z floor plan (floor layout)
    show_titles=True, # show subplot titles
    panel_spacing=80,  # vertical spacing between panels in pixels
    panel_heights=None,  # dict: panel_index -> height_px, overrides default per-panel height
    panel_annotations=None,  # dict: panel_index -> wildcard pattern, e.g. {0: 'IPM*', 2: 'QF*,QD*'}
    legend_positions=None,  # dict: panel_index or 'floor-xz'/'floor-yz' -> [x, y] in normalized 0-1 coords
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
        # In lite mode build element name lookup array once for all panels
        _elem_names = _make_elem_name_array(s, elements) if bar_lite else None
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
                                  show_markers=show_markers_bar, bar_lite=bar_lite)
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
                        uni_idx=_ui, elem_names=_elem_names)
            else:
                # ── Single universe: original path ────────────────────────────
                result = _build_panel3(fig, p, s, ba, bb, ex, ey, ox, oy, pa, pb,
                              row=current_row, legend_name=legend_n,
                              row3_secondary=has_secondary,
                              beam_params=bp_full, elem_names=_elem_names)

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

        _lpos = legend_positions or {}

        def _lgd(row_idx, pos_key=None):
            y_top = row_tops[row_idx] if row_idx < len(row_tops) else 1.0
            usr = _lpos.get(pos_key) if pos_key is not None else None
            if usr and len(usr) == 2:
                try:
                    ux, uy = float(usr[0]), float(usr[1])
                    # Convert per-panel normalized coords to paper coords:
                    # ux: 0=left edge, 1=right edge (already paper space)
                    # uy: 0=bottom of this panel, 1=top of this panel
                    row_h = row_heights_norm[row_idx] if row_idx < len(row_heights_norm) else 0.1
                    paper_y = y_top - (1.0 - uy) * row_h
                    return dict(
                        itemsizing='constant',
                        bgcolor=LEGEND_BG,
                        bordercolor=LEGEND_BC,
                        borderwidth=1,
                        x=ux, xanchor='left',
                        y=paper_y, yanchor='top')
                except (ValueError, TypeError):
                    pass
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
            for _fi, _fp in enumerate(_floor_panels):
                for _ui in range(len(_plot_unis)):
                    _fp_leg = 'legend91' if _ui == 0 else f'legend{90 + _ui * 2 + 1}'
                    _el_leg = 'legend90' if _ui == 0 else f'legend{90 + _ui * 2}'
                    if row_idx < len(row_tops):
                        lkw[_fp_leg] = _lgd(row_idx, pos_key=_fp)
                        lkw[_el_leg] = _lgd(row_idx, pos_key=_fp)
                row_idx += 1

        for i, p in enumerate(panels):
            if p in ('floor-xz', 'floor-yz'): continue
            if row_idx < len(row_tops):
                # Use spec string for preset panels, index for custom/expr (data panels preserve order)
                _PRESET_SPECS = {'twiss','beta','dispersion','alpha','orbit','phase','beamsize','twiss_disp','bar','summary','latdiff'}
                _pk = p if (isinstance(p, str) and p in _PRESET_SPECS) else (p.get('name', i) if isinstance(p, dict) else i)
                lkw[f'legend{i+1}'] = _lgd(row_idx, pos_key=_pk)
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
                                  show_markers=show_markers_bar, bar_lite=bar_lite)
                bar_fig.update_xaxes(title_text='s (m)' if not normalize_s else 's/s_max',
                                     row=1, col=1)
                bar_fig.update_yaxes(title_text='', showticklabels=False,
                                     range=[-0.4, 0.4], row=1, col=1)
                for ri, c in enumerate(_csep, start=2):
                    _build_layout_bar(bar_fig, c['elems'], show_element_labels, row=ri,
                                      show_markers=show_markers_bar, bar_lite=bar_lite)
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