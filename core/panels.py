# =============================================================================
# core/panels.py — RanOptics panel and floor plan builders
# =============================================================================

from __future__ import annotations
import copy, fnmatch, re, traceback
import numpy as np

from core.utils import (
    element_color, element_thickness, element_polygon, element_oval,
    make_hover, panel_title, THIN_ELEMENT_THRESHOLD,
)
from core.expr import _build_expr_namespace, _eval_expression, _eval_expression_tao

def _read_tunnel_wall(filepath, log_fn=None):
    """Read tunnel wall coordinate file.

    Format: each row has 6 values (any delimiter — comma, tab, space):
        x_inner  y_inner  z_inner  x_outer  y_outer  z_outer

    Returns dict with keys:
        xi, yi, zi  — inner wall arrays
        xo, yo, zo  — outer wall arrays
    or None on failure.
    """
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
                parts = re.split(r'[,\t\s]+', line)
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


def _build_layout_bar(fig, elements, show_labels, row=4, show_markers=False, bar_lite=False):
    import plotly.graph_objects as go
    _MARKER_MONITOR_KEYS = {'marker','mark','monitor','hmon','vmon','instrument','bpm'}
    if not show_markers:
        elements = [e for e in elements if e['key'].lower() not in _MARKER_MONITOR_KEYS]

    if bar_lite:
        # ── Lite mode: use exact same two-trace method as floor plan ──────────
        # Elements laid out linearly (theta=0). No bends, no dead-reckoning.
        # One invisible hover line + one visible polygon per element, same as
        # _build_floor_plan. No add_shape calls at all.
        import plotly.graph_objects as go
        th = 0.2  # fixed element thickness for bar
        la = set()
        _lt = {'quadrupole':'Quadrupole','sbend':'Dipole','sextupole':'Sextupole',
               'kicker':'Kicker','hkicker':'Kicker','vkicker':'Kicker',
               'monitor':'Monitor','rfcavity':'RF Cavity','lcavity':'RF Cavity'}
        _lg = f'bar_lite_{row}'
        for elem in elements:
            s0=elem['s_start']; L_=elem['length']; key=elem['key']
            color=element_color(key); kl=key.lower()
            hover=make_hover(elem)
            ll=_lt.get(kl)
            if L_==0:
                if color is None: continue
                fig.add_trace(go.Scatter(
                    x=[s0,s0], y=[-th/2, th/2], mode='lines',
                    line=dict(color=color, width=2),
                    hoverlabel=dict(bgcolor=color),
                    legendgroup=f'{_lg}_{ll or key}', showlegend=False,
                    hovertemplate=hover), row=row, col=1)
                continue
            if color is None: continue
            if L_<THIN_ELEMENT_THRESHOLD:
                fig.add_trace(go.Scatter(
                    x=[s0,s0], y=[-th/2, th/2], mode='lines',
                    line=dict(color=color, width=2),
                    hoverlabel=dict(bgcolor=color),
                    legendgroup=f'{_lg}_{ll or key}', showlegend=False,
                    hovertemplate=hover), row=row, col=1)
                continue
            if 'rfcavity' in kl or 'lcavity' in kl:
                ov_x, ov_y = element_oval(s0, 0.0, 0.0, L_, th/2)
                # Trace 1: invisible hover line
                fig.add_trace(go.Scatter(
                    x=[s0, s0+L_], y=[0.0, 0.0], mode='lines',
                    line=dict(color='rgba(0,0,0,0)', width=max(8, th*60)),
                    hoverlabel=dict(bgcolor=color),
                    legendgroup=f'{_lg}_h_{ll or "o"}', showlegend=False,
                    hovertemplate=hover), row=row, col=1)
                # Trace 2: visible oval
                fig.add_trace(go.Scatter(
                    x=ov_x, y=ov_y, mode='lines', fill='toself',
                    fillcolor=color, line=dict(color=color, width=0),
                    opacity=0.8, name=ll, legendgroup=f'{_lg}_{ll or "o"}',
                    showlegend=False, hoverinfo='skip'), row=row, col=1)
                continue
            if 'quadrupole' in kl:
                pol = elem.get('profile', 0.0) or elem.get('k1', 0.0)
                y0v=0.0 if pol>0 else (-0.2 if pol<0 else -0.1)
                y1v=0.2 if pol>0 else (0.0  if pol<0 else  0.1)
            elif 'sextupole' in kl:
                pol = elem.get('k2', 0.0)
                y0v=0.0 if pol>0 else (-0.2 if pol<0 else -0.1)
                y1v=0.2 if pol>0 else (0.0  if pol<0 else  0.1)
            else:
                y0v,y1v=-th/2,th/2
            ymid=(y0v+y1v)/2.0
            # Trace 1: invisible wide hover line over full element length
            fig.add_trace(go.Scatter(
                x=[s0, s0+L_], y=[ymid, ymid], mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=max(8, abs(y1v-y0v)*60)),
                hoverlabel=dict(bgcolor=color),
                legendgroup=f'{_lg}_h_{ll or "o"}', showlegend=False,
                hovertemplate=hover), row=row, col=1)
            # Trace 2: visible filled polygon — hoverinfo skip
            rx=[s0, s0+L_, s0+L_, s0, s0]
            ry=[y0v, y0v, y1v, y1v, y0v]
            fig.add_trace(go.Scatter(
                x=rx, y=ry, mode='lines', fill='toself',
                fillcolor=color, line=dict(color='black', width=0.5),
                opacity=0.8, name=ll, legendgroup=f'{_lg}_{ll or "o"}',
                showlegend=False, hoverinfo='skip'), row=row, col=1)
    else:
        # ── Standard mode — original add_shape + invisible point scatter ──────
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
                ov_x, ov_y = element_oval(s0, 0.0, 0.0, L_, 0.1)
                fig.add_trace(go.Scatter(x=ov_x, y=ov_y,
                    mode='lines',fill='toself',fillcolor=color,line=dict(color='black',width=0.5),
                    opacity=0.8,name=short,showlegend=False,hoverinfo='skip'),row=row,col=1)
                fig.add_trace(go.Scatter(x=[s0+L_/2],y=[0.0],mode='markers',
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
        new_annots = []
        for elem in elements:
            if elem['key'].lower() not in ('quadrupole','sbend') or elem['length']==0: continue
            new_annots.append(dict(
                x=elem['s_start']+elem['length']/2, y=0.15,
                text=elem['name'].split('\\')[-1], showarrow=False, textangle=-90,
                font=dict(size=7), xref=xref, yref=yref))
        existing_annots = list(fig.layout.annotations) if fig.layout.annotations else []
        fig.update_layout(annotations=existing_annots + new_annots)

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

def _make_elem_name_array(s, elements):
    """Return an array of element names, one per s point.
    Each s value (s_end of element) is matched to its element by index.
    Falls back to empty string if no match found.
    """
    names = np.empty(len(s), dtype=object)
    names[:] = ''
    for e in elements:
        s0 = e['s_start']; s1 = s0 + e['length']; ename = e['name'].split('\\')[-1]
        mask = (s >= s0) & (s <= s1 + 1e-9)
        names[mask] = ename
    return names

def _dot_trace(fig, x, y, name, color, legend_name, legendgroup, row, col,
               hovertemplate='', secondary_y=False, customdata=None):
    """Add a line trace using a colored dot as the legend symbol."""
    import plotly.graph_objects as go
    # Dummy single-point trace for the legend dot
    fig.add_trace(go.Scatter(
        x=[None], y=[None], name=name, mode='markers',
        marker=dict(symbol='circle', size=10, color=color),
        legend=legend_name, legendgroup=legendgroup, showlegend=True,
        hoverinfo='skip'),
        row=row, col=col, **({'secondary_y': secondary_y} if secondary_y is not None else {}))
    # If element names provided, append to hover and attach as customdata
    if customdata is not None:
        ht = hovertemplate.replace('<extra></extra>', '<br>%{customdata}<extra></extra>')
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode='lines',
            line=dict(color=color),
            customdata=customdata,
            legend=legend_name, legendgroup=legendgroup, showlegend=False,
            hovertemplate=ht),
            row=row, col=col, **({'secondary_y': secondary_y} if secondary_y is not None else {}))
    else:
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
                        al_a, al_b, beam_params, row, legend_name, elem_names=None):
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
                   secondary_y=False, customdata=elem_names)

    for dtype, axis in y2_ds:
        y, name, _, fmt = _get_dataset(dtype, axis, s, ba, bb, ex, ey,
                                       ox, oy, pa, pb, al_a, al_b, beam_params)
        color = color_map[(dtype, axis)]
        _dot_trace(fig, s, y, name, color, legend_name, name, row, 1,
                   hovertemplate=f's=%{{x:.3f}} m<br>{fmt}<extra></extra>',
                   secondary_y=True, customdata=elem_names)

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

def _build_twiss_panel(fig, s, ba, bb, ex, ey, row=2, legend_name='legend1', elem_names=None):
    import plotly.graph_objects as go
    for y,name,col,sec,fmt in [
        (ba,'βₓ','blue',False,'βₓ=%{y:.3f} m'),
        (bb,'βᵧ','red', False,'βᵧ=%{y:.3f} m'),
        (ex,'ηₓ','green',True,'ηₓ=%{y:.4f} m'),
        (ey,'ηᵧ','brown',True,'ηᵧ=%{y:.4f} m'),
    ]:
        _dot_trace(fig, s, y, name, col, legend_name, name, row, 1,
            hovertemplate=f's=%{{x:.3f}} m<br>{fmt}<extra></extra>', secondary_y=sec,
            customdata=elem_names)

def _build_beta_panel(fig, s, ba, bb, row=2, legend_name='legend1', elem_names=None):
    import plotly.graph_objects as go
    for y,name,col in [(ba,'βₓ','blue'),(bb,'βᵧ','red')]:
        _dot_trace(fig, s, y, name, col, legend_name, name, row, 1,
            hovertemplate=f's=%{{x:.3f}} m<br>{name}=%{{y:.3f}} m<extra></extra>',
            customdata=elem_names)

def _build_dispersion_panel(fig, s, ex, ey, row=2, legend_name='legend1', elem_names=None):
    import plotly.graph_objects as go
    for y,name,col in [(ex,'ηₓ','green'),(ey,'ηᵧ','brown')]:
        _dot_trace(fig, s, y, name, col, legend_name, name, row, 1,
            hovertemplate=f's=%{{x:.3f}} m<br>{name}=%{{y:.4f}} m<extra></extra>',
            customdata=elem_names)

def _build_alpha_panel(fig, s, al_a, al_b, row=2, legend_name='legend1', elem_names=None):
    import plotly.graph_objects as go
    for y,name,col in [(al_a,'αₓ','blue'),(al_b,'αᵧ','red')]:
        _dot_trace(fig, s, y, name, col, legend_name, name, row, 1,
            hovertemplate=f's=%{{x:.3f}} m<br>{name}=%{{y:.4f}}<extra></extra>',
            customdata=elem_names)

def _build_panel3(fig, panel3, s, ba, bb, ex, ey, ox, oy, pa, pb,
                  row=3, legend_name='legend2', row3_secondary=False, beam_params=None,
                  elem_names=None):
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
                                   pa, pb, al_a, al_b, bp, row, legend_name,
                                   elem_names=elem_names)
    if panel3=='phase':
        for y,n,c in [(pa,'μₓ','blue'),(pb,'μᵧ','red')]:
            _dot_trace(fig, s, y, n, c, legend_name, n, row, 1,
                hovertemplate=f's=%{{x:.3f}} m<br>{n}=%{{y:.4f}}<extra></extra>',
                customdata=elem_names)
    elif panel3=='orbit':
        for y,n,c in [(ox,'x orbit','blue'),(oy,'y orbit','red')]:
            _dot_trace(fig, s, y, n, c, legend_name, n, row, 1,
                hovertemplate=f's=%{{x:.3f}} m<br>{n}=%{{y:.6f}} m<extra></extra>',
                customdata=elem_names)
    elif panel3=='beamsize':
        ex_v=bp.get('emit_x',0.0); ey_v=bp.get('emit_y',0.0); sdp=bp.get('sigma_dp',0.0)
        n_sig=bp.get('n_sigma',1.0)
        sx=n_sig*np.sqrt(np.maximum(ex_v*ba+(ex*sdp)**2,0))*1e3
        sy=n_sig*np.sqrt(np.maximum(ey_v*bb+(ey*sdp)**2,0))*1e3
        n_lbl = f'{n_sig:.4g}·' if n_sig != 1.0 else ''
        for y,n,c in [(sx,f'{n_lbl}σₓ','blue'),(sy,f'{n_lbl}σᵧ','red')]:
            _dot_trace(fig, s, y, n, c, legend_name, n, row, 1,
                hovertemplate=f's=%{{x:.3f}} m<br>{n}=%{{y:.4f}} mm<extra></extra>',
                customdata=elem_names)
    elif panel3=='twiss':
        _build_twiss_panel(fig,s,ba,bb,ex,ey,row=row,legend_name=legend_name,elem_names=elem_names)
    elif panel3=='beta':
        _build_beta_panel(fig,s,ba,bb,row=row,legend_name=legend_name,elem_names=elem_names)
    elif panel3=='dispersion':
        _build_dispersion_panel(fig,s,ex,ey,row=row,legend_name=legend_name,elem_names=elem_names)
    elif panel3=='alpha':
        al_a=beam_params.get('alpha_a',np.zeros_like(s)) if bp else np.zeros_like(s)
        al_b=beam_params.get('alpha_b',np.zeros_like(s)) if bp else np.zeros_like(s)
        _build_alpha_panel(fig,s,al_a,al_b,row=row,legend_name=legend_name,elem_names=elem_names)

def _build_panel3_uni(fig, panel3, s, ba, bb, ex, ey, ox, oy, pa, pb,
                      al_a, al_b, beam_params, row, legend_name,
                      uni_label='u1', palette=None, uni_idx=0, elem_names=None):
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
                   secondary_y=sec, customdata=elem_names)

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


# ─── Annotation helpers ─────────────────────────────────────────────────────

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