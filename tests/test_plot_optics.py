# =============================================================================
# tests/test_plot_optics.py — regression coverage for plot_optics()'s three
# layout modes (floor / stacked panels / grid).
#
# Uses the synthetic xsuite ring in tests/fixtures.py, so it needs no
# external lattice files — only xtrack, an optional RanOptics dependency.
# =============================================================================

import pytest

xt = pytest.importorskip('xtrack')

from tests.fixtures import build_ring_json
from core.engine import plot_optics


@pytest.fixture(scope='module')
def ring_json(tmp_path_factory):
    path = tmp_path_factory.mktemp('fixture') / 'ring.json'
    build_ring_json(path)
    return str(path)


def _assert_rendered(output_file):
    assert output_file.exists()
    html = output_file.read_text()
    assert len(html) > 10_000
    assert 'plotly' in html.lower()


def test_floor_layout(ring_json, tmp_path):
    out = tmp_path / 'floor.html'
    plot_optics(ring_json, code='xsuite', output_file=str(out),
                layout='floor', xsuite_twiss='4d')
    _assert_rendered(out)


def test_panels_layout(ring_json, tmp_path):
    out = tmp_path / 'panels.html'
    plot_optics(ring_json, code='xsuite', output_file=str(out),
                layout='panels', xsuite_twiss='4d',
                panels=['floor-xz', 'twiss', 'orbit', 'bar'])
    _assert_rendered(out)


def test_grid_layout(ring_json, tmp_path):
    out = tmp_path / 'grid.html'
    plot_optics(ring_json, code='xsuite', output_file=str(out),
                layout='panels', xsuite_twiss='4d',
                panels=['twiss', 'orbit'],
                grid_layout={
                    'rows': 2, 'cols': 1,
                    'cells': [
                        {'row': 1, 'col': 1, 'spec': 'twiss', 'name': 'Twiss'},
                        {'row': 2, 'col': 1, 'spec': 'orbit', 'name': 'Orbit'},
                    ],
                })
    _assert_rendered(out)
