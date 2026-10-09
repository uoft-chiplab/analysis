"""
Magnetic field model of the MOT, transfer and bias coils (thesis Fig. 4.9 / Tab. 4.3),
built from current filaments with magpylib (v5, SI units).

Coordinates: origin = geometric centre of the MOT coil pair. Lengths in m, currents in A,
fields in T internally; plots and printouts use cm and G.
    +z : MOT coil axis, towards NW (thesis -x)
    +y : up, towards the transfer coil and the chip (thesis -z)
    +x : completes the right-handed set, towards SW (thesis +y)

Coils (Tab. 4.3 names) in this frame:
    MOT        circular pair along z, anti-Helmholtz
    transfer   single circular coil along y, above the MOT
    Zbias      circular pair along z, wound around the outside of the MOT coils
    Xbias      rectangular (Ioffe) pair along x, in the gap between the MOT coils
    Ybias      circular pair along y (up/down), top coil sitting on the transfer coil

Not in Tab. 4.3 (assumptions - edit COILS and the constants below if you know better):
    - "# turns" is per coil; the tabulated wire lengths agree with this.
    - MOT and transfer winding packs have a square cross-section (axial thickness = radial
      build). This reproduces the tabulated MOT gradient of 1.56 G/cm/A.
    - 14 AWG bias packs: 20 turns of ~1.8 mm wire in the 5 mm radial build -> 13 mm thick,
      modelled with 2 x 4 filaments (the winding pattern is unknown).
    - Y_XFER and Y_CHIP are solved from the transfer coil's 1.60 G/A (MOT) and 1.83 G/A
      (chip). Cross-check: the MOT pair in Helmholtz then gives 6.7 G/A at the chip (Tab: 7.0).
    - Xbias is centred on the MOT; transfer coil is coaxial with the vertical through the MOT.
    - The transfer coil's field points down (polarity -1), pushing the quadrupole zero to
      y < 0: at the QMT currents the atoms sit ~5 cm below the chip.
Tab. 4.3's calculated values are good to ~5%. Run print_calibration() to compare the model
with them and print_trap_positions() for the atom position relative to the chip.

Usage:
    mot = CoilSetup({'MOT': 7.0})              # currents in A per turn
    mot.set_currents(MOT=45.0)                 # change currents later
    full = CoilSetup(CURRENTS_QMT)             # all coils, Tab. 4.3 QMT currents
    plot_slice(full, plane='zy', offset=0)     # field lines on a slice ('xy', 'xz', 'zy', ...)
    plot_slice(full, 'xz', offset=0.02, style='map', quantity='y')
    plot_homogeneity(mot, extent=0.02)         # field or gradient uniformity
    plot_line(full, axis='y')                  # 1D cut of the field components
    find_field_zero(full)                      # quadrupole centre
    plot_field_lines_3d(CoilSetup({'MOT': 7.0}, grid=(2, 2)))  # traced 3D field lines
    show_coils(full, backend='plotly')         # interactive 3D view of the coils
"""
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from dataclasses import dataclass, field
from matplotlib.colors import ListedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle, Wedge, Circle as CirclePatch
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
from scipy.optimize import root
from scipy.spatial.transform import Rotation

T2G = 1e4  # T -> G
CM = 1e2   # m -> cm, for plot axes

#%% Coil geometry (Tab. 4.3)

@dataclass
class CoilSpec:
    """One coil or coil pair. Lengths in m.

    A pair is mirror-symmetric about `center` along `axis`, with its inner faces `inner_sep`
    apart; a single coil's winding pack is centred on `center`. Circular coils take inner and
    outer diameters. Rectangular coils take (u, v) side lengths, where u, v are the axes that
    follow `axis` cyclically: axis 'x' -> (y, z), 'y' -> (z, x), 'z' -> (x, y).
    Each winding pack is modelled as grid[0] (radial) x grid[1] (axial) filaments that
    together carry `turns` x current. `polarity` (+1/-1) sets the field direction for a
    positive current, so currents can be given as the (positive) supply values of Tab. 4.3.
    """
    name: str
    axis: str
    inner: object
    outer: object
    thickness: float
    turns: int
    grid: tuple
    inner_sep: float = None
    config: str = 'helmholtz'       # 'helmholtz', 'anti-helmholtz' or 'single'
    center: tuple = (0.0, 0.0, 0.0)
    shape: str = 'circular'         # or 'rectangular'
    polarity: int = 1               # +1: positive current gives a field along +axis
    color: str = '0.4'              # overlay colour (Fig. 4.9: light grey MOT/transfer, dark grey bias)
    table: dict = field(default_factory=dict)  # Tab. 4.3 values: G/A at MOT/chip, G/cm/A

Y_XFER = 0.0569         # m, centre of the transfer-coil winding pack above the MOT centre
Y_CHIP = 0.0300         # m, chip above the MOT centre
XFER_THICKNESS = 4.2e-2
YBIAS_SEP = 13e-2
CHIP = (0.0, Y_CHIP, 0.0)
MARKS = {'MOT': (0.0, 0.0, 0.0), 'chip': CHIP}  # points labelled on plots

COILS = {
    'MOT': CoilSpec('MOT', axis='z', inner=10e-2, outer=18.4e-2, thickness=4.2e-2,
                    turns=100, grid=(10, 10), inner_sep=8.5e-2, config='anti-helmholtz',
                    color='0.8', table=dict(MOT=7.5, chip=7.0, gradient=1.56)),
    'transfer': CoilSpec('transfer', axis='y', inner=28e-2, outer=36.4e-2,
                         thickness=XFER_THICKNESS, turns=49, grid=(7, 7), config='single',
                         center=(0.0, Y_XFER, 0.0), color='0.8',
                         polarity=-1,  # field along -y: pushes the quadrupole zero down, away from the chip
                         table=dict(MOT=1.60, chip=1.83)),
    'Zbias': CoilSpec('Zbias', axis='z', inner=19e-2, outer=20e-2, thickness=1.3e-2,
                      turns=20, grid=(2, 4), inner_sep=8.5e-2, table=dict(MOT=2.2, chip=2.2)),
    'Xbias': CoilSpec('Xbias', axis='x', shape='rectangular', inner=(19.5e-2, 7.5e-2),
                      outer=(20.5e-2, 8.5e-2), thickness=1.3e-2, turns=20, grid=(2, 4),
                      inner_sep=8.5e-2, table=dict(MOT=1.1, chip=1.1)),
    'Ybias': CoilSpec('Ybias', axis='y', inner=33e-2, outer=34e-2, thickness=1.3e-2,
                      turns=20, grid=(2, 4), inner_sep=YBIAS_SEP,
                      center=(0.0, Y_XFER + XFER_THICKNESS/2 - YBIAS_SEP/2, 0.0),
                      table=dict(MOT=1.8, chip=1.8)),
}

def biasToCurr(bias, val):
    '''
    Convert bias value on sequencer to amps
    :string bias: 'z', 'x', or 'y'
    :double val: The value as input on the sequencer gui
    val to V is defined on sequencer. They convert the seq val into a voltage (V) that goes to a PS which finally outputs current (A).
    As of today: PS's are Agilent 6552A for z, and HighFinesse BCS-5/5 for x and y.
    '''
    match bias:
        case 'zrev':
            V_in = val * 0.196
            I_out = -V_in * 5
        case 'zfor':
            V_in = val * 0.196
            I_out = -V_in * 5
        case 'y':
            V_in = val * 0.333 - 0.9
            I_out = V_in / 2
        case 'x':
            V_in = val * 1.0117 - 0.0547
            I_out = V_in / 2
    return I_out

# Tab. 4.3 operating currents (A). A positive current gives the field direction set by
# COILS[name].polarity (see build_coil); a negative current reverses that coil.
# Inputs are Amps. For bias coils, there is an extra processing step to turn it into current. MOT and XFER are already in amps.
# for Z bias, also specify if coil is in reverse or forward mode. (PS is not bipolar, needs switch).
CURRENTS_MOT = dict(MOT=6.6, transfer=6.2, Zbias=biasToCurr('zfor', 0), Xbias=biasToCurr('x', 0), Ybias=biasToCurr('y', 0))
# CURRENTS_QMT_INIT_OLD = dict(MOT=26, transfer=21.75, Zbias=biasToCurr('zrev', 0.9), Xbias=biasToCurr('x', 0.9), Ybias=biasToCurr('y', -10))
CURRENTS_QMT_INIT_OLD = dict(MOT=26, transfer=21.75, Zbias=biasToCurr('zrev', 1.2), Xbias=biasToCurr('x', 1), Ybias=biasToCurr('y', -5))
CURRENTS_QMT_INIT = dict(MOT=26.65, transfer=23.4, Zbias=biasToCurr('zrev', 1.2), Xbias=biasToCurr('x', 1), Ybias=biasToCurr('y', -5))
CURRENTS_QMT_FINAL = dict(MOT=45.0, transfer=43, Zbias=biasToCurr('zrev', 0), Xbias=biasToCurr('x', 0), Ybias=biasToCurr('y', 0))
#CURRENTS_ZTRAP = dict(MOT=0.0, transfer=0.0, Zbias=0.0, Xbias=2.0, Ybias=10.0)  # chip Z-wire not modelled

#%% Coil building

_LOCAL = {'z': (0, 1, 2), 'x': (1, 2, 0), 'y': (2, 0, 1)}  # global index of local (u, v, normal)

def _rotation(axis):
    """Rotation taking the local (u, v, normal) frame to the global frame."""
    M = np.zeros((3, 3))
    M[_LOCAL[axis], range(3)] = 1
    return Rotation.from_matrix(M)

def _pack_ranges(spec, config):
    """(start, end, current sign) of each winding pack, along the axis relative to center."""
    if config == 'single':
        return [(-spec.thickness/2, spec.thickness/2, 1)]
    if spec.inner_sep is None:
        raise ValueError(f"{spec.name}: a {config} pair needs inner_sep")
    s = spec.inner_sep/2
    return [(s, s + spec.thickness, 1),
            (-s, -s - spec.thickness, -1 if config == 'anti-helmholtz' else 1)]

def build_coil(spec, current, config=None, grid=None):
    """magpylib Collection for one coil or pair carrying `current` (A) per turn.

    Positive current (with spec.polarity = +1) gives a field along +axis at the centre of the
    coil on the +axis side; the other coil of a pair carries the same (helmholtz) or opposite
    (anti-helmholtz) current. `config` overrides spec.config; `grid` overrides spec.grid
    (fewer filaments is faster and fine away from the windings).
    """
    config = config or spec.config
    n_r, n_z = grid or spec.grid
    rot = _rotation(spec.axis)
    center = np.asarray(spec.center, float)
    I_fil = spec.polarity * current * spec.turns / (n_r * n_z)
    rad_frac = (np.arange(n_r) + 0.5) / n_r
    sources = []
    for t0, t1, sign in _pack_ranges(spec, config):
        for t in t0 + (np.arange(n_z) + 0.5) / n_z * (t1 - t0):
            pos = center + rot.apply([0, 0, t])
            for f in rad_frac:
                if spec.shape == 'circular':
                    src = magpy.current.Circle(current=sign*I_fil, position=pos, orientation=rot,
                                               diameter=spec.inner + f*(spec.outer - spec.inner))
                else:
                    u, v = (np.asarray(spec.inner) + f*(np.subtract(spec.outer, spec.inner))) / 2
                    verts = [(u, v, 0), (-u, v, 0), (-u, -v, 0), (u, -v, 0), (u, v, 0)]
                    src = magpy.current.Polyline(current=sign*I_fil, vertices=verts,
                                                 position=pos, orientation=rot)
                sources.append(src)
    return magpy.Collection(*sources, style_label=spec.name)

class CoilSetup:
    """A set of coils with currents (A per turn), e.g. CoilSetup({'MOT': 7.0, 'Zbias': 0.06}).

    configs : {name: 'helmholtz' | 'anti-helmholtz' | 'single'} overrides
    grid    : (n_radial, n_axial) filaments per winding pack for every coil, overriding COILS.
              Field evaluation time scales with the filament count: (3, 3) is ~5x faster
              than the default one-filament-per-turn MOT/transfer model and agrees to ~0.3%
              (worst near the windings).
    """
    def __init__(self, currents, coils=COILS, configs=None, grid=None):
        self.coils = coils
        self.configs = dict(configs or {})
        self.grid = grid
        self.currents, self.parts = {}, {}
        self.set_currents(**currents)

    def set_currents(self, **currents):
        """Update (or add) coil currents, e.g. setup.set_currents(MOT=45, transfer=48)."""
        for name, I in currents.items():
            self.currents[name] = I
            self.parts[name] = build_coil(self.coils[name], I, self.configs.get(name), self.grid)
        self.collection = magpy.Collection()
        self.collection.add(*self.parts.values(), override_parent=True)
        return self

    @property
    def specs(self):
        return {name: self.coils[name] for name in self.currents}

    def config(self, name):
        return self.configs.get(name) or self.coils[name].config

    def getB(self, points):
        """B (T) at points (..., 3) in m; output has the same shape."""
        points = np.asarray(points, float)
        return np.reshape(magpy.getB(self.collection, points), points.shape)

    def __str__(self):
        on = [f"{name} {I:g} A" for name, I in self.currents.items() if I != 0]
        return ', '.join(on) or 'all coils off'

#%% Field evaluation

def _plane_axes(plane):
    """Global indices (horizontal, vertical, normal) of a plane such as 'zy'."""
    if len(plane) != 2 or plane[0] == plane[1] or set(plane) - set('xyz'):
        raise ValueError(f"plane must be two different letters from 'xyz', got {plane!r}")
    ih, iv = 'xyz'.index(plane[0]), 'xyz'.index(plane[1])
    return ih, iv, 3 - ih - iv

def _range(extent, center=0.0):
    """(min, max) from a half-width about `center`, or an explicit (min, max)."""
    return (center - extent, center + extent) if np.isscalar(extent) else tuple(extent)

def plane_grid(plane='zy', offset=0.0, extent=0.1, n=101, center=(0.0, 0.0, 0.0)):
    """Points on a planar slice: returns h, v (1D, m) and points with shape (len(v), len(h), 3).

    plane  : horizontal axis then vertical axis, e.g. 'zy'; the slice sits at `offset` (m)
             along the remaining axis
    extent : half-width (m) about the in-plane coordinates of `center`, a half-width per
             axis, or ((hmin, hmax), (vmin, vmax))
    n      : points along the longer side (square pixels)
    """
    ih, iv, inn = _plane_axes(plane)
    if np.isscalar(extent):
        extent = (extent, extent)
    (h0, h1), (v0, v1) = _range(extent[0], center[ih]), _range(extent[1], center[iv])
    step = max(h1 - h0, v1 - v0) / (n - 1)
    h = np.linspace(h0, h1, int(round((h1 - h0) / step)) + 1)
    v = np.linspace(v0, v1, int(round((v1 - v0) / step)) + 1)
    pts = np.empty((len(v), len(h), 3))
    pts[..., ih], pts[..., iv], pts[..., inn] = h, v[:, None], offset
    return h, v, pts

def find_field_zero(setup, guess=(0.0, 0.0, 0.0)):
    """Position (m) where B = 0 (quadrupole centre) and the residual |B| (G) there."""
    sol = root(lambda p: setup.getB(p) * T2G, np.asarray(guess, float))
    if not sol.success:
        print(f"find_field_zero: {sol.message}")
    return sol.x, np.linalg.norm(setup.getB(sol.x)) * T2G

def gradient(setup, points, component='z', direction='z', step=1e-4):
    """dB_component/d_direction (G/cm) at points (..., 3), by central difference."""
    points = np.asarray(points, float)
    ic, e = 'xyz'.index(component), np.eye(3)['xyz'.index(direction)] * step
    dB = setup.getB(points + e)[..., ic] - setup.getB(points - e)[..., ic]
    return dB / (2*step) * T2G / CM

def homogeneity(setup, plane='zy', offset=None, extent=0.02, n=101, ref=(0.0, 0.0, 0.0),
                kind='auto', component='z', direction='z'):
    """Relative deviation (%) over a planar slice centred on `ref` (offset defaults to the
    slice through ref).

    kind='field'    : |B(r) - B(ref)| / |B(ref)|      uniformity of a bias field
    kind='gradient' : |G(r) - G(ref)| / |G(ref)|, G = dB_component/d_direction
                      uniformity of a quadrupole gradient
    kind='auto'     : 'gradient' if |B(ref)| is small compared with the field on the slice
    Returns h, v, deviation (%), kind and the reference value (G or G/cm).
    """
    ih, iv, inn = _plane_axes(plane)
    ref = np.asarray(ref, float)
    offset = ref[inn] if offset is None else offset
    h, v, pts = plane_grid(plane, offset, extent, n, center=ref)
    B, B0 = setup.getB(pts), setup.getB(ref)
    if kind == 'auto':
        small = np.linalg.norm(B0) < 0.1 * np.median(np.linalg.norm(B, axis=-1))
        kind = 'gradient' if small else 'field'
    if kind == 'field':
        dev, ref_val = np.linalg.norm(B - B0, axis=-1) / np.linalg.norm(B0), np.linalg.norm(B0) * T2G
    elif kind == 'gradient':
        ic, idir = 'xyz'.index(component), 'xyz'.index(direction)
        if idir in (ih, iv):  # in-plane derivative: difference the grid already computed
            s = h if idir == ih else v
            step = s[1] - s[0]
            G = np.gradient(B[..., ic], s, axis=1 if idir == ih else 0, edge_order=2) * T2G / CM
        else:
            step = 1e-4
            G = gradient(setup, pts, component, direction, step)
        ref_val = gradient(setup, ref, component, direction, step)
        dev = np.abs(G / ref_val - 1)
    else:
        raise ValueError(f"kind must be 'auto', 'field' or 'gradient', got {kind!r}")
    return h, v, 100*dev, kind, ref_val

def print_calibration(coils=COILS, chip=CHIP, tol=0.05):
    """Model field per amp at the MOT centre and chip vs Tab. 4.3 [table, deviation];
    '!' marks deviations beyond `tol`."""
    def fmt(x, key, table):
        x = round(x, 3) + 0.0  # no '-0.000'
        if key not in table:
            return f"{x:7.3f} [  -       ] "
        dev = x / table[key] - 1
        return f"{x:7.3f} [{table[key]:4.2f} {100*dev:+4.0f}%]{'!' if abs(dev) > tol else ' '}"
    print(f"\n{'coil':10s}{'config':16s}{'|B| MOT (G/A)':>22s}{'|B| chip (G/A)':>22s}"
          f"{'dB/d(axis) (G/cm/A)':>24s}")
    for name, spec in coils.items():
        configs = [spec.config] + (['helmholtz'] if spec.config == 'anti-helmholtz' else [])
        for config in configs:
            s = CoilSetup({name: 1.0}, coils, configs={name: config})
            b_mot = np.linalg.norm(s.getB((0, 0, 0))) * T2G
            b_chip = np.linalg.norm(s.getB(chip)) * T2G
            grad = gradient(s, (0, 0, 0), spec.axis, spec.axis)
            # Tab. 4.3 quotes the MOT-pair field in Helmholtz and its gradient in anti-Helmholtz
            quad = config == 'anti-helmholtz'
            table = {k: val for k, val in spec.table.items() if (k == 'gradient') == quad}
            print(f"{name:10s}{config:16s}{fmt(b_mot, 'MOT', table):>22s}"
                  f"{fmt(b_chip, 'chip', table):>22s}{fmt(grad, 'gradient', table):>24s}")
    print()

def print_trap_positions(presets=None, chip=CHIP, grid=None):
    """Quadrupole zero (atom position) for each current preset, and its distance below the chip."""
    presets = presets or {'MOT': CURRENTS_MOT, 
                          'QMT_INIT_OLD':CURRENTS_QMT_INIT_OLD,
                          'QMT_INIT': CURRENTS_QMT_INIT, 'QMT_FINAL':CURRENTS_QMT_FINAL}
    for name, currents in presets.items():
        s = CoilSetup(currents, grid=grid)
        zero, _ = find_field_zero(s)
        print(f"{name} currents: B = 0 at (x, y, z) = ({', '.join(_cm(c) for c in zero)}) cm, "
              f"{(chip[1] - zero[1])*CM:.2f} cm below the chip; dBz/dz = {gradient(s, zero):.1f} G/cm")
    print()

#%% Plotting

def _seq_cmap(name, lo=0.3):
    """Single-hue sequential colormap without its near-white end."""
    return ListedColormap(plt.get_cmap(name)(np.linspace(lo, 1, 256)))

_CMAP_B = _seq_cmap('Blues')
_CMAP_DEV = _seq_cmap('Purples', 0.15)
_COMPONENT_COLORS = {'x': '#2a78d6', 'y': '#eb6834', 'z': '#1baf7a', 'm': '#0b0b0b'}
_MARK_STYLES = {'MOT': dict(marker='+', ms=12, mew=1.5), 'chip': dict(marker='s', ms=6, mfc='none')}

def _quantity(B, quantity, log):
    """Scalar (G) to colour by, with its norm, colormap and label."""
    if quantity == 'mag':
        q = np.linalg.norm(B, axis=-1)
        vmax = np.percentile(q, 99)  # ignore the near-singular field right at the wires
        norm = LogNorm(max(np.percentile(q, 1), vmax*1e-3), vmax) if log else Normalize(0, vmax)
        return q, norm, _CMAP_B, '|B| (G)'
    q = B[..., 'xyz'.index(quantity)]
    vmax = max(np.percentile(np.abs(q), 99), 1e-12)
    return q, TwoSlopeNorm(0, -vmax, vmax), 'RdBu_r', f'B{quantity} (G)'

def _new_ax(ax, figsize=(6.5, 5.5)):
    return ax if ax is not None else plt.figure(figsize=figsize, layout='constrained').gca()

def _colorbar(mappable, ax, label, ticks=None):
    """Colorbar with plain-number labels (1-2-5 ticks on a log scale)."""
    cb = plt.colorbar(mappable, ax=ax, label=label, shrink=0.85)
    if isinstance(mappable.norm, LogNorm):
        cb.ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)) if ticks is None else plt.FixedLocator(ticks))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        cb.ax.yaxis.set_minor_formatter(NullFormatter())
    return cb

def draw_coils(ax, setup, plane='zy', offset=0.0, projections=True):
    """Overlay winding-pack cross-sections (cm) on a slice. Packs facing the slice but not
    cut by it are drawn as dashed outlines if `projections`."""
    ih, iv, inn = _plane_axes(plane)
    for name, spec in setup.specs.items():
        gu, gv, gn = _LOCAL[spec.axis]
        c = np.asarray(spec.center, float)
        if spec.shape == 'circular':
            half_in, half_out = (spec.inner/2,)*2, (spec.outer/2,)*2
        else:
            half_in, half_out = np.divide(spec.inner, 2), np.divide(spec.outer, 2)
        kw = dict(fc=spec.color, ec='k', lw=0.5, zorder=3)
        dashed = dict(fill=False, ec='0.5', lw=0.6, ls='--', zorder=3)
        for t0, t1 in [sorted(r[:2]) for r in _pack_ranges(spec, setup.config(name))]:
            a0, a1 = c[gn] + t0, c[gn] + t1  # pack extent along the coil axis
            if gn == inn:  # coil axis normal to the slice: seen face-on
                cut = a0 <= offset <= a1
                if not (cut or projections):
                    continue
                cen = (c[ih]*CM, c[iv]*CM)
                if spec.shape == 'circular':
                    r_in, r_out = half_in[0]*CM, half_out[0]*CM
                    if cut:
                        ax.add_patch(Wedge(cen, r_out, 0, 360, width=r_out - r_in, **kw))
                    else:
                        for r in (r_in, r_out):
                            ax.add_patch(CirclePatch(cen, r, **dashed))
                    continue
                # rectangular frame; local u lies along the slice's horizontal or vertical axis
                order = (0, 1) if gu == ih else (1, 0)
                (H_in, V_in), (H_out, V_out) = [(hw[order[0]]*CM, hw[order[1]]*CM) for hw in (half_in, half_out)]
                if cut:
                    for x0, y0, w, h in [(-H_out, V_in, 2*H_out, V_out - V_in),
                                         (-H_out, -V_out, 2*H_out, V_out - V_in),
                                         (-H_out, -V_in, H_out - H_in, 2*V_in),
                                         (H_in, -V_in, H_out - H_in, 2*V_in)]:
                        ax.add_patch(Rectangle((cen[0] + x0, cen[1] + y0), w, h, **kw))
                else:
                    for H, V in ((H_in, V_in), (H_out, V_out)):
                        ax.add_patch(Rectangle((cen[0] - H, cen[1] - V), 2*H, 2*V, **dashed))
                continue
            # coil axis lies in the slice: cross-section of the pack
            k = 0 if inn == gu else 1           # local transverse axis normal to the slice
            gw = gv if k == 0 else gu           # local transverse axis within the slice
            d = abs(offset - c[inn])
            if d >= half_out[k]:
                continue
            if spec.shape == 'circular':
                w_out = np.sqrt(half_out[k]**2 - d**2)
                w_in = np.sqrt(half_in[k]**2 - d**2) if d < half_in[k] else None
            else:
                w_out = half_out[1 - k]
                w_in = half_in[1 - k] if d < half_in[k] else None
            spans = [(w_in, w_out), (-w_out, -w_in)] if w_in is not None else [(-w_out, w_out)]
            for w0, w1 in spans:
                span = {gn: (a0, a1), gw: (c[gw] + w0, c[gw] + w1)}
                (x0, x1), (y0, y1) = span[ih], span[iv]
                ax.add_patch(Rectangle((x0*CM, y0*CM), (x1 - x0)*CM, (y1 - y0)*CM, **kw))

def draw_marks(ax, marks, plane='zy', offset=0.0, tol=5e-3):
    """Label points (m) lying within `tol` of the slice."""
    ih, iv, inn = _plane_axes(plane)
    drawn = 0
    for name, p in (marks or {}).items():
        p = np.asarray(p, float)
        if abs(p[inn] - offset) > tol:
            continue
        style = _MARK_STYLES.get(name, dict(marker='x', ms=7, mew=1.5))
        ax.plot(p[ih]*CM, p[iv]*CM, color='k', ls='none', zorder=5, **style)
        side = 1 if drawn % 2 == 0 else -1  # alternate up-right / down-left so nearby labels don't collide
        ax.annotate(name, (p[ih]*CM, p[iv]*CM), xytext=(6*side, 4*side), textcoords='offset points',
                    ha='left' if side > 0 else 'right', va='bottom' if side > 0 else 'top',
                    fontsize=9, zorder=5, annotation_clip=True)
        drawn += 1

def _cm(x):
    """Length (m) as a cm string without '-0.00'."""
    return f"{round(x*CM, 2) + 0.0:.2f}"

def _style_slice(ax, plane, offset, h, v, title):
    inn = 'xyz'[_plane_axes(plane)[2]]
    ax.set(xlabel=f'{plane[0]} (cm)', ylabel=f'{plane[1]} (cm)', aspect='equal',
           xlim=(h[0]*CM, h[-1]*CM), ylim=(v[0]*CM, v[-1]*CM),
           title=title + f'\n{plane} slice at {inn} = {_cm(offset)} cm')

def plot_slice(setup, plane='zy', offset=0.0, extent=0.12, n=101, style='stream',
               quantity='mag', log=True, density=1.5, center=(0.0, 0.0, 0.0), ax=None,
               coils=True, marks=MARKS, title=None):
    """Field on a planar slice.

    plane    : horizontal then vertical axis. 'zy' = side view with the MOT axis horizontal
               and up vertical; 'xy' = looking along the MOT axis; 'xz' = top view.
               Any order works ('yz', 'zx', ...). The slice is at `offset` (m) along the
               remaining axis.
    extent   : half-width (m) about the in-plane coordinates of `center`, or
               ((hmin, hmax), (vmin, vmax))
    style    : 'stream'     field lines of the in-plane components, coloured by `quantity`
               'quiver'     unit arrows of the in-plane direction, coloured by `quantity`
               'map'        colour map of `quantity` with contours
               'map+stream' colour map with white field lines on top
    quantity : 'mag' for |B|, or a component 'x', 'y', 'z' (G)
    log      : log colour scale for |B|
    """
    ih, iv, _ = _plane_axes(plane)
    h, v, pts = plane_grid(plane, offset, extent, n, center)
    B = np.nan_to_num(setup.getB(pts) * T2G)
    q, norm, cmap, label = _quantity(B, quantity, log)
    ax = _new_ax(ax)
    H, V = h*CM, v*CM
    if style in ('map', 'map+stream'):
        mappable = ax.pcolormesh(H, V, q, norm=norm, cmap=cmap, shading='auto')
        if style == 'map':
            levels = norm.inverse(np.linspace(0.1, 0.9, 5)) if quantity == 'mag' else 7
            cs = ax.contour(H, V, q, levels=levels, colors='0.2', linewidths=0.5)
            ax.clabel(cs, fmt='%.3g', fontsize=7)
        else:
            ax.streamplot(H, V, B[..., ih], B[..., iv], density=density, color='w',
                          linewidth=0.6, arrowsize=0.6)
    elif style == 'stream':
        mappable = ax.streamplot(H, V, B[..., ih], B[..., iv], density=density, color=q,
                                 norm=norm, cmap=cmap, linewidth=0.8, arrowsize=0.7).lines
    elif style == 'quiver':
        k = max(1, len(h) // 25)
        Bh, Bv = B[::k, ::k, ih], B[::k, ::k, iv]
        mag = np.where(np.hypot(Bh, Bv) > 0, np.hypot(Bh, Bv), 1)
        mappable = ax.quiver(H[::k], V[::k], Bh/mag, Bv/mag, q[::k, ::k], norm=norm, cmap=cmap,
                             pivot='mid', angles='xy', scale_units='width',
                             scale=1.2*len(H[::k]), width=0.004)
    else:
        raise ValueError(f"style must be 'stream', 'quiver', 'map' or 'map+stream', got {style!r}")
    _colorbar(mappable, ax, label)
    if coils:
        draw_coils(ax, setup, plane, offset)
    draw_marks(ax, marks, plane, offset)
    _style_slice(ax, plane, offset, h, v, title or str(setup))
    return ax

def plot_homogeneity(setup, plane='zy', offset=None, extent=0.02, n=101, ref=(0.0, 0.0, 0.0),
                     kind='auto', component='z', direction='z', levels=None, ax=None,
                     coils=True, marks=MARKS, title=None):
    """Contours of the relative field (or gradient) deviation from its value at `ref`, on a
    slice centred on ref; see homogeneity() for `kind`, `component` and `direction`."""
    ref = np.asarray(ref, float)
    offset = ref[_plane_axes(plane)[2]] if offset is None else offset
    h, v, dev, kind, ref_val = homogeneity(setup, plane, offset, extent, n, ref, kind,
                                           component, direction)
    levels = np.asarray(levels if levels is not None else [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10])
    dev = np.clip(dev, levels[0]/2, None)
    ax = _new_ax(ax)
    H, V = h*CM, v*CM
    cf = ax.contourf(H, V, dev, levels=levels, norm=LogNorm(levels[0], levels[-1]),
                     cmap=_CMAP_DEV, extend='both')
    decades = levels[np.isclose(np.log10(levels) % 1, 0) | np.isclose(np.log10(levels) % 1, 1)]
    cs = ax.contour(H, V, dev, levels=levels, colors='0.25', linewidths=0.5)
    ax.clabel(cs, levels=decades if len(decades) else levels, fmt='%g%%', fontsize=7)
    _colorbar(cf, ax, 'deviation (%)', ticks=levels)
    if coils:
        draw_coils(ax, setup, plane, offset)
    draw_marks(ax, marks, plane, offset)
    ref_str = f"|B| = {ref_val:.4g} G" if kind == 'field' else f"dB{component}/d{direction} = {ref_val:.4g} G/cm"
    _style_slice(ax, plane, offset, h, v,
                 (title or str(setup)) + f'\n{kind} homogeneity vs ref: {ref_str}')
    return ax

def plot_line(setup, axis='y', through=(0.0, 0.0, 0.0), extent=0.1, n=401, components='xyzm',
              ax=None, marks=MARKS, title=None):
    """Field components (G) along a line parallel to `axis` through the point `through`.
    components: any of 'x', 'y', 'z' and 'm' (|B|).
    extent: half-width (m) about `through`, or (min, max) along the axis."""
    i = 'xyz'.index(axis)
    through = np.asarray(through, float)
    s = np.linspace(*_range(extent, through[i]), n)
    pts = np.tile(through, (n, 1))
    pts[:, i] = s
    B = setup.getB(pts) * T2G
    ax = _new_ax(ax, figsize=(6.5, 4.5))
    for c in components:
        y = np.linalg.norm(B, axis=-1) if c == 'm' else B[:, 'xyz'.index(c)]
        ax.plot(s*CM, y, color=_COMPONENT_COLORS[c], lw=2 if c == 'm' else 1.5,
                ls='--' if c == 'm' else '-', label='|B|' if c == 'm' else f'B{c}')
    others = [j for j in range(3) if j != i]
    drawn = 0
    for name, p in (marks or {}).items():
        p = np.asarray(p, float)
        if np.allclose(p[others], pts[0, others], atol=1e-3) and s[0] <= p[i] <= s[-1]:
            ax.axvline(p[i]*CM, color='0.5', ls=':', lw=1)
            ax.annotate(name, (p[i]*CM, 1), xycoords=('data', 'axes fraction'),
                        xytext=(3, -12 - 11*(drawn % 3)), textcoords='offset points', fontsize=9)
            drawn += 1
    ax.axhline(0, color='0.7', lw=0.8)
    pos = ', '.join(f"{'xyz'[j]} = {_cm(pts[0, j])} cm" for j in others)
    ax.set(xlabel=f'{axis} (cm)', ylabel='B (G)', title=(title or str(setup)) + f'\nline along {axis} at {pos}')
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    return ax

def _sphere_points(n):
    """n roughly evenly spaced unit vectors (Fibonacci sphere)."""
    k = np.arange(n) + 0.5
    phi, cos_t = np.pi * (1 + 5**0.5) * k, 1 - 2*k/n
    sin_t = np.sqrt(1 - cos_t**2)
    return np.c_[sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t]

def trace_field_lines(setup, seeds, step=1e-3, max_steps=400, bounds=0.15):
    """Field lines (list of (M, 3) arrays, m) through `seeds` (N, 3), traced both ways with
    fixed-step RK4 along B/|B|. A line stops when it leaves the cube |x|,|y|,|z| < bounds,
    returns to its seed, or reaches a field zero."""
    seeds = np.atleast_2d(np.asarray(seeds, float))
    N = len(seeds)
    X0 = np.vstack([seeds, seeds])
    sgn = np.r_[np.ones(N), -np.ones(N)][:, None]
    X, path = X0.copy(), np.full((max_steps + 1, 2*N, 3), np.nan)
    path[0] = X0
    alive = np.ones(2*N, bool)

    def direction(p, s):
        B = setup.getB(p)
        mag = np.linalg.norm(B, axis=-1, keepdims=True)
        return s * B / np.where(mag > 0, mag, np.inf)

    for k in range(1, max_steps + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        p, s = X[idx], sgn[idx]
        k1 = direction(p, s)
        k2 = direction(p + step/2*k1, s)
        k3 = direction(p + step/2*k2, s)
        k4 = direction(p + step*k3, s)
        p = p + step/6 * (k1 + 2*k2 + 2*k3 + k4)
        X[idx], path[k, idx] = p, p
        stop = (np.any(np.abs(p) > bounds, axis=1) | ~np.any(k1, axis=1)
                | ((k > 10) & (np.linalg.norm(p - X0[idx], axis=1) < step)))
        alive[idx[stop]] = False
    lines = []
    for i in range(N):
        fwd, bwd = path[:, i], path[:, N + i]
        fwd, bwd = fwd[~np.isnan(fwd[:, 0])], bwd[~np.isnan(bwd[:, 0])]
        lines.append(np.vstack([bwd[::-1], fwd[1:]]))
    return lines

def plot_field_lines_3d(setup, seeds=None, n_seeds=30, seed_radius=0.02, center=(0.0, 0.0, 0.0),
                        extent=0.12, step=1e-3, max_steps=400, ax=None, coils=True, title=None):
    """3D field lines with the coil filaments; y (up) is drawn vertical.
    Default seeds: n_seeds points on a sphere of seed_radius (m) around `center`.
    Tracing calls getB ~4*max_steps times, so build the setup with a coarse grid for speed,
    e.g. CoilSetup({'MOT': 7}, grid=(2, 2))."""
    if seeds is None:
        seeds = np.asarray(center, float) + seed_radius * _sphere_points(n_seeds)
    lines = trace_field_lines(setup, seeds, step, max_steps, bounds=extent)
    if ax is None:
        ax = plt.figure(figsize=(8, 7), layout='constrained').add_subplot(projection='3d')
    if coils:
        draw_coils_3d(ax, setup)
    for line in lines:
        ax.plot(*(line.T*CM), color=_COMPONENT_COLORS['x'], lw=0.8)
    lim = (-extent*CM, extent*CM)
    ax.set(xlim=lim, ylim=lim, zlim=lim, xlabel='x (cm)', ylabel='y (cm)', zlabel='z (cm)',
           title=title or str(setup))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-50, vertical_axis='y')
    return ax

def draw_coils_3d(ax, setup, color='0.55'):
    """Draw every current filament of the setup on a 3D axis (cm)."""
    t = np.linspace(0, 2*np.pi, 73)
    for part in setup.parts.values():
        for src in part.children:
            if isinstance(src, magpy.current.Circle):
                r = src.diameter / 2
                local = np.c_[r*np.cos(t), r*np.sin(t), 0*t]
            else:
                local = np.asarray(src.vertices)
            xyz = src.orientation.apply(local) + src.position
            ax.plot(*(xyz.T*CM), color=color, lw=0.4, alpha=0.6)

def show_coils(setup, backend='matplotlib', **kwargs):
    """magpylib's own 3D view of the coils ('plotly' gives an interactive browser view)."""
    return magpy.show(setup.collection, backend=backend, **kwargs)

#%% Main

if __name__ == '__main__':
    RUN_MOT_TEST = False   # MOT coils only
    RUN_FULL = True       # all coils, Tab. 4.3 QMT currents
    RUN_BIAS = False       # field homogeneity of each bias pair at 1 A
    RUN_3D = False   # traced 3D field lines (slower)
    GRID = None           # filaments per winding pack: None = one per turn; (3, 3) is ~5x faster

    print_calibration()
    print_trap_positions()

    if RUN_MOT_TEST:
        mot = CoilSetup({'MOT': 7.0}, grid=GRID)  # edit currents here, or later: mot.set_currents(MOT=45)
        print(f"{mot}: dBz/dz = {gradient(mot, (0, 0, 0)):.2f} G/cm, "
              f"dBy/dy = {gradient(mot, (0, 0, 0), 'y', 'y'):.2f} G/cm")
        fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
        plot_slice(mot, 'zy', extent=0.12, ax=axs[0])               # side view through the axis
        plot_slice(mot, 'xy', offset=0.0, extent=0.12, ax=axs[1])   # mid-plane, looking along z
        fig, axs = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
        plot_homogeneity(mot, 'zy', extent=0.02, ax=axs[0])         # auto -> gradient homogeneity
        plot_line(mot, 'z', extent=0.12, ax=axs[1])
        plot_line(mot, 'y', extent=0.12, ax=axs[2])

    if RUN_FULL:
        full = CoilSetup(CURRENTS_QMT_INIT_OLD, grid=GRID)  # or CURRENTS_MOT, CURRENTS_ZTRAP, your own dict
        zero, b_res = find_field_zero(full)
        marks = {**MARKS, 'B = 0': zero}
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), layout='constrained')
        plot_slice(full, 'zy', extent=0.2, ax=axs[0], marks=marks)
        plot_slice(full, 'xy', extent=0.2, ax=axs[1], marks=marks)
        plot_slice(full, 'xz', offset=zero[1], extent=0.2, ax=axs[2], marks=marks)  # top view through B = 0
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), layout='constrained')
        plot_slice(full, 'zy', offset=zero[0], center=(zero + CHIP)/2, extent=0.03,
                   style='map+stream', ax=axs[0], marks=marks)  # atoms to chip
        plot_line(full, 'y', through=zero, extent=0.1, ax=axs[1], marks=marks)
        plot_homogeneity(full, 'zy', ref=zero, extent=0.02, ax=axs[2], marks=marks)

    if RUN_BIAS:
        fig, axs = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
        for ax, name in zip(axs, ['Zbias', 'Xbias', 'Ybias']):
            plot_homogeneity(CoilSetup({name: 1.0}, grid=GRID), 'zy', extent=0.04, ax=ax)

    if RUN_3D:
        plot_field_lines_3d(CoilSetup({'MOT': 7.0}, grid=(2, 2)))

    plt.show()
