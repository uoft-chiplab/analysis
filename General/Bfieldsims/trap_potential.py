"""
Trapping potential U = mu |B| + m g y felt by atoms in the magnetic quadrupole trap (QMT),
using the coil model in B_field_sim.py (same frame: +y is up, so gravity points along -y).

Atoms (constants from Steck, "Rubidium 87 D Line Data", and Tiecke, "Properties of Potassium"):
    Rb87 |F=2, mF=2>      low-field seeker, mu = mF gF mu_B = 0.99967 mu_B
    K40  |F=9/2, mF=9/2>  low-field seeker, mu = 1.00185 mu_B (K40's hyperfine structure is
                           inverted, but F = 9/2 is still the F = I + 1/2 manifold)
Both are stretched states, whose Zeeman shift is exactly linear in |B| at any field, so
U = mu |B| is exact for them (for other |F, mF> it is the low-field approximation).

Potentials are plotted relative to the trap centre (the field zero, where U is lowest when the
atoms are held against gravity), in uK by default.

Usage:
    setup = CoilSetup(QMT_PRESETS['QMT_INIT'])
    print_trap_summary(setup)                          # |B| slopes vs gravity for each atom
    potential(setup, points, 'Rb87')                   # J, at points (..., 3) in m
    plot_potential_slices(setup, planes=('zy', 'xy'))  # maps through the trap centre
    plot_potential_lines(setup, axes='y')              # 1D cuts, with and without gravity
    plot_qmt_potentials('QMT_FINAL')                   # all of the above for a preset
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy.constants import atomic_mass as AMU, g as G_EARTH, h as H_PLANCK, k as K_B
from scipy.constants import physical_constants
from B_field_sim import (CHIP, CM, MARKS, QMT_PRESETS, T2G, CoilSetup, draw_marks,
                         find_field_zero, plane_grid, _cm, _plane_axes, _range, _style_slice)
MU_B = physical_constants['Bohr magneton'][0]  # J/T

#%% Atoms

@dataclass(frozen=True)
class Atom:
    """Alkali ground state |F, mF> (J = 1/2) with nuclear spin I. g_I is in units of mu_B
    with the convention H = mu_B (g_J J + g_I I).B (so g_I < 0 for Rb87)."""
    name: str
    label: str
    mass: float  # kg
    I: float
    F: float
    mF: float
    g_J: float
    g_I: float

    @property
    def g_F(self):
        FF, II, JJ = self.F*(self.F + 1), self.I*(self.I + 1), 0.75
        return self.g_J*(FF - II + JJ)/(2*FF) + self.g_I*(FF + II - JJ)/(2*FF)

    @property
    def mu(self):
        """Magnetic moment (J/T) in U = mu |B|; positive for a low-field seeker."""
        return self.mF * self.g_F * MU_B

ATOMS = {
    'Rb87': Atom('Rb87', r'$^{87}$Rb $|2, 2\rangle$', mass=86.909180520*AMU,
                 I=1.5, F=2, mF=2, g_J=2.00233113, g_I=-0.0009951414),
    'K40': Atom('K40', r'$^{40}$K $|9/2, 9/2\rangle$', mass=39.96399848*AMU,
                I=4, F=4.5, mF=4.5, g_J=2.00229421, g_I=0.000176490),
}

UNITS = {'uK': (K_B*1e-6, r'U/k$_B$ (µK)'), 'mK': (K_B*1e-3, r'U/k$_B$ (mK)'),
         'MHz': (H_PLANCK*1e6, 'U/h (MHz)')}  # energy scale (J) and axis label

_ATOM_COLORS = {'Rb87': '#2a78d6', 'K40': '#eb6834'}
_CMAP_U = 'Greens'

def _atom(atom):
    return ATOMS[atom] if isinstance(atom, str) else atom

#%% Potential

def potential(setup, points, atom, gravity=True, B=None):
    """Potential energy (J) of `atom` at points (..., 3) in m: mu |B| (+ m g y with gravity).
    Pass the field B (T) at those points to skip evaluating it again."""
    atom, points = _atom(atom), np.asarray(points, float)
    B = setup.getB(points) if B is None else B
    U = atom.mu * np.linalg.norm(B, axis=-1)
    return U + atom.mass * G_EARTH * points[..., 1] if gravity else U

def gravity_gradient(atom):
    """|B| gradient (G/cm) whose magnetic force on `atom` balances gravity: m g / mu."""
    atom = _atom(atom)
    return atom.mass * G_EARTH / atom.mu * T2G / CM

def field_jacobian(setup, point, step=1e-5):
    """J[i, j] = dB_i/dx_j (G/cm) at a point, by central difference."""
    p = np.asarray(point, float)
    J = np.empty((3, 3))
    for j in range(3):
        e = np.eye(3)[j] * step
        J[:, j] = (setup.getB(p + e) - setup.getB(p - e)) / (2*step)
    return J * T2G / CM

def print_trap_summary(setup, atoms=('Rb87', 'K40'), name=None, chip=CHIP):
    """Field zero (trap centre), |B| slopes away from it, and how each atom is held against
    gravity. Returns the field zero (m)."""
    zero, _ = find_field_zero(setup)
    slope = np.linalg.norm(field_jacobian(setup, zero), axis=0)  # |B| slope along x, y, z (G/cm)
    print(f"{name or setup}\n  B = 0 at ({', '.join(_cm(c) for c in zero)}) cm, "
          f"{(chip[1] - zero[1])*CM:.2f} cm below the chip; |B| slope along x, y, z = "
          f"{slope[0]:.1f}, {slope[1]:.1f}, {slope[2]:.1f} G/cm")
    for atom in map(_atom, atoms):
        per_G_cm = atom.mu * (CM / T2G) / K_B * 1e3  # uK/mm of potential per G/cm of slope
        mg = atom.mass * G_EARTH / K_B * 1e3          # gravity in uK/mm
        mag = slope * per_G_cm
        ratio = slope[1] / gravity_gradient(atom)
        held = '' if ratio > 1 else '  -> NOT HELD: gravity beats the vertical gradient'
        print(f"  {atom.name:5s} gravity = {gravity_gradient(atom):.2f} G/cm, vertical slope is "
              f"{ratio:.2f}x that. U rises {mag[1] - mg:.0f} uK/mm downwards, {mag[1] + mg:.0f} "
              f"uK/mm upwards, {mag[0]:.0f} (x) / {mag[2]:.0f} (z) uK/mm sideways{held}")
    print()
    return zero

#%% Plotting

def _gravity_arrow(ax, plane):
    """Small 'g' arrow pointing along -y, if y lies in the slice."""
    ih, iv, _ = _plane_axes(plane)
    if 1 not in (ih, iv):
        return
    start, end = ((0.93, 0.93), (0.93, 0.80)) if iv == 1 else ((0.95, 0.93), (0.82, 0.93))
    ax.annotate('', xy=end, xytext=start, xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='k', lw=1.2))
    ax.annotate('g', start, xycoords='axes fraction', xytext=(-10, -2), textcoords='offset points',
                ha='right', va='top', fontsize=10)

def _slice_potentials(setup, atoms, plane, center, extent, n, gravity, unit):
    """h, v (m), slice offset and {atom name: U - U(field zero)} in `unit` on a slice through
    `center`; the field is evaluated once for all atoms."""
    zero, _ = find_field_zero(setup)
    center = zero if center is None else np.asarray(center, float)
    offset = center[_plane_axes(plane)[2]]
    h, v, pts = plane_grid(plane, offset, extent, n, center)
    B = setup.getB(pts)
    scale = UNITS[unit][0]
    U = {a.name: (potential(setup, pts, a, gravity, B) - potential(setup, zero, a, gravity)) / scale
         for a in atoms}
    return h, v, offset, U, zero

def _draw_potential(ax, plane, offset, h, v, U, norm, levels, label, marks):
    H, V = h*CM, v*CM
    mesh = ax.pcolormesh(H, V, U, norm=norm, cmap=_CMAP_U, shading='auto')
    cs = ax.contour(H, V, U, levels=levels, colors='0.25', linewidths=0.5)
    ax.clabel(cs, fmt='%g', fontsize=7)
    draw_marks(ax, marks, plane, offset)
    _gravity_arrow(ax, plane)
    _style_slice(ax, plane, offset, h, v, label)
    return mesh

def _norm_levels(maps, levels):
    vmin = min(U.min() for U in maps)
    vmax = max(U.max() for U in maps)
    if levels is None:
        levels = [l for l in MaxNLocator(10).tick_values(vmin, vmax) if vmin < l < vmax]
    return Normalize(vmin, vmax), levels

def plot_potential_slice(setup, atom='Rb87', plane='zy', center=None, extent=0.005, n=101,
                         gravity=True, unit='uK', levels=None, ax=None, marks=None, title=None):
    """Map of U - U(trap centre) for one atom on a slice through `center` (default: the
    field zero). extent: half-width (m) or ((hmin, hmax), (vmin, vmax))."""
    atom = _atom(atom)
    h, v, offset, U, zero = _slice_potentials(setup, [atom], plane, center, extent, n, gravity, unit)
    norm, levels = _norm_levels(U.values(), levels)
    ax = ax if ax is not None else plt.figure(figsize=(6.5, 5.5), layout='constrained').gca()
    marks = {**MARKS, 'B = 0': zero} if marks is None else marks
    mesh = _draw_potential(ax, plane, offset, h, v, U[atom.name], norm, levels,
                           f"{title or setup}\n{atom.label}", marks)
    plt.colorbar(mesh, ax=ax, label=UNITS[unit][1], shrink=0.85)
    return ax

def plot_potential_slices(setup, atoms=('Rb87', 'K40'), planes=('zy', 'xy', 'zx'), center=None,
                          extent=0.005, n=101, gravity=True, unit='uK', levels=None, marks=None,
                          title=None):
    """Potential maps, one row per atom and one column per plane, all through `center`
    (default: the field zero) on one shared colour scale."""
    atoms = [_atom(a) for a in atoms]
    slices = [_slice_potentials(setup, atoms, plane, center, extent, n, gravity, unit)
              for plane in planes]
    norm, levels = _norm_levels([U for s in slices for U in s[3].values()], levels)
    fig, axs = plt.subplots(len(atoms), len(planes), squeeze=False, layout='constrained',
                            figsize=(4.8*len(planes) + 1, 4.6*len(atoms)))
    for col, (plane, (h, v, offset, U, zero)) in enumerate(zip(planes, slices)):
        for row, atom in enumerate(atoms):
            mesh = _draw_potential(axs[row, col], plane, offset, h, v, U[atom.name], norm, levels,
                                   atom.label, {**MARKS, 'B = 0': zero} if marks is None else marks)
    fig.colorbar(mesh, ax=axs, label=UNITS[unit][1], shrink=0.85)
    fig.suptitle(title or str(setup) + ('' if gravity else ' (no gravity)'))
    return fig, axs

def plot_potential_lines(setup, atoms=('Rb87', 'K40'), axes='xyz', through=None, extent=0.005,
                         n=401, gravity=True, unit='uK', title=None):
    """U - U(trap centre) along lines parallel to each of `axes` through `through` (default:
    the field zero), one panel per axis and one curve per atom. Along y, dotted curves show
    the magnetic part alone."""
    atoms = [_atom(a) for a in atoms]
    zero, _ = find_field_zero(setup)
    through = zero if through is None else np.asarray(through, float)
    scale, label = UNITS[unit]
    fig, axs = plt.subplots(1, len(axes), squeeze=False, sharey=True, layout='constrained',
                            figsize=(4.8*len(axes), 4.2))
    for ax, axis in zip(axs[0], axes):
        i = 'xyz'.index(axis)
        s = np.linspace(*_range(extent, through[i]), n)
        pts = np.tile(through, (n, 1))
        pts[:, i] = s
        B = setup.getB(pts)
        for k, atom in enumerate(atoms):
            color = _ATOM_COLORS.get(atom.name, f'C{k}')
            lw = max(3.0 - 1.2*k, 1.2)  # later atoms thinner: curves with equal mu stay visible
            U = potential(setup, pts, atom, gravity, B) - potential(setup, zero, atom, gravity)
            ax.plot(s*CM, U/scale, color=color, lw=lw, label=atom.label)
            if gravity and axis == 'y':
                ax.plot(s*CM, potential(setup, pts, atom, False, B)/scale, color=color, lw=lw/2,
                        ls=':', label=f'{atom.label}, no gravity')
        others = [j for j in range(3) if j != i]
        if np.allclose(through[others], zero[others], atol=1e-4):
            ax.axvline(zero[i]*CM, color='0.5', ls=':', lw=1)
        pos = ', '.join(f"{'xyz'[j]} = {_cm(through[j])} cm" for j in others)
        ax.set(xlabel=f'{axis} (cm)', title=f'line along {axis} at {pos}')
        ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel(label)
    axs[0, axes.index('y') if 'y' in axes else 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title or str(setup) + ('' if gravity else ' (no gravity)'))
    return fig, axs

def plot_qmt_potentials(preset='QMT_INIT', atoms=('Rb87', 'K40'), planes=('zy', 'xy', 'zx'),
                        extent=0.005, n=101, gravity=True, unit='uK', grid=None, summary=True):
    """Trap summary, potential maps and line cuts for a QMT preset: a QMT_PRESETS key or a
    currents dict. Returns the CoilSetup."""
    named = isinstance(preset, str)
    setup = CoilSetup(QMT_PRESETS[preset] if named else preset, grid=grid)
    title = f"{preset}: {setup}" if named else str(setup)
    if summary:
        print_trap_summary(setup, atoms, title)
    plot_potential_slices(setup, atoms, planes, extent=extent, n=n, gravity=gravity, unit=unit,
                          title=title)
    plot_potential_lines(setup, atoms, extent=extent, gravity=gravity, unit=unit, title=title)
    return setup

#%% Main

if __name__ == '__main__':
    PLOT_PRESETS = ['QMT_INIT_OLD', 'QMT_INIT', 'QMT_FINAL']  # keys of QMT_PRESETS (B_field_sim.py)
    ATOM_KEYS = ('Rb87', 'K40')
    EXTENT = 0.005    # m, half-width of the plots about the trap centre
    UNIT = 'uK'       # 'uK', 'mK' or 'MHz'
    GRID = None       # filaments per winding pack: None = one per turn; (3, 3) is ~5x faster

    for name, currents in QMT_PRESETS.items():
        print_trap_summary(CoilSetup(currents, grid=GRID), ATOM_KEYS, name)
    for name in PLOT_PRESETS:
        plot_qmt_potentials(name, ATOM_KEYS, extent=EXTENT, unit=UNIT, grid=GRID, summary=False)
    plt.show()
