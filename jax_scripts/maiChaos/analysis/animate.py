"""
Animation of MHD RPOs.

Migrated from jax_scripts/legacy/animation.py and made callable as a function.

Produces a GIF (default) or MP4 showing vorticity ω and current j over one
full period of the RPO.

Usage
-----
    from maiChaos.analysis.animate import make_animation
    make_animation(input_dict, param_dict, "output.gif", fps=10)
"""

import io
import os
import sys
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB  = os.path.join(_HERE, '..', '..')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO


def make_animation(
    input_dict:   Dict[str, Any],
    param_dict:   Dict[str, Any],
    output_path:  str,
    fps:          int   = 10,
    save_every:   int   = 32,
    vmin:         float = -10.0,
    vmax:         float = 10.0,
    double_domain: bool  = True,
) -> str:
    """
    Animate one period of an RPO and write to ``output_path``.

    Parameters
    ----------
    input_dict    : RPO solution ``{'fields', 'T', 'sx'}``.
    param_dict    : physics parameters (must include 'steps').
    output_path   : destination file path (extension determines format:
                    ``.gif`` or ``.mp4``).
    fps           : frames per second.
    save_every    : timesteps between saved frames.
    vmin, vmax    : colour-bar limits.
    double_domain : tile the domain 2×2 for visualisation.

    Returns
    -------
    output_path : str (same as input, for convenience)
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError("imageio is required: pip install imageio[ffmpeg]")

    from matplotlib.colors import LinearSegmentedColormap

    f  = input_dict['fields']
    T  = float(input_dict['T'])
    sx = float(input_dict['sx'])

    steps = int(param_dict['steps'])
    if steps % save_every != 0:
        # Round save_every to nearest divisor
        for s in range(save_every, 1, -1):
            if steps % s == 0:
                save_every = s
                break
        else:
            save_every = 1

    nt = steps // save_every
    dt = T / steps

    # Custom colourmap (same as legacy animation.py)
    colors = [
        [0, 0, 0.5],
        [0, 0.5, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0, 0],
    ]
    my_cmap = LinearSegmentedColormap.from_list("custom_bkb", colors, N=256)

    bg_color   = "black"
    font_color = "white"

    update = jax.jit(lambda f: mhd_jax.eark4(f, dt, save_every, param_dict))

    frames = []
    for t in range(nt):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), facecolor=bg_color)
        plt.subplots_adjust(wspace=0.3)

        for ax, field in zip(axs, [f[0], f[1]]):
            ax.set_axis_off()
            display = jnp.tile(field, (2, 2)) if double_domain else field
            im = ax.imshow(
                np.array(display).T,
                cmap=my_cmap, origin="lower",
                interpolation="none",
                vmin=vmin, vmax=vmax,
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([vmin, 0, vmax])
            cbar.ax.tick_params(colors=font_color)
            plt.setp(cbar.ax.get_yticklabels(), color=font_color)

        axs[0].set_title(r"$\nabla \times \mathbf{u}$", fontsize=12,
                         color=font_color)
        axs[1].set_title(r"$\nabla \times \mathbf{B}$", fontsize=12,
                         color=font_color)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    pad_inches=0)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

        # Advance one save_every segment
        f_spec = jnp.fft.rfft2(f)
        f_spec = update(f_spec)
        # Apply fractional spatial shift (co-moving frame)
        f_spec = jnp.exp(-1j * sx / nt * param_dict['kx']) * f_spec
        f = jnp.fft.irfft2(f_spec)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        imageio.mimsave(output_path, frames, palettesize=256,
                        duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(output_path, frames, fps=fps)

    print(f"[animate] Saved {len(frames)} frames to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(solution_file: str, output_path: Optional[str] = None,
         fps: int = 10, save_every: int = 32):
    input_dict, param_dict = dictionaryIO.load_dicts(solution_file)

    if output_path is None:
        base = os.path.splitext(solution_file)[0]
        output_path = base + ".gif"

    make_animation(input_dict, param_dict, output_path,
                   fps=fps, save_every=save_every)
