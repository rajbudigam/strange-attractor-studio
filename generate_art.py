#!/usr/bin/env python3
"""
Generative art using the Clifford attractor.

Running this script will create one or more high‑resolution images of the Clifford
attractor with random parameters.  Each image is saved as a PNG file in the
specified output directory.  You can optionally fix the random seed for
reproducibility or provide your own parameters via the `--params` option.

Usage::

    python3 generate_art.py --count 5 --iterations 500000 --resolution 3000 3000

By default a single 2000×2000 image is generated using 600 000 iterations.
"""
import argparse
import os
import random
from datetime import datetime
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt


def clifford_attractor(a: float, b: float, c: float, d: float,
                       iterations: int = 600_000,
                       discard: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate points for the Clifford attractor.

    Parameters
    ----------
    a, b, c, d: float
        Parameters controlling the attractor.
    iterations: int
        Total number of points to compute.  Higher numbers yield smoother images.
    discard: int
        Number of initial points to discard to avoid transient behaviour.

    Returns
    -------
    x, y: numpy.ndarray
        Arrays of x and y coordinates of length ``iterations - discard``.
    """
    # Preallocate arrays for speed
    x = np.empty(iterations, dtype=np.float64)
    y = np.empty(iterations, dtype=np.float64)

    # Initialise with a small non‑zero value to break symmetry
    x[0] = 0.1
    y[0] = 0.0

    # Iterate the map
    for i in range(1, iterations):
        x_prev, y_prev = x[i - 1], y[i - 1]
        x[i] = np.sin(a * y_prev) + c * np.cos(a * x_prev)
        y[i] = np.sin(b * x_prev) + d * np.cos(b * y_prev)

    return x[discard:], y[discard:]


def render_density(x: np.ndarray, y: np.ndarray,
                   resolution: Tuple[int, int] = (2000, 2000)) -> np.ndarray:
    """Create a log‑scaled density image from x/y points.

    The histogram is normalised to lie between 0 and 1.

    Returns a 2D array of shape ``resolution``.
    """
    width, height = resolution
    # Compute 2D histogram.  Note that numpy.histogram2d returns shape (x_bins, y_bins),
    # but we interpret width as the horizontal axis (columns) and height as vertical axis.
    H, xedges, yedges = np.histogram2d(x, y, bins=[width, height])

    # Log scale to enhance details
    H = np.log1p(H)  # log(1 + H) to avoid -inf for zero counts
    # Normalise
    if H.max() > 0:
        H /= H.max()

    return H.T  # Transpose so that the first dimension corresponds to y (rows)


def choose_colormap() -> str:
    """Return the name of a random Matplotlib colormap suitable for dark backgrounds."""
    # These colormaps have good contrast on a dark background
    palettes = [
        'magma', 'inferno', 'plasma', 'viridis', 'cividis',
        'twilight', 'hot', 'cool', 'turbo', 'cubehelix',
        'gist_earth', 'jet', 'rainbow', 'gist_rainbow'
    ]
    return random.choice(palettes)


def save_image(density: np.ndarray, colormap: str, filename: str) -> None:
    """Save the density array as a coloured PNG file using the given colormap."""
    plt.figure(figsize=(density.shape[1] / 100.0, density.shape[0] / 100.0), dpi=100)
    # Use extent so that the aspect ratio remains square regardless of DPI
    plt.imshow(density, cmap=colormap, origin='lower', aspect='equal')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def parse_params(s: str) -> Tuple[float, float, float, float]:
    """Parse a comma‑separated list of four floats from the command line."""
    parts = s.split(',')
    if len(parts) != 4:
        raise ValueError('Parameter string must contain exactly four comma‑separated values (a,b,c,d).')
    return tuple(float(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description='Generate chaotic art using the Clifford attractor.')
    parser.add_argument('--count', type=int, default=1, help='Number of images to generate (default: 1)')
    parser.add_argument('--iterations', type=int, default=600_000,
                        help='Number of points to compute per image (default: 600000)')
    parser.add_argument('--resolution', nargs=2, type=int, default=[2000, 2000],
                        metavar=('WIDTH', 'HEIGHT'), help='Output resolution in pixels (width height)')
    parser.add_argument('--output_dir', type=str, default='images', help='Directory to save images')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--params', type=str, default=None,
                        help='Comma‑separated parameters (a,b,c,d).  If supplied, overrides random parameters.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # If user provided custom parameters, parse once and reuse for all images
    user_params = None
    if args.params:
        user_params = parse_params(args.params)

    for i in range(args.count):
        if user_params is not None:
            a, b, c, d = user_params
        else:
            # Draw random parameters uniformly from [-3, 3]
            a, b, c, d = [random.uniform(-3.0, 3.0) for _ in range(4)]

        print(f'Generating image {i + 1}/{args.count} with parameters a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}')

        x, y = clifford_attractor(a, b, c, d, iterations=args.iterations)
        density = render_density(x, y, resolution=tuple(args.resolution))
        cmap = choose_colormap()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = os.path.join(args.output_dir, f'attractor_{i + 1}_{timestamp}.png')
        save_image(density, cmap, filename)
        print(f'  saved to {filename} using colormap {cmap}')

    print('Done.')


if __name__ == '__main__':
    main()