from __future__ import annotations

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_smooth_system(n_freqs: int, alpha: float) -> np.ndarray:
    """
    Build (I + alpha * B^T B), the left-hand-side matrix for the Sx sub-problem.

    B is the (n_freqs-1) x n_freqs first-order finite-difference matrix:
        (B @ x)[i] = x[i+1] - x[i]

    B^T B is tridiagonal:
        main diagonal:  [1, 2, 2, ..., 2, 1]
        off-diagonals:  -1
    """
    N = n_freqs
    diag = np.full(N, 2.0)
    diag[0] = 1.0
    diag[-1] = 1.0
    off = np.full(N - 1, -1.0)
    BtB = np.diag(diag) + np.diag(off, k=1) + np.diag(off, k=-1)
    return np.eye(N) + alpha * BtB


def _prox_l1(u: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-thresholding: proximal operator of the scaled L1 norm."""
    return np.sign(u) * np.maximum(np.abs(u) - threshold, 0.0)


def _fista_sparse(
    rhs: np.ndarray,
    sy_init: np.ndarray,
    beta: float,
    gamma: float,
    n_inner_iter: int,
) -> np.ndarray:
    """
    FISTA (Fast Iterative Shrinkage-Thresholding) update for the Sy sub-problem:

        minimize_{Sy}  ||rhs - Sy||^2 + beta * ||Sy||_1

    The gradient of the data term is 2*(Sy - rhs), so the proximal gradient
    step with step size gamma is:

        Sy_new = prox_{gamma*beta * ||.||_1}( Sy - 2*gamma*(Sy - rhs) )

    Uses Nesterov momentum (FISTA) for faster convergence.
    """
    t = 1.0
    z = sy_init.copy()
    x = sy_init.copy()

    for _ in range(n_inner_iter):
        x_new = _prox_l1(z - 2.0 * gamma * (z - rhs), gamma * beta)
        t_new = (1.0 + np.sqrt(4.0 * t ** 2 + 1.0)) / 2.0
        eta = (t - 1.0) / t_new
        z = x_new + eta * (x_new - x)
        x = x_new
        t = t_new

    return x


def _cost(
    sz: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """
    Objective value:
        ||Sz - Sx - Sy||_F  +  alpha * ||diff(Sx, axis=0)||_F  +  beta * ||Sy||_1

    np.diff(sx, axis=0) is equivalent to B @ Sx where B is the first-difference
    matrix, so this matches costFunction() from the original implementation.
    """
    term1 = float(np.linalg.norm(sz - sx - sy))
    term2 = alpha * float(np.linalg.norm(np.diff(sx, axis=0)))
    term3 = beta * float(np.linalg.norm(sy.ravel(), ord=1))
    return term1 + term2 + term3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def separate_spectrogram(
    mixture: np.ndarray,
    n_sources: int = 2,
    *,
    alpha: float = 1.0,
    beta: float = 1e-5,
    gamma: float = 0.15,
    max_iter: int = 300,
    tol: float = 1e-3,
    n_inner_iter: int = 10,
) -> List[np.ndarray]:
    """
    Separate a mixture spectrogram into two components.

    The mixture is decomposed as  Sz ≈ Sx + Sy  by solving:

        minimize_{Sx, Sy}
            ||Sz - Sx - Sy||_F          (reconstruction fidelity)
            + alpha * ||B Sx||_F        (Sx smooth along frequency axis)
            + beta  * ||Sy||_1          (Sy sparse in time-frequency)

    where B is the first-order finite-difference matrix along the frequency
    axis.  The algorithm alternates between:
      - A closed-form linear solve for Sx  (tridiagonal system per frame)
      - A FISTA proximal-gradient step for Sy  (soft-thresholding)

    Args:
        mixture: 2D non-negative spectrogram, shape (n_freqs, n_frames).
                 Accepts magnitude, power, or log-magnitude — same format
                 as compute_representation() output.
        n_sources: Must be 2. The algorithm always produces exactly two
                   components. Provided for API consistency.
        alpha: Smoothness penalty for Sx (frequency-axis variation).
               Larger values → Sx more tonal / band-limited.
               Typical range: 0.1 – 10.
        beta: Sparsity penalty for Sy (L1 norm).
              Larger values → Sy sparser, more impulsive.
              Typical range: 1e-6 – 1e-3.
        gamma: FISTA step size. Must satisfy 0 < gamma < 0.5 for convergence.
               Default 0.15 works well for normalised spectrograms.
        max_iter: Maximum number of outer alternating iterations.
        tol: Convergence threshold (relative decrease in cost). Iteration
             stops early when  (cost_prev - cost) / cost_prev  < tol.
        n_inner_iter: Number of FISTA inner iterations per outer step.

    Returns:
        [Sx, Sy] — list of two arrays, each with shape (n_freqs, n_frames).

        Sx — spectrally smooth component  (tonal / harmonic content)
        Sy — sparse component             (transient / impulsive content)

    Notes:
        Output assignment:  the algorithm has no knowledge of the original
        sources.  Which of [Sx, Sy] corresponds to which source depends on
        the spectral structure of the signals; compare against reference
        spectrograms to determine the correct assignment.

        Non-negativity:  outputs are not constrained to be non-negative.
        Clip if needed:  Sx = np.clip(Sx, 0, None)

        Reference:  algorithm adapted from Meynard (2023),
        SpectrogramSeparation/Estimation_algorithm.py.
    """
    if n_sources != 2:
        raise ValueError(
            f"Only n_sources=2 is supported. Got n_sources={n_sources}."
        )

    sz = np.array(mixture, dtype=float)
    n_freqs, _ = sz.shape

    mat = _build_smooth_system(n_freqs, alpha)
    sy = np.zeros_like(sz)
    cost_prev = np.inf

    for k in range(max_iter):
        # ── Sx sub-problem: (I + alpha * B^T B) @ Sx = Sz - Sy ──────────
        sx = np.linalg.solve(mat, sz - sy)

        # ── Sy sub-problem: FISTA proximal gradient ───────────────────────
        sy = _fista_sparse(sz - sx, sy, beta, gamma, n_inner_iter)

        # ── Convergence check ─────────────────────────────────────────────
        cost = _cost(sz, sx, sy, alpha, beta)
        if k > 0:
            relative_decrease = (cost_prev - cost) / (abs(cost_prev) + 1e-12)
            if relative_decrease < tol:
                break
        cost_prev = cost

    return [sx, sy]
