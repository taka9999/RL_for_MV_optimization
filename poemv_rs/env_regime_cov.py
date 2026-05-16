from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RSGBMParamsRegimeCov:
    mu1: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.18], dtype=float))
    mu2: np.ndarray = field(default_factory=lambda: np.array([-0.73, -0.40], dtype=float))
    Sigma1: np.ndarray = field(default_factory=lambda: np.array(
        [[0.22**2, 0.22 * 0.18 * 0.30],
         [0.22 * 0.18 * 0.30, 0.18**2]],
        dtype=float,
    ))
    Sigma2: np.ndarray = field(default_factory=lambda: np.array(
        [[0.30**2, 0.30 * 0.24 * 0.55],
         [0.30 * 0.24 * 0.55, 0.24**2]],
        dtype=float,
    ))
    lam1: float = 0.36
    lam2: float = 2.89
    r: float = 0.01


@dataclass
class EpisodeConfigRegimeCov:
    T_years: float = 10.0
    dt: float = 1 / 252
    x0: float = 1.0
    s0: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0], dtype=float))
    p0: float = 0.5
    a_max: float | None = None
    seed: int = 0
    apply_action_projection: bool = True


def _project_to_gross_leverage(u: np.ndarray, wealth: float, gross_cap: float | None) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(-1)
    if gross_cap is None:
        return u
    denom = max(abs(float(wealth)), 1e-12)
    gross = float(np.sum(np.abs(u)) / denom)
    if gross <= gross_cap or gross <= 1e-12:
        return u
    return u * (gross_cap / gross)


class RSGBMEnvRegimeCov:
    """Regime-switching GBM with regime-dependent covariance matrices."""

    def __init__(self, params: RSGBMParamsRegimeCov, cfg: EpisodeConfigRegimeCov):
        self.params = params
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.reset()

    @property
    def n_steps(self) -> int:
        return int(round(self.cfg.T_years / self.cfg.dt))

    def _mu_Sigma(self, I: int):
        if I == 1:
            return np.asarray(self.params.mu1, dtype=float), np.asarray(self.params.Sigma1, dtype=float)
        return np.asarray(self.params.mu2, dtype=float), np.asarray(self.params.Sigma2, dtype=float)

    def reset(self):
        self.t = 0.0
        self.k = 0
        self.I = 1 if self.rng.random() < self.cfg.p0 else 2
        self.S = np.array(self.cfg.s0, dtype=float).copy()
        self.X = float(self.cfg.x0)
        return self._obs()

    def step(self, u: np.ndarray):
        dt = self.cfg.dt
        u = np.asarray(u, dtype=float).reshape(-1)
        if self.cfg.apply_action_projection:
            u = _project_to_gross_leverage(u, self.X, self.cfg.a_max)

        if self.I == 1:
            if self.rng.random() < self.params.lam1 * dt:
                self.I = 2
        else:
            if self.rng.random() < self.params.lam2 * dt:
                self.I = 1

        mu, Sigma = self._mu_Sigma(self.I)
        eps = self.rng.normal(size=mu.shape[0])
        L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(mu.shape[0]))
        dlogS = (mu - 0.5 * np.diag(Sigma)) * dt + L @ (np.sqrt(dt) * eps)
        S_next = self.S * np.exp(dlogS)
        discounted_ret = np.exp(dlogS - self.params.r * dt) - 1.0
        X_next = self.X + float(np.dot(u, discounted_ret))

        if not np.isfinite(X_next):
            raise FloatingPointError(
                f"X_next became non-finite: X={self.X}, u={u}, discounted_ret={discounted_ret}"
            )

        self.S = np.asarray(S_next, dtype=float)
        self.X = float(X_next)
        self.t += dt
        self.k += 1
        done = self.k >= self.n_steps
        return self._obs(), discounted_ret, done

    def _obs(self):
        return {"t": self.t, "k": self.k, "S": self.S, "X": self.X, "I_true": self.I}
