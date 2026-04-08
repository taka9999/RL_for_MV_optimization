from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class RSGBMParams:
    mu1: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.18], dtype=float))
    mu2: np.ndarray = field(default_factory=lambda: np.array([-0.73, -0.40], dtype=float))
    Sigma: np.ndarray = field(default_factory=lambda: np.array(
        [[0.22**2, 0.22 * 0.18 * 0.3],
         [0.22 * 0.18 * 0.3, 0.18**2]],
        dtype=float,
    ))
    lam1: float = 0.36   # intensity 1->2
    lam2: float = 2.89   # intensity 2->1
    r: float = 0.01

    @property
    def Sigma1(self) -> np.ndarray:
        return np.asarray(self.Sigma, dtype=float)

    @property
    def Sigma2(self) -> np.ndarray:
        return np.asarray(self.Sigma, dtype=float)

@dataclass
class RSGBMParams2:
    """Backward-compatible alias for two-asset regime-switching GBM params."""
    mu1: np.ndarray
    mu2: np.ndarray
    Sigma: np.ndarray
    lam1: float = 0.36
    lam2: float = 2.89
    r: float = 0.01

    @property
    def Sigma1(self) -> np.ndarray:
        return np.asarray(self.Sigma, dtype=float)

    @property
    def Sigma2(self) -> np.ndarray:
        return np.asarray(self.Sigma, dtype=float)

@dataclass
class EpisodeConfig:
    T_years: float = 10.0
    dt: float = 1/252
    x0: float = 1.0
    s0: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0], dtype=float))
    p0: float = 0.5
    a_max: float = 2.0
    omega: float = 0.0
    seed: int = 0
    apply_action_projection: bool = True


def _project_to_gross_leverage(u: np.ndarray, wealth: float, gross_cap: float | None) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(2,)
    if gross_cap is None:
        return u
    denom = max(abs(float(wealth)), 1e-12)
    gross = float(np.sum(np.abs(u)) / denom)
    if gross <= gross_cap or gross <= 1e-12:
        return u
    return u * (gross_cap / gross)


class RSGBMEnv:
    """
    Simulate price and discounted wealth under hidden 2-state CTMC regime.

    Regime I_t in {1,2}, with transition intensities:
      1 -> 2: lam1
      2 -> 1: lam2
    Risky price:
      d log S = (mu_{I_t} - 0.5 diag(Sigma)) dt + L dB,  Sigma = L L^T
    Discounted wealth:
      X_{k+1} = X_k + u_k^T * (exp(d log S_k - r dt) - 1)
    where u_k is the vector of discounted dollar amounts invested in the risky assets.
    """

    def __init__(self, params: RSGBMParams, cfg: EpisodeConfig):
        self.params = params
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.reset()

    @property
    def n_steps(self) -> int:
        return int(round(self.cfg.T_years / self.cfg.dt))

    def reset(self):
        self.t = 0.0
        self.k = 0
        self.I = 1 if self.rng.random() < self.cfg.p0 else 2
        self.S = np.array(self.cfg.s0, dtype=float).copy()  # (2,)
        self.X = float(self.cfg.x0)
        return self._obs()

    def _mu_Sigma(self, I: int):
        if I == 1:
            return np.asarray(self.params.mu1, dtype=float), np.asarray(self.params.Sigma, dtype=float)
        return np.asarray(self.params.mu2, dtype=float), np.asarray(self.params.Sigma, dtype=float)

    def step(self, u: np.ndarray):
        dt = self.cfg.dt
        u = np.asarray(u, dtype=float).reshape(2,)
        # For policy-gradient training we want the environment to execute the
        # same action that the actor assigned log-probability to. The agent
        # already applies a smooth leverage-aware squash, so the extra hard
        # projection is disabled by default and only kept as an optional guard.
        if self.cfg.apply_action_projection:
            u = _project_to_gross_leverage(u, self.X, self.cfg.a_max)
        # regime transition
        if self.I == 1:
            if self.rng.random() < self.params.lam1 * dt:
                self.I = 2
        else:
            if self.rng.random() < self.params.lam2 * dt:
                self.I = 1

        # price update
        eps = self.rng.normal(size=2)
        mu, Sigma = self._mu_Sigma(self.I)  # mu:(2,), Sigma:(2,2)
        L = np.linalg.cholesky(Sigma + 1e-12*np.eye(2))
        dlogS = (mu - 0.5*np.diag(Sigma)) * dt + (L @ (np.sqrt(dt)*eps))
        S_next = self.S * np.exp(dlogS)

        discounted_ret = np.exp(dlogS - self.params.r * dt) - 1.0
        X_next = self.X + float(np.dot(u, discounted_ret))

        # stop immediately if non-finite shows up (prevents silent CSV corruption)
        if not np.isfinite(X_next):
            raise FloatingPointError(
                f"X_next became non-finite: X={self.X}, u={u}, discounted_ret={discounted_ret}, S={self.S}, S_next={S_next}"
            )

        self.S, self.X = np.array(S_next, dtype=float), float(X_next)
        self.t += dt
        self.k += 1
        done = self.k >= self.n_steps
        return self._obs(), discounted_ret, done

    def _obs(self):
        return {
            "t": self.t,
            "k": self.k,
            "S": self.S,
            "X": self.X,
            "I_true": self.I,
        }

def simulate_price_only(params: RSGBMParams, T_years: float, dt: float, s0: float=1.0, seed: int=0):
    """Convenience: simulate (S_k, I_k) without wealth."""
    rng = np.random.default_rng(seed)
    n = int(round(T_years/dt))
    S = np.empty((n+1, 2), dtype=float); I = np.empty(n+1, dtype=int)
    S[0]=np.array([s0, s0], dtype=float); I[0]=1
    for k in range(n):
        i = I[k]
        # transition
        if i==1:
            if rng.random() < params.lam1*dt: i=2
        else:
            if rng.random() < params.lam2*dt: i=1
        I[k+1]=i
        mu = np.asarray(params.mu1 if i==1 else params.mu2, dtype=float)
        Sigma = np.asarray(params.Sigma, dtype=float)
        L = np.linalg.cholesky(Sigma + 1e-12*np.eye(2))
        eps = rng.normal(size=2)
        dlogS = (mu - 0.5*np.diag(Sigma))*dt + (L @ (np.sqrt(dt)*eps))
        S[k+1]=S[k]*np.exp(dlogS)
    return S, I
