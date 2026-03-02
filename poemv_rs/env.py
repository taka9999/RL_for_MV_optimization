from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RSGBMParams:
    # 2-asset: mu_i is (2,), Sigma_i is (2,2)
    mu1: np.ndarray = np.array([0.25, 0.18], dtype=float)
    mu2: np.ndarray = np.array([-0.73, -0.40], dtype=float)
    Sigma1: np.ndarray = np.array([[0.22**2, 0.22*0.18*0.3],
                                   [0.22*0.18*0.3, 0.18**2]], dtype=float)
    Sigma2: np.ndarray = np.array([[0.22**2, 0.22*0.18*0.5],
                                   [0.22*0.18*0.5, 0.18**2]], dtype=float)
    lam1: float = 0.36   # intensity 1->2
    lam2: float = 2.89   # intensity 2->1
    r: float = 0.0

@dataclass
class EpisodeConfig:
    T_years: float = 10.0
    dt: float = 1/252
    x0: float = 1.0
    s0: np.ndarray = np.array([1.0, 1.0], dtype=float)
    p0: float = 0.5
    a_max: float = 2.0
    omega: float = 0.0
    seed: int = 0

class RSGBMEnv:
    """Simulate price and wealth under hidden 2-state CTMC regime.

    Regime I_t in {1,2}, with transition intensities:
      1 -> 2: lam1
      2 -> 1: lam2
    Risky price:
      d log S = (mu_{I_t} - 0.5 diag(Sigma_{I_t})) dt + L_{I_t} dB,  Sigma = L L^T
    Wealth (discounted, r=0 by default):
      X_{k+1} = X_k + u_k^T * ((S_{k+1}-S_k)/S_k)
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
        self.I = 1  # start in regime 1 by default
        self.S = np.array(self.cfg.s0, dtype=float).copy()  # (2,)
        self.X = float(self.cfg.x0)
        return self._obs()

    def _mu_Sigma(self, I: int):
        if I == 1:
            return np.asarray(self.params.mu1, dtype=float), np.asarray(self.params.Sigma1, dtype=float)
        else:
            return np.asarray(self.params.mu2, dtype=float), np.asarray(self.params.Sigma2, dtype=float)

    def step(self, u: np.ndarray):
        dt = self.cfg.dt
        # Action bounds are enforced by the policy via tanh-squashing.
        u = np.asarray(u, dtype=float).reshape(2,)
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

        # wealth update (discounted; if r!=0 you can discount explicitly)
        ret = (S_next - self.S) / self.S     # (2,)
        X_next = self.X + float(np.dot(u, ret))

        # stop immediately if non-finite shows up (prevents silent CSV corruption)
        if not np.isfinite(X_next):
            raise FloatingPointError(
                f"X_next became non-finite: X={self.X}, u={u}, ret={ret}, S={self.S}, S_next={S_next}"
            )

        self.S, self.X = np.array(S_next, dtype=float), float(X_next)
        self.t += dt
        self.k += 1
        done = (self.k >= self.n_steps)
        return self._obs(), ret, done

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
        Sigma = np.asarray(params.Sigma1 if i==1 else params.Sigma2, dtype=float)
        L = np.linalg.cholesky(Sigma + 1e-12*np.eye(2))
        eps = rng.normal(size=2)
        dlogS = (mu - 0.5*np.diag(Sigma))*dt + (L @ (np.sqrt(dt)*eps))
        S[k+1]=S[k]*np.exp(dlogS)
    return S, I

class HistoricalLogReturnEnv:
    """Replay environment driven by historical log-returns.

    Inputs:
      logret: np.ndarray shape (T, d), where logret[t] = log(S_{t+1}/S_t) for each asset.
    Dynamics:
      S_{t+1} = S_t * exp(logret_t)
      r_t (simple) = exp(logret_t) - 1
      X_{t+1} = X_t + u_t^T r_t
    """
    def __init__(self, logret: np.ndarray, dt: float, x0: float = 1.0, s0=None, seed: int = 0):
        self.logret = np.asarray(logret, dtype=float)
        if self.logret.ndim != 2:
            raise ValueError("logret must be 2D (T,d)")
        self.T, self.d = self.logret.shape
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        if s0 is None:
            self.s0 = np.ones(self.d, dtype=float)
        else:
            self.s0 = np.asarray(s0, dtype=float).reshape(self.d,)
        self.x0 = float(x0)
        self.reset(0)

    @property
    def n_steps(self) -> int:
        return self.T

    def reset(self, start_idx: int = 0):
        if not (0 <= start_idx <= self.T):
            raise ValueError("start_idx out of range")
        self.start_idx = int(start_idx)
        self.k = 0
        self.t = 0.0
        self.S = self.s0.copy()
        self.X = self.x0
        return self._obs()

    def step(self, u: np.ndarray):
        if self.k >= self.T:
            return self._obs(), np.zeros(self.d), True
        u = np.asarray(u, dtype=float).reshape(self.d,)
        lr = self.logret[self.start_idx + self.k]  # (d,)
        S_next = self.S * np.exp(lr)
        ret = np.exp(lr) - 1.0                     # (d,)
        X_next = self.X + float(np.dot(u, ret))
        if not np.isfinite(X_next):
            raise FloatingPointError(f"X_next non-finite: X={self.X}, u={u}, ret={ret}")
        self.S = S_next
        self.X = float(X_next)
        self.k += 1
        self.t += self.dt
        done = (self.k >= self.T)
        return self._obs(), ret, done

    def _obs(self):
        return {"t": self.t, "k": self.k, "S": self.S, "X": self.X}