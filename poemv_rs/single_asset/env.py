from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RSGBMParams:
    mu1: float = 0.25
    mu2: float = -0.73
    sigma: float = 0.22
    lam1: float = 0.36   # intensity 1->2
    lam2: float = 2.89   # intensity 2->1
    r: float = 0.0

@dataclass
class EpisodeConfig:
    T_years: float = 10.0
    dt: float = 1/252
    x0: float = 1.0
    s0: float = 1.0
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
      dS/S = mu_{I_t} dt + sigma dB
    Wealth (discounted, r=0 by default):
      X_{k+1} = X_k + u_k * (S_{k+1}-S_k)/S_k
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
        self.S = float(self.cfg.s0)
        self.X = float(self.cfg.x0)
        return self._obs()

    def _mu(self, I: int) -> float:
        return self.params.mu1 if I == 1 else self.params.mu2

    def step(self, u: float):
        dt = self.cfg.dt
        # Action bounds are enforced by the policy via tanh-squashing.
        u = float(u)
        # regime transition
        if self.I == 1:
            if self.rng.random() < self.params.lam1 * dt:
                self.I = 2
        else:
            if self.rng.random() < self.params.lam2 * dt:
                self.I = 1

        # price update
        eps = self.rng.normal()
        mu = self._mu(self.I)
        sigma = self.params.sigma
        S_next = self.S * np.exp((mu - 0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*eps)

        # wealth update (discounted; if r!=0 you can discount explicitly)
        ret = (S_next - self.S) / self.S
        a = u
        #growth = 1.0 + a * ret
        #growth = max(growth, 1e-6)
        #X_next = self.X * growth
        X_next = self.X + a * ret

        # stop immediately if non-finite shows up (prevents silent CSV corruption)
        if not np.isfinite(X_next):
            raise FloatingPointError(
                f"X_next became non-finite: X={self.X}, u={u}, ret={ret}, S={self.S}, S_next={S_next}"
            )

        self.S, self.X = float(S_next), float(X_next)
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
    S = np.empty(n+1); I = np.empty(n+1, dtype=int)
    S[0]=s0; I[0]=1
    for k in range(n):
        i = I[k]
        # transition
        if i==1:
            if rng.random() < params.lam1*dt: i=2
        else:
            if rng.random() < params.lam2*dt: i=1
        I[k+1]=i
        mu = params.mu1 if i==1 else params.mu2
        eps = rng.normal()
        S[k+1]=S[k]*np.exp((mu-0.5*params.sigma**2)*dt + params.sigma*np.sqrt(dt)*eps)
    return S, I
