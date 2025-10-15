#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BB-BPRF Minimal Demo — Population-of-One Behavioral Calibration
License: MIT (c) 2025 Charles Thomas

What this shows
♦ population-of-one baselines via exponential filters
♦ variance-reduction vs. a naive (demographic) group baseline
♦ convergence metric (moving KL to baseline proxy)
♦ ephemeral initialization vectors (EIV) with cryptographic decay
♦ simple adversarial/baseline-poisoning guardrails
"""
from __future__ import annotations
import hashlib, hmac, os, time, math, random
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

# -----------------------
# Config
# -----------------------
RNG_SEED = 42
random.seed(RNG_SEED); np.random.seed(RNG_SEED)

# Two "modalities" for demo: response latency (ms), keystroke interval (ms)
MODALITIES = ["latency_ms", "keystroke_ms"]

# Exponential moving average/variance smoothing for population-of-one
EMA_ALPHA_MEAN = 0.12
EMA_ALPHA_VAR = 0.10
EPS = 1e-9

# EIV parameters
EPOCH_SECONDS = 120.0         # EIV changes every 2 minutes
EIV_KEY_BYTES = 32            # master secret length
EIV_TRUNC = 16                # truncate tag for storage/transport

# Adversarial guard: reject updates if z-score spike suggests poisoning
MAX_ABS_Z_FOR_UPDATE = 4.0

# -----------------------
# Utilities
# -----------------------
def epoch_index(now_s: float, epoch_s: float) -> int:
    return int(now_s // epoch_s)

def hkdf_sha256(key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    # Minimal HKDF (RFC5869) style derivation using hashlib; demo only.
    prk = hmac.new(salt, key, hashlib.sha256).digest()
    t = b""
    okm = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:length]

@dataclass
class EIV:
    """Ephemeral Initialization Vector with cryptographic decay."""
    master_secret: bytes

    def derive(self, user_id: str, now_s: float) -> bytes:
        # Derive per-epoch IV bound to user_id and time-epoch; decays when epoch rolls.
        ep_idx = epoch_index(now_s, EPOCH_SECONDS)
        salt = hashlib.sha256(f"{user_id}:{ep_idx}".encode()).digest()
        info = b"BB-BPRF:EIV:v1"
        iv = hkdf_sha256(self.master_secret, salt, info, 32)
        return iv[:EIV_TRUNC]  # transport-friendly truncated tag

# -----------------------
# Population-of-One Baseline
# -----------------------
@dataclass
class Po1Stats:
    mean: float = 0.0
    var: float = 1.0  # start non-zero to avoid div-by-zero
    initialized: bool = False

    def update(self, x: float) -> None:
        if not self.initialized:
            self.mean = x
            self.var = 1.0  # conservative start
            self.initialized = True
            return
        # EMA for mean and variance (population-of-one, no group terms)
        delta = x - self.mean
        self.mean += EMA_ALPHA_MEAN * delta
        # Update var with EMA of squared deviation
        dev2 = delta * delta
        self.var = (1 - EMA_ALPHA_VAR) * self.var + EMA_ALPHA_VAR * dev2
        # Lower bound to avoid collapse
        self.var = max(self.var, 1e-6)

@dataclass
class Po1Model:
    per_modality: Dict[str, Po1Stats]

    def zscore(self, modality: str, x: float) -> float:
        st = self.per_modality[modality]
        return (x - st.mean) / math.sqrt(st.var + EPS)

    def maybe_update(self, modality: str, x: float) -> bool:
        """Update baseline unless the sample looks adversarial/poisoning."""
        z = self.zscore(modality, x) if self.per_modality[modality].initialized else 0.0
        if abs(z) > MAX_ABS_Z_FOR_UPDATE:
            # Reject extreme update to guard against poisoning
            return False
        self.per_modality[modality].update(x)
        return True

# -----------------------
# Variance Reduction & Convergence
# -----------------------
def variance_reduction(group_var: float, po1_var: float) -> float:
    """How much variance we shaved off vs. naive group variance."""
    group_var = max(group_var, EPS)
    return max(0.0, 1.0 - (po1_var / group_var))

def moving_kl(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """
    KL divergence D_KL(N(mu_p, var_p) || N(mu_q, var_q)) in 1D.
    Used as a convergence proxy: as personal baseline stabilizes, KL -> smaller.
    """
    mu_p, var_p = p
    mu_q, var_q = q
    var_p = max(var_p, EPS); var_q = max(var_q, EPS)
    return 0.5 * ( (var_p/var_q) + ((mu_q - mu_p)**2)/var_q - 1.0 + math.log(var_q/var_p) )

# -----------------------
# Synthetic Data Generator
# -----------------------
def synth_session(n: int, drift: float=0.0, jitter: float=1.0) -> Dict[str, np.ndarray]:
    """
    Generate synthetic behavioral samples for a session.
    latency_ms ~ baseline 250ms +/- jitter; keystroke_ms ~ 140ms +/- jitter
    Optional drift simulates learning/fatigue or context shift.
    """
    t = np.arange(n)
    latency = 250 + drift * (t / max(1, n-1)) * 40.0 + np.random.randn(n) * (12.0*jitter)
    kstroke = 140 + drift * (t / max(1, n-1)) * (-20.0) + np.random.randn(n) * (8.0*jitter)
    return {"latency_ms": latency, "keystroke_ms": kstroke}

# -----------------------
# Demo
# -----------------------
def main():
    # Master secret for EIV (normally in secure storage/HSM)
    eiv = EIV(master_secret=os.urandom(EIV_KEY_BYTES))
    user_id = "user_7fd3"
    print(f"[EIV tag @ t0]: {eiv.derive(user_id, time.time()).hex()}")

    # Naive "group" baselines (pretend demographic average); used only for comparison
    group_mean = {"latency_ms": 270.0, "keystroke_ms": 150.0}
    group_var  = {"latency_ms": 25.0**2, "keystroke_ms": 18.0**2}

    # Initialize population-of-one model
    po1 = Po1Model({m: Po1Stats() for m in MODALITIES})

    # Simulate 3 sessions: warmup, stable, slight drift
    sessions = [
        synth_session(120, drift=0.2, jitter=1.3),   # S1: settling in
        synth_session(150, drift=0.0, jitter=0.9),   # S2: stable flow
        synth_session(160, drift=0.15, jitter=1.0),  # S3: mild context shift
    ]

    # Track metrics over time for reporting
    logs: List[Dict[str, float]] = []

    for s_idx, sess in enumerate(sessions, start=1):
        # New epoch ⇒ EIV rotates; privacy-preserving per-session tag
        now = time.time() + s_idx * (EPOCH_SECONDS + 1)
        print(f"[EIV tag @ session {s_idx}]: {eiv.derive(user_id, now).hex()}")

        for i in range(len(next(iter(sess.values())))):
            row = {}
            for m in MODALITIES:
                x = float(sess[m][i])

                # Current z-score before update
                z_before = po1.zscore(m, x) if po1.per_modality[m].initialized else 0.0

                updated = po1.maybe_update(m, x)

                # After update, capture mean/var
                mu = po1.per_modality[m].mean
                var = po1.per_modality[m].var

                # Convergence proxy: KL(po1 || group)
                kl = moving_kl((mu, var), (group_mean[m], group_var[m]))

                # Variance reduction vs group
                vr = variance_reduction(group_var[m], var)

                row.update({
                    f"{m}_x": x,
                    f"{m}_z_before": z_before,
                    f"{m}_mu": mu,
                    f"{m}_var": var,
                    f"{m}_kl": kl,
                    f"{m}_var_reduction": vr,
                    f"{m}_updated": 1.0 if updated else 0.0,
                })
            logs.append(row)

    # Summarize end-state metrics
    def pct(x): return f"{x*100:5.1f}%"

    print("\n=== Summary (end of Session 3) ===")
    for m in MODALITIES:
        mu = po1.per_modality[m].mean
        var = po1.per_modality[m].var
        vr = variance_reduction(group_var[m], var)

        # Convergence across last 50 samples
        tail = logs[-50:]
        avg_kl = np.mean([r[f"{m}_kl"] for r in tail])
        upd_rate = np.mean([r[f"{m}_updated"] for r in tail])

        print(f"[{m}] mu={mu:7.3f}  var={var:7.3f}  "
              f"variance_reduction={pct(vr)}  tail_avg_KL={avg_kl:6.3f}  "
              f"accepted_updates(last50)={pct(upd_rate)}")

    # Sanity check: are we getting a multi-modal advantage?
    # A simple heuristic: average per-modality variance reduction
    avg_vr = np.mean([variance_reduction(group_var[m], po1.per_modality[m].var) for m in MODALITIES])
    print(f"\n[Overall] average variance reduction vs. group: {pct(avg_vr)}")

    # Demonstrate adversarial guard: inject an extreme outlier, ensure it's rejected
    print("\n=== Adversarial Injection Test ===")
    for m, bad in [("latency_ms", 1000.0), ("keystroke_ms", 10.0)]:
        before_mu, before_var = po1.per_modality[m].mean, po1.per_modality[m].var
        accepted = po1.maybe_update(m, bad)
        after_mu, after_var = po1.per_modality[m].mean, po1.per_modality[m].var
        print(f"{m}: injected={bad}, accepted={accepted}, "
              f"mu {before_mu:.2f}->{after_mu:.2f}, var {before_var:.2f}->{after_var:.2f}")

if __name__ == "__main__":
    main()
