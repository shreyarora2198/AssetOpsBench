# confidence_evaluator.py
#
# Matches main.tex §5 (Confidence Evaluator): decides whether to invoke full
# statistical TSFM after FMSR's first diagnosis.  The numerical confidence
# itself comes from tools.score_diagnosis_confidence (heuristic proxy until
# FMSR exposes native scores).

from __future__ import annotations

import os


def theta_from_env() -> float:
    """Confidence threshold θ; proposal evaluates θ ∈ {0.7, 0.8, 0.9}.

    Default **0.85** so medium-severity FMSR proxy scores (0.84) fall below θ and
    conditional deep TSFM can run; set ``RCA_CONFIDENCE_THETA=0.8`` to skip deep
    more aggressively (see ``tools.score_diagnosis_confidence``).
    """
    return float(os.getenv("RCA_CONFIDENCE_THETA", "0.85"))


def conditional_deep_tsfm_enabled() -> bool:
    return os.getenv("ENABLE_CONDITIONAL_DEEP_TSFM", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def should_invoke_deep_tsfm(
    diagnosis_confidence: float,
    *,
    theta: float | None = None,
    conditional_enabled: bool | None = None,
) -> bool:
    """Return True iff we should run Deep TSFM validation (proposal Alg.~2, No path).

    High FMSR confidence (≥ θ) → skip expensive TSFM.  Low confidence (< θ) → run
    ``deep_tsfm_refine_anomalies`` then re-run FMSR mapping in RCA.
    """
    if conditional_enabled is None:
        conditional_enabled = conditional_deep_tsfm_enabled()
    if not conditional_enabled:
        return False
    t = theta if theta is not None else theta_from_env()
    return diagnosis_confidence < t
