"""Sheared-frame chain machinery vs direct methods (run: pytest)."""

import numpy as np
import qutip as qt

from channel import qfi_fock, gaussian_qfi, sqz_cov
from chains import qfi_chain
from probes import make_probe


def _dm(kind, n, N, **kw):
    return qt.ket2dm(make_probe(kind, n, N, **kw))


def test_frame_vs_direct_loss_placements():
    """At K = 1 (cheap for the direct method) the sheared-frame chain must
    match the direct Fock channel for all three loss placements."""
    K, eta, N = 1.0, 0.8, 220
    rho = _dm("cat", 2.0, N)
    for place, direct_name in (("pre", "inj"), ("post", "det"),
                               ("conc", "conc")):
        q_frame = qfi_chain(rho, K, N, [("loss", eta, place)])
        q_direct = qfi_fock(rho, K, N, direct_name, eta)
        tol = 5e-3 if place == "conc" else 1e-6   # Trotter slices
        assert abs(q_frame - q_direct) / q_direct < tol, (place, q_frame,
                                                          q_direct)


def test_frame_high_K_matches_gaussian():
    """At K = 9.4 (10 Hz) the direct method is intractable; the frame
    method must still match the exact Gaussian result for a squeezed probe."""
    K, eta = 9.4, 0.9
    r = np.arcsinh(np.sqrt(2.0))
    for place in ("pre", "post", "conc"):
        gq = gaussian_qfi(sqz_cov(r, 0.0), K,
                          {"pre": "inj", "post": "det", "conc": "conc"}[place],
                          eta)
        N = 260
        q = qfi_chain(_dm("sqz_vac", 2.0, N, theta=0.0), K, N,
                      [("loss", eta, place)])
        tol = 6e-3 if place == "conc" else 2e-3
        assert abs(q - gq) / gq < tol, (place, q, gq)


def test_pn_chain_orderings_differ():
    """PN before vs after the SB block are different channels at K > 0."""
    K, chi, N = 2.0, 0.1, 200
    rho = _dm("sqz_vac", 2.0, N, theta=0.0)
    q_pre = qfi_chain(rho, K, N, [("pn", chi, "pre")])
    q_post = qfi_chain(rho, K, N, [("pn", chi, "post")])
    assert abs(q_pre - q_post) / q_pre > 1e-3


def test_pn_at_zero_K_placement_irrelevant():
    """With no back-action, PN placement around a commuting block is moot."""
    chi, N = 0.1, 160
    rho = _dm("cat", 2.0, N)
    q_pre = qfi_chain(rho, 0.0, N, [("pn", chi, "pre")])
    q_post = qfi_chain(rho, 0.0, N, [("pn", chi, "post")])
    assert abs(q_pre - q_post) / q_pre < 1e-8


def test_two_noise_chain_runs():
    """PN -> SB -> loss vs loss -> SB -> PN: both run and differ at K > 0."""
    K, N = 2.0, 220
    rho = _dm("sqz_vac", 2.0, N, theta=0.0)
    q1 = qfi_chain(rho, K, N, [("pn", 0.1, "pre"), ("loss", 0.9, "post")])
    q2 = qfi_chain(rho, K, N, [("loss", 0.9, "pre"), ("pn", 0.1, "post")])
    assert q1 > 0 and q2 > 0 and abs(q1 - q2) / q1 > 1e-3
