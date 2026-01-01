"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import jax
import jax.numpy as jnp
from diffres.tools import logpdf_mvn_chol, leading_concat, chol_solve


def kf_pred(mf, vf, semigroup, trans_cov):
    mp = semigroup @ mf
    vp = semigroup @ vf @ semigroup.T + trans_cov
    return mp, vp


def kf_update(mp, vp, y, obs_op, obs_cov):
    S = obs_op @ vp @ obs_op.T + obs_cov
    ch = jnp.linalg.cholesky(S, upper=False)
    K = chol_solve(ch, obs_op @ vp).T
    mf = mp + K @ (y - obs_op @ mp)
    vf = vp - K @ S @ K.T
    nll = -logpdf_mvn_chol(y, obs_op @ mp, ch)
    return mf, vf, nll


def kf(ys, m0, v0, semigroup, trans_cov, obs_op, obs_cov):
    mf0, vf0, nll0 = kf_update(m0, v0, ys[0], obs_op, obs_cov)

    def scan_body(carry, elem):
        mf, vf, nll_ = carry
        y = elem

        mp, vp = kf_pred(mf, vf, semigroup, trans_cov)
        mf, vf, nll_k = kf_update(mp, vp, y, obs_op, obs_cov)
        return (mf, vf, nll_ + nll_k), (mf, vf, mp, vp)

    (_, _, nll), (mfs, vfs, mps, vps) = jax.lax.scan(scan_body, (mf0, vf0, nll0), ys[1:])
    return leading_concat(mf0, mfs), leading_concat(vf0, vfs), nll, mps, vps


def rts(mfs, vfs, mps, vps, semigroup):
    def scan_body(carry, elem):
        ms, vs = carry
        mf, vf, mp, vp = elem

        chol = jax.scipy.linalg.cho_factor(vp)
        G = jax.scipy.linalg.cho_solve(chol, semigroup.T @ vf).T
        ms = mf + G @ (ms - mp)
        vs = vf + G @ (vs - vp) @ G.T
        return (ms, vs), (ms, vs)

    _, (mss, vss) = jax.lax.scan(scan_body, (mfs[-1], vfs[-1]), (mfs[:-1], vfs[:-1], mps, vps), reverse=True)
    return leading_concat(mss, mfs[-1]), leading_concat(vss, vfs[-1])
