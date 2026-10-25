# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gluon kernel of the fused Kimi-K3 KDA speculative verify.

See kda_fused_spec_verify.py for what it computes and how it is used; this
module is imported only when the kernel is launched (it needs Triton with
Gluon and an NVIDIA GPU).
"""

from vllm.triton_utils import gl, gluon

gac = gl.nvidia.ampere.async_copy

# --------------------------------------------------------------------------
# Explicit-rounding arithmetic (inline PTX).
#   conv:   acc = fma(w0, c0, 0); fma(w1, c1, acc); fma(w2, c2, acc);
#           fma(w3, x, acc)
#   l2norm: (fma(x0, x0, x2*x2) + fma(x1, x1, x3*x3)), 32-lane butterfly
#   rowstep (one token for whole state rows):
#     decay   S' = S * dec
#     k dot   ((fma(S'0,k0,S'2*k2) + fma(S'4,k4,S'6*k6))
#              + (fma(S'1,k1,S'3*k3) + fma(S'5,k5,S'7*k7))), 16-lane butterfly
#     update  S_new = fma(S, dec, k * v_new)
#     q dot   p_i = S_new_i * q_i; ((p0+p2)+(p4+p6)) + ((p1+p3)+(p5+p7)),
#             16-lane butterfly. The products are added with scalar
#             add.rn.f32: a packed mul.rn.f32x2 -> add.rn.f32x2 pair may be
#             fused into a packed FMA by some ptxas versions.
#   norm:   (fma(x0,x0,x2*x2) + fma(x4,x4,x6*x6))
#           + (fma(x1,x1,x3*x3) + fma(x5,x5,x7*x7)), 16-lane butterfly
# --------------------------------------------------------------------------
_ASM_CONV4 = gl.constexpr(
    "{ .reg .f32 a; fma.rn.f32 a, $5, $1, 0f00000000; fma.rn.f32 a, $6, $2, a; "
    "fma.rn.f32 a, $7, $3, a; fma.rn.f32 $0, $8, $4, a; }"
)
_C9 = gl.constexpr("=r,r,r,r,r,r,r,r,r")
_C4P = gl.constexpr("=r,=r,=r,=r,r,r,r,r")
_C8P1 = gl.constexpr("=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r")


def _normvar_asm() -> str:
    X = [f"${8 + i}" for i in range(8)]
    a = [
        "{ .reg .f32 p2, p3, p6, p7, a0, a1, a4, a5, b0, b1, r, t;",
        f"mul.rn.f32 p2, {X[2]}, {X[2]};",
        f"mul.rn.f32 p3, {X[3]}, {X[3]};",
        f"mul.rn.f32 p6, {X[6]}, {X[6]};",
        f"mul.rn.f32 p7, {X[7]}, {X[7]};",
        f"fma.rn.f32 a0, {X[0]}, {X[0]}, p2;",
        f"fma.rn.f32 a1, {X[1]}, {X[1]}, p3;",
        f"fma.rn.f32 a4, {X[4]}, {X[4]}, p6;",
        f"fma.rn.f32 a5, {X[5]}, {X[5]}, p7;",
        "add.rn.f32 b0, a0, a4;",
        "add.rn.f32 b1, a1, a5;",
        "add.rn.f32 r, b0, b1;",
    ]
    for m in (8, 4, 2, 1):
        a += [f"shfl.sync.bfly.b32 t, r, {m}, 0x1f, 0xffffffff;", "add.rn.f32 r, r, t;"]
    for i in range(8):
        a.append(f"mov.b32 ${i}, r;")
    a.append("}")
    return " ".join(a)


def _ssq4_asm() -> str:
    a = [
        "{ .reg .f32 p2, p3, a0, a1, r, t;",
        "mul.rn.f32 p2, $6, $6;",
        "mul.rn.f32 p3, $7, $7;",
        "fma.rn.f32 a0, $4, $4, p2;",
        "fma.rn.f32 a1, $5, $5, p3;",
        "add.rn.f32 r, a0, a1;",
    ]
    for m in (16, 8, 4, 2, 1):
        a += [f"shfl.sync.bfly.b32 t, r, {m}, 0x1f, 0xffffffff;", "add.rn.f32 r, r, t;"]
    a += ["mov.b32 $0, r;", "mov.b32 $1, r;", "mov.b32 $2, r;", "mov.b32 $3, r;", "}"]
    return " ".join(a)


def _rowstep_asm() -> str:
    """One token for whole state rows, with packed f32x2 math where the
    rounding is the same as the scalar ops."""
    S = [f"${16 + i}" for i in range(8)]
    DC = [f"${24 + i}" for i in range(8)]
    K = [f"${32 + i}" for i in range(8)]
    Q = [f"${40 + i}" for i in range(8)]
    V, BT = "$48", "$56"
    a = [
        "{ .reg .b64 S<4>, D<4>, K<4>, Q<4>, P<4>, X67, X23, A01, A45, B, VV, "
        "KV<4>, N<4>, QP<4>, C01, C45, E; .reg .f32 r, t, lo, hi, vn, o;"
    ]
    for j in range(4):
        a += [
            f"mov.b64 S{j}, {{{S[2 * j]}, {S[2 * j + 1]}}};",
            f"mov.b64 D{j}, {{{DC[2 * j]}, {DC[2 * j + 1]}}};",
            f"mov.b64 K{j}, {{{K[2 * j]}, {K[2 * j + 1]}}};",
            f"mov.b64 Q{j}, {{{Q[2 * j]}, {Q[2 * j + 1]}}};",
        ]
    for j in range(4):
        a.append(f"mul.rn.f32x2 P{j}, S{j}, D{j};")
    a += [
        "mul.rn.f32x2 X67, P3, K3;",
        "fma.rn.f32x2 A45, P2, K2, X67;",
        "mul.rn.f32x2 X23, P1, K1;",
        "fma.rn.f32x2 A01, P0, K0, X23;",
        "add.rn.f32x2 B, A01, A45;",
        "mov.b64 {lo, hi}, B;",
        "add.rn.f32 r, lo, hi;",
    ]
    for m in (8, 4, 2, 1):
        a += [f"shfl.sync.bfly.b32 t, r, {m}, 0x1f, 0xffffffff;", "add.rn.f32 r, r, t;"]
    a += [
        f"sub.rn.f32 vn, {V}, r;",
        f"mul.rn.f32 vn, vn, {BT};",
        "mov.b64 VV, {vn, vn};",
    ]
    for j in range(4):
        a += [
            f"mul.rn.f32x2 KV{j}, K{j}, VV;",
            f"fma.rn.f32x2 N{j}, S{j}, D{j}, KV{j};",
        ]
    # q dot: packed products, scalar adds (see the table above).
    a.append("{ .reg .f32 qp<8>, c0, c1, c4, c5, d0, d1;")
    for j in range(4):
        a.append(f"mul.rn.f32x2 QP{j}, N{j}, Q{j};")
    for j in range(4):
        a.append(f"mov.b64 {{qp{2 * j}, qp{2 * j + 1}}}, QP{j};")
    a += [
        "add.rn.f32 c0, qp0, qp2;",
        "add.rn.f32 c1, qp1, qp3;",
        "add.rn.f32 c4, qp4, qp6;",
        "add.rn.f32 c5, qp5, qp7;",
        "add.rn.f32 d0, c0, c4;",
        "add.rn.f32 d1, c1, c5;",
        "add.rn.f32 o, d0, d1; }",
    ]
    for m in (8, 4, 2, 1):
        a += [f"shfl.sync.bfly.b32 t, o, {m}, 0x1f, 0xffffffff;", "add.rn.f32 o, o, t;"]
    for j in range(4):
        a.append(f"mov.b64 {{${2 * j}, ${2 * j + 1}}}, N{j};")
    for i in range(8):
        a.append(f"mov.b32 ${8 + i}, o;")
    a.append("}")
    return " ".join(a)


_ASM_NORMVAR = gl.constexpr(_normvar_asm())
_ASM_SSQ4 = gl.constexpr(_ssq4_asm())
_ASM_ROWSTEP = gl.constexpr(_rowstep_asm())
_C_ROWSTEP = gl.constexpr(",".join(["=r"] * 16 + ["r"] * 48))


@gluon.jit
def _conv4(c0, c1, c2, x, w0, w1, w2, w3):
    return gl.inline_asm_elementwise(
        _ASM_CONV4,
        _C9,
        [c0, c1, c2, x, w0, w1, w2, w3],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _ssq4(x):
    """sum(x*x) over a [4]x[32] vector (l2 norm tree), replicated."""
    return gl.inline_asm_elementwise(
        _ASM_SSQ4, _C4P, [x], dtype=gl.float32, is_pure=False, pack=4
    )


@gluon.jit
def _normvar(x):
    """sum(x*x) over a row (norm tree + 16-lane butterfly), replicated."""
    return gl.inline_asm_elementwise(
        _ASM_NORMVAR, _C8P1, [x], dtype=gl.float32, is_pure=False, pack=8
    )


@gluon.jit
def _rowstep(s, dec, k, q, v, bt):
    """One token for whole rows: (S_new, o) with o replicated over the row."""
    return gl.inline_asm_elementwise(
        _ASM_ROWSTEP,
        _C_ROWSTEP,
        [s, dec, k, q, v, bt],
        dtype=(gl.float32, gl.float32),
        is_pure=False,
        pack=8,
    )


@gluon.jit
def _first(a, b):
    return a


@gluon.jit
def _sig(x):
    # tl.sigmoid as lowered by Triton: 1 / (exp(-x) + 1)
    return 1.0 / (gl.exp(-x) + 1.0)


@gluon.jit
def _conv_group(
    x_ptr,
    stride_x_tok,
    bos,
    cbase,
    w_ptr,
    ch,
    a0,
    m1,
    m2,
    m3,
    cv_smem,
    row0: gl.constexpr,
    T: gl.constexpr,
    W: gl.constexpr,
):
    w0 = gl.load(w_ptr + ch * W)
    w1 = gl.load(w_ptr + ch * W + 1)
    w2 = gl.load(w_ptr + ch * W + 2)
    w3 = gl.load(w_ptr + ch * W + 3)
    c0 = gl.load(cbase + a0)
    c1 = gl.load(cbase + a0 + 1)
    c2 = gl.load(cbase + a0 + 2)
    xp = x_ptr + bos * stride_x_tok + ch
    x0 = gl.load(xp)
    x1 = gl.load(xp + stride_x_tok, mask=m1 & (ch >= 0), other=0.0)
    x2 = gl.load(xp + 2 * stride_x_tok, mask=m2 & (ch >= 0), other=0.0)
    x3 = gl.load(xp + 3 * stride_x_tok, mask=m3 & (ch >= 0), other=0.0)
    f0 = c0.to(gl.float32)
    f1 = c1.to(gl.float32)
    f2 = c2.to(gl.float32)
    g0 = x0.to(gl.float32)
    g1 = x1.to(gl.float32)
    g2 = x2.to(gl.float32)
    g3 = x3.to(gl.float32)
    a = _conv4(f0, f1, f2, g0, w0, w1, w2, w3)
    cv_smem.index(row0 + 0).store((a / (gl.exp(-a) + 1.0)).to(gl.bfloat16))
    a = _conv4(f1, f2, g0, g1, w0, w1, w2, w3)
    cv_smem.index(row0 + 1).store((a / (gl.exp(-a) + 1.0)).to(gl.bfloat16))
    a = _conv4(f2, g0, g1, g2, w0, w1, w2, w3)
    cv_smem.index(row0 + 2).store((a / (gl.exp(-a) + 1.0)).to(gl.bfloat16))
    a = _conv4(g0, g1, g2, g3, w0, w1, w2, w3)
    cv_smem.index(row0 + 3).store((a / (gl.exp(-a) + 1.0)).to(gl.bfloat16))
    # New conv window [c1, c2, x0 .. x_{L-1}] (positions < 2 + L, as stock).
    gl.store(cbase, c1)
    gl.store(cbase + 1, c2)
    gl.store(cbase + 2, x0)
    gl.store(cbase + 3, x1, mask=m1 & (ch >= 0))
    gl.store(cbase + 4, x2, mask=m2 & (ch >= 0))
    gl.store(cbase + 5, x3, mask=m3 & (ch >= 0))


@gluon.jit
def _token_vectors(
    cv_smem,
    t: gl.constexpr,
    T: gl.constexpr,
    graw,
    bias,
    A,
    lower_bound,
    qn_smem,
    kn_smem,
    dc_smem,
    L4: gl.constexpr,
    X1: gl.constexpr,
    SCALE: gl.constexpr,
):
    # Sums of squares in the [4]x[32] l2 norm layout; element-wise math once
    # per element (X1).
    qrow = cv_smem.index(t)
    krow = cv_smem.index(T + t)
    q4 = qrow.load(L4).to(gl.float32)
    k4 = krow.load(L4).to(gl.float32)
    qs = gl.sqrt(gl.reduce(_ssq4(q4), 0, _first) + 1e-6)
    ks = gl.sqrt(gl.reduce(_ssq4(k4), 0, _first) + 1e-6)
    qn = qrow.load(X1).to(gl.float32) / qs
    kn = krow.load(X1).to(gl.float32) / ks
    qn = qn * SCALE
    gate = lower_bound * _sig(A * (graw.to(gl.float32) + bias))
    qn_smem.index(t).store(qn)
    kn_smem.index(t).store(kn)
    dc_smem.index(t).store(gl.exp(gate))


@gluon.jit
def _step(
    S,
    cv_smem,
    qn_smem,
    kn_smem,
    dc_smem,
    o_smem,
    t: gl.constexpr,
    T: gl.constexpr,
    c: gl.constexpr,
    RC: gl.constexpr,
    bt,
    slot,
    st_ptr,
    stride_st_slot,
    head_off,
    offs_fold,
    KF: gl.constexpr,
    VF: gl.constexpr,
    D: gl.constexpr,
):
    kn = kn_smem.index(t).load(KF)[None, :]
    qn = qn_smem.index(t).load(KF)[None, :]
    dc = dc_smem.index(t).load(KF)[None, :]
    v = cv_smem.index(2 * T + t).slice(c * RC, RC).load(VF).to(gl.float32)
    S, orep = _rowstep(S, dc, kn, qn, v[:, None], bt)
    o = gl.reduce(orep, 1, _first)
    o_smem.index(t).slice(c * RC, RC).store(o.to(gl.bfloat16))
    if slot > 0:
        gl.store(
            st_ptr + slot * stride_st_slot + head_off + c * (RC * D) + offs_fold,
            S.to(gl.bfloat16),
        )
    return S


@gluon.jit
def _norm_store(
    o_smem,
    t: gl.constexpr,
    gv,
    wv,
    eps,
    dmask,
    out_ptr,
    orow,
    kx,
    KF: gl.constexpr,
    X1: gl.constexpr,
    D: gl.constexpr,
    mask,
):
    # Variance with the [1,8]x[2,16] norm tree; element-wise output once per
    # element (X1).
    row = o_smem.index(t)
    x = row.load(KF).to(gl.float32)[None, :]
    xbar = gl.where(dmask[None, :], x, 0.0)
    var = gl.reduce(gl.reduce(_normvar(xbar), 1, _first), 0, _first) / D
    rstd = 1.0 / gl.sqrt(var + eps)
    x1 = row.load(X1).to(gl.float32)
    y = x1 * rstd
    y = y * wv
    y = y * _sig(gv.to(gl.float32))
    gl.store(out_ptr + orow + kx, y.to(gl.bfloat16), mask=mask & (kx < D))


@gluon.jit
def _kda_fused_spec_verify_kernel(
    x_ptr,
    stride_x_tok,
    cs_ptr,
    stride_cs_slot,
    w_ptr,
    g1_ptr,
    stride_g1_tok,
    beta_ptr,
    stride_beta_tok,
    g2_ptr,
    stride_g2_tok,
    nw_ptr,
    eps,
    A_log_ptr,
    dt_bias_ptr,
    lower_bound,
    st_ptr,
    stride_st_slot,
    idx_ptr,
    stride_idx,
    qsl_ptr,
    nacc_ptr,
    out_ptr,
    H: gl.constexpr,
    D: gl.constexpr,
    T: gl.constexpr,
    SL: gl.constexpr,
    W: gl.constexpr,
    NW: gl.constexpr,
    RC: gl.constexpr,
    SCALE: gl.constexpr,
):
    gl.static_assert(T == 4)
    HD: gl.constexpr = H * D
    NCH: gl.constexpr = D // RC
    FOLD: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [NW, 1], [1, 0])
    KF: gl.constexpr = gl.SliceLayout(0, FOLD)
    VF: gl.constexpr = gl.SliceLayout(1, FOLD)
    L4: gl.constexpr = gl.BlockedLayout([4], [32], [NW], [0])
    CPT: gl.constexpr = D // (32 * NW) if D >= 32 * NW else 1
    CV: gl.constexpr = gl.BlockedLayout([CPT], [32], [NW], [0])
    X1: gl.constexpr = CV
    SM2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SM1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    s_smem = gl.allocate_shared_memory(gl.bfloat16, [D, D], SM2)
    # conv outputs, row = group * T + t for the q, k and v groups
    cv_smem = gl.allocate_shared_memory(gl.bfloat16, [4 * T, D], SM1)
    qn_smem = gl.allocate_shared_memory(gl.float32, [T, D], SM1)
    kn_smem = gl.allocate_shared_memory(gl.float32, [T, D], SM1)
    dc_smem = gl.allocate_shared_memory(gl.float32, [T, D], SM1)
    o_smem = gl.allocate_shared_memory(gl.bfloat16, [T, D], SM1)

    pid = gl.program_id(0)
    n = pid // H
    h = pid % H
    bos = gl.load(qsl_ptr + n).to(gl.int64)
    eos = gl.load(qsl_ptr + n + 1).to(gl.int64)
    L = eos - bos
    ip = idx_ptr + n * stride_idx
    cslot = gl.load(ip).to(gl.int64)
    # Padded requests (no tokens or state index 0) are skipped, as by stock.
    if (L > 0) & (cslot > 0):
        a0 = gl.load(nacc_ptr + n).to(gl.int64) - 1
        s_init = gl.load(ip + a0).to(gl.int64)
        rf = gl.arange(0, RC, layout=VF)
        kf = gl.arange(0, D, layout=KF)
        offs_fold = rf[:, None] * D + kf[None, :]
        head_off = h * (D * D)
        # 1. initial state tile -> shared memory, one cp.async group per chunk
        if s_init > 0:
            src = st_ptr + s_init * stride_st_slot + head_off
            for c in gl.static_range(NCH):
                gac.async_copy_global_to_shared(
                    s_smem.slice(c * RC, RC, dim=0), src + c * (RC * D) + offs_fold
                )
                gac.commit_group()
        m1 = L > 1
        m2 = L > 2
        m3 = L > 3
        kmask = kf < D
        # 2. the small per-request inputs, all issued now
        slot0 = gl.load(ip).to(gl.int64)
        slot1 = gl.load(ip + 1).to(gl.int64)
        slot2 = gl.load(ip + 2).to(gl.int64)
        slot3 = gl.load(ip + 3).to(gl.int64)
        kx = gl.arange(0, D, layout=X1)
        xmask = kx < D
        gp = g1_ptr + bos * stride_g1_tok + h * D + kx
        gr0 = gl.load(gp)
        gr1 = gl.load(gp + stride_g1_tok, mask=xmask & m1, other=0.0)
        gr2 = gl.load(gp + 2 * stride_g1_tok, mask=xmask & m2, other=0.0)
        gr3 = gl.load(gp + 3 * stride_g1_tok, mask=xmask & m3, other=0.0)
        bp = beta_ptr + bos * stride_beta_tok + h
        b0 = gl.load(bp).to(gl.float32)
        b1 = gl.load(bp + stride_beta_tok, mask=m1, other=0.0).to(gl.float32)
        b2 = gl.load(bp + 2 * stride_beta_tok, mask=m2, other=0.0).to(gl.float32)
        b3 = gl.load(bp + 3 * stride_beta_tok, mask=m3, other=0.0).to(gl.float32)
        op = g2_ptr + bos * stride_g2_tok + h * D + kx
        o0 = gl.load(op)
        o1 = gl.load(op + stride_g2_tok, mask=xmask & m1, other=0.0)
        o2 = gl.load(op + 2 * stride_g2_tok, mask=xmask & m2, other=0.0)
        o3 = gl.load(op + 3 * stride_g2_tok, mask=xmask & m3, other=0.0)
        bias = gl.load(dt_bias_ptr + h * D + kx)
        A = gl.exp(gl.load(A_log_ptr + h))
        wv = gl.load(nw_ptr + kx, mask=xmask).to(gl.float32)
        # 3. conv1d update of the head's q, k and v channels
        cc = gl.arange(0, D, layout=CV)
        for gi in gl.static_range(3):
            ch = gi * HD + h * D + cc
            _conv_group(
                x_ptr,
                stride_x_tok,
                bos,
                cs_ptr + cslot * stride_cs_slot + ch * SL,
                w_ptr,
                ch,
                a0,
                m1,
                m2,
                m3,
                cv_smem,
                gi * T,
                T,
                W,
            )
        gl.barrier()
        # 4. per-token vectors (all 4 rows; rows t >= L are never used)
        _token_vectors(
            cv_smem, 0, T, gr0, bias, A, lower_bound, qn_smem, kn_smem, dc_smem,
            L4, X1, SCALE,
        )  # fmt: skip
        _token_vectors(
            cv_smem, 1, T, gr1, bias, A, lower_bound, qn_smem, kn_smem, dc_smem,
            L4, X1, SCALE,
        )  # fmt: skip
        _token_vectors(
            cv_smem, 2, T, gr2, bias, A, lower_bound, qn_smem, kn_smem, dc_smem,
            L4, X1, SCALE,
        )  # fmt: skip
        _token_vectors(
            cv_smem, 3, T, gr3, bias, A, lower_bound, qn_smem, kn_smem, dc_smem,
            L4, X1, SCALE,
        )  # fmt: skip
        bt0 = _sig(b0)
        bt1 = _sig(b1)
        bt2 = _sig(b2)
        bt3 = _sig(b3)
        gl.barrier()
        # 5. recurrence, chunk by chunk
        if s_init > 0:
            for c in gl.static_range(NCH):
                gac.wait_group(NCH - 1 - c)
                gl.barrier()
                S = s_smem.slice(c * RC, RC, dim=0).load(FOLD).to(gl.float32)
                S = _step(
                    S, cv_smem, qn_smem, kn_smem, dc_smem, o_smem, 0, T, c, RC,
                    bt0, slot0, st_ptr, stride_st_slot, head_off, offs_fold,
                    KF, VF, D,
                )  # fmt: skip
                if m1:
                    S = _step(
                        S, cv_smem, qn_smem, kn_smem, dc_smem, o_smem, 1, T, c,
                        RC, bt1, slot1, st_ptr, stride_st_slot, head_off,
                        offs_fold, KF, VF, D,
                    )  # fmt: skip
                if m2:
                    S = _step(
                        S, cv_smem, qn_smem, kn_smem, dc_smem, o_smem, 2, T, c,
                        RC, bt2, slot2, st_ptr, stride_st_slot, head_off,
                        offs_fold, KF, VF, D,
                    )  # fmt: skip
                if m3:
                    S = _step(
                        S, cv_smem, qn_smem, kn_smem, dc_smem, o_smem, 3, T, c,
                        RC, bt3, slot3, st_ptr, stride_st_slot, head_off,
                        offs_fold, KF, VF, D,
                    )  # fmt: skip
        else:
            # As stock: token 0 output := 0, later tokens keep the buffer.
            orow = (bos * H + h) * D
            o_smem.index(0).store(gl.zeros([D], gl.bfloat16, layout=KF))
            o_smem.index(1).store(
                gl.load(out_ptr + orow + HD + kf, mask=kmask & m1, other=0.0)
            )
            o_smem.index(2).store(
                gl.load(out_ptr + orow + 2 * HD + kf, mask=kmask & m2, other=0.0)
            )
            o_smem.index(3).store(
                gl.load(out_ptr + orow + 3 * HD + kf, mask=kmask & m3, other=0.0)
            )
        gl.barrier()
        # 6. gated RMSNorm in the stock layer_norm_gated_fwd_kernel order
        orow = (bos * H + h) * D
        _norm_store(o_smem, 0, o0, wv, eps, kmask, out_ptr, orow, kx, KF, X1, D, L > 0)
        _norm_store(
            o_smem, 1, o1, wv, eps, kmask, out_ptr, orow + HD, kx, KF, X1, D, m1
        )
        _norm_store(
            o_smem, 2, o2, wv, eps, kmask, out_ptr, orow + 2 * HD, kx, KF, X1, D, m2
        )
        _norm_store(
            o_smem, 3, o3, wv, eps, kmask, out_ptr, orow + 3 * HD, kx, KF, X1, D, m3
        )
