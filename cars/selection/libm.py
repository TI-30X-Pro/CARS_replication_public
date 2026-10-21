"""Vectorised float32 ``atan2`` with stable cross-platform rounding.

The original ran NumPy 1.21 on Linux, where np.arctan2 on float32 calls glibc's atan2f. Other
libms (macOS, or SVML in NumPy 2 on AVX-512 machines) round differently in about one case in
seven, by one ulp. That is enough to move a split threshold of the selection classifier, so
the features use this port. Every operation is float32 without fused multiply-add, as glibc
is built for generic x86-64. Within one ulp of the correctly rounded result.
"""
import numpy as np

f32 = np.float32
_ATANHI = np.array([4.6364760399e-01, 7.8539812565e-01, 9.8279368877e-01, 1.5707962513e+00], dtype=f32)
_ATANLO = np.array([5.0121582440e-09, 3.7748947079e-08, 3.4473217170e-08, 7.5497894159e-08], dtype=f32)
_AT = [f32(v) for v in (3.3333334327e-01, -2.0000000298e-01, 1.4285714924e-01, -1.1111110449e-01,
                        9.0908870101e-02, -7.6918758452e-02, 6.6610731184e-02, -5.8335702866e-02,
                        4.9768779427e-02, -3.6531571299e-02, 1.6285819933e-02)]
_PI, _PI_O_2, _PI_LO = f32(3.1415927410e+00), f32(1.5707963705e+00), f32(-8.7422776573e-08)


def _bits(x):
    return np.asarray(x, dtype=f32).view(np.int32)


def atanf(x):
    x = np.asarray(x, dtype=f32)
    hx = _bits(x)
    ix = hx & 0x7fffffff
    ax = np.abs(x)
    one = f32(1)
    # argument reduction by |x|: < 7/16, < 11/16, < 19/16, < 39/16, larger
    m0 = (ix >= 0x3ee00000) & (ix < 0x3f300000)
    m1 = (ix >= 0x3f300000) & (ix < 0x3f980000)
    m2 = (ix >= 0x3f980000) & (ix < 0x401c0000)
    m3 = ix >= 0x401c0000
    with np.errstate(all="ignore"):
        r = np.where(m0, (f32(2) * ax - one) / (f32(2) + ax), x)
        r = np.where(m1, (ax - one) / (ax + one), r)
        r = np.where(m2, (ax - f32(1.5)) / (one + f32(1.5) * ax), r)
        r = np.where(m3, -one / ax, r)
    k = np.select([m0, m1, m2, m3], [0, 1, 2, 3], -1)
    z = r * r
    w = z * z
    s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
    s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
    kk = np.clip(k, 0, 3)
    big = _ATANHI[kk] - ((r * (s1 + s2) - _ATANLO[kk]) - r)
    out = np.where(k < 0, r - r * (s1 + s2), np.where(hx < 0, -big, big))
    out = np.where(ix >= 0x4c000000, np.where(hx > 0, _ATANHI[3] + _ATANLO[3], -_ATANHI[3] - _ATANLO[3]), out)
    out = np.where(ix < 0x31000000, x, out)
    return out.astype(f32)


def atan2f(y, x):
    y, x = np.broadcast_arrays(np.asarray(y, dtype=f32), np.asarray(x, dtype=f32))
    hx, hy = _bits(x), _bits(y)
    ix, iy = hx & 0x7fffffff, hy & 0x7fffffff
    m = ((hy >> 31) & 1) | ((hx >> 30) & 2)          # 2 * sign(x) + sign(y)
    with np.errstate(all="ignore"):
        z = atanf(np.abs(y / x))
    k = (iy - ix) >> 23
    z = np.where(k > 60, _PI_O_2 + f32(0.5) * _PI_LO, z)
    z = np.where((hx < 0) & (k < -60), f32(0), z)
    out = np.select([m == 0, m == 1, m == 2], [z, -z, _PI - (z - _PI_LO)], (z - _PI_LO) - _PI)
    out = np.where(hx == 0x3f800000, atanf(y), out)                                   # x = 1
    out = np.where(iy == 0, np.select([m <= 1, m == 2], [y, _PI], -_PI), out)           # y = 0
    out = np.where((ix == 0) & (iy != 0), np.where(hy < 0, -_PI_O_2, _PI_O_2), out)     # x = 0
    out = np.where((ix > 0x7f800000) | (iy > 0x7f800000), x + y, out)                  # NaN
    return out.astype(f32)
