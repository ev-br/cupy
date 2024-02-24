import operator
from math import prod

from numpy.core.multiarray import normalize_axis_index

import cupy
from cupy._core._scalar import get_typename

from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import _get_dtype, _as_float_array, _get_module_func

from cupyx.scipy.interpolate._bspline import (
    _get_module_func, INTERVAL_MODULE, D_BOOR_MODULE, BSpline)


#################################
#  Interpolating spline helpers #
#################################

def _not_a_knot(x, k):
    """Given data x, construct the knot vector w/ not-a-knot BC.
    cf de Boor, XIII(12)."""
    x = cupy.asarray(x)
    if k % 2 != 1:
        raise ValueError("Odd degree for now only. Got %s." % k)

    m = (k - 1) // 2
    t = x[m+1:-m-1]
    t = cupy.r_[(x[0],)*(k+1), t, (x[-1],)*(k+1)]
    return t


def _augknt(x, k):
    """Construct a knot vector appropriate for the order-k interpolation."""
    return cupy.r_[(x[0],)*k, x, (x[-1],)*k]


def _periodic_knots(x, k):
    """Returns vector of nodes on a circle."""
    xc = cupy.copy(x)
    n = len(xc)
    if k % 2 == 0:
        dx = cupy.diff(xc)
        xc[1: -1] -= dx[:-1] / 2
    dx = cupy.diff(xc)
    t = cupy.zeros(n + 2 * k)
    t[k: -k] = xc
    for i in range(0, k):
        # filling first `k` elements in descending order
        t[k - i - 1] = t[k - i] - dx[-(i % (n - 1)) - 1]
        # filling last `k` elements in ascending order
        t[-k + i] = t[-k + i - 1] + dx[i % (n - 1)]
    return t


def _convert_string_aliases(deriv, target_shape):
    if isinstance(deriv, str):
        if deriv == "clamped":
            deriv = [(1, cupy.zeros(target_shape))]
        elif deriv == "natural":
            deriv = [(2, cupy.zeros(target_shape))]
        else:
            raise ValueError("Unknown boundary condition : %s" % deriv)
    return deriv


def _process_deriv_spec(deriv):
    if deriv is not None:
        try:
            ords, vals = zip(*deriv)
        except TypeError as e:
            msg = ("Derivatives, `bc_type`, should be specified as a pair of "
                   "iterables of pairs of (order, value).")
            raise ValueError(msg) from e
    else:
        ords, vals = [], []
    return cupy.atleast_1d(ords, vals)


def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0,
                       check_finite=True):
    """Compute the (coefficients of) interpolating B-spline.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n, ...)
        Ordinates.
    k : int, optional
        B-spline degree. Default is cubic, ``k = 3``.
    t : array_like, shape (nt + k + 1,), optional.
        Knots.
        The number of knots needs to agree with the number of data points and
        the number of derivatives at the edges. Specifically, ``nt - n`` must
        equal ``len(deriv_l) + len(deriv_r)``.
    bc_type : 2-tuple or None
        Boundary conditions.
        Default is None, which means choosing the boundary conditions
        automatically. Otherwise, it must be a length-two tuple where the first
        element (``deriv_l``) sets the boundary conditions at ``x[0]`` and
        the second element (``deriv_r``) sets the boundary conditions at
        ``x[-1]``. Each of these must be an iterable of pairs
        ``(order, value)`` which gives the values of derivatives of specified
        orders at the given edge of the interpolation interval.
        Alternatively, the following string aliases are recognized:

        * ``"clamped"``: The first derivatives at the ends are zero. This is
           equivalent to ``bc_type=([(1, 0.0)], [(1, 0.0)])``.
        * ``"natural"``: The second derivatives at ends are zero. This is
          equivalent to ``bc_type=([(2, 0.0)], [(2, 0.0)])``.
        * ``"not-a-knot"`` (default): The first and second segments are the
          same polynomial. This is equivalent to having ``bc_type=None``.
        * ``"periodic"``: The values and the first ``k-1`` derivatives at the
          ends are equivalent.

    axis : int, optional
        Interpolation axis. Default is 0.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.

    Returns
    -------
    b : a BSpline object of the degree ``k`` and with knots ``t``.

    """
    # convert string aliases for the boundary conditions
    if bc_type is None or bc_type == 'not-a-knot' or bc_type == 'periodic':
        deriv_l, deriv_r = None, None
    elif isinstance(bc_type, str):
        deriv_l, deriv_r = bc_type, bc_type
    else:
        try:
            deriv_l, deriv_r = bc_type
        except TypeError as e:
            raise ValueError("Unknown boundary condition: %s" % bc_type) from e

    y = cupy.asarray(y)

    axis = normalize_axis_index(axis, y.ndim)

    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)

    y = cupy.moveaxis(y, axis, 0)    # now internally interp axis is zero

    # sanity check the input
    if bc_type == 'periodic' and not cupy.allclose(y[0], y[-1], atol=1e-15):
        raise ValueError("First and last points does not match while "
                         "periodic case expected")
    if x.size != y.shape[0]:
        raise ValueError('Shapes of x {} and y {} are incompatible'
                         .format(x.shape, y.shape))
    if (x[1:] == x[:-1]).any():
        raise ValueError("Expect x to not have duplicates")
    if x.ndim != 1 or (x[1:] < x[:-1]).any():
        raise ValueError("Expect x to be a 1D strictly increasing sequence.")

    # special-case k=0 right away
    if k == 0:
        if any(_ is not None for _ in (t, deriv_l, deriv_r)):
            raise ValueError("Too much info for k=0: t and bc_type can only "
                             "be None.")
        t = cupy.r_[x, x[-1]]
        c = cupy.asarray(y)
        c = cupy.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)

    # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
    if k == 1 and t is None:
        if not (deriv_l is None and deriv_r is None):
            raise ValueError(
                "Too much info for k=1: bc_type can only be None.")
        t = cupy.r_[x[0], x, x[-1]]
        c = cupy.asarray(y)
        c = cupy.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)

    k = operator.index(k)

    if bc_type == 'periodic' and t is not None:
        raise NotImplementedError("For periodic case t is constructed "
                                  "automatically and can not be passed "
                                  "manually")

    # come up with a sensible knot vector, if needed
    if t is None:
        if deriv_l is None and deriv_r is None:
            if bc_type == 'periodic':
                t = _periodic_knots(x, k)
            elif k == 2:
                # OK, it's a bit ad hoc: Greville sites + omit
                # 2nd and 2nd-to-last points, a la not-a-knot
                t = (x[1:] + x[:-1]) / 2.
                t = cupy.r_[(x[0],)*(k+1),
                            t[1:-1],
                            (x[-1],)*(k+1)]
            else:
                t = _not_a_knot(x, k)
        else:
            t = _augknt(x, k)

    t = _as_float_array(t, check_finite)

    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t.ndim != 1 or (t[1:] < t[:-1]).any():
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    if t.size < x.size + k + 1:
        raise ValueError('Got %d knots, need at least %d.' %
                         (t.size, x.size + k + 1))
    if (x[0] < t[k]) or (x[-1] > t[-k]):
        raise ValueError('Out of bounds w/ x = %s.' % x)

    if bc_type == 'periodic':
        return _make_periodic_spline(x, y, t, k, axis)

    # Here : deriv_l, r = [(nu, value), ...]
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]

    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]

    # have `n` conditions for `nt` coefficients; need nt-n derivatives
    n = x.size
    nt = t.size - k - 1

    if nt - n != nleft + nright:
        raise ValueError("The number of derivatives at boundaries does not "
                         "match: expected %s, got %s + %s" %
                         (nt-n, nleft, nright))

    # bail out if the `y` array is zero-sized
    if y.size == 0:
        c = cupy.zeros((nt,) + y.shape[1:], dtype=float)
        return BSpline.construct_fast(t, c, k, axis=axis)

    # Consruct the colocation matrix of b-splines + boundary conditions.
    # The coefficients of the interpolating B-spline function are the solution
    # of the linear system `A @ c = rhs` where `A` is the colocation matrix
    # (i.e., each row of A corresponds to a data point in the `x` array and
    #  contains b-splines which are non-zero at this value of x)
    # Each boundary condition is a fixed value of a certain derivative
    # at the edge, so each derivative adds a row to `A`.
    # The `rhs` is the array of data values, `y`, plus derivatives from
    # boundary conditions, if any.
    # The colocation matrix is banded (has at most k+1 diagonals). Since LAPACK
    # linear algebra (?gbsv) is not available, we store it as a CSR array

    # 1. Construct the colocation matrix itself.
    matr = BSpline.design_matrix(x, t, k)

    # 2. Boundary conditions: need to augment the design matrix with additional
    # rows, one row per derivative at the left and right edges.
    # The left-side boundary conditions go to the first rows of the matrix
    # and the right-side boundary conditions go to the last rows.
    # Will need a python loop for each derivative because in general they
    # can be of any order, `m`.
    # To compute the derivatives, will invoke the de Boor D kernel.
    if nleft > 0 or nright > 0:
        # Prepare the I/O arrays for the kernels. We only need the non-zero
        # b-splines at x[0] and x[-1], but the kernel wants more arrays which
        # we allocate and ignore (mode != 1)
        temp = cupy.zeros((nt, ), dtype=float)
        num_c = 1
        dummy_c = cupy.empty((nt, num_c), dtype=float)
        out = cupy.empty((1, 1), dtype=dummy_c.dtype)

        d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', dummy_c)

        # find the intervals for x[0] and x[-1]
        intervals_bc = cupy.empty(2, dtype=cupy.int64)
        interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
        interval_kernel((1,), (2,),
                        (t, cupy.r_[x[0], x[-1]], intervals_bc, k, nt,
                         False, 2))

    # 3. B.C.s at x[0]
    if nleft > 0:
        x0 = cupy.array([x[0]], dtype=x.dtype)
        rows = cupy.zeros((nleft, nt), dtype=float)

        for j, m in enumerate(deriv_l_ords):
            # place the derivatives of the order m at x[0] into `temp`
            d_boor_kernel((1,), (1,),
                          (t, dummy_c, k, int(m), x0, intervals_bc,
                           out,     # ignore (mode !=1),
                           temp,    # non-zero b-splines
                           num_c,   # the 2nd dimension of `dummy_c`. Ignore.
                           0,       # mode != 1 => do not touch dummy_c array
                           1))      # the length of the `x0` array
            left = intervals_bc[0]
            rows[j, left-k:left+1] = temp[:k+1]

        matr = sparse.vstack([sparse.csr_matrix(rows),   # A[:nleft, :]
                              matr])

    # 4. Repeat for B.C.s at x[-1]
    if nright > 0:
        intervals_bc[0] = intervals_bc[-1]   # use the intervals for x[-1]
        x0 = cupy.array([x[-1]], dtype=x.dtype)
        rows = cupy.zeros((nright, nt), dtype=float)

        for j, m in enumerate(deriv_r_ords):
            # place the derivatives of the order m at x[0] into `temp`
            d_boor_kernel((1,), (1,),
                          (t, dummy_c, k, int(m), x0, intervals_bc,
                           out,     # ignore (mode !=1),
                           temp,    # non-zero b-splines
                           num_c,   # the 2nd dimension of `dummy_c`. Ignore.
                           0,       # mode != 1 => do not touch dummy_c array
                           1))      # the length of the `x0` array
            left = intervals_bc[0]
            rows[j, left-k:left+1] = temp[:k+1]

        matr = sparse.vstack([matr,
                              sparse.csr_matrix(rows)])  # A[nleft+len(x):, :]

    # 5. Prepare the RHS: `y` values to interpolate (+ derivatives, if any)
    extradim = prod(y.shape[1:])
    rhs = cupy.empty((nt, extradim), dtype=y.dtype)
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)

    # 6. Finally, solve the linear system for the coefficients.
    if cupy.issubdtype(rhs.dtype, cupy.complexfloating):
        # avoid upcasting the l.h.s. to complex (that doubles the memory)
        coef = (spsolve(matr, rhs.real) +
                spsolve(matr, rhs.imag) * 1.j)
    else:
        coef = spsolve(matr, rhs)
    coef = cupy.ascontiguousarray(coef.reshape((nt,) + y.shape[1:]))
    return BSpline(t, coef, k)


def _make_interp_spline_full_matrix(x, y, k, t, bc_type):
    """ Construct the interpolating spline spl(x) = y with *full* linalg.

        Only useful for testing, do not call directly!
        This version is O(N**2) in memory and O(N**3) in flop count.
    """
    # convert string aliases for the boundary conditions
    if bc_type is None or bc_type == 'not-a-knot':
        deriv_l, deriv_r = None, None
    elif isinstance(bc_type, str):
        deriv_l, deriv_r = bc_type, bc_type
    else:
        try:
            deriv_l, deriv_r = bc_type
        except TypeError as e:
            raise ValueError("Unknown boundary condition: %s" % bc_type) from e

    # Here : deriv_l, r = [(nu, value), ...]
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]

    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]

    # have `n` conditions for `nt` coefficients; need nt-n derivatives
    n = x.size
    nt = t.size - k - 1
    # Here : deriv_l, r = [(nu, value), ...]
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]

    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]

    # have `n` conditions for `nt` coefficients; need nt-n derivatives
    n = x.size
    nt = t.size - k - 1
    assert nt - n == nleft + nright

    # Consruct the colocation matrix of b-splines + boundary conditions.
    # The coefficients of the interpolating B-spline function are the solution
    # of the linear system `A @ c = rhs` where `A` is the colocation matrix
    # (i.e., each row of A corresponds to a data point in the `x` array and
    #  contains b-splines which are non-zero at this value of x)
    # Each boundary condition is a fixed value of a certain derivative
    # at the edge, so each derivative adds a row to `A`.
    # The `rhs` is the array of data values, `y`, plus derivatives from
    # boundary conditions, if any.

    # 1. Compute intervals for each value
    intervals = cupy.empty_like(x, dtype=cupy.int64)
    interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
    interval_kernel(((x.shape[0] + 128 - 1) // 128,), (128,),
                    (t, x, intervals, k, nt, False, x.shape[0]))

    # 2. Compute non-zero b-spline basis elements for each value in `x`
    # The way de_Boor_D kernel is written, it wants `c` and `out` arrays
    # which we do not use (but need to provide to the kernel), and the
    # `temp` array contains non-zero b-spline basis elements, which we do want.
    dummy_c = cupy.empty((nt, 1), dtype=float)
    out = cupy.empty(
        (len(x), prod(dummy_c.shape[1:])), dtype=dummy_c.dtype)

    num_c = prod(dummy_c.shape[1:])
    temp = cupy.empty(x.shape[0] * (2 * k + 1))
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', dummy_c)
    d_boor_kernel(((x.shape[0] + 128 - 1) // 128,), (128,),
                  (t, dummy_c, k, 0, x, intervals, out, temp, num_c, 0,
                   x.shape[0]))

    # 3. Construct the colocation matrix.
    # For each value in `x`, the `temp` array contains 2k+1 entries : first
    # k+1 elements are b-splines, followed by k entries used for work storage
    # which we ignore.
    # XXX: full matrices! Can / should use banded linear algebra instead.
    A = cupy.zeros((nt, nt), dtype=float)
    offset = nleft
    for j in range(len(x)):
        left = intervals[j]
        A[j + offset, left-k:left+1] = temp[j*(2*k+1):j*(2*k+1)+k+1]

    # 4. Handle boundary conditions: The colocation matrix is augmented with
    # additional rows, one row per derivative at the left and right edges.
    # We need a python loop for each derivative because in general they can be
    # of any order, `m`.
    # The left-side boundary conditions go to the first rows of the matrix
    # and the right-side boundary conditions go to the last rows.
    intervals_bc = cupy.empty(1, dtype=cupy.int64)
    if nleft > 0:
        intervals_bc[0] = intervals[0]
        x0 = cupy.array([x[0]], dtype=x.dtype)
        for j, m in enumerate(deriv_l_ords):
            # place the derivatives of the order m at x[0] into `temp`
            d_boor_kernel((1,), (1,),
                          (t, dummy_c, k, int(m), x0, intervals_bc, out, temp,
                           num_c, 0, 1))
            left = intervals_bc[0]
            A[j, left-k:left+1] = temp[:k+1]

    # repeat for the b.c. at the right edge.
    if nright > 0:
        intervals_bc[0] = intervals[-1]
        x0 = cupy.array([x[-1]], dtype=x.dtype)
        for j, m in enumerate(deriv_r_ords):
            # place the derivatives of the order m at x[0] into `temp`
            d_boor_kernel((1,), (1,),
                          (t, dummy_c, k, int(m), x0, intervals_bc, out, temp,
                           num_c, 0, 1))
            left = intervals_bc[0]
            row = nleft + len(x) + j
            A[row, left-k:left+1] = temp[:k+1]

    # 5. Prepare the RHS: `y` values to interpolate (+ derivatives, if any)
    extradim = prod(y.shape[1:])
    rhs = cupy.empty((nt, extradim), dtype=y.dtype)
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)

    # 6. Finally, solve for the coefficients.
    from cupy.linalg import solve
    coef = solve(A, rhs)

    coef = cupy.ascontiguousarray(coef.reshape((nt,) + y.shape[1:]))
    return BSpline(t, coef, k)


def _make_periodic_spline(x, y, t, k, axis):
    n = x.size

    # 1. Construct the colocation matrix.
    matr = BSpline.design_matrix(x, t, k)

    # 2. Boundary conditions: need to augment the design matrix with additional
    # rows, one row per derivative at the left and right edges.
    # The k-1 boundary conditions go to the first rows of the matrix
    # To compute the derivatives, will invoke the de Boor D kernel.

    # Prepare the I/O arrays for the kernels. We only need the non-zero
    # b-splines at x[0] and x[-1], but the kernel wants more arrays which
    # we allocate and ignore (mode != 1)
    temp = cupy.zeros(2*(2*k+1), dtype=float)
    num_c = 1
    dummy_c = cupy.empty((t.size - k - 1, num_c), dtype=float)
    out = cupy.empty((2, 1), dtype=dummy_c.dtype)

    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', dummy_c)

    # find the intervals for x[0] and x[-1]
    x0 = cupy.r_[x[0], x[-1]]
    intervals_bc = cupy.array([k, n + k - 1], dtype=cupy.int64)   # match scipy

    # 3. B.C.s
    rows = cupy.zeros((k-1, n + k - 1), dtype=float)

    for m in range(k-1):
        # place the derivatives of the order m at x[0] into `temp`
        d_boor_kernel((1,), (2,),
                      (t, dummy_c, k, m+1, x0, intervals_bc,
                       out,     # ignore (mode !=1),
                       temp,    # non-zero b-splines
                       num_c,   # the 2nd dimension of `dummy_c`. Ignore.
                       0,       # mode != 1 => do not touch dummy_c array
                       2))      # the length of the `x0` array
        rows[m, :k+1] = temp[:k+1]
        rows[m, -k:] -= temp[2*k + 1:(2*k + 1) + k+1][:-1]

    matr_csr = sparse.vstack([sparse.csr_matrix(rows),   # A[:nleft, :]
                              matr])

    # r.h.s.
    extradim = prod(y.shape[1:])
    rhs = cupy.empty((n + k - 1, extradim), dtype=float)
    rhs[:(k - 1), :] = 0
    rhs[(k - 1):, :] = (y.reshape(n, 0) if y.size == 0 else
                        y.reshape((-1, extradim)))

    # solve for the coefficients
    coef = spsolve(matr_csr, rhs)
    coef = cupy.ascontiguousarray(coef.reshape((n + k - 1,) + y.shape[1:]))
    return BSpline.construct_fast(t, coef, k,
                                  extrapolate='periodic', axis=axis)


# ### LSQ spline helpers

def _lsq_solve_qr(x, y, t, k, w):
    """Solve for the LSQ spline coeffs given x, y and knots.
    `y` is always 2D: for 1D data, the shape is ``(m, 1)``.
    `w` is always 1D: one weight value per `x` value.
    """

    """

    # prepare the l.h.s.
    # A, offset, nc = _bspl._data_matrix(x, t, k, w)
    a_csr = BSpline.design_matrix(x, t, k)
    a_w = (a_csr * w[:, None]).tocsr()
    A = a_w.data.reshape((m, k+1))
    offset = a_w.indices[::(k+1)]
    nc = t.shape[0] - k - 1

    # prepare the r.h.s.
    assert y.ndim == 2
    y_w = y * w[:, None]


    c = cupy.ones(1)

#    num_c = int(np.prod(c.shape[1:]))
#    temp = cupy.empty(xp.shape[0] * (2 * k + 1))
    """

    qr_kernel = _get_module_func(QR_MODULE, 'dlartg', cupy.ones(1))

    f = 0.5
    g = 0.5
    r = cupy.empty(1)
    cs = cupy.empty(1)
    sn = cupy.empty(1)

    qr_kernel((1,), (1,),
                  (f, g, cs, sn, r))

    breakpoint()

#    R = PackedMatrix(A, offset, nc)
#    _bspl._qr_reduce(R, y_w)         # modifies arguments in-place
#    c = _bspl._fpback(R, y_w)

    return R, y_w, c



QR_KERNEL = r'''
# include <cuda/std/cmath>

/* 
 * Compute the parameters of the Givens transformation: LAPACK's dlartg replacement.
 *
 * Naive computation, following
 * https://github.com/scipy/scipy/blob/v1.12.0/scipy/interpolate/fitpack/fpgivs.f
 */
template<typename T>
__global__ void dlartg(const T f, const T g, T *cs, T *sn, T *r) {

    T piv = cuda::std::abs(f);

    if (piv >= g) {
        T sq = g / f;
        *r = piv * cuda::std::sqrt(1.0 + sq*sq);
    } else {
        T sq = f / g;
        *r = g * cuda::std::sqrt(1.0 + sq*sq);
    }

    *cs = g / *r;
    *sn = f / *r;
};

'''

TYPES = ['double']

QR_MODULE = cupy.RawModule(
    code=QR_KERNEL, options=('-std=c++14',),
    name_expressions=[f'dlartg<{type_name}>' for type_name in TYPES])





def make_lsq_spline(x, y, t, k=3, w=None, axis=0, check_finite=True, *, method="qr"):
    r"""Construct a BSpline via an LSQ (Least SQuared) fit.

    The result is a linear combination

        .. math::
            S(x) = \sum_j c_j B_j(x; t)

    of the B-spline basis elements, :math:`B_j(x; t)`, which minimizes

        .. math::
            \sum_{j} \left( w_j \times (S(x_j) - y_j) \right)^2

    Parameters
    ----------
    x : array_like, shape (m,)
        Abscissas.
    y : array_like, shape (m, ...)
        Ordinates.
    t : array_like, shape (n + k + 1,).
        Knots.
        Knots and data points must satisfy Schoenberg-Whitney conditions.
    k : int, optional
        B-spline degree. Default is cubic, ``k = 3``.
    w : array_like, shape (m,), optional
        Weights for spline fitting. Must be positive. If ``None``,
        then weights are all equal.
        Default is ``None``.
    axis : int, optional
        Interpolation axis. Default is zero.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.

    Returns
    -------
    b : a BSpline object of the degree ``k`` with knots ``t``.

    See Also
    --------
    scipy.interpolate.make_lsq_spline
    BSpline : base class representing the B-spline objects
    make_interp_spline : a similar factory function for interpolating splines


    Notes
    -----
    The number of data points must be larger than the spline degree ``k``.
    Knots ``t`` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    """
    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    t = _as_float_array(t, check_finite)
    if w is not None:
        w = _as_float_array(w, check_finite)
    else:
        w = cupy.ones_like(x)
    k = operator.index(k)
    axis = normalize_axis_index(axis, y.ndim)

    y = cupy.moveaxis(y, axis, 0)    # now internally interp axis is zero

    if x.ndim != 1 or any(x[1:] - x[:-1] <= 0):
        raise ValueError("Expect x to be a 1-D sorted array_like.")
    if x.ndim != 1:
        raise ValueError("Expect x to be a 1-D sequence.")
    if x.shape[0] < k+1:
        raise ValueError("Need more x points.")
    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t.ndim != 1 or any(t[1:] - t[:-1] < 0):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
        raise ValueError("Expect t to be a 1D strictly increasing sequence.")
    if x.size != y.shape[0]:
        raise ValueError(f'Shapes of x {x.shape} and y {y.shape} are incompatible')
    if k > 0 and any((x < t[k]) | (x > t[-k])):
        raise ValueError('Out of bounds w/ x = %s.' % x)
    if x.size != w.size:
        raise ValueError(f'Shapes of x {x.shape} and w {w.shape} are incompatible')
    if method != "qr":
        raise ValueError(f"{method = } is not supported.")
    if any(x[1:] - x[:-1] < 0):
        raise ValueError("Expect x to be a 1D non-decreasing sequence.")

    # number of coefficients
    nc = t.size - k - 1

    # multiple r.h.s
    extradim = prod(y.shape[1:])
    yy = y.reshape(-1, extradim)

    # solve
    _, _, c = _lsq_solve_qr(x, yy, t, k, w)

    # restore the shape of `c` for both single and multiple r.h.s.
    c = c.reshape((nc,) + y.shape[1:])
    c = np.ascontiguousarray(c)
    return BSpline.construct_fast(t, c, k, axis=axis)
