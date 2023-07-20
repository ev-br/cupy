import cupy


def _pad_h(h, up):
    """Store coefficients in a transposed, flipped arrangement.
    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].
    Then the internal buffer will look like this::
       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
    """
    h_padlen = len(h) + (-len(h) % up)
    h_full = cupy.zeros(h_padlen, h.dtype)
    h_full[: len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full


class _UpFIRDn(object):
    def __init__(self, h, x_dtype, up, down):
        """Helper for resampling"""
        h = cupy.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError("h must be 1D with non-zero length")

        self._output_type = cupy.result_type(h.dtype, x_dtype, cupy.float32)
        h = cupy.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError("Both up and down must be >= 1")
        # This both transposes, and "flips" each phase for filtering
        self._h_trans_flip = _pad_h(h, self._up)
        self._h_trans_flip = cupy.asarray(self._h_trans_flip)
        self._h_trans_flip = cupy.ascontiguousarray(self._h_trans_flip)
        self._h_len_orig = len(h)

    def apply_filter(
        self,
        x,
        axis,
    ):
        """Apply the prepared filter to the specified axis of a nD signal x"""

        x = cupy.asarray(x, self._output_type)

        output_len = _output_len(self._h_len_orig, x.shape[axis], self._up, self._down)
        output_shape = list(x.shape)
        output_shape[axis] = output_len
        out = cupy.empty(output_shape, dtype=self._output_type, order="C")
        axis = axis % x.ndim

        # Precompute variables on CPU
        x_shape_a = x.shape[axis]
        h_per_phase = len(self._h_trans_flip) // self._up
        padded_len = x.shape[axis] + (len(self._h_trans_flip) // self._up) - 1

        if out.ndim > 1:
            threadsperblock = (8, 8)
            blocks = ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_x = blocks if blocks < _get_max_gdx() else _get_max_gdx()

            blocks = ceil(out.shape[1] / threadsperblock[1])
            blockspergrid_y = blocks if blocks < _get_max_gdy() else _get_max_gdy()

            blockspergrid = (blockspergrid_x, blockspergrid_y)

        else:
            threadsperblock, blockspergrid = _get_tpb_bpg()

        if out.ndim == 1:
            k_type = "upfirdn1D"

            _populate_kernel_cache(out.dtype, k_type)

            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                k_type,
            )
        elif out.ndim == 2:
            k_type = "upfirdn2D"

            _populate_kernel_cache(out.dtype, k_type)

            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                k_type,
            )
        else:
            raise NotImplementedError("upfirdn() requires ndim <= 2")

        kernel(
            x,
            self._h_trans_flip,
            self._up,
            self._down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
        )

        _print_atts(kernel)

        return out


def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
):
    """
    Upsample, FIR filter, and downsample.

    Parameters
    ----------
    h : array_like
        1-dimensional FIR (finite-impulse response) filter coefficients.
    x : array_like
        Input signal array.
    up : int, optional
        Upsampling rate. Default is 1.
    down : int, optional
        Downsampling rate. Default is 1.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.

    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.

    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).

    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).

    See Also
    --------
    scipy.signal.upfirdn

    References
    ----------
    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
       Prentice Hall, 1993.
    """

    ufd = _UpFIRDn(h, x.dtype, up, down)
    # This is equivalent to (but faster than) using cp.apply_along_axis
    return ufd.apply_filter(x, axis)
