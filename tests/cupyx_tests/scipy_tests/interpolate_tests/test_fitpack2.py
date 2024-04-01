import pytest
import cupy
from cupy import testing
from cupy.cuda import runtime

import numpy as _np
import cupyx.scipy.interpolate as csi  # NOQA

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass


@testing.with_requires("scipy")
class TestUnivariateSpline:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_linear_constant(self, xp, scp):
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([3, 3, 3])
        lut = scp.interpolate.UnivariateSpline(x, y, k=1)
        return lut.get_knots(), lut.get_coeffs(), lut(x)

#        assert_array_almost_equal(lut.get_knots(),[1,3])
#        assert_array_almost_equal(lut.get_coeffs(),[3,3])
#        assert_almost_equal(lut.get_residual(),0.0)
#        assert_array_almost_equal(lut([1,1.5,2]),[3,3,3])
#
#        spl = make_splrep(x, y, k=1, s=len(x))
#        assert_allclose(spl.t[1:-1], lut.get_knots(), atol=1e-15)
#        assert_allclose(spl.c, lut.get_coeffs(), atol=1e-15)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_preserve_shape(self, xp, scp):
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([0, 2, 4])
        lut = scp.interpolate.UnivariateSpline(x, y, k=1)
        arg = 2
        return lut(arg), lut(arg, nu=1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_preserve_shape_2(self, xp, scp):
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([0, 2, 4])
        lut = scp.interpolate.UnivariateSpline(x, y, k=1)
        arg = xp.asarray([1.5, 2, 2.5])
        return lut(arg), lut(arg, nu=1)

#        assert_equal(shape(arg), shape(lut(arg)))
#        assert_equal(shape(arg), shape(lut(arg, nu=1)))

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_linear_1d(self, xp, scp):
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([0, 2, 4])
        lut = scp.interpolate.UnivariateSpline(x, y, k=1)
        return lut.get_knots(), lut.get_coeffs(), lut(x)

#        assert_array_almost_equal(lut.get_knots(),[1,3])
#        assert_array_almost_equal(lut.get_coeffs(),[0,4])
#        assert_almost_equal(lut.get_residual(),0.0)
#        assert_array_almost_equal(lut([1,1.5,2]),[0,1,2])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_empty_input(self, xp, scp):
        # Test whether empty input returns an empty output. Ticket 1014
        x = xp.asarray([1, 3, 5, 7, 9])
        y = xp.asarray([0, 4, 9, 12, 21])
        spl = scp.interpolate.UnivariateSpline(x, y, k=3)
        return spl([])
#        assert_array_equal(spl([]), array([]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_derivatives(self, xp, scp):
        x = xp.asarray([1, 3, 5, 7, 9])
        y = xp.asarray([0, 4, 9, 12, 21])
        spl = scp.interpolate.UnivariateSpline(x, y, k=3)
        return spl.derivatives(3.5)

#        assert_almost_equal(spl.derivatives(3.5),
#                            [5.5152902, 1.7146577, -0.1830357, 0.3125])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_derivatives_2(self, xp, scp):
        x = xp.arange(8)
        y = x**3 + 2.*x**2

        spl = scp.interpolate.UnivariateSpline(x, y, s=0, k=3)
        return spl.derivatives(3)

#        assert_allclose(spl.derivatives(3),
#                        ders,
#                        atol=1e-15)

    @pytest.mark.parametrize('klass',
        ['UnivariateSpline', 'InterpolatedUnivariateSpline']
    )
    @pytest.mark.parametrize('ext', ['extrapolate', 'zeros', 'const'])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_out_of_range_regression(self, klass, ext, xp, scp):
        # Test different extrapolation modes. See ticket 3557
        x = xp.arange(5, dtype=float)
        y = x**3

        xp = xp.linspace(-8, 13, 100)
#        xp_zeros = xp.copy()
#        xp_zeros[np.logical_or(xp_zeros < 0., xp_zeros > 4.)] = 0
#        xp_clip = xp.copy()
#        xp_clip[xp_clip < x[0]] = x[0]
#        xp_clip[xp_clip > x[-1]] = x[-1]

        cls = getattr(scp.interpolate, klass)
        spl = cls(x=x, y=y)
        return spl(xp, ext=ext)

#        for cls in [UnivariateSpline, InterpolatedUnivariateSpline]:
#        spl = cls(x=x, y=y)
#        for ext in [0, 'extrapolate']:
#            assert_allclose(spl(xp, ext=ext), xp**3, atol=1e-16)
#            assert_allclose(cls(x, y, ext=ext)(xp), xp**3, atol=1e-16)
#        for ext in [1, 'zeros']:
#            assert_allclose(spl(xp, ext=ext), xp_zeros**3, atol=1e-16)
#            assert_allclose(cls(x, y, ext=ext)(xp), xp_zeros**3, atol=1e-16)
#        for ext in [2, 'raise']:
#            assert_raises(ValueError, spl, xp, **dict(ext=ext))
#        for ext in [3, 'const']:
#            assert_allclose(spl(xp, ext=ext), xp_clip**3, atol=1e-16)
#            assert_allclose(cls(x, y, ext=ext)(xp), xp_clip**3, atol=1e-16)


    def test_lsq_fpchec(self):
        xs = cupy.arange(100) * 1.
        ys = cupy.arange(100) * 1.
        knots = cupy.linspace(0, 99, 10)
        bbox = (-1, 101)
        with pytest.raises(ValueError):
            csi.LSQUnivariateSpline(xs, ys, knots, bbox=bbox)

#        assert_raises(ValueError, LSQUnivariateSpline, xs, ys, knots,
#                      bbox=bbox)

    @pytest.mark.parametrize('ext', [0, 1, 2, 3])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integral_out_of_bounds(self, xp, scp, ext):
        # Regression test for gh-7906: .integral(a, b) is wrong if both
        # a and b are out-of-bounds
        x = xp.linspace(0., 1., 7)
        f = scp.interpolate.UnivariateSpline(x, x, s=0, ext=ext)
        vals = [f.integral(a, b)
                for (a, b) in [(1, 1), (1, 5), (2, 5),
                               (0, 0), (-2, 0), (-2, -1)]
        ]
        # NB: scipy returns python floats, cupy returns 0D arrays
        return xp.asarray(vals)

#            for (a, b) in [(1, 1), (1, 5), (2, 5),
#                           (0, 0), (-2, 0), (-2, -1)]:
#                assert_allclose(f.integral(a, b), 0, atol=1e-15)


    @pytest.mark.parametrize('s', [0, 0.1, 0.01])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_values(self, xp, scp, s):
        x = xp.arange(8) + 0.5
        y = x + 1 / (1 - x)
        spl = scp.interpolate.UnivariateSpline(x, y, s=s)
        return spl(x)

