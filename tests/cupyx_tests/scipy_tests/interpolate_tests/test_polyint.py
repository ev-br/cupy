import io
import warnings

import numpy
import pytest

import cupy
from cupy import testing
import cupyx.scipy.interpolate  # NOQA

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass


@testing.with_requires("scipy")
class TestBarycentric:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_lagrange(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 100, dtype=dtype)
        xs = xp.linspace(-5, 5, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(test_xs)

    @testing.for_all_dtypes(no_bool=True, no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-5)
    def test_scalar(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'iu':
            pytest.skip()
        true_poly = numpy.poly1d([-1, 2, 6, -3, 2])
        xs = numpy.linspace(-1, 1, 10, dtype=dtype)
        ys = true_poly(xs)
        if xp is cupy:
            xs = cupy.asarray(xs)
            ys = cupy.asarray(ys)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(xp.array(7, dtype=dtype))

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_delayed(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 100, dtype=dtype)
        xs = xp.linspace(-5, 5, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs)
        P.set_yi(ys)
        return P(test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_append(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 100, dtype=dtype)
        xs = xp.linspace(-5, 5, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs[:3], ys[:3])
        P.add_xi(xs[3:], ys[3:])
        return P(test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vector(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        xs = xp.array([0, 1, 2], dtype=dtype)
        ys = xp.array([[0, 1], [1, 0], [2, 1]], dtype=dtype)
        test_xs = xp.linspace(-1, 3, 100, dtype=dtype)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(test_xs)

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_scalarvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_vectorvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([4, -5, 3, 2, -4])
        xs = xp.linspace(-10, 10, 20)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(
            xs, xp.outer(ys, xp.arange(3)))
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_1d_vectorvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-3, -1, 4, 9, 8])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(
            xs, xp.outer(ys, xp.array([1])))
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @testing.with_requires("scipy>=1.8.0")
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_large_chebyshev(self, xp, scp, dtype):
        n = 100
        j = numpy.arange(n + 1, dtype=dtype).astype(numpy.float64)
        x = numpy.cos(j * numpy.pi / n)

        if xp is cupy:
            j = cupy.asarray(j)
            x = cupy.asarray(x)
        # The weights for Chebyshev points against SciPy counterpart
        return scp.interpolate.BarycentricInterpolator(x).wi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex(self, xp, scp):
        x = xp.array([1, 2, 3, 4])
        y = xp.array([1, 2, 1j, 3])
        xi = xp.array([0, 8, 1, 5])
        return scp.interpolate.BarycentricInterpolator(x, y)(xi)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-6, rtol=1e-6)
    def test_wrapper(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-2, 2, 5, dtype=dtype)
        xs = xp.linspace(-2, 2, 5, dtype=dtype)
        ys = true_poly(xs)
        return scp.interpolate.barycentric_interpolate(xs, ys, test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_array_input(self, xp, scp, dtype):
        x = 1000 * xp.arange(1, 11, dtype=dtype)
        y = xp.arange(1, 11, dtype=dtype)
        xi = xp.array(1000 * 9.5)
        return scp.interpolate.barycentric_interpolate(x, y, xi)


@testing.with_requires("scipy")
class TestKrogh:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_lagrange(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(test_xs)

    @testing.for_all_dtypes(no_bool=True, no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-6, rtol=1e-6)
    def test_scalar(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = numpy.poly1d([-1, 2, 6, -3, 2])
        xs = numpy.linspace(-1, 1, 10, dtype=dtype)
        ys = true_poly(xs)
        if xp is cupy:
            xs = cupy.asarray(xs)
            ys = cupy.asarray(ys)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(xp.array(7, dtype=dtype))

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_derivatives(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        D = P.derivatives(test_xs)
        return D

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_low_derivatives(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        D = P.derivatives(test_xs, len(xs) + 2)
        return D

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_derivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        m = 10
        return [P.derivative(test_xs, i) for i in range(m)]

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_high_derivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        D = P.derivative(test_xs, 2 * len(xs))
        return D

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_hermite(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        D = P(test_xs)
        return D

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_hermite_derivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        xs = xp.array([0, 0, 0], dtype=dtype)
        ys = xp.array([1, 2, 3], dtype=dtype)
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_vector(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        xs = xp.array([0, 1, 2], dtype=dtype)
        ys = xp.array([[0, 1], [1, 0], [2, 1]], dtype=dtype)
        test_xs = xp.linspace(-1, 3, 5, dtype=dtype)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(test_xs)

    @testing.for_dtypes('bhilfd')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_empty(self, xp, scp, dtype):
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        xs = xp.linspace(-5, 5, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(xp.array([]))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_scalarvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_scalarvalue_derivatives(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        test_xs = xp.array(test_xs)
        return xp.shape(P.derivatives(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_vectorvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([4, -5, 3, 2, -4])
        xs = xp.linspace(-10, 10, 20)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(
            xs, xp.outer(ys, xp.arange(3)))
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_vectorvalue_derivative(self, xp, scp, test_xs):
        true_poly = xp.poly1d([4, -5, 3, 2, -4])
        xs = xp.linspace(-10, 10, 20)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(
            xs, xp.outer(ys, xp.arange(3)))
        test_xs = xp.array(test_xs)
        return xp.shape(P.derivatives(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_1d_vectorvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-3, -1, 4, 9, 8])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(
            xs, xp.outer(ys, xp.array([1])))
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_wrapper(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-2, 2, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        return scp.interpolate.krogh_interpolate(xs, ys, test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_wrapper2(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-2, 2, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        return scp.interpolate.krogh_interpolate(xs, ys, test_xs, der=3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_derivatives_complex(self, xp, scp):
        x = xp.array([-1, -1, 0, 1, 1])
        y = xp.array([1, 1.0j, 0, -1, 1.0j])
        P = scp.interpolate.KroghInterpolator(x, y)
        D = P.derivatives(xp.array(0))
        return D


@testing.with_requires("scipy")
class TestAkima1DInterpolator:
    # in scipy, these tests are in test_interpolate.py

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_eval(self, xp, scp):
        x = xp.arange(0., 11.)
        y = xp.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        ak = scp.interpolate.Akima1DInterpolator(x, y)
        xi = xp.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
            8.6, 9.9, 10.])
        return ak(xi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_eval_2d(self, xp, scp):
        x = xp.arange(0., 11.)
        y = xp.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        y = xp.column_stack((y, 2. * y))
        ak = scp.interpolate.Akima1DInterpolator(x, y)
        xi = xp.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
                       8.6, 9.9, 10.])
        return ak(xi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_eval_3d(self, xp, scp):
        x = xp.arange(0., 11.)
        y_ = xp.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        y = xp.empty((11, 2, 2))
        y[:, 0, 0] = y_
        y[:, 1, 0] = 2. * y_
        y[:, 0, 1] = 3. * y_
        y[:, 1, 1] = 4. * y_
        ak = scp.interpolate.Akima1DInterpolator(x, y)
        xi = xp.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
                       8.6, 9.9, 10.])
        return ak(xi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_case_multidimensional(self, xp, scp):
        # This test is for the scipy issue gh-5683.
        x = xp.array([0, 1, 2])
        y = xp.vstack((x, x**2)).T
        ak = scp.interpolate.Akima1DInterpolator(x, y)
        x_eval = np.array([0.5, 1.5])
        return ak(x_eval)


@testing.with_requires("scipy")
class TestPCHIP:
    def _make_random(self, xp, npts=20):
        # NB: deliberately sample from numpy.random (random streams differ
        # for numpy.random and cupy.random!)
        numpy.random.seed(1234)
        xi = numpy.sort(numpy.random.random(npts))
        yi = numpy.random.random(npts)
        if xp is cupy:
            xi = cupy.asarray(xi)
            yi = cupy.asarray(yi)
        return xi, yi

    def test_overshoot(self):
        # PCHIP should not overshoot
        xi, yi = self._make_random(cupy)
        p = cupyx.scipy.interpolate.PchipInterpolator(xi, yi)
        for i in range(len(xi) - 1):
            x1, x2 = xi[i], xi[i + 1]
            y1, y2 = yi[i], yi[i + 1]
            if y1 > y2:
                y1, y2 = y2, y1
            xp = cupy.linspace(x1, x2, 10)
            yp = p(xp)
            assert (((y1 <= yp + 1e-15) & (yp <= y2 + 1e-15)).all())

    def test_monotone(self):
        # PCHIP should preserve monotonicty
        xi, yi = self._make_random(cupy)
        p = cupyx.scipy.interpolate.PchipInterpolator(xi, yi)
        for i in range(len(xi) - 1):
            x1, x2 = xi[i], xi[i + 1]
            y1, y2 = yi[i], yi[i + 1]
            xp = cupy.linspace(x1, x2, 10)
            yp = p(xp)
            assert (((y2-y1) * (yp[1:] - yp[:1]) > 0).all())

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cast(self, xp, scp):
        # regression test for integer input data, see gh-3453
        data = xp.array([[0, 4, 12, 27, 47, 60, 79, 87, 99, 100],
                         [-33, -33, -19, -2, 12, 26, 38, 45, 53, 55]])
        data = data.astype(int)
        xx = xp.arange(100)
        curve = pchip(data[0], data[1])(xx)
        return curve

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=5e-5)
    def test_nag(self, xp, scp):
        # Example from NAG C implementation,
        # http://nag.com/numeric/cl/nagdoc_cl25/html/e01/e01bec.html
        # suggested in gh-5326 as a smoke test for the way the derivatives
        # are computed (see also gh-3453)
        dataStr = '''
          7.99   0.00000E+0
          8.09   0.27643E-4
          8.19   0.43750E-1
          8.70   0.16918E+0
          9.20   0.46943E+0
         10.00   0.94374E+0
         12.00   0.99864E+0
         15.00   0.99992E+0
         20.00   0.99999E+0
        '''
        data = cupy.loadtxt(io.StringIO(dataStr))
        pch = pchip(data[:, 0], data[:, 1])

        resultStr = '''
           7.9900       0.0000
           9.1910       0.4640
          10.3920       0.9645
          11.5930       0.9965
          12.7940       0.9992
          13.9950       0.9998
          15.1960       0.9999
          16.3970       1.0000
          17.5980       1.0000
          18.7990       1.0000
          20.0000       1.0000
        '''
        result = cupy.loadtxt(io.StringIO(resultStr))
        return pch(result(:, 0))
#        assert_allclose(result[:,1], pch(result[:,0]), rtol=0., atol=5e-5)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_endslopes(self, xp, scp):
        # this is a smoke test for gh-3453: PCHIP interpolator should not
        # set edge slopes to zero if the data do not suggest zero edge
        # derivatives
        x = xp.array([0.0, 0.1, 0.25, 0.35])
        y1 = xp.array([279.35, 0.5e3, 1.0e3, 2.5e3])
        y2 = xp.array([279.35, 2.5e3, 1.50e3, 1.0e3])
        return (scp.interpolate.PchipInterpolator(x, y1), 
                scp.interpolate.PchipInterpolator(x, y2))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_all_zeros(self, xp, scp):
        x = xp.arange(10)
        y = xp.zeros_like(x)

        # this should work and not generate any warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            pch = scp.interpolate.PchipInterpolator(x, y)

        xx = np.linspace(0, 9, 101)
        return pch(xx)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_two_points(self, xp, scp):
        # regression test for gh-6222: pchip([0, 1], [0, 1]) fails because
        # it tries to use a three-point scheme to estimate edge derivatives,
        # while there are only two points available.
        # Instead, it should construct a linear interpolator.
        x = xp.linspace(0, 1, 11)
        p = scp.interpolate.PchipInterpolator([0, 1], [0, 2])
        return p(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots(self, xp, scp):
        # regression test for gh-6357: .roots method should work
        p = scp.interpolate.PchipInterpolator([0, 1], [-1, 1])
        r = p.roots()
        return r


@testing.with_requires("scipy")
class TestCubicHermiteSpline:

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_correctness(self, xp, scp):
        x = [0, 2, 7]
        y = [-1, 2, 3]
        dydx = [0, 3, 7]
        s = scp.interpolate.CubicHermiteSpline(x, y, dydx)
        return s(x), s(x, 1)

    def test_CubicHermiteSpline_error_handling(self):
        x = [1, 2, 3]
        y = [0, 3, 5]
        dydx = [1, -1, 2, 3]
        with pytest.raises(ValueError):
            scp.interpolate.CubicHermiteSpline(x, y, dydx)

        dydx_with_nan = [1, 0, cupy.nan]
        with pytest.raises(ValueError):
            scp.interpolate.CubicHermiteSpline(x, y, dydx_with_nan)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots_extrapolate_gh_11185(self, xp, scp):
        x = xp.array([0.001, 0.002])
        y = xp.array([1.66066935e-06, 1.10410807e-06])
        dy = xp.array([-1.60061854, -1.600619])
        p = scp.interpolate.CubicHermiteSpline(x, y, dy)

        # roots(extrapolate=True) for a polynomial with a single interval
        # should return all three real roots
        r = p.roots(extrapolate=True)
        return p.c.shape[1], r.size


@testing.with_requires("scipy>=1.10.0")
class TestZeroSizeArrays:
    # regression tests for SciPy/gh-17241 : CubicSpline et al must not segfault
    # when y.size == 0
    # The two methods below are _almost_ the same, but not quite:
    # one is for objects which have the `bc_type` argument (CubicSpline)
    # and the other one is for those which do not (Pchip, Akima1D)

    # XXX: add CubicSpline to the test loop, when implemented

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('y_shape', [(10, 0, 5), (10, 5, 0)])
    @pytest.mark.parametrize('bc_type',
                             ['not-a-knot', 'periodic', 'natural', 'clamped'])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('klass', ['make_interp_spline', ])
    def test_zero_size(self, xp, scp, klass, y_shape, bc_type, axis):
        x = xp.arange(10)
        y = xp.zeros(y_shape)
        xval = xp.arange(3)

        cls = getattr(scp.interpolate, klass)
        obj = cls(x, y, bc_type=bc_type)
        assert obj(xval).size == 0
        assert obj(xval).shape == xval.shape + y.shape[1:]

        # Also check with an explicit non-default axis
        yt = xp.moveaxis(y, 0, axis)  # (10, 0, 5) --> (0, 10, 5) if axis=1 etc

        obj = cls(x, yt, bc_type=bc_type, axis=axis)
        sh = yt.shape[:axis] + (xval.size, ) + yt.shape[axis+1:]
        assert obj(xval).size == 0
        assert obj(xval).shape == sh

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('y_shape', [(10, 0, 5), (10, 5, 0)])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('klass', ['PchipInterpolator',
                                       'Akima1DInterpolator'])
    def test_zero_size_2(self, xp, scp, klass, y_shape, axis):
        x = xp.arange(10)
        y = xp.zeros(y_shape)
        xval = xp.arange(3)

        cls = getattr(scp.interpolate, klass)
        obj = cls(x, y)
        assert obj(xval).size == 0
        assert obj(xval).shape == xval.shape + y.shape[1:]

        # Also check with an explicit non-default axis
        yt = xp.moveaxis(y, 0, axis)  # (10, 0, 5) --> (0, 10, 5) if axis=1 etc

        obj = cls(x, yt, axis=axis)
        sh = yt.shape[:axis] + (xval.size, ) + yt.shape[axis+1:]
        assert obj(xval).size == 0
        assert obj(xval).shape == sh


def check_shape(xp, scp, interpolator_cls, x_shape, y_shape, deriv_shape=None,
                axis=0, extra_args={}):
    np.random.seed(1234)

    x = [-1, 0, 1, 2, 3, 4]
    s = list(range(1, len(y_shape)+1))
    s.insert(axis % (len(y_shape)+1), 0)
    y = np.random.rand(*((6,) + y_shape)).transpose(s)
    if xp is cupy:
        y = cupy.asarray(y)

    xi = xp.zeros(x_shape)

    if interpolator_cls is scp.interpolate.CubicHermiteSpline:
        dydx = np.random.rand(*((6,) + y_shape)).transpose(s)
        if xp is cupy:
            dydx = cupy.asarray(dydx)

        yi = interpolator_cls(x, y, dydx, axis=axis, **extra_args)(xi)
    else:
        yi = interpolator_cls(x, y, axis=axis, **extra_args)(xi)

    target_shape = ((deriv_shape or ()) + y.shape[:axis]
                    + x_shape + y.shape[axis:][1:])
    testing.assert_equal(yi.shape, target_shape)

    # check it works also with lists
    if x_shape and y.size > 0:
        if interpolator_cls is scp.interpolate.CubicHermiteSpline:
            interpolator_cls(list(x), list(y), list(dydx), axis=axis,
                             **extra_args)(list(xi))
        else:
            interpolator_cls(list(x), list(y), axis=axis,
                             **extra_args)(list(xi))

    # check also values
    if xi.size > 0 and deriv_shape is None:
        bs_shape = y.shape[:axis] + (1,)*len(x_shape) + y.shape[axis:][1:]
        yv = y[((slice(None,),)*(axis % y.ndim)) + (1,)]
        yv = yv.reshape(bs_shape)

        yi, y = np.broadcast_arrays(yi, yv)
        testing.assert_allclose(yi, y)


SHAPES = [(), (0,), (1,), (6, 2, 5)]

@testing.numpy_cupy_allclose(scipy_name='scp')
def test_shapes(xp, scp):

    def spl_interp(x, y, axis):
        return scp.interpolate.make_interp_spline(x, y, axis=axis)

    for ip in [scp.interpolate.KroghInterpolator,
               scp.interpolate.BarycentricInterpolator,
               scp.interpolate.CubicHermiteSpline,
               scp.interpolate.PchipInterpolator,
               scp.interpolate.Akima1DInterpolator,
               # CubicSpline,   # XXX: add when implemented
               spl_interp]:
        for s1 in SHAPES:
            for s2 in SHAPES:
                for axis in range(-len(s2), len(s2)):
                    check_shape(xp, scp, ip, s1, s2, None, axis)


@testing.numpy_cupy_allclose(scipy_name='scp')
def test_deriv_shapes(xp, scp):
    def krogh_deriv(x, y, axis=0):
        return scp.interpolate.KroghInterpolator(x, y, axis).derivative

    def pchip_deriv(x, y, axis=0):
        return scp.interpolate.PchipInterpolator(x, y, axis).derivative()

    def pchip_deriv2(x, y, axis=0):
        return scp.interpolate.PchipInterpolator(x, y, axis).derivative(2)

    def pchip_antideriv(x, y, axis=0):
        return scp.interpolate.PchipInterpolator(x, y, axis).derivative()

    def pchip_antideriv2(x, y, axis=0):
        return scp.interpolate.PchipInterpolator(x, y, axis).antiderivative(2)

    def pchip_deriv_inplace(x, y, axis=0):
        class P(scp.interpolate.PchipInterpolator):
            def __call__(self, x):
                return super().__call__(self, x, 1)
            pass
        return P(x, y, axis)

    def akima_deriv(x, y, axis=0):
        return scp.interpolate.Akima1DInterpolator(x, y, axis).derivative()

    def akima_antideriv(x, y, axis=0):
        return scp.interpolate.Akima1DInterpolator(x, y, axis).antiderivative()

    def bspl_deriv(x, y, axis=0):
        return scp.interpolate.make_interp_spline(x, y, axis=axis).derivative()

    def bspl_antideriv(x, y, axis=0):
        func = scp.interpolate.make_interp_spline
        return func(x, y, axis=axis).antiderivative()

    for ip in [krogh_deriv, pchip_deriv, pchip_deriv2, pchip_deriv_inplace,
               pchip_antideriv, pchip_antideriv2, akima_deriv, akima_antideriv,
               bspl_deriv, bspl_antideriv]:
        for s1 in SHAPES:
            for s2 in SHAPES:
                for axis in range(-len(s2), len(s2)):
                    check_shape(xp, scp, ip, s1, s2, (), axis)


