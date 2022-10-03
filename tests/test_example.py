from PEPLProbes.utils.differentiate import central_difference_coeffs
import math

def test_central_diffs_even():
    h = 0.1
    x0 = -h
    x1 = 0*h
    x2 = h
    (d0, d1, d2) = central_difference_coeffs(x0, x1, x2)
    # Check to make sure we recover classic second order central difference
    # on evenly-spaced data
    assert math.isclose(d0, -0.5 / h) and math.isclose(d1, 0.0) and math.isclose(d2, 0.5 / h)

class TestDerivatives:

    tol = 1e-5

    def deriv_test_helper(self, func=math.sin, deriv=math.cos, difference_operator=central_difference_coeffs):
        h = self.tol**2
        x0 = -h
        x1 = 0.1 * h
        x2 = 2 * h

        (d0, d1, d2) = difference_operator(x0, x1, x2)

        f0 = func(x0)
        f1 = func(x1)
        f2 = func(x2)

        deriv_exact  = deriv(x0)
        deriv_approx = d0*f0 + d1*f1 + d2*f2
        assert math.isclose(deriv_exact, deriv_approx, abs_tol=self.tol)

    def test_sin(self):
        func = math.sin
        deriv = math.cos
        self.deriv_test_helper(func, deriv)

    def test_quadratic(self):
        func = lambda x: x**2 + 3*x + 2
        deriv = lambda x: 2*x + 3
        self.deriv_test_helper(func, deriv)

    def test_exp(self):
        func = math.exp
        deriv = math.exp
        self.deriv_test_helper(func, deriv)
