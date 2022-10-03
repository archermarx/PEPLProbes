import math
import numpy as np

class DifferenceOperator:
    def __init__(self, kernel, stencil_width):

        if stencil_width % 2 == 0:
            raise ValueError("Finite difference stencil width must be odd!")

        if stencil_width < 3:
            raise ValueError("Finite difference stencil width must be 3 or more!")

        self.stencil_width = stencil_width
        self.kernel = kernel

class CentralDifferenceOperator(DifferenceOperator):
    def __init__(self):
        DifferenceOperator.__init__(self, self.central_difference_kernel, 3)

    @staticmethod
    def central_difference_kernel(x, f):
        # Spacing between x0 and x1
        h_01 = x[1] - x[0]

        # Spacing between x1 and x2
        h_12 = x[2] - x[1]

        # Difference coefficients
        d0 = -h_12 / (h_01 * (h_01 + h_12))
        d1 = -(h_01 - h_12) / (h_01 * h_12)
        d2 =  h_01 / (h_12 * (h_01 + h_12))

        return d0*f[0] + d1*f[1] + d2*f[2]

class SmoothDifferenceOperator(DifferenceOperator):
    '''
    Smooth noise-robust differentiator, derived from
    http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
    Should be exact on 1, x, and x^2
    '''
    def __init__(self, stencil_width=5):

        kernel = self.smooth_difference_kernel
        if stencil_width < 5:
            kernel = CentralDifferenceOperator.central_difference_kernel

        DifferenceOperator.__init__(self, kernel, stencil_width)

    @staticmethod
    def binomial_coeff(n, k):
        if k < 0 or k > n:
            return 0
        else:
            return math.comb(n, k)

    @staticmethod
    def smooth_difference_coefficient(k, m):
        term_1 = SmoothDifferenceOperator.binomial_coeff(2 * m, m - k + 1)
        term_2 = SmoothDifferenceOperator.binomial_coeff(2 * m, m - k - 1)

        return (term_1 - term_2) / 2**(2*m + 1)

    def smooth_difference_kernel(self, x, f):
        N = self.stencil_width
        M = (N-1)//2
        m = (N-3)//2

        deriv = 0.0

        for k in range(1, M+1):
            ck = SmoothDifferenceOperator.smooth_difference_coefficient(k, m)
            deriv = deriv + ck * (f[M+k] - f[M-k]) / (x[M+k] - x[M-k]) * 2 * k

        return deriv

def derivative(xs, fs, operator = CentralDifferenceOperator()):

    if len(xs) != len(fs):
        raise ValueError("x and f must be arrays of the same length")

    if (operator.stencil_width % 2) == 0:
        raise ValueError("Derivative stencil width must be odd")

    num_pts = len(xs)

    M = (operator.stencil_width - 1) // 2

    # pad xs and fs with ghost cells
    xs_padded = np.pad(xs, (M,), 'reflect', reflect_type='odd')
    fs_padded = np.pad(fs, (M,), 'reflect', reflect_type='odd')

    # Allocate derivative array
    deriv = np.zeros_like(fs)

    for i in range(num_pts):
        deriv[i] = operator.kernel(xs_padded[i:i+2*M+1], fs_padded[i:i+2*M+1])

    return deriv
