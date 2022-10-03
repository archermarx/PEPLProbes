def central_difference_coeffs(x0, x1, x2):
    '''
    For some function evaluated on discrete x locations, x0, x1, x2, this function
    computes the coefficients for a central finite difference approximation to the
    derivative of that function evaluated at x1. For example, if x0, x1, and x2 are
    evenly spaced with (x1 - x0) == (x2 - x1) == h, we would get

    central_diff_coeffs(x0, x1, x2) == (-1/2/h, 0, 1/2/h)

    Corresponding to df/dx = (f(x2) - f(x0)) / 2 / h

    For function values f0, f1, f2 evaluated at x0, x1, x2, the derivative will simply be:

    (d0, d1, d2) = central_diff_coeffs(x0, x1, x2)
    f0 * d0 + f1 * d1 + f2 * d2

    NOTE: This kind of derivative is not robust to numerical noise. For noisy data,
    consider smoothing first, fitting a spline, or employing a different noise-robust
    derivative operator.
    '''

    # Spacing between x0 and x1
    h_01 = x1 - x0

    # Spacing between x1 and x2
    h_12 = x2 - x1

    # Difference coefficients
    d0 = -h_12 / (h_01 * (h_01 + h_12))
    d1 = -(h_01 - h_12) / (h_01 * h_12)
    d2 =  h_01 / (h_12 * (h_01 + h_12))

    return (d0, d1, d2)
