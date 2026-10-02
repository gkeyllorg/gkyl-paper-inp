import numpy as np
def kx_to_x(var_kx, nx0, axis=-3):
    """ Perform inverse FFT on kx spectral direction of variable

    Note: The fft and ifft in python and GENE/IDL have the factor 1/N switched!
    :param var_kx: Variable in kx space
    :param nx0: Number of kx grid points
    :param axis: Which axis of var_kx is the x direction, by default the third last one
    :returns: variable in real x space
    """
    var_x = np.fft.ifft(nx0*var_kx, axis=axis)
    var_x = np.real_if_close(var_x, tol=1e5)
    return var_x


def ky_to_y(var_ky, nky0, axis=-2):
    """ Perform inverse FFT on ky spectral direction of variable

    The GENE data only include the non-negative ky components, so we need to use the real
    valued FFT routines
    Note: The fft and ifft in python and GENE/IDL have the factor 1/N switched!
    :param var_ky: Variable in kx space
    :param nky0: Number of ky grid points
    :param axis: Which axis of var_kx is the x direction, by default the third last one
    :returns: variable in real y space
    """
    # The GENE data only include the non-negative ky components, so we need to use the real
    # valued FFT routines
    var_y = np.fft.irfft(2*nky0*var_ky, n=2*nky0, axis=axis)
    var_y = np.real_if_close(var_y, tol=1e5)
    return var_y
