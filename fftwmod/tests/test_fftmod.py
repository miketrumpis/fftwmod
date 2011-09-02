import numpy as np
import numpy.testing as npt
from nose.tools import assert_true, assert_false, raises

from decotest import parametric
from fftwmod import fft1, ifft1, fft2, ifft2
from fftwmod.wraps import _fftn, _ifftn

def direct_dft(v):
    assert v.dtype.char.isupper()
    L = v.shape[0]
    Lax = np.linspace(0,L,L,endpoint=False)
    basis = np.exp((Lax[:,None] * -2j*np.pi*Lax/L)).astype(v.dtype)
    w = np.dot(basis, v.reshape(L, np.prod(v.shape[1:])))
    w.shape = v.shape
    return w

def direct_idft(v):
    assert v.dtype.char.isupper()
    L = v.shape[0]
    Lax = np.linspace(0,L,L,endpoint=False)
    basis = np.exp((Lax[:,None] * 2j*np.pi*Lax/L)).astype(v.dtype)
    w = np.dot(basis, v.reshape(L, np.prod(v.shape[1:])))
    w.shape = v.shape
    return w
    
def direct_dft_centered(v):
    assert v.dtype.char.isupper()
    L = v.shape[0]
    grid = np.linspace(-L/2,L/2,num=L,endpoint=False)
    basis = np.exp((grid[:,None] * -2j*np.pi*grid/L)) #.astype(v.dtype)
    w = np.dot(basis, v.reshape(L, np.prod(v.shape[1:])))
    w.shape = v.shape
    return w.astype(v.dtype)

def direct_idft_centered(v):
    assert v.dtype.char.isupper()
    L = v.shape[0]
    grid = np.linspace(-L/2,L/2,num=L,endpoint=False)
    basis = np.exp((grid[:,None] * 2j*np.pi*grid/L)).astype(v.dtype)
    w = np.dot(basis, v.reshape(L, np.prod(v.shape[1:])))
    w.shape = v.shape
    return w

def checkerline(cols):
    return np.ones(cols) - 2*(np.arange(cols)%2)

def checkerboard(rows, cols):
    return np.outer(checkerline(rows), checkerline(cols))

## def reference_fftn(a, axes=(0,), shift=True):
##     a_s = np.fft.fftshift(a,axes=axes) if shift else a
##     b = np.fft.fftn(a_s, axes=axes)
##     return np.fft.fftshift(b, axes=axes) if shift else b

def reference_fftn(a, axes=(0,), shift=True):
    dft_func = direct_dft_centered if shift else direct_dft
    a_dft = a.copy()
    for ax in axes:
        b = np.rollaxis(a_dft, ax)
        b[:] = dft_func(b)
    return a_dft

## def reference_ifftn(a, axes=(0,), shift=True):
##     a_s = np.fft.fftshift(a,axes=axes) if shift else a
##     b = np.fft.ifftn(a_s, axes=axes)
##     return np.fft.fftshift(b, axes=axes) if shift else b

def reference_ifftn(a, axes=(0,), shift=True):
    dft_func = direct_idft_centered if shift else direct_idft
    a_dft = a.copy()
    for ax in axes:
        b = np.rollaxis(a_dft, ax)
        b[:] = dft_func(b)
    return a_dft

def sum_of_sqr_comp(a1, a2, err=''):
    dec = 6 if a1.dtype.char=='F' else 12
    a = np.dot(a1.flatten(), a1.conj().flatten()).real
    b = np.dot(a2.flatten(), a2.conj().flatten()).real
    return npt.assert_almost_equal((a-b)/a, 0, decimal=dec, err_msg=err)
##     return npt.assert_almost_equal(
##         np.dot(a1.flatten(), a1.conj().flatten()).real,
##         np.dot(a2.flatten(), a2.conj().flatten()).real,
##         err_msg=err
##         )

centered_sgrid = np.linspace(0, 1, num=128, endpoint=False) - 0.5
sgrid = np.linspace(0, 1, num=128, endpoint=False)
def ref_1D_32hz(centered, dtype):
    # a 32Hz complex exponential (sampling rate = 128Hz)
    if centered:
        return np.exp(2j*np.pi*32*centered_sgrid).astype(dtype)
    return np.exp(2j*np.pi*32*sgrid).astype(dtype)

def ref_2D_grating(centered, dtype):
    # a separable complex exponential with fy = 13Hz, fx = 4Hz
    if centered:
        return np.exp( (2j*np.pi*13*centered_sgrid)[:,None] + \
                       (2j*np.pi*4*centered_sgrid)[None,:] ).astype(dtype)
    return np.exp( (2j*np.pi*13*sgrid)[:,None] + \
                   (2j*np.pi*4*sgrid)[None,:] ).astype(dtype)


def _get_1D_fft(shift, dt):
    c = ref_1D_32hz(shift, dt)
    c2 = c.copy()
    _fftn(c, axes=(0,), shift=shift, inplace=True)
    c_np = reference_fftn(c2, axes=(0,), shift=shift)
    return c, c_np

@parametric
def test_unnormalized_1F():
    x = np.random.randn(128) + 1j*np.random.randn(128)
    x = x.astype('F')
    z = _fftn(x, axes=(0,), shift=0, inplace=False)
    x2 = _ifftn(z, axes=(0,), shift=0, inplace=False, normalize=False)
    yield npt.assert_almost_equal(
        np.linalg.norm(x2), np.linalg.norm(x)*len(x)
        )
    z = _fftn(x, axes=(0,), shift=1, inplace=False, normalize=False)
    x2 = _ifftn(z, axes=(0,), shift=1, inplace=False, normalize=False)
    yield npt.assert_almost_equal(
        np.linalg.norm(x2), np.linalg.norm(x)*len(x)
        )

@parametric
def test_unnormalized_1D():
    x = np.random.randn(128) + 1j*np.random.randn(128)
    z = _fftn(x, axes=(0,), shift=0, inplace=False)
    x2 = _ifftn(z, axes=(0,), shift=0, inplace=False, normalize=False)
    yield npt.assert_almost_equal(
        np.linalg.norm(x2), np.linalg.norm(x)*len(x)
        )
    z = _fftn(x, axes=(0,), shift=1, inplace=False, normalize=False)
    x2 = _ifftn(z, axes=(0,), shift=1, inplace=False, normalize=False)
    yield npt.assert_almost_equal(
        np.linalg.norm(x2), np.linalg.norm(x)*len(x)
        )

@parametric
def test_simple_1D_fft_0_F():
    c, c_np = _get_1D_fft(0, 'F')
    # test sum-of-squares equality
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    # test point-wise equality
    yield npt.assert_array_almost_equal(c, c_np), 'not equal pointwise'
    # analytically, the DFT indexed from 0,127 of s
    # is a weighted delta at k=32
    yield npt.assert_almost_equal(c[32], 128.0), 'delta function error'

@parametric
def test_simple_1D_fft_1_F():
    c, c_np = _get_1D_fft(1, 'F')
    # test sum-of-squares equality
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    # test point-wise equality
    yield npt.assert_array_almost_equal(c, c_np), 'not equal pointwise'
    # analytically, the DFT indexed from 0,127 of s
    # is a weighted delta at k=32
    yield npt.assert_almost_equal(c[64+32], 128.0), 'delta function error'

@parametric
def test_simple_1D_fft_0_D():
    c, c_np = _get_1D_fft(0, 'D')
    # test sum-of-squares equality
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    # test point-wise equality
    yield npt.assert_array_almost_equal(c, c_np), 'not equal pointwise'
    # analytically, the DFT indexed from 0,127 of s
    # is a weighted delta at k=32
    yield npt.assert_almost_equal(c[32], 128.0), 'delta function error'

@parametric
def test_simple_1D_fft_1_D():
    c, c_np = _get_1D_fft(1, 'D')
    # test sum-of-squares equality
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    # test point-wise equality
    yield npt.assert_array_almost_equal(c, c_np), 'not equal pointwise'
    # analytically, the DFT indexed from 0,127 of s
    # is a weighted delta at k=32
    yield npt.assert_almost_equal(c[64+32], 128.0), 'delta function error'


def _get_2D_fft(shift, dt):
    c = ref_2D_grating(shift, dt)
    c2 = c.copy()
    _fftn(c, axes=(0,1), shift=shift, inplace=True)
    c_np = reference_fftn(c2, axes=(0,1), shift=shift)
    return c, c_np

@parametric
def test_simple_2D_fft_0_F():
    c, c_np = _get_2D_fft(0, 'F')
    # test sum-of-squares equality
    yield npt.assert_almost_equal(np.dot(c,c.conj()).real,
                                  np.dot(c_np, c_np.conj()).real), \
                                  'total energy not equal'
    # test point-wise equality
    yield npt.assert_array_almost_equal(c, c_np), 'not equal pointwise'
    # analytically, the DFT indexed from 0,127 of s
    # is a weighted delta at i=13, j=4
    yield npt.assert_almost_equal(c[13,4], 128.0**2), 'delta function error'

@parametric
def test_simple_2D_fft_0_D():
    c, c_np = _get_2D_fft(0, 'D')
    # test sum-of-squares equality
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    # test point-wise equality
    yield npt.assert_array_almost_equal(c, c_np), 'not equal pointwise'
    # analytically, the DFT indexed from 0,127 of s
    # is a weighted delta at i=13, j=4
    yield npt.assert_almost_equal(c[13,4], 128.0**2), 'delta function error'

@parametric
def test_simple_2D_fft_1_F():
    c, c_np = _get_2D_fft(1, 'F')
    # test sum-of-squares equality
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    # test point-wise equality
    yield npt.assert_array_almost_equal(c, c_np), 'not equal pointwise'
    # analytically, the DFT indexed from 0,127 of s
    # is a weighted delta at i=13, j=4
    yield npt.assert_almost_equal(c[64+13,64+4], 128.0**2), \
          'delta function error'

@parametric
def test_simple_2D_fft_1_D():
    c, c_np = _get_2D_fft(1, 'D')
    # test sum-of-squares equality
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    # test point-wise equality
    yield npt.assert_array_almost_equal(c, c_np), 'not equal pointwise'
    # analytically, the DFT indexed from 0,127 of s
    # is a weighted delta at i=13, j=4
    yield npt.assert_almost_equal(c[64+13,64+4], 128.0**2), \
          'delta function error'

@parametric
def test_simple_multi_fft_0_F():
    shift = 0
    dt = 'F'
    c = np.outer(np.ones(64), ref_1D_32hz(shift, dt))
    c2 = c.copy()
    ct = c.copy().T
    ct2 = c.copy().T
        
    _fftn(c, axes=(-1,), inplace=True, shift=shift)
    _fftn(ct, axes=(0,), inplace=True, shift=shift)
    c_np = reference_fftn(c2, axes=(-1,), shift=shift)
    ct_np = reference_fftn(ct2, axes=(0,), shift=shift)
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    yield sum_of_sqr_comp(ct, ct_np, 'total energy not equal')

@parametric
def test_simple_multi_fft_1_F():
    shift = 1
    dt = 'F'
    c = np.outer(np.ones(64), ref_1D_32hz(shift, dt))
    c2 = c.copy()
    ct = c.copy().T
    ct2 = c.copy().T
        
    _fftn(c, axes=(-1,), inplace=True, shift=shift)
    _fftn(ct, axes=(0,), inplace=True, shift=shift)
    c_np = reference_fftn(c2, axes=(-1,), shift=shift)
    ct_np = reference_fftn(ct2, axes=(0,), shift=shift)
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    yield sum_of_sqr_comp(ct, ct_np, 'total energy not equal')

@parametric
def test_simple_multi_fft_0_D():
    shift = 0
    dt = 'D'
    c = np.outer(np.ones(64), ref_1D_32hz(shift, dt))
    c2 = c.copy()
    ct = c.copy().T
    ct2 = c.copy().T
        
    _fftn(c, axes=(-1,), inplace=True, shift=shift)
    _fftn(ct, axes=(0,), inplace=True, shift=shift)
    c_np = reference_fftn(c2, axes=(-1,), shift=shift)
    ct_np = reference_fftn(ct2, axes=(0,), shift=shift)
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    yield sum_of_sqr_comp(ct, ct_np, 'total energy not equal')

@parametric
def test_simple_multi_fft_1_D():
    shift = 1
    dt = 'D'
    c = np.outer(np.ones(64), ref_1D_32hz(shift, dt))
    c2 = c.copy()
    ct = c.copy().T
    ct2 = c.copy().T
        
    _fftn(c, axes=(-1,), inplace=True, shift=shift)
    _fftn(ct, axes=(0,), inplace=True, shift=shift)
    c_np = reference_fftn(c2, axes=(-1,), shift=shift)
    ct_np = reference_fftn(ct2, axes=(0,), shift=shift)
    yield sum_of_sqr_comp(c, c_np, 'total energy not equal')
    yield sum_of_sqr_comp(ct, ct_np, 'total energy not equal')


@parametric
def test_strided_1d_fft_0_F():
    shift = 0
    dt = 'F'
    rand_3d = (np.random.randn(40, 50, 60) + \
               1j*np.random.randn(40, 50, 60)).astype(dt)
    
    r1 = rand_3d.copy().transpose(0,2,1)
    r1_2 = rand_3d.copy().transpose(0,2,1)
    r2 = rand_3d.copy().transpose(1,0,2)
    r2_2 = rand_3d.copy().transpose(1,0,2)

    _fftn(r1, axes=(0,), inplace=True, shift=shift)
    _fftn(r2, axes=(1,), inplace=True, shift=shift)
    r1_np = reference_fftn(r1_2, axes=(0,), shift=shift)
    r2_np = reference_fftn(r2_2, axes=(1,), shift=shift)

    yield sum_of_sqr_comp(r1, r1_np, 'axis0 dtype='+dt)
    yield sum_of_sqr_comp(r2, r2_np, 'axis1 dtype='+dt)

@parametric
def test_strided_1d_fft_0_D():
    shift = 0
    dt = 'D'
    rand_3d = (np.random.randn(40, 50, 60) + \
               1j*np.random.randn(40, 50, 60)).astype(dt)
    
    r1 = rand_3d.copy().transpose(0,2,1)
    r1_2 = rand_3d.copy().transpose(0,2,1)
    r2 = rand_3d.copy().transpose(1,0,2)
    r2_2 = rand_3d.copy().transpose(1,0,2)

    _fftn(r1, axes=(0,), inplace=True, shift=shift)
    _fftn(r2, axes=(1,), inplace=True, shift=shift)
    r1_np = reference_fftn(r1_2, axes=(0,), shift=shift)
    r2_np = reference_fftn(r2_2, axes=(1,), shift=shift)

    yield sum_of_sqr_comp(r1, r1_np, 'axis0 dtype='+dt)
    yield sum_of_sqr_comp(r2, r2_np, 'axis1 dtype='+dt)

@parametric
def test_strided_1d_fft_1_F():
    shift = 1
    dt = 'F'
    rand_3d = (np.random.randn(40, 50, 60) + \
               1j*np.random.randn(40, 50, 60)).astype(dt)
    
    r1 = rand_3d.copy().transpose(0,2,1)
    r1_2 = rand_3d.copy().transpose(0,2,1)
    r2 = rand_3d.copy().transpose(1,0,2)
    r2_2 = rand_3d.copy().transpose(1,0,2)

    _fftn(r1, axes=(0,), inplace=True, shift=shift)
    _fftn(r2, axes=(1,), inplace=True, shift=shift)
    r1_np = reference_fftn(r1_2, axes=(0,), shift=shift)
    r2_np = reference_fftn(r2_2, axes=(1,), shift=shift)

    yield sum_of_sqr_comp(r1, r1_np, 'axis0 dtype='+dt)
    yield sum_of_sqr_comp(r2, r2_np, 'axis1 dtype='+dt)

@parametric
def test_strided_1d_fft_1_D():
    shift = 1
    dt = 'D'
    rand_3d = (np.random.randn(40, 50, 60) + \
               1j*np.random.randn(40, 50, 60)).astype(dt)
    
    r1 = rand_3d.copy().transpose(0,2,1)
    r1_2 = rand_3d.copy().transpose(0,2,1)
    r2 = rand_3d.copy().transpose(1,0,2)
    r2_2 = rand_3d.copy().transpose(1,0,2)

    _fftn(r1, axes=(0,), inplace=True, shift=shift)
    _fftn(r2, axes=(1,), inplace=True, shift=shift)
    r1_np = reference_fftn(r1_2, axes=(0,), shift=shift)
    r2_np = reference_fftn(r2_2, axes=(1,), shift=shift)

    yield sum_of_sqr_comp(r1, r1_np, 'axis0 dtype='+dt)
    yield sum_of_sqr_comp(r2, r2_np, 'axis1 dtype='+dt)


@parametric
def test_strided_2d_fft_0_F():
    shift = 0
    dt = 'F'

    rand_3d = (np.random.randn(40, 50, 60) + \
               1j*np.random.randn(40, 50, 60)).astype(dt)
    
    r1 = rand_3d.copy().transpose(0,2,1)
    r1_2 = rand_3d.copy().transpose(0,2,1)
    r2 = rand_3d.copy().transpose(1,0,2)
    r2_2 = rand_3d.copy().transpose(1,0,2)

    _fftn(r1, axes=(0,2), inplace=True, shift=shift)
    _fftn(r2, axes=(1,2), inplace=True, shift=shift)
    r1_np = reference_fftn(r1_2, axes=(0,2), shift=shift)
    r2_np = reference_fftn(r2_2, axes=(1,2), shift=shift)

    yield sum_of_sqr_comp(r1, r1_np, 'dtype = '+dt)
    yield sum_of_sqr_comp(r2, r2_np, 'dtype = '+dt)

@parametric
def test_strided_2d_fft_0_D():
    shift = 0
    dt = 'D'

    rand_3d = (np.random.randn(40, 50, 60) + \
               1j*np.random.randn(40, 50, 60)).astype(dt)
    
    r1 = rand_3d.copy().transpose(0,2,1)
    r1_2 = rand_3d.copy().transpose(0,2,1)
    r2 = rand_3d.copy().transpose(1,0,2)
    r2_2 = rand_3d.copy().transpose(1,0,2)

    _fftn(r1, axes=(0,2), inplace=True, shift=shift)
    _fftn(r2, axes=(1,2), inplace=True, shift=shift)
    r1_np = reference_fftn(r1_2, axes=(0,2), shift=shift)
    r2_np = reference_fftn(r2_2, axes=(1,2), shift=shift)

    yield sum_of_sqr_comp(r1, r1_np, 'dtype = '+dt)
    yield sum_of_sqr_comp(r2, r2_np, 'dtype = '+dt)

@parametric
def test_strided_2d_fft_1_F():
    shift = 1
    dt = 'F'

    rand_3d = (np.random.randn(40, 50, 60) + \
               1j*np.random.randn(40, 50, 60)).astype(dt)
    
    r1 = rand_3d.copy().transpose(0,2,1)
    r1_2 = rand_3d.copy().transpose(0,2,1)
    r2 = rand_3d.copy().transpose(1,0,2)
    r2_2 = rand_3d.copy().transpose(1,0,2)

    _fftn(r1, axes=(0,2), inplace=True, shift=shift)
    _fftn(r2, axes=(1,2), inplace=True, shift=shift)
    r1_np = reference_fftn(r1_2, axes=(0,2), shift=shift)
    r2_np = reference_fftn(r2_2, axes=(1,2), shift=shift)

    yield sum_of_sqr_comp(r1, r1_np, 'dtype = '+dt)
    yield sum_of_sqr_comp(r2, r2_np, 'dtype = '+dt)

@parametric
def test_strided_2d_fft_1_D():
    shift = 1
    dt = 'D'

    rand_3d = (np.random.randn(40, 50, 60) + \
               1j*np.random.randn(40, 50, 60)).astype(dt)
    
    r1 = rand_3d.copy().transpose(0,2,1)
    r1_2 = rand_3d.copy().transpose(0,2,1)
    r2 = rand_3d.copy().transpose(1,0,2)
    r2_2 = rand_3d.copy().transpose(1,0,2)

    _fftn(r1, axes=(0,2), inplace=True, shift=shift)
    _fftn(r2, axes=(1,2), inplace=True, shift=shift)
    r1_np = reference_fftn(r1_2, axes=(0,2), shift=shift)
    r2_np = reference_fftn(r2_2, axes=(1,2), shift=shift)

    yield sum_of_sqr_comp(r1, r1_np, 'dtype = '+dt)
    yield sum_of_sqr_comp(r2, r2_np, 'dtype = '+dt)



@parametric
def test_roundtrip_inplace_0_F():
    shift = 0
    dt = 'F'
    grid = np.arange(128)
    mu = 43.
    stdv = 3.
    g = (np.exp(-(grid-mu)**2 / (2*stdv**2)) / (2*np.pi*stdv**2)).astype(dt)
    g2 = g.copy()
    g_bkp = g.copy()
    _fftn(g, inplace=True, shift=shift)
    gw_np = reference_fftn(g2, shift=shift, axes=(0,))
    yield npt.assert_array_almost_equal(
        g, gw_np, err_msg='differs from numpy fft ref, shift=%d'%shift
        )
    yield sum_of_sqr_comp(g, gw_np)

    _ifftn(g, inplace=True, shift=shift)
    yield npt.assert_array_almost_equal(
        g_bkp, g, err_msg='roundtrip transforms diverge, shift=%d'%shift
        )
    yield sum_of_sqr_comp(g_bkp, g)

@parametric
def test_roundtrip_inplace_1_F():
    shift = 1
    dt = 'F'
    grid = np.arange(128)
    mu = 43.
    stdv = 3.
    g = (np.exp(-(grid-mu)**2 / (2*stdv**2)) / (2*np.pi*stdv**2)).astype(dt)
    g2 = g.copy()
    g_bkp = g.copy()
    _fftn(g, inplace=True, shift=shift)
    gw_np = reference_fftn(g2, shift=shift, axes=(0,))
    yield npt.assert_array_almost_equal(
        g, gw_np, err_msg='differs from numpy fft ref, shift=%d'%shift
        )
    yield sum_of_sqr_comp(g, gw_np)

    _ifftn(g, inplace=True, shift=shift)
    yield npt.assert_array_almost_equal(
        g_bkp, g, err_msg='roundtrip transforms diverge, shift=%d'%shift
        )
    yield sum_of_sqr_comp(g_bkp, g)

@parametric
def test_roundtrip_inplace_0_D():
    shift = 0
    dt = 'D'
    grid = np.arange(128)
    mu = 43.
    stdv = 3.
    g = (np.exp(-(grid-mu)**2 / (2*stdv**2)) / (2*np.pi*stdv**2)).astype(dt)
    g2 = g.copy()
    g_bkp = g.copy()
    _fftn(g, inplace=True, shift=shift)
    gw_np = reference_fftn(g2, shift=shift, axes=(0,))
    yield npt.assert_array_almost_equal(
        g, gw_np, err_msg='differs from numpy fft ref, shift=%d'%shift
        )
    yield sum_of_sqr_comp(g, gw_np)

    _ifftn(g, inplace=True, shift=shift)
    yield npt.assert_array_almost_equal(
        g_bkp, g, err_msg='roundtrip transforms diverge, shift=%d'%shift
        )
    yield sum_of_sqr_comp(g_bkp, g)

@parametric
def test_roundtrip_inplace_1_D():
    shift = 1
    dt = 'D'
    grid = np.arange(128)
    mu = 43.
    stdv = 3.
    g = (np.exp(-(grid-mu)**2 / (2*stdv**2)) / (2*np.pi*stdv**2)).astype(dt)
    g2 = g.copy()
    g_bkp = g.copy()
    _fftn(g, inplace=True, shift=shift)
    gw_np = reference_fftn(g2, shift=shift, axes=(0,))
    yield npt.assert_array_almost_equal(
        g, gw_np, err_msg='differs from numpy fft ref, shift=%d'%shift
        )
    yield sum_of_sqr_comp(g, gw_np)

    _ifftn(g, inplace=True, shift=shift)
    yield npt.assert_array_almost_equal(
        g_bkp, g, err_msg='roundtrip transforms diverge, shift=%d'%shift
        )
    yield sum_of_sqr_comp(g_bkp, g)

@raises(ValueError)
def test_fails_for_odd_1():
    a = np.empty((3,4,5)).astype('D')
    fft1(a)

@raises(ValueError)
def test_fails_for_odd_1():
    a = np.empty((3,4,5)).astype('D')
    fft2(a, axes=(0,1))

def test_passes_for_even():
    a = np.empty((3,4,5)).astype('D')
    try:
        fft1(a, axis=1)
        assert True, 'passed'
    except:
        assert False, 'did not attempt to transform even dimension'
    
def test_twice_odd_length():
    # functions with dimension lengths such that N_i/2 is odd need
    # special treatment in the output modulation
    a = np.random.randn(30).astype('D')
    A_ref = reference_fftn(a, shift=True)
    A_test = fft1(a, shift=True)
    assert np.allclose(A_ref, A_test), 'twice-odd length FT fails'

def test_twice_odd_length2():
    # functions with dimension lengths such that N_i/2 is odd need
    # special treatment in the output modulation
    a = np.random.randn(30,40).astype('D')
    A_ref = reference_fftn(a, axes=(0,1), shift=True)
    A_test = fft2(a, shift=True)
    assert np.allclose(A_ref, A_test), 'twice-odd length FT fails in 2D'
