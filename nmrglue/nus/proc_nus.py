"""
Iterative Soft Thresholding for Processing NUS datasets
REF: Application of Iterative Soft Thresholding for Fast
Reconstruction of NMR Data Non-uniformly Sampled with
Multidimensional Poisson Gap Scheduling
Sven G. Hyberts, Alexander G. Milbradt, Andreas B. Wagner, 
Haribabu Arthanari and Gerhard Wagner*
J Biomol NMR. 2012 52(4): 315–327. 
doi:10.1007/s10858-012-9611-z.
"""

import numpy as np

def _setzeroes(data, non_sampled_indices):
    """
    Sets the non-sampled points in the fid to zero

    Parameters
    ----------
    data : ndarray
        non uniformly sampled dataset 
    non_sampled_indices : list of integers
        list of the indices that were not sampled 
        this list can be generated from the _nonsampled_indices function

    Returns
    -------
    data : ndarray
        non uniformly sampled dataset with non-sampled points set to zero

    """
    data[non_sampled_indices] = 0 + 1j*0
    return data
    
def _nonsampled_indices(datasize, sampling):
    """
    Returns a list of non-sampled indices from the list of sampled indices

    Parameters
    ----------
    datasize : int
        size of the full dataset
    sampling : list
        list of the indices that were sampled

    Returns
    -------
    nonsampled_list : list
        list of non-sampled indices

    """
    ns_list = []
    for i in range(datasize):
        if i in sampling:
            return 1
        else:
            ns_list.append(i)
    return ns_list
            
def _threshold(data, threshold):
    """
    Sets all fourier coeffs below the threshold to zero

    Parameters
    ----------
    data : ndarray
        FT dataset generated from the NUS dataset
    threshold : float
        Percentage of max FT-coeff below which all wll be set to zero

    Returns
    -------
    data : ndarray
        FT dataset whose coefs below the threshold are all set to zero

    """
    data = data.copy()
    data[data < threshold] = 0 + 1j*0
    return data   
    
def _makecompletefid(data, sampling, size):
    """
    Takes in the raw data and generates a complete fid based on
    the sampling schedule

    Parameters
    ----------
    data : ndarray
    sampling : list
    size : size of us fid
    """
    fid = np.zeros(size).astype('complex128')
    for i in range(len(sampling)):
        fid[sampling[i]] = data[i].astype('complex128')
    return fid


def _get_l2norm(data):
    """
    Calculates the l2-norm of the data

    Parameters
    ----------
    data : ndarray
  
    Returns
    -------
    l2norm : float

    """
    l2norm = np.linalg.norm(data, 2)
    return l2norm
    
def ist(data, sampling, size, maxiter=500, cutoff=0.001, threshold=0.98):
    """
    Reconstructs the NUS dataset with IST

    Parameters
    ----------
    data : complex ndarray
        Raw NUS dataset
    sampling : int array
        Sampling schedule to match with the dataset
    size : int
        Size of the nyquist grid on which the data was sampled
    maxiter : int
        Maximum number of iterations for IST
    cutoff : float
        Convergence criteria. terminates the cycle when increase in 
        the l2 norm decreases below cutoff
    threshold : float
        Fraction below which all fourier coeffs are set to zero in each
        iteration

    Returns
    -------
    reconstructedft : complex ndarry 
        FT dataset reconstructed after IST
    l2list : list of floats
        List containing the l2 calculated at each step for each FT

    """    

    # make a complete fid 
    fid_original = _makecompletefid(data, sampling=sampling, size=size)
    
    # make a copy of the original to perform checks on
    fid_temp = fid_original.copy() 

    # initialize a array with zeros to add reconstructed data into
    reconstructured_ft = np.zeros(fid_original.size).astype('complex128')

    # make list of non sampled points
    nslist = _nonsampled_indices(datasize=size, sampling=sampling)
   
    # initialize a list to check for l2 norm convergence
    l2normlist = []
 
    # iterate till convergence is reached or maxiter are satisfied
    for i in range(maxiter):
        
        # FT the fid
        ft_temp = np.fft.fft(fid_temp)
        
        # apply thresholding
        ft_temp_thr = _threshold(ft_temp, np.max(ft_temp)*threshold)
        
        # iFT of the residual FT 
        fid_temp = np.fft.ifft(ft_temp - ft_temp_thr)
        fid_temp = _setzeroes(fid_temp, non_sampled_indices=nslist) 

        # add the thresholded FT to the dataset and break
        reconstructured_ft += ft_temp_thr

        # check convergence by comparing residual fid with original fid
        convergence = _get_l2norm(reconstructured_ft)

        # normalize the l2 norm and add to list
        if i > 0:
            l2normlist.append(convergence / startl2)
        else:
            l2normlist.append(1)
            startl2 = convergence    

        # check for convergence
        # decrease in l2-norm-increase is < cutoff 
        # to avoid unlucky terminations, average of last five
        # steps is taken to ensure cutoff is valid
        if len(l2normlist) > 6:
            if l2normlist[-1] - np.mean(l2normlist[-5:-1]) < cutoff: 
                break
        
    # shift the reconstructed ft dataset
    reconstructured_ft = np.fft.fftshift(reconstructured_ft) 
       
    return reconstructured_ft, l2normlist


from scipy.fftpack import dst, idst, dct, idct
from sklearn.linear_model import Lasso


def _make_fourier_matrices(size, sampling):
    """
    Creates truncated Fourier matrices for Linear Regression
    
    Parameters
    ----------
    size : Integer
        Length of vector for Uniformly smapled data
    sampling : dict
        Array of integers corresponding to the acquired data
        
    Returns
    -------
    Ac : ndarray
        Discrete cosine transform of Identity matrix
        with only the sampled indices
    As : ndarray
        Discrete sine transform of Identity matrix
        with only the sampled indices
           
    """
    Ac = dct(np.eye(size))[sampling] 
    As = dst(np.eye(size))[sampling]   
    return Ac, As
    
    
def _set_lasso(alpha, tol, max_iter):
    """
    Creates a Lasso Model for use with dataset
    
    Parameters
    ----------
    alpha : Float
        Weight for the l1 norm
    tol : Float
        Convergence criteria
    max_iter : Integer
        Maximum number of iterations to try before convergence

    Returns
    -------
    lc, ls : Models
        Models to be used in l1-norm fits

    """
    lassoc = Lasso(alpha=alpha, tol=tol, max_iter=max_iter)
    lassos = Lasso(alpha=alpha, tol=tol, max_iter=max_iter)

    return lassoc, lassos    

def l1norm_lasso(data, size, sampling, alpha, tol=10e-9, max_iter=20):
    
    Ac, As = _make_fourier_matrices(size=size, sampling=sampling)   
    lassoc, lassos = _set_lasso(alpha=alpha, tol=tol, max_iter=max_iter)
   
    lassoc.fit(Ac, np.real(data))
    lassos.fit(As, np.imag(data))
        
    Xhat = idct(lassoc.coef_) + 1j*idst(lassos.coef_)
    Yhat = np.fft.fftshift(np.fft.fft(Xhat))
        
   
    return Yhat, Xhat       
        
        

