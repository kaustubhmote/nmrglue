"""
Schedule Generator for Poisson Sampling
REF: Poisson-Gap Sampling and Forward Maximum Entropy Reconstruction 
for Enhancing the Resolution and Sensitivity of Protein NMR Data
Sven G. Hyberts, Koh Takeuchi, Gerhard Wagner
J. Am. Chem. Soc., 2010, 132, 2145-2147
DOI: 10.1021/ja908004w
"""

import numpy as np

def _gen_poisson_schedule(l_array, samples):
    """
    Generates gaps sampled according to poisson sampling
                                                                                
    Gaps are genarted by the following function:                                
                                                                                
    .. math::                                                                   
        f(k; \\lambda) = (\\lambda^k) * \\exp(-\\lambda) / (k!)                 
                                                                                
        \\lambda = \\Lambda * \\sin(\\theta)                                    
                                                                                
        \\theta \\ varies \\ from \\ 0 \\ to \\ \\pi/2                          

    Parameters
    ----------
    samples : np.ndarray
        array of sampling points
    l_array : np.ndarray
        the weighting function for generating possion distribution

    Returns
    -------
    nuslist : list of integers
        List of indices that must be sampled 

    """

    # generate gaps : defaults to sin function for lambda
    gaps = np.random.poisson(l_array, samples)
   
    # based on the gaps, generate list of indices to be sampled
    nuslist = np.arange(samples) + np.cumsum(gaps)
        
    return nuslist

def _check_sampling(samples, nuslist):
    """
    checks whether each indirect dimension has atleast one sampling point
    for 3D or higher datasets
    """
    rval = True

    if len(samples) > 1: # for a 3D or higher-D dataset only
        nuslist = nuslist.T
        for i, dimlist in enumerate(nuslist):
            if len(np.unique(dimlist) < samples[i]):
                rval = False
                break

    return rval

def poisson_1d(size=256, sampling=0.5, fudge=1.0, tolerance=0.02, max_iter=1000, verbosity=False, weight=1):
    """
    Generates gaps sampled according to poisson sampling that satisfies
    tolerance criteria

    Gaps are genarted by the following function:

    .. math::
        f(k; \\lambda) = (\\lambda^k) * \\exp(-\\lambda) / (k!)
    
        \\lambda = \\Lambda * \\sin(\\theta)

        \\theta \\ varies \\ from \\ 0 \\ to \\ \\pi/2   

    Parameters
    ----------
    size : int
       Total size of the Nyquist grid for uniformly sampled data
    sampling : float
       Undersampling, number between 0 and 1
    fudge :  float
        and adjustable parameter used in generating gaps
    tolerance : float
        devaition allowed for the sampling (default = 0.02)
    max_iter : int
        maximum number of iterations for generating a schedule
           

    Returns
    -------
    nuslist : list of integers
        List of indices that must be sampled 
    flist : list of floats
        List of fudge values used

    """
    
    # make an empty list to keep values of fudge    
    flist = []

    # number of samples to generate ~ number of gaps
    samples = np.floor(size * sampling).astype(int) 
 
    # generate an array of lambda values to be used for generating gaps   
    l_array = np.sin(np.linspace(0, np.pi/2, samples))**weight

    for i in range(max_iter):
        
        if verbosity:
            if i % 10*verbosity == 0:
                print('Iteration', i, 'of', max_iter)
    
        # generate a schedule   
        nlist = _gen_poisson_schedule(l_array=fudge*l_array, 
                                      samples=samples)
    
        # check whether it satisfies the criteria of tolerance
        if nlist[-1] > size-1:
            fudge = fudge - 0.05
            flist.append(fudge)
        elif nlist[-1] < (1-tolerance) * (size-1):
            fudge = fudge + 0.05
            flist.append(fudge)
        else:
            break

    # Tell the user if the tolerance criteria was not satisfied
    if i < max_iter - 1:
        if verbosity:
            print('Success! Done in', i, 'iterations')
        nlist_flag = True
        return nlist, flist, nlist_flag
    else:
        nlist_flag = False
        print("Maximum number of iterations reached.") 
        print("Either increase max_iter or increase tolerance.")
        print("Current error is ", (1 - nlist[-1]/size)*100, "%.")
        return nlist, flist, nlist_flag


def poisson(sampling=0.1, samples=256, tolerance=0.001,
            max_iter=500, verbosity=False, check_nd_sampling=True, 
            weight=1):
    """
    Poisson sampling for more 3 or more dimensions (no upper limit)
    for the number of dimensions

    Gaps are genarted by the following function:                
                                                                 
    .. math::                                                   
         f(k; \\lambda) = (\\lambda^k) * \\exp(-\\lambda) / (k!) 
                                                                 
         \\lambda = \\Lambda * \\sin(\\theta)                    
                                                                 
         \\theta \\ varies \\ from \\ 0 \\ to \\ \\pi/2          

    A 1D array is generated and sorted according to distance from origin
    Then a linear poission sampling applied.

    Parameters
    ----------
    sampling : float
        Fraction of the total points to be sampled (between 0 and 1)
    samples : int for 2D, list of integers for 3D or higher
        number of points to be sampled on the nyquist grid for each dimension
    tolerance : float
        error in accepting a sampling
    max_iter : int
        number if iterations 

    Returns
    -------
    nuslist : ndarray
        array of indices to be sampled

    """
    from itertools import product

    if isinstance(samples, int):
        samples = [samples]

    # coordinates for each axis
    coords = [range(npoints) for npoints in samples]

    # index list
    index_list = list(product(*coords))

    # sort the index list by virtual distance from 0    
    index_list = np.array(sorted(index_list, 
        key=lambda x: np.linalg.norm(np.array(x))))                       
            
    # get indices to sample on a the index list
    nusindices = poisson_1d(size=len(index_list), 
                            sampling=sampling, 
                            tolerance=tolerance, 
                            max_iter=max_iter, 
                            verbosity=verbosity,
                            weight=weight) 

    # return the indices that need to be sampled
    nuslist = np.array([index_list[coord] for coord in nusindices[0]])

    # include the last point in the list
    nuslist[-1] = index_list[-1]

    if check_nd_sampling and nusindices[2]:
        nd_sampling = _check_sampling(samples, nuslist)
    else:
        nd_sampling = True

    if nd_sampling:
        return nuslist 
    elif check_nd_sampling:
        print('N-dimensional sampling error. Restarting ...')
        return poisson(sampling, samples, tolerance,
                               max_iter, verbosity)
    else:
        print('N-dimensional sampling error. Ignored')
        return nuslist


