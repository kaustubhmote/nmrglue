
"""
Schedule Generator for Poisson Sampling
REF: Poisson-Gap Sampling and Forward Maximum Entropy Reconstruction 
for Enhancing the Resolution and Sensitivity of Protein NMR Data
Sven G. Hyberts, Koh Takeuchi, Gerhard Wagner
J. Am. Chem. Soc., 2010, 132, 2145-2147
DOI: 10.1021/ja908004w
"""

import numpy as np

def _gen_poisson_schedule(size=256, sampling=0.5, fudge=1.0):
    """
    Generates gaps sampled according to poisson sampling
                                                                                
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
    fudge :  an adjustable parameter to generate gaps

    Returns
    -------
    nuslist : list of integers
        List of indices that must be sampled 

    """

    # number of samples to generate ~ number of gaps
    samples = np.floor(size * sampling).astype(int) 

    # generate an array of lambda values to be used for generating gaps   
    lambdaarray = np.linspace(0, np.pi/2, samples)

    # make an array of zeros to be used for gaps
    gaps = np.zeros(samples)

    # generate gaps : defauilts to sin function for lambda
    for i in range(samples):
        gaps[i] += np.random.poisson(fudge * np.sin(lambdaarray[i]), 1)       
   
    # based on the gaps, generate liost of indices to be sampled
    nuslist=[0] 
    for i in range(1, len(gaps)):
        indexi = nuslist[i-1] + 1 + gaps[i]
        nuslist.append(indexi)
        
    return nuslist


def poisson(size=256, sampling=0.5, fudge=1.0, tolerance=0.02, max_iter=1000):
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
    
    for i in range(max_iter):
        # generate a schedule 
        nlist = _gen_poisson_schedule(size=size, sampling=sampling, 
                                      fudge=fudge)
    
        # check whether it satisfies the criteria of tolerance
        if nlist[-1] >  size:
            fudge = fudge - 0.05
            flist.append(fudge)
        elif nlist[-1] < (1-tolerance) * size:
            fudge = fudge + 0.05
            flist.append(fudge)
        else:
            break

    # Tell the user if the tolerance criteria was not satisfied
    if i < 1000:
        return nlist, flist
    else:
        print("Maximum number of iterations reached.") 
        print("Either increase max_iter or increase tolerance.")
        print("Current error is ", (1 - nlist[-1]/size)*100, "%.")
        return nlist, flist


def poisson_nD(sampling=0.1, samples=[32, 32], tolerance=0.001,
               max_iter=500):
    """
    Poisson sampling for more than 2 dimensions (3, 4 or 5)

     Gaps are genarted by the following function:                
                                                                 
     .. math::                                                   
         f(k; \\lambda) = (\\lambda^k) * \\exp(-\\lambda) / (k!) 
                                                                 
         \\lambda = \\Lambda * \\sin(\\theta)                    
                                                                 
         \\theta \\ varies \\ from \\ 0 \\ to \\ \\pi/2          

    A 2D array is generated and sorted according to distance from [0,0]
    Then a linear poission sampling applied.

    Parameters
    ----------
    sampling : float
        Fraction of the total points to be sampled (between 0 and 1)
    samples : list of integers
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
# TODO: recursive list implementation

    if len(samples) == 2:   # 3D dataset
        index_list = [ [i, j] 
            for i in list(range(samples[0])) 
            for j in list(range(samples[1])) 
            ]
        index_list = np.array( sorted(index_list, key=lambda x: 
                            x[0]**2 + x[1]**2) ) 

    if len(samples) == 3:  # 4D dataset                                                      
        index_list = [ [i, j, k]                                                    
            for i in list(range(samples[0]))                                    
            for j in list(range(samples[1]))                                    
            for k in list(range(samples[2]))                                    
            ]                                                                   
        index_list = np.array( sorted(index_list, key=lambda x:                          
                            x[0]**2 + x[1]**2 + x[2]**2) )                        

    if len(samples) == 4:   # 5D dataset                                                      
        index_list = [ [i, j, k]                                                    
            for i in list(range(samples[0]))                                    
            for j in list(range(samples[1]))                                    
            for k in list(range(samples[2]))
            for l in list(range(samples[3]))                                    
            ]                                                                   
        index_list = np.array( sorted(index_list, key=lambda x:                          
                            x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) )                       
            
    
    for i in range(500):
        nusindices = poisson(size=len(index_list), sampling=sampling, 
                        tolerance=tolerance, max_iter=max_iter)[0] 

        nuslist = np.array(index_list[nusindices])
            for j in range(samples[0]):
                if j not in nuslist.T[0]



    # TODO: check whether each index has atleast 1 sampling point

    return nuslist 

   

                    











   
