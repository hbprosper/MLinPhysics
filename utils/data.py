# ----------------------------------------------------------------------------
# Machine Learning in Physics Course at Florida State University.
# This contains work developed with Claire David and Tlotlo Oepeng in the
# contect PINN black hole project.
#
# Harrison B. Prosper
# Created: Mon Aug 25 2025
# ----------------------------------------------------------------------------
import os, sys, re
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
import scipy.stats as st
# ----------------------------------------------------------------------------
# Using a Sobol sequence to created a sample of points
# ----------------------------------------------------------------------------
class SobolSample(np.ndarray):
    def __new__(cls,
                 lower_bounds,
                 upper_bounds,
                 num_points_exp=17, # of points = 2^num_points_exp
                 verbose=1):
       
        # Generate Sobol points in the unit D-cube and scale to bounds
        D = len(lower_bounds)
        sampler = st.qmc.Sobol(d=D, scramble=True)
        sample  = sampler.random_base2(m=num_points_exp) 
        sample  = st.qmc.scale(sample, lower_bounds, upper_bounds)

        if verbose:
            print("SobolSample")
            print(f"  {2**num_points_exp} Sobol points created in unit {D}-cube.")

        # Cast the numpy array to the type SobolSample
        sample = np.asarray(sample).view(cls)
        return sample
# ----------------------------------------------------------------------------
# Use uniform sampling to create a sample of points
# ----------------------------------------------------------------------------
class UniformSample(np.ndarray):
    def __new__(cls,
                 lower_bounds,
                 upper_bounds,
                 num_points,   # of points
                 verbose=1):

        # Generate points in the unit D-cube and scale to bounds
        D = len(lower_bounds)
        sample = np.random.uniform(0, 1, D*num_points).reshape((num_points, D))
        sample = st.qmc.scale(sample, lower_bounds, upper_bounds)
        
        if verbose:
            print("UniformSample")
            print(f"  {num_points} uniformly sampled points created in unit {D}-cube.")

        # Cast the numpy array to the type UniformSample
        sample = np.asarray(sample).view(cls)
        return sample
# ---------------------------------------------------------------------------
# Custom Dataset that takes (N, D) array of N points in the unit D-cube,
# Taken from AIMS PINN project
# ---------------------------------------------------------------------------
class Dataset(td.Dataset):
    
    PINNSPLIT=(0, True, False)
    
    def __init__(self, data, start, end,
                 # Or a 3-tuple: (column, requires_grad, requires_grad)
                 split_data=None, 
                 random_sample_size=None,
                 device=torch.device("cpu"),
                 verbose=1):
        
        super().__init__()

        self.verbose = verbose

        # # Check that we have the right data types
        # if not isinstance(data, (SobolSample, UniformSample)):
        #     raise TypeError('''
        #     The object at argument 1 must be an instance of SobolSample
        #     or UniformSample
        #     ''')

        if random_sample_size == None:
            tdata = torch.Tensor(data[start:end])
        else:
            # create a random sample from items in the specified range (start, end)
            assert(type(random_sample_size) == type(0))
            
            length  = end - start
            assert(length > 0)
            
            indices = torch.randint(0, length-1, size=(random_sample_size,))
            tdata   = torch.Tensor(data[indices])

        # check whether to split data
        if type(split_data) != type(None):
            
            self.split = True
            
            try:
                col, req_grad1, req_grad2 = split_data
            except:
                raise ValueError('split_data should be a 3-tuple!')

            # get data shape
            if tdata.ndim < 2:
                tdata = tdata.view(-1, 1)
                
            nrows, ncols = tdata.shape
            
            x = tdata[:, :col]
            if req_grad1:
                self.x = x.reshape(-1, col+1).requires_grad_().to(device)
            else:
                self.x = x.to(device)
                
            z = tdata[:, col+1:]
            if req_grad2:
                self.z = z.reshape(-1, ncols-col-1).requires_grad_().to(device)
            else:
                self.z = z.to(device)
        else:
            self.split = False
            # do not split data
            self.x = tdata.requires_grad_().to(device)

        if verbose:
            print('Dataset')
            print(f"  shape of x: {self.x.shape}")
            if self.split:
                print(f"  shape of z: {self.z.shape}")
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.split:
            return self.x[idx], self.z[idx]
        else:
            return self.x[idx]
# ---------------------------------------------------------------------------
# Custom DataLoader that is much faster than the default usage of the PyTorch
# DataLoader.
# ---------------------------------------------------------------------------
class DataLoader:
    '''
    A data loader that is much faster than the default PyTorch DataLoader.
    
    Notes:
    
    1. If used, "sampler" must be a PyTorch sampler and the arguments
       batch_size, shuffle, num_iterations are ignored.
       
    2. If num_iterations is specified, it is assumed that this is the
       desired maximum number of iterations, maxiter, per for-loop. 
       The flag shuffle is automatically set to True and an internal 
       count, defined by shuffle_step = floor(len(dataset) / batch_size) 
       is computed. The indices for accessing items from the dataset 
       are shuffled every time the following condition is True

           itnum % shuffle_step == 0,

       where itnum is an internal counter that keeps track of the iteration
       number. If num_iterations is not specified (the default), then
       the maximum number of iterations, maxiter = shuffle_step.
       
       This data loader, unlike the PyTorch data loader does not provide the 
       option to return the last batch if the latter is shorter than batch_size.
    '''
    def __init__(self, dataset, 
                 batch_size=None,
                 num_iterations=None,
                 verbose=1,
                 debug=0,
                 shuffle=False):

        # Note: sampler and (batch_size, shuffle, num_iterations) are 
        # mutually exclusive
        self.dataset = dataset
        self.size    = batch_size
        self.niterations = num_iterations
        self.verbose = verbose
        self.debug   = debug
        self.shuffle = shuffle
        
        # Not using a sampler, so need batch_size
        if self.size == None:
            raise ValueError("you must specify a batch_size!")
            
        # If shuffle, then shuffle the dataset every shuffle_step iterations
        self.shuffle_step = int(len(dataset) / self.size)

        if self.niterations != None:
            # The user has specified the maximum number of iterations 
            assert(type(self.niterations)==type(0))
            assert(self.niterations > 0)
            
            self.maxiter = self.niterations
            
            # IMPORTANT: shuffle indices every self.shuffle_step iterations
            self.shuffle = True

            if self.verbose:
                print('DataLoader')
                print('  Maximum number of iterations has been specified')
                print(f'  maxiter:      {self.maxiter:10d}')
                print(f'  batch_size:   {self.size:10d}')
                print(f'  shuffle_step: {self.shuffle_step:10d}')
                
        elif len(dataset) > self.size:
            self.maxiter = self.shuffle_step
            
            if self.verbose:
                print('DataLoader')
                print(f'  maxiter:      {self.maxiter:10d}')
                print(f'  batch_size:   {self.size:10d}')
                print(f'  shuffle_step: {self.shuffle_step:10d}')

        else:
            # Note: this could be = 2 for a 2-tuple of tensors!
            self.size = len(dataset)
            self.shuffle_step = 1
            self.maxiter = self.shuffle_step

        assert(self.maxiter > 0)

        # initialize iteration number
        # IMPORTANT: must start at -1 so that itnum goes from
        # 0 to size - 1
        self.itnum = -1

    # Tell Python to make objects of type DataLoader iterable
    def __iter__(self):
        return self

    # This method implements and terminates iterations
    def __next__(self): 

        # IMPORTANT: increment iteration number here!
        self.itnum += 1

        if self.itnum < self.maxiter:

            if self.sampler:
                # Create a new tensor indexing dataset using the sequence
                # returned by the PyTorch sampler
                indices = list(sampler)[0]
                return self.dataset[indices]

            elif self.shuffle:
                # Create a new tensor indexing dataset via a random
                # sequence of indices
                jtnum = self.itnum % self.shuffle_step
                if jtnum == 0:
                    if self.debug > 0:
                        print(f'DataLoader/shuffling indices @ index {self.itnum}')
      
                    self.indices = torch.randperm(len(self.dataset))

                start = jtnum * self.size
                end = start + self.size
                indices = self.indices[start:end]
                return self.dataset[indices]
                
            else:
                # Create a new tensor directly indexing dataset
                start = self.itnum * self.size
                end = start + self.size
                return self.dataset[start:end]
        else:
            # Terminate iteration and reset iteration counter
            # IMPORTANT: must start at -1 so that itnum goes from
            # 0 to size - 1
            self.itnum = -1
            raise StopIteration

    def __len__(self):
        return self.maxiter