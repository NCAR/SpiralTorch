"""
loss functions that can be employed with sparsa optimization
"""
import torch
from typing import List

pi_const = 3.14159265359




class loss_fnc_base:
    def __init__(self,name_str:str):
        # self.fwd_var_lst = fwd_var_lst
        # self.noise_var_lst = noise_var_lst
        self.forward = None
        self.name_str = None
        
        if name_str in noise_model_dct.keys():
            self.name_str = name_str
            self.forward = noise_model_dct[self.name_str]['function']
        else:
            print("loss_fnc_base in loss.py initialized with distribution "+name_str)
            print("This distribution does not exist.  Possible options are:")
            for key in noise_model_dct.keys():
                print(key,end=", ")
        
    def __str__(self):
        return self.name_str
    
    # def forward(self,**fwd_model_vars,**noise_model_vars):
    #     return None



def pois_loss_fn(
        y_mean_est:torch.tensor=None,
        counts:torch.tensor=None,
        shot_count:torch.tensor=1.0,
        channel_mask:torch.tensor=1.0,
        channel_weight:torch.tensor=1.0)->torch.tensor:
    """
    single channel Poisson loss for loss function definitions.
    Any argument with default None needs to be provided

    y_mean_est:torch.tensor=None,
        from forward model
    counts:torch.tensor=None,
        from observations
        Photon counts in each histogram bin
    shot_count:torch.tensor=None,
        from observations
        number of laser shots.  can also include 
        bin accumulation time and other forward model scalars
    channel_mask:torch.tensor=1.0,
        from observations
        pixel based masking or weighting
    channel_weight:torch.tensor=1.0
        from observations
        total channel weighting
    
    """
    return channel_weight*(channel_mask*(shot_count*y_mean_est-counts*torch.log(y_mean_est)))

def pois_bg_loss_fn(
        y_mean_est:torch.tensor=None,
        counts:torch.tensor=None,
        bg:torch.tensor=None,
        shot_count:torch.tensor=1.0,
        channel_mask:torch.tensor=1.0,
        channel_weight:torch.tensor=1.0)->torch.tensor:
    """
    single channel Poisson loss for loss function definitions.
    Any argument with default None needs to be provided

    y_mean_est:torch.tensor=None,
        from forward model
    counts:torch.tensor=None,
        from observations
        Photon counts in each histogram bin
    bg:torch.tensor=None,
        from observations, estimated background for
        this set of observations
        this is expected sum sum with the y_mean_est
    shot_count:torch.tensor=None,
        from observations
        number of laser shots.  can also include 
        bin accumulation time and other forward model scalars
    channel_mask:torch.tensor=1.0,
        from observations
        pixel based masking or weighting
    channel_weight:torch.tensor=1.0
        from observations
        total channel weighting
    
    """
    return channel_weight*(channel_mask*(shot_count*(y_mean_est+bg)-counts*torch.log(y_mean_est+bg)))

def deadtime_loss_fn(
        y_mean_est:torch.tensor=None,
        counts:torch.tensor=None,
        active_time:torch.tensor=None,
        channel_mask:torch.tensor=1.0,
        channel_weight:torch.tensor=1.0)->torch.tensor:
    """
    single channel deadtime loss for loss function definitions.
    Any argument with default None needs to be provided 

    y_mean_est:torch.tensor=None,
        from forward model
    
    counts:torch.tensor=None,
        from observations
        Photon counts in each histogram bin
    active_time:torch.tensor=None,
        from observations
        detector active_time for each histogram bin
    channel_mask:torch.tensor=1.0,
        from observations
        pixel based masking or weighting
    channel_weight:torch.tensor=1.0
        from observations
        total channel weighting  
    """

    return channel_weight*(channel_mask*(active_time*y_mean_est-counts*torch.log(y_mean_est)))

def deadtime_bg_loss_fn(
        y_mean_est:torch.tensor=None,
        counts:torch.tensor=None,
        active_time:torch.tensor=None,
        bg:torch.tensor=None,
        channel_mask:torch.tensor=1.0,
        channel_weight:torch.tensor=1.0)->torch.tensor:
    """
    single channel deadtime loss for loss function definitions.
    Any argument with default None needs to be provided 

    y_mean_est:torch.tensor=None,
        from forward model
    
    counts:torch.tensor=None,
        from observations
        Photon counts in each histogram bin
    active_time:torch.tensor=None,
        from observations
        detector active_time for each histogram bin
    bg:torch.tensor=None,
        from observations, estimated background for
        this set of observations
        this is expected sum sum with the y_mean_est
    channel_mask:torch.tensor=1.0,
        from observations
        pixel based masking or weighting
    channel_weight:torch.tensor=1.0
        from observations
        total channel weighting  
    """

    return channel_weight*(channel_mask*(active_time*(y_mean_est+bg)-counts*torch.log(y_mean_est+bg)))

def gaus_fn(
        y_mean_est:torch.tensor=None,
        y_var_est:torch.tensor=None,
        counts:torch.tensor=None,
        shot_count:torch.tensor=1.0,
        channel_mask:torch.tensor=1.0,
        channel_weight:torch.tensor=1.0)->torch.tensor:
    """
    y_mean_est:torch.tensor=None,
        from forward model
    y_var_est:torch.tensor=None,
        from forward model
    
    counts:torch.tensor=None,
        from observations
        Photon counts in each histogram bin
    shot_count:torch.tensor=None,
        from observations
        number of laser shots.  can also include 
        bin accumulation time and other forward model scalars
    channel_mask:torch.tensor=1.0,
        from observations
        pixel based masking or weighting
    channel_weight:torch.tensor=1.0
        from observations
        total channel weighting 
    """ 
    return channel_weight*(channel_mask*(0.5*torch.log(2.0*pi_const*y_var_est*shot_count**2)+(y_mean_est*shot_count-counts)**2/(2*y_var_est*shot_count**2)))
    

def gaus_mean_fn(
        y_mean_est:torch.tensor=None,
        counts:torch.tensor=None,
        shot_count:torch.tensor=1.0,
        variance:torch.tensor=1.0,
        channel_mask:torch.tensor=1.0,
        channel_weight:torch.tensor=1.0)->torch.tensor:
    """
    # Gaussian noise with known variance

    y_mean_est:torch.tensor=None,
        from forward model
    
    counts:torch.tensor=None,
        from observations
        Photon counts in each histogram bin
    shot_count:torch.tensor=None,
        from observations
        number of laser shots.  can also include 
        bin accumulation time and other forward model scalars
    variance:torch.tensor=None,
        from observations
        known variance of the observed signal
    channel_mask:torch.tensor=1.0,
        from observations
        pixel based masking or weighting
    channel_weight:torch.tensor=1.0
        from observations
        total channel weighting 
    """ 

    return channel_weight*(channel_mask*((y_mean_est*shot_count-counts)**2/(2*variance*shot_count**2)))


noise_model_dct = {
    'poisson':{
        'forward_model_inputs':['y_mean_est'],
        'noise_model_inputs':['counts'],
        'noise_optional':['shot_count'],
        'fixed_optional':['channel_mask','channel_weight'],
        'function':pois_loss_fn,
    },
    'poisson_bg':{
        'forward_model_inputs':['y_mean_est'],
        'noise_model_inputs':['counts','bg'],
        'noise_optional':['shot_count'],
        'fixed_optional':['channel_mask','channel_weight'],
        'function':pois_bg_loss_fn,
    },
    'deadtime':{
        'forward_model_inputs':['y_mean_est'],
        'noise_model_inputs':['counts','active_time'],
        'noise_optional':[],
        'fixed_optional':['channel_mask','channel_weight'],
        'function':deadtime_loss_fn,
    },
    'deadtime_bg':{
        'forward_model_inputs':['y_mean_est'],
        'noise_model_inputs':['counts','active_time','bg'],
        'noise_optional':[],
        'fixed_optional':['channel_mask','channel_weight'],
        'function':deadtime_bg_loss_fn,
    },
    'gaussian':{
        'forward_model_inputs':['y_mean_est','y_var_est'],
        'noise_model_inputs':['counts'],
        'noise_optional':['shot_count'],
        'fixed_optional':['channel_mask','channel_weight'],
        'function':gaus_fn,
    },
    'gaussian_mean':{
        'forward_model_inputs':['y_mean_est'],
        'noise_model_inputs':['counts'],
        'noise_optional':['shot_count','variance'],
        'fixed_optional':['channel_mask','channel_weight'],
        'function':gaus_mean_fn,
    },
}