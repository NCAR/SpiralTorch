"""
loss functions that can be employed with sparsa optimization
"""
import torch
from typing import List

pi_const = 3.14159265359


# class loss_fnc_base:
#     def __init__(self,fwd_var_lst:List[str],noise_var_lst:List[str],name_str:str):
#         self.fwd_var_lst = fwd_var_lst
#         self.noise_var_lst = noise_var_lst
#         self.name_str = name_str
#     def __str__(self):
#         return self.name_str
    
#     def forward(self,**fwd_model_vars,**noise_model_vars):
#         return None



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
    return channel_weight*(channel_mask*(shot_count*y_mean_est-counts*torch.log(y_mean_est))).sum()

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

    return channel_weight*(channel_mask*(active_time*y_mean_est-counts*torch.log(y_mean_est))).sum()

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
    return channel_weight*(channel_mask*(0.5*torch.log(2.0*pi_const*y_var_est*shot_count**2)+(y_mean_est*shot_count-counts)**2/(2*y_var_est*shot_count**2))).sum()
    

def gaus_mean_fn(
        y_mean_est:torch.tensor=None,
        counts:torch.tensor=None,
        shot_count:torch.tensor=1.0,
        variance:torch.tensor=None,
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

    return channel_weight*(channel_mask*((y_mean_est*shot_count-counts)**2/(2*variance*shot_count**2))).sum()