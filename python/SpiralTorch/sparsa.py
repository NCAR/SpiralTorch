"""
functions to implement FISTA
in PyTorch
"""

import torch
import numpy as np
import time
# from torch.special import gammaln
import xarray as xr

import fista

from typing import List, Dict, Callable

import copy

torch.autograd.set_detect_anomaly(True)

def poisson_thin(signal:np.ndarray,n:int=2)->List:
    """
    Poisson thin an array of observations
    inputs:
        signal - observations of Poisson random variables
        n - number of thinned profiles desired
    
    returns:
        list of the thinned data
    """

    thin_set = []
    # save the data type so we can restore it after thinning
    signal_dtype = signal.dtype 
    s = signal.astype(int)
    thin_total = np.zeros(s.shape)

    for ai in range(n-1):
        thin_data = np.random.binomial(s.flatten(),1.0/(n-ai),size=s.size).reshape(s.shape)
        thin_set += [thin_data.astype(signal_dtype)]
        thin_total += thin_data
        s -= thin_data

    thin_set += [s.astype(signal_dtype)]

    return thin_set

def create_x_kwargs(x:dict,key:str):
    """
    Create a dict for all the variables
    not optimized in the SpaRSA subproblem
    x: dictionary containing all estimated variables
    key: key to be removed
    """
    x_new = x.copy()
    x_new.pop(key)
    return x_new


class sparsa_torch_autograd:
    """
    solves a spiral/sparsa subproblem for an individual
    estimated variable.

    This routine uses the pytorch autograd feature to
    compute gradients which may be slower than functions
    based on analytical gradient calculations.

    This class must be initialized through a series of 
    steps.

    subprob = sparsa_torch_autograd(device,dtype)
    subprob.load_fit_parameters(x_dct, y_obs_dct_lst, fwd_model_dct_lst, 'backscatter')
    subprob.set_penalty_weight(4.0)  # defaults to 1
    subprob.set_loss_poisson()  # set the loss function
    subprob.set_fista()   # compile the jit fista script

    in the above example
        x_dct is a dictionary of all optimized variables
        y_obs_lst is a list of the photon count observations
        fwd_model_lst is a list of the forward model functions
            corresponding to the above observations
        'backscatter' is the name of the fit variable for this
            subproblem



    Initialization methods
    """

    # TODO change y_obs_lst to y_obs_dct_lst such that
    # each sublist is a set of observational terms needed for the loss function.
    # In this way any number of terms can be passed in.
    # e.g. [{'counts':ch1_counts,'active_time':ch1_active_time}, {'counts':ch2_counts,'active_time':ch2_active_time},...]
    # then the loss function is called on an element as loss(**y_obs_dct_lst[idx])
    # such that the loss function definition uses keyword arguments
    
    def __init__(self,device,dtype):
        self.device = device
        self.dtype = dtype
        
        """
        set the default values for the 
        optimization routine.  These can be
        set to non-default values by calling
        the methods individually
        """
        
        # set default value for max iterations
        self.set_max_iter(100)
        self.min_iter = 1

        self.pi = torch.tensor(np.pi,device=self.device,dtype=self.dtype)

        # set the default number of seconds before
        # the sparsa loop times out
        self.timeout = 1000 
        
        # set default value for eta
        self.set_eta(2.0)
        
        # set default value for FISTA criteria
        self.set_sigma(1e-5)
        
        # set default value for terminating SpaRSA
        self.set_eps(1e-5)
        
        # set default value for TV penalty
        self.set_penalty_weight(1.0)
        
        # set alpha
        self.set_alpha(1.0)
        self.set_alpha_min(1.0)
        self.set_alpha_max(1e20)
        
        # set default on history used for
        # alpha acceptance criteria
        self.M_hist = 100

        # initialize the total subproblem step to zero0
        self.total_rel_step = 0.0

        # set lower and upper bounds to none before they are set
        self.x_lb = None
        self.x_ub = None

        # TODO Remove
        # set the channel weights to be empty to start
        # self.chan_weight_lst = None
        
    def set_max_iter(self,max_iter:int):
        self.max_iter = max_iter
        self.objective_tnsr = torch.zeros(max_iter+1,device=self.device, dtype=self.dtype)
        self.rel_step_tnsr = torch.zeros(max_iter+1,device=self.device, dtype=self.dtype)
        self.alpha_tnsr = torch.zeros(max_iter+1,device=self.device, dtype=self.dtype)
    
    def set_eta(self,eta:float):
        self.eta = torch.tensor(eta,device=self.device, dtype=self.dtype)
        
    def set_sigma(self,sigma:float):
        self.sigma = torch.tensor(sigma,device=self.device, dtype=self.dtype)
        
    def set_alpha(self,alpha:float):
        self.alpha = torch.tensor(alpha,device=self.device, dtype=self.dtype)

    def set_alpha_min(self,alpha_min:float):
        self.alpha_min = torch.tensor(alpha_min,device=self.device, dtype=self.dtype)

    def set_alpha_max(self,alpha_max:float):
        self.alpha_max = torch.tensor(alpha_max,device=self.device, dtype=self.dtype)
        
    def set_eps(self,eps:float):
        self.eps = torch.tensor(eps,device=self.device, dtype=self.dtype)
    
    def set_penalty_weight(self,tau:float):
        self.penalty_weight = torch.tensor(tau,device=self.device, dtype=self.dtype)

    # TODO Remove
    # def set_channel_weights(self,channel_weights:List[float]):
    #     self.chan_weight_lst = []
    #     for ch_weight in channel_weights:
    #         self.chan_weight_lst.append(torch.tensor(ch_weight, device=self.device, dtype=self.dtype))

    # TODO Remove
    # def set_channel_masks(self,channel_masks:List[torch.tensor]):
    #     self.channel_mask_lst = []  # set mask to default value
    #     for ch_mask in channel_masks:
    #         self.channel_mask_lst.append(ch_mask.type(self.dtype))

        
    def load_fit_parameters(self,x:Dict[str,torch.tensor],
                            y_obs_dct_lst:List[Dict[str,torch.tensor]],
                            fwd_model_lst:List[Callable],
                            fit_var:str,
                            model_scale:float=1.0):
        """
        Set all the fitting terms
            x: Dict[torch.tensor]
                dictionary containing all the fit variables as
                torch tensors
            
            y_obs_dct_lst: List[Dict[str:torch.tensor]]
                observation and optimization terms for each channel as 
                torch tensors.  The order of this list must
                correspond to fwd_model_lst
            
            fwd_model_lst: List[Callable]
                list of functions that evaluate the forward
                model for each channel for a given **x (dictionary)
                input.  These functions will be converted to
                Functions with a single x (tensor) input in this
                method.
                The order of this list
                must correspond with y_obs_lst.
            
            fit_var: str
                string identifying the variable we are solving
                for in this subproblem.
            
            # TODO: get rid of this because this should be handled with the loss arguments
            model_scale: float (defaults 1.0)
                set the forward model multiplier to account for
                Poisson thinning
        """
        
        # store the functions used to compute the
        # forward models using the dictionary input
        self.base_fwd_model_lst = fwd_model_lst

        # store which variable this subproblem is
        # configured to optimize
        self.fit_var = fit_var
        
        # asve the observation data
        # force it to be the correct data type
        self.y_obs_dct_lst = []
        self.channel_mask_lst = []  # set mask to default value
        for y_obs in y_obs_lst:
            self.y_obs_lst.append(y_obs.type(self.dtype))
            self.channel_mask_lst.append(torch.ones(y_obs.shape,dtype=self.dtype,device=self.device))

        # set the channel weights to 1 if they have not
        # been initialized
        if self.chan_weight_lst is None:
            self.chan_weight_lst = [torch.ones(1,dtype=self.dtype,device=self.device)]*len(self.y_obs_lst)

        # setup the optimization variable
        # and subproblem functions
        self.set_fit_param(x,model_scale=model_scale)
        
    def set_loss_poisson(self):
        self.loss_fn = self.pois_loss_fn
        # print("spiral loss function set to \'poisson\'")

    def set_loss_deadtime(self,y_active_lst:List[torch.tensor]):
        # requires that the active time histograms for each channel
        # are passed into the this function
        self.y_active_lst = []
        for y_act in y_active_lst:
            self.y_active_lst.append(y_act.type(self.dtype))
        self.loss_fn = self.deadtime_loss_fn
        # print("spiral loss function set to \'deadtime\'")

    def set_loss_gaus(self):
        self.loss_fn = self.gaus_fn
        # print("spiral loss function set to \'gaussian\'")

    def set_loss_gaus_approx(self):
        self.loss_fn = self.gaus_approx_fn

        # set estimated variance weighting
        for idx, y_obs in enumerate(self.y_obs_lst):
            self.channel_mask_lst[idx]*=1.0/torch.maximum(1.0,y_obs)
        # print("spiral loss function set to \'gaussian approx\'")
        
    def set_fista(self,fista_ver_str, order:int=1):
        """
        compile fista jit.script
        order:int
            set the subproblem order
            defaults to first order (order=1)
            but it can also be set to second order (order=2)
        """
        if order == 2:
            # print("Fista configured for 2nd Order")
            self.fista = torch.jit.trace(fista.solve_FISTA_subproblem_2ndOrder_jit,(self.x,self.alpha,
                                                self.x_lb,self.x_ub))
            self.pen_fn = self.pen_fn_2ndOrder
        elif order == 3:
            # print("Fista configured for 2nd Order in range, 1st order in time")
            # special case where the 0 dimension (time) is first order
            # and the 1 dimension (range) is second order
            self.fista = torch.jit.trace(fista.solve_FISTA_subproblem_1st2ndOrder_jit,(self.x,self.alpha,
                                                self.x_lb,self.x_ub))
            self.pen_fn = self.pen_fn_1st2ndOrder
        else:
            # print("Fista configured for 1st Order")
            if fista_ver_str == "cuda-fista":
                import cuda.fista_cuf as fista_cuf
                self.fista = fista_cuf.solve_FISTA_subproblem_kernel
            else:
                self.fista = torch.jit.trace(fista.solve_FISTA_subproblem_jit,(self.x,self.alpha,
                                                    self.x_lb,self.x_ub))
            self.pen_fn = self.pen_fn_1stOrder

        
    def reset_iterations(self):
        """
        Reset the iterations
        This allows us to use the same sparsa object
        without having to reload all the static parameters
        """
        self.set_max_iter(self.max_iter)

        # reset the total subproblem step to zero
        self.total_rel_step = 0
    
    def set_fit_param(self,x:Dict[str,torch.tensor],model_scale:float=1.0):
        """
        Configure the optimization routine for a particular
        state of x, with a predefined optimization variable

        model_scale: float (defaults to 1.0)
            Scalar multiplier for the model.  Used to account for
            Poisson thinning
        """

        # set the additional parameters not being fit
        self.x_kwargs = create_x_kwargs(x,self.fit_var)
        
        # setup the forward models to take a torch.tensor
        # input from one variable
        self.fwd_model_lst = []
        for idx,_ in enumerate(self.base_fwd_model_lst):
            # input arguments include one that identifies what variable is being fit ('fit_var') as well as 
            # specifying the value of the fit variable as a variable in the function (self.fit_var:x)
            # by specifying the fit variable to the function, it allows us to selectively call update steps
            # and save time on the forward model.
            self.fwd_model_lst+=[lambda x,idx_loc=idx: model_scale*self.base_fwd_model_lst[idx_loc](**{'fit_var':self.fit_var,self.fit_var:x},**self.x_kwargs)]
        
        # set the fit parameter from the dictionary
        self.x = copy.deepcopy(x[self.fit_var]).requires_grad_(True) # this sets the initial condition

        # create the bounds on x if they haven't been set yet
        if self.x_lb is None:
            self.set_x_lower_bound(torch.zeros(self.x.shape,device=self.device,dtype=self.dtype)-np.inf)
        if self.x_ub is None:
            self.set_x_upper_bound(torch.zeros(self.x.shape,device=self.device,dtype=self.dtype)+np.inf)

        # if init_bounds:
        #     # automatically configure the bounds to be infinite
        #     # this can be overwritten using the set_x_lower_bounds()/set_x_upper_bounds() methods
        #     self.set_x_lower_bound(torch.zeros(self.x.shape,device=self.device,dtype=self.dtype)-np.inf)
            # self.set_x_upper_bound(torch.zeros(self.x.shape,device=self.device,dtype=self.dtype)+np.inf)
        
    def set_x_lower_bound(self,xlb:torch.tensor):
        self.x_lb = copy.deepcopy(xlb)
    
    def set_x_upper_bound(self,xub:torch.tensor):
        self.x_ub = copy.deepcopy(xub)
        
        
    
    """
    General Definitions - possibly move to subclass
    """
        
    def pois_loss_fn(self,x:torch.tensor)->torch.tensor:
        """
        x: torch.tensor
            estimated state variable     
        """

        loss = torch.tensor(0, device=self.device, dtype=self.dtype) # torch.zeros(1, device=self.device, dtype=self.dtype)
        for idx,mod in enumerate(self.fwd_model_lst):
            y_est = mod(x)
            # loss += self.chan_weight_lst[idx]*(y_est-self.y_obs_lst[idx]*torch.log(y_est)).sum()
            loss += self.chan_weight_lst[idx]*(self.channel_mask_lst[idx]*y_est-self.channel_mask_lst[idx]*self.y_obs_lst[idx]*torch.log(y_est)).sum()

        return loss   
    
    def deadtime_loss_fn(self,x:torch.tensor)->torch.tensor:
        """
        x: torch.tensor
            estimated state variable     
        """

        loss = torch.tensor(0, device=self.device, dtype=self.dtype) # torch.zeros(1, device=self.device, dtype=self.dtype)
        for idx,mod in enumerate(self.fwd_model_lst):
            y_est = mod(x)
            loss += self.chan_weight_lst[idx]*(self.channel_mask_lst[idx]*self.y_active_lst[idx]*y_est-self.channel_mask_lst[idx]*self.y_obs_lst[idx]*torch.log(y_est)).sum()

        return loss

    def gaus_fn(self,x:torch.tensor)->torch.tensor:
        """
        x: torch.tensor
            estimated state variable 
        Gaussian loss function where mean = variance
        """ 

        loss = torch.tensor(0, device=self.device, dtype=self.dtype) # torch.zeros(1, device=self.device, dtype=self.dtype)
        for idx,mod in enumerate(self.fwd_model_lst):
            y_est = mod(x)
            loss += self.chan_weight_lst[idx]*(self.channel_mask_lst[idx]*(0.5*torch.log(2.0*self.pi*y_est)+(y_est-self.y_obs_lst[idx])**2/y_est)).sum()

        return loss

    def gaus_approx_fn(self,x:torch.tensor)->torch.tensor:
        """
        x: torch.tensor
            estimated state variable 
        approximate Gaussian loss function with known variance
        The variance is folded into the channel mask
        """ 

        loss = torch.tensor(0, device=self.device, dtype=self.dtype) # torch.zeros(1, device=self.device, dtype=self.dtype)
        for idx,mod in enumerate(self.fwd_model_lst):
            y_est = mod(x)
            loss += self.chan_weight_lst[idx]*(self.channel_mask_lst[idx]*(y_est-self.y_obs_lst[idx])**2).sum()

        return loss   

    def pen_fn_1stOrder(self,x):
        tv = torch.sum(torch.abs(torch.diff(x,dim=0))) + torch.sum(torch.abs(torch.diff(x,dim=1)))
        return self.penalty_weight*tv

    def pen_fn_2ndOrder(self,x):  
        tv = torch.sum(torch.abs(torch.diff(torch.diff(x,dim=0),dim=0))) + torch.sum(torch.abs(torch.diff(torch.diff(x,dim=1),dim=1)))
        return self.penalty_weight*tv

    def pen_fn_1st2ndOrder(self,x):
        tv = torch.sum(torch.abs(torch.diff(x,dim=0))) + torch.sum(torch.abs(torch.diff(torch.diff(x,dim=1),dim=1)))
        return self.penalty_weight*tv
    
    """
    SpaRSA definitions
    """
        
    def get_new_alpha(self,x_diff,x_grad_diff):
        return torch.sum(x_diff*x_grad_diff)/torch.linalg.norm(x_diff.ravel(), 2)**2
        
    def prox_gradient(self,x,x_grad,alpha):
        x_p1 = self.fista(x-x_grad/alpha,self.penalty_weight/alpha,
                        self.x_lb,self.x_ub)
            
        obj_p1 = self.loss_fn(x_p1) + self.pen_fn(x_p1)
        dx_l2_norm_p1 = torch.linalg.norm((x_p1 - x).ravel(), 2)**2

        # print(f"{self.loop_iter}, {self.alpha}: {dx_l2_norm_p1}, {obj_p1}")

        # if torch.isnan(x).sum() > 0:
        #     print(f"{torch.isnan(x_p1).sum()} nans in x")

        # if torch.isnan(x_grad).sum() > 0:
        #     print(f"{torch.isnan(x_grad).sum()} nans in gradient")

        # if torch.isnan(x_p1).sum() > 0:
        #     print(f"{torch.isnan(x_p1).sum()} nans in x_p1")

        
        
        return x_p1, obj_p1, dx_l2_norm_p1
    
    def acceptance_criteria(self,obj_value,alpha,dx_norm_l2):
        hist_idx = np.maximum(self.loop_iter-self.M_hist,0)
        max_history = self.objective_tnsr[hist_idx:self.loop_iter+1].max()
        step_inc = 0.5*self.sigma*alpha*dx_norm_l2
        accept_result = obj_value <= max_history - step_inc
        
#         print(f"{obj_value[0].item()}  :  {hist_idx}, {max_history}, {step_inc}, {accept_result[0].item()}")
        
        return accept_result
    
    
    def solve_sparsa_subprob(self)->str:
        
        status_str = ""
        self.loop_iter = 0
        terminate_opt = False
        self.start_time = time.time()

        loss = self.loss_fn(self.x)
        self.objective_tnsr[0] = loss + self.pen_fn(self.x)
        self.alpha_tnsr[self.loop_iter] = self.alpha
        loss.backward()  # backprop the gradient
        x_grad = copy.deepcopy(self.x.grad)
        x_sqrt_l2_norm = torch.linalg.norm(self.x.ravel(), 2)
        # store the initial state to calculate the step size of
        # the entire subproblem
        x_sqrt_l2_norm_init = x_sqrt_l2_norm
        x0 = copy.deepcopy(self.x.data)  

        # print(f"  start alpha {self.alpha.item()}")
        # print(f"  start loss {loss.item()}")
        # print(f"  start objective {self.objective_tnsr[0].item()}")
        # print(f"  x[1,60]: {self.x[1,60].item()}")

        while True: 
            with torch.no_grad(): 
                x_p1, obj_p1, dx_l2_norm_p1 = self.prox_gradient(self.x,x_grad,self.alpha)
                
                while (not self.acceptance_criteria(obj_p1,self.alpha,dx_l2_norm_p1)) and (self.alpha <= self.alpha_max):
                    self.alpha *= self.eta
                    x_p1, obj_p1, dx_l2_norm_p1 = self.prox_gradient(self.x,x_grad,self.alpha)

                # print(f"  x[1,60]: {self.x[1,60].item()}")
                # print(f"  xp1[1,60]: {x_p1[1,60].item()}")

                if (obj_p1 >= self.objective_tnsr[self.loop_iter] and self.alpha >= self.alpha_max) or torch.isnan(obj_p1) or (obj_p1 > 0):
                    status_str += f"Step produced increased objective with alpha: {self.alpha.item()}"
                    status_str += f"\n      objective change {obj_p1.item()-self.objective_tnsr[self.loop_iter].item()}"
                    termimate_opt = True
                    rel_step = 0.0
                    self.rel_step_tnsr[self.loop_iter] = rel_step
                    # # self.x.data = copy.deepcopy(x_bu)
                    # # obj_ret = self.loss_fn(self.x) + self.pen_fn(self.x)
                    # # status_str += f"\n      returning objective to {obj_ret.item()}"
                    # status_str += f"\n      previous objective was {obj_m1.item()}"
                    # status_str += f"\n      previous tnsr objective was {self.objective_tnsr[self.loop_iter].item()}"
                    # hist_idx = np.maximum(self.loop_iter-self.M_hist-1,0)
                    # max_history = self.objective_tnsr[hist_idx:self.loop_iter+1].max()
                    # status_str += f"\n      max history {max_history.item()}"
                    # status_str += f"\n      history {self.objective_tnsr[hist_idx:self.loop_iter+2].data}"
                    # status_str += f"\n      loop_iter: {self.loop_iter}"
                    # status_str += f"\n      dx_l2_norm_p1 {dx_l2_norm_p1}"
                    # status_str += f"\n      step_inc {step_inc}"
                    # status_str += f"\n      accept criteria {max_history - step_inc}"
                    # print("step did not decrease")
                    self.x.grad = None
                    break
                else:
                    # if obj_p1 >= self.objective_tnsr[self.loop_iter]:
                    #     status_str += f"Step produced increased objective with alpha: {self.alpha.item()}"
                    #     status_str += f"\n      objective change {obj_p1.item()-self.objective_tnsr[self.loop_iter].item()}"
                    #     rel_step = 0.0
                    #     self.rel_step_tnsr[self.loop_iter] = rel_step
                    #     # self.x.data = copy.deepcopy(x_bu)
                    #     # obj_ret = self.loss_fn(self.x) + self.pen_fn(self.x)
                    #     status_str += f"\n      new objective {obj_p1.item()}"
                    #     status_str += f"\n      previous objective was {obj_m1.item()}"
                    #     hist_idx = np.maximum(self.loop_iter-self.M_hist,0)
                    #     max_history = self.objective_tnsr[hist_idx:self.loop_iter+1].max()
                    #     status_str += f"\n      max history {max_history.item()}"
                    #     status_str += f"\n      history {self.objective_tnsr[hist_idx:self.loop_iter+1]}"
                    #     status_str += f"\n      loop_iter: {self.loop_iter}"
                    #     status_str += f"\n      dx_l2_norm_p1 {dx_l2_norm_p1}"
                    #     status_str += f"\n      step_inc {step_inc}"
                    #     status_str += f"\n      accept criteria {max_history - step_inc}"
                    #     status_str += f"\nNo loop termination is occuring\n"

                    # update for next iteration
                    self.loop_iter+=1

                    # calculate relative step size
                    if x_sqrt_l2_norm == 0:
                        rel_step = np.inf
                    else:
                        rel_step = torch.sqrt(dx_l2_norm_p1) / x_sqrt_l2_norm
                        
                    self.objective_tnsr[self.loop_iter] = obj_p1
                    self.rel_step_tnsr[self.loop_iter] = rel_step

                    # update x values
                    x_diff = x_p1-self.x

                    self.x.grad = None
                    # obj_m2 = self.loss_fn(x_p1) + self.pen_fn(x_p1)
                    # self.x.data = copy.deepcopy(x_p1.data)
                    # self.x.data = x_p1.data.clone()
                    self.x.data = x_p1.clone()  # alters the value of both
                    # self.x = x_p1.clone()  # produces nans
                    # self.x.data = self.x.data.copy_(x_p1.data)  # still changes objective
                    # self.x.data = x_p1.data
                    # self.x.data += x_diff.data

                    # obj_m2 = self.loss_fn(x_p1) + self.pen_fn(x_p1)
                    self.x.grad = None

                    # x_bu = self.x.data # store current state in case of increased step size # changes obj value
                    # x_bu = copy.deepcopy(x_p1.data) # store current state in case of increased step size

                    # x_bu_str2 = f"{x_p1[0,0].item()}"
                    # max_x_diff = torch.abs(x_p1-x_bu).max()
                    # print(f"  xp1[1,60]: {x_p1[1,60].item()}")

                    # obj_m1 = self.loss_fn(self.x) + self.pen_fn(self.x)
                    
                    # if obj_m1 != obj_p1 or obj_m2 != obj_p1:
                    #     status_str += "On copy step, objective function has changed value"
                    #     status_str += f"\n      loop_iter: {self.loop_iter}"
                    #     status_str += f"\n      objective from self.x {obj_m1}"
                    #     status_str += f"\n      objective from x_p1 {obj_m2}"
                    #     status_str += f"\n      objective from precopy x_p1 {obj_m3}"
                    #     status_str += f"\n      objective from obj_p1 {obj_p1}"
                    #     status_str += f"\n      max x difference {max_x_diff}"
                    #     status_str += f"\n      x1: {x_bu_str1}"
                    #     status_str += f"\n      x2: {x_bu_str2}\n"
            
            if terminate_opt:
                break

            # print(f"      step alpha {self.alpha.item()}")
            # print(f"      objective change {obj_p1.item()-self.objective_tnsr[self.loop_iter-1].item()}")
            loss = self.loss_fn(self.x)
            loss.backward()
            x_m1_grad = copy.deepcopy(x_grad) # .clone()
            x_grad = copy.deepcopy(self.x.grad)

            # update for relative step calculation
            x_sqrt_l2_norm = torch.linalg.norm(self.x.ravel(), 2)
            
            self.alpha_tnsr[self.loop_iter] = self.alpha
            
            if dx_l2_norm_p1 == 0:
                self.alpha = self.alpha_min
            else:
                self.alpha = self.get_new_alpha(x_diff, x_grad - x_m1_grad)
                self.alpha = torch.clamp(self.alpha,min=self.alpha_min,max=self.alpha_max)
            
            
            if (rel_step < self.eps) and (self.loop_iter > self.min_iter):
                # print("Found Minimum")
                status_str = f"Found Minimum for eps {self.eps}"
                break
            
            if self.loop_iter >= self.max_iter:
                # print("Maximum Iterations Exceeded")
                status_str = "Maximum Iterations Exceeded"
                break

            if time.time() - self.start_time > self.timeout:
                status_str = f"Exceeded maximum SPARSA time of {self.timeout} seconds"
                break

        # Evaluate the total step size from this subproblem
        dx_total_l2_norm_p1 = torch.linalg.norm((self.x - x0).ravel(), 2)**2
        if x_sqrt_l2_norm_init == 0:
            self.total_rel_step = np.inf
        else:
            self.total_rel_step = (torch.sqrt(dx_total_l2_norm_p1) / x_sqrt_l2_norm_init).item()      
                
        return status_str



class multiSpiral_autograd:
    def __init__(self,device,dtype,max_iterations=100,min_iterations=0,spiral_eps=1e-5,timeout=21600):
        self.device = device
        self.dtype = dtype

        # default fista is the jit compiled pytorch version
        self.fista_ver_str = "jit-fista"
        self.valid_fista_ver = ["jit-fista","cuda-fista"]
        
        self.subprob_dct = {}
        
        self.max_iterations=max_iterations
        self.min_iterations=min_iterations
        self.spiral_eps = spiral_eps  # precision for terminating the Spiral loop (not the subproblem)

        self.tv_order = {}  # set the total variation order (defaults to 1)
        self.alpha0 = {}  # set the total variation initial step size (defaults to 1)

        self.step_size_lst = []
        self.loss_lst = []

        self.x0 = {}  # initial conditions for optimization (stored as np.ndarrays)
        self.x = {}   # estimated variables (torch.tensors with autograd enabled)
        self.x_lb = {}  # lower bounds on estimated variables
        self.x_ub = {}  # upper bounds on estimated variables

        # initialize the channel weight list
        self.chan_weight_lst = None

        # initialize masks
        self.channel_mask_lst = None

        self.verbose = False  # verbose output during optimization

        self.start_time = 0.0
        self.stop_time = 0.0
        self.timeout = timeout

        # TODO set this in configuration so we can use deadtime
        self.noise_model = 'poisson'  # poisson, deadtime, gaussian, gaussian_approx
        self.nll = self.pois_nll

        self.pi = self.to_tensor(np.pi)

        # TODO reconsider how to handle scaling due to thinning
        # this is not robust for cases where thinning may not be
        # exactly 50:50
        # set thin_factor to 1.0 and scale shot counts appropriately in the mpd_model?

        # model adjustment for poisson thinning
        self.thin_factor = 0.5
        
        # self.fwd_model_lst
        # self.y_obs_lst
        # self.set_estimate_lst(estimate_lst)
        
    def to_tensor(self,arr:np.ndarray):
        return torch.tensor(arr,device=self.device,dtype=self.dtype)
    
    def set_fista_version(self,fista_str:str):
        if fista_str in self.valid_fista_ver:
            self.fista_ver_str = fista_str

            # if self.fista_ver_str == "cuda-fista":
                # if using the custom cuda version of fista,
                # import the library
                # import cuda.fista_cuf as fista_cuf
        else:
            print(f"{fista_str} is not a valid version of fista, you must choose one of the following:")
            print(self.valid_fista_ver.join(', '))
            print("using default version: "+self.fista_ver_str)
    
    def set_fwd_model_lst(self, fwd_model_lst:List[Callable]):
        self.fwd_model_lst = fwd_model_lst

    def set_noise_model(self,noise_model:str):
        """
        set the noise model
        
        noise_model:str
            string identifying the noise model
            accepts: 'poisson', 'gaussian', 'gaussian_approx'
        """

        if noise_model.lower() == 'poisson':
            self.noise_model = noise_model.lower()
            self.nll = self.pois_nll
            self.output("Using Poisson Noise Model")
        elif noise_model.lower() == 'deadtime':
            self.noise_model = noise_model.lower()
            self.nll = self.deadtime_nll
            self.output("Using Deadtime Noise Model")
        elif noise_model.lower() == 'gaussian':
            self.noise_model = noise_model.lower()
            self.nll = self.gaus_nll
            self.output("Using Gaussian Noise Model")
        elif noise_model.lower() == 'gaussian_approx':
            self.noise_model = noise_model.lower()
            self.nll = self.gaus_approx_nll
            self.output("Using Gaussian Approximation Noise Model")
        else:
            self.output("Noise model, "+noise_model+ ", is not a valid option")
    
    def set_y_obs_lst(self,y_obs_lst:List[np.ndarray]):
        """
        Create lists of observations
        split them into fit and validation sets
        """
        
        # TODO update to work with more than Poisson noise model
        # this isn't generally used in the temperature processing
        # routine.  For that we use load_y_obs()

        self.y_obs_lst = []
        self.y_fit_lst = []
        self.y_val_lst = []

        self.y_fit_act_lst = []
        self.y_val_act_lst = []
        for y_obs in y_obs_lst:
            self.y_obs_lst.append(self.to_tensor(y_obs))
            y_thin_lst = poisson_thin(y_obs,n=2)
            self.y_fit_lst.append(self.to_tensor(y_thin_lst[0]))
            self.y_val_lst.append(self.to_tensor(y_thin_lst[1]))
            
        # if y_act_lst is not None:
        
        self.thin_factor = 0.5

        if self.chan_weight_lst is None:
            self.chan_weight_lst = [1.0]*len(self.y_obs_lst)
            
    def load_y_obs(self,load_ds:xr.Dataset,channel_lst:List[str],rethin=False):
        """
        Load the observation
        data from an xarray dataset
        
        rethin: bool
            create a new thinned version of fit and validation data
            defaults to False (using the thinned data as defined in
            the input dataset).  This is usually just for bootstrap
            uncertainty analysis.
        """
        self.y_obs_lst = []
        self.y_fit_lst = []
        self.y_val_lst = []

        self.y_fit_act_lst = []
        self.y_val_act_lst = []

        for ch_name in channel_lst:
            self.y_obs_lst.append(self.to_tensor(load_ds[ch_name+"_raw"].values))
            if rethin:
                y_thin_lst = poisson_thin(load_ds[ch_name+"_raw"].values,n=load_ds.attrs.get('thin_count',2))
                self.y_fit_lst.append(self.to_tensor(y_thin_lst[0]))
                self.y_val_lst.append(self.to_tensor(y_thin_lst[1]))
            else:
                self.y_fit_lst.append(self.to_tensor(load_ds[ch_name+"_thin0"].values))
                self.y_val_lst.append(self.to_tensor(load_ds[ch_name+"_thin1"].values))

                if self.noise_model == 'deadtime':
                    self.y_fit_act_lst.append(self.to_tensor(load_ds[ch_name+"_ActiveTime_thin0"].values))
                    self.y_val_act_lst.append(self.to_tensor(load_ds[ch_name+"_ActiveTime_thin1"].values))
                else:
                    self.y_fit_act_lst.append(1.0)
                    self.y_val_act_lst.append(1.0)

        self.thin_factor = 1.0/load_ds.attrs.get('thin_count',2.0)
        # if self.noise_model != 'deadtime':
        #     self.thin_factor = 1.0/load_ds.attrs.get('thin_count',2.0)
        # else:
        #     self.thin_factor = 1.0

        if self.chan_weight_lst is None:
            self.chan_weight_lst = [1.0]*len(self.y_obs_lst)

    def set_channel_weights(self,channel_weights:List[float]):
        self.chan_weight_lst = []
        for chan_weight in channel_weights:
            self.chan_weight_lst.append(chan_weight)

    def set_channel_masks(self,channel_masks:List[np.ndarray]):
        self.channel_mask_lst = []
        for ch_mask in channel_masks:
            self.channel_mask_lst.append(torch.tensor(ch_mask,device=self.device, dtype=self.dtype))

    def save_y_obs(self,filename:str):
        """
        save the observation data to a netcdf file
        """
        ds = xr.Dataset({},attrs={
            'data_channel_count':len(self.y_obs_lst),
                    })

        for idx_y,y in enumerate(self.y_obs_lst):
            ds[f"y_obs_{idx_y}"] = xr.DataArray(y.cpu().numpy(),["time","range"])
            ds[f"y_fit_{idx_y}"] = xr.DataArray(self.y_fit_lst[idx_y].cpu().numpy(),["time","range"])
            ds[f"y_val_{idx_y}"] = xr.DataArray(self.y_val_lst[idx_y].cpu().numpy(),["time","range"])
            if self.noise_model == "deadtime":
                ds[f"y_fit_act_{idx_y}"] = xr.DataArray(self.y_fit_act_lst[idx_y].cpu().numpy(),["time","range"])
                ds[f"y_val_act_{idx_y}"] = xr.DataArray(self.y_val_act_lst[idx_y].cpu().numpy(),["time","range"])


    # TODO
    # def save_data(self,filename:str):
    #     """
    #     Save the data to a pickle file
    #     """
    #     # import pickle
    #     # with open(filename, 'wb') as f:
    #     #     pickle.dump(self, f)

    #     import xarray as xr

    #     ds = xr.Dataset({},attrs={
    #         'data_channel_count':len(self.y_obs_lst),
    #                 })

    #     for idx_y,y in enumerate(self.y_obs_lst):
    #         ds[f"y_obs_{idx_y}"] = xr.DataArray(y.cpu().numpy(),["time","range"])
    #         ds[f"y_fit_{idx_y}"] = xr.DataArray(self.y_fit_lst[idx_y].cpu().numpy(),["time","range"])
    #         ds[f"y_val_{idx_y}"] = xr.DataArray(self.y_val_lst[idx_y].cpu().numpy(),["time","range"])

    #     # Save solutions

    #     # Save regularizer values

    #     # Save validation NLL
        

    def set_estimate_lst(self,estimate_lst:List[str]):
        """
        Set the list of variables to estimate
        and the order in which to do it
        """
        self.estimate_lst = list(estimate_lst)
        
    def set_tv_penalties(self,tv_penalty_dct:Dict[str,float]):
        """
        set the total variation penalty for multiple
        variables
        """
        for var in tv_penalty_dct:
            self.set_tv_penalty(tv_penalty_dct[var],var)
    
    def set_tv_penalty(self,tv_penalty_flt,var:str):
        """
        set the total variation penalty for one variable
        """
        if var in self.subprob_dct.keys():
            self.subprob_dct[var].set_penalty_weight(tv_penalty_flt)
        else:
            print("passed TV penalty for "+var+" but this is not defined as a subproblem in the current multiSpiral instance")

            
    def add_sparsa_config(self,sparsa_config_dct:Dict[str,dict]):
        """
        Add variables to the sparsa subproblem configurations.
        Note that this will append these settings as long as the
        variables are not repeated.

        Expects a dictionary entry for each estimated variable.
            Each variable entry can have the following entries
            (Any not supplied will be set to default values)
                M_hist: int
                    number of previous iterations for the acceptance calculation
                alpha: float
                    initial value of alpha in spiral subproblem
                alpha_max: float
                    max value of alpha in spiral subproblem
                alpha_min: float
                    min value of alpha in spiral subproblem
                eta: float
                    alpha multiplier for when an optimization step fails
                max_iter: int
                    maximum iterations for the subproblem
                min_iter: int
                    minimum iterations for the subproblem
                eps: float
                    precision for termination of the subproblem
                x_lower_bound: np.ndarray
                    lower bound for estimated variable
                x_upper_bound: np.ndarray
                    upper bound for estimated variable
                tv_penalty: np.ndarray
                    total variation multiplier
                tv_order: int
                    order of total variation penalty (1 or 2)
                x0: np.ndarray
                    initial value of the optimization variable
                
        """
        for var in sparsa_config_dct:
            self.subprob_dct[var] = sparsa_torch_autograd(self.device,self.dtype)
            if 'M_hist' in sparsa_config_dct[var].keys():
                self.subprob_dct[var].M_hist = sparsa_config_dct[var]['M_hist']
            
            if 'alpha_min' in sparsa_config_dct[var].keys():
                self.subprob_dct[var].set_alpha_min(sparsa_config_dct[var]['alpha_min'])

            if 'alpha_max' in sparsa_config_dct[var].keys():
                self.subprob_dct[var].set_alpha_max(sparsa_config_dct[var]['alpha_max'])

            if 'alpha' in sparsa_config_dct[var].keys():
                self.set_alpha0(sparsa_config_dct[var]['alpha'],var)
                # self.subprob_dct[var].set_alpha(sparsa_config_dct[var]['alpha'])

            if 'eta' in sparsa_config_dct[var].keys():
                self.subprob_dct[var].set_eta(sparsa_config_dct[var]['eta'])

            if 'max_iter' in sparsa_config_dct[var].keys():
                self.subprob_dct[var].set_max_iter(sparsa_config_dct[var]['max_iter'])

            if 'min_iter' in sparsa_config_dct[var].keys():
                self.subprob_dct[var].min_iter = sparsa_config_dct[var]['min_iter']
            
            if 'eps' in sparsa_config_dct[var].keys():
                self.subprob_dct[var].set_eps(sparsa_config_dct[var]['eps'])

            if 'x_lower_bound' in sparsa_config_dct[var].keys():
                self.x_lb[var] = sparsa_config_dct[var]['x_lower_bound']
                self.set_x_lb(self.x_lb[var],var)

            if 'x_upper_bound' in sparsa_config_dct[var].keys():
                self.x_ub[var] = sparsa_config_dct[var]['x_upper_bound']
                self.set_x_ub(self.x_ub[var],var)

            if 'tv_penalty' in sparsa_config_dct[var].keys():
                self.subprob_dct[var].set_penalty_weight(sparsa_config_dct[var]['tv_penalty'])

            if 'sparsa_timeout' in sparsa_config_dct[var].keys():
                self.subprob_dct[var].timeout = sparsa_config_dct[var]['sparsa_timeout']

            # set the total variation order to 1 if not set explicity
            self.set_tv_order(sparsa_config_dct[var].get('tv_order',1),var)
            
            if 'x0' in sparsa_config_dct[var].keys():
                self.set_initial_condiation(var,sparsa_config_dct[var]['x0'])

    def set_alpha0(self,alpha0,var):
        self.alpha0[var] = alpha0

    def set_x_lb(self,x_lb:np.ndarray,var:str):
        self.subprob_dct[var].set_x_lower_bound(self.to_tensor(x_lb))
    
    def set_x_ub(self,x_ub:np.ndarray,var:str):
        self.subprob_dct[var].set_x_upper_bound(self.to_tensor(x_ub))

    def set_tv_order(self,order:int,var:str):
        """
        Set the total variation order of the
        optimiztaion variable.  Defaults to 1
        """
        if order == 2:
            if self.fista_ver_str == 'cuda-fista':
                print(f"{var} set for 2nd order TV and using {self.fista_ver_str}")
                print(f"  this version of fista is not writtenf or 2nd order")
                print(f"  reverting to jit-fista for this variable")
            self.tv_order[var] = 2
        elif order == 3:
            if self.fista_ver_str == 'cuda-fista':
                print(f"{var} set for 2nd order TV in range and using {self.fista_ver_str}")
                print(f"  this version of fista is not writtenf or 2nd order")
                print(f"  reverting to jit-fista for this variable")
            self.tv_order[var] = 3
        else:
            self.tv_order[var] = 1

    def set_initial_conditions(self,x_init_dct):
        """
        Set the initial conditions using a dict input
        """
        for var in x_init_dct:
            self.set_initial_condition(var,x_init_dct[var])

    def set_initial_condition(self,var,x_init:np.ndarray):
        """
        set the initial condition of a variable
        """
        self.x0[var] = x_init.copy()
        if var in self.x_lb:
            self.x0[var] = np.maximum(self.x0[var],self.x_lb[var])
        if var in self.x_ub:
            self.x0[var] = np.minimum(self.x0[var],self.x_ub[var])
        
    def load_subproblem_data(self):
        """
        Initialize the subproblems with the forward models,
        data and templates for the estimated variables
        also reset the step size to alpha0 as previously defined
            (defaults to 1.0 if not defined)
        """

        # make sure the observation data and
        # forward models have been loaded
        assert len(self.y_fit_lst) > 0
        assert len(self.fwd_model_lst) > 0

        for var in self.subprob_dct:
            self.subprob_dct[var].load_fit_parameters(self.x, self.y_fit_lst, self.fwd_model_lst,var, model_scale=self.thin_factor)
            if self.channel_mask_lst is not None:
                self.subprob_dct[var].set_channel_masks(self.channel_mask_lst)
            self.subprob_dct[var].set_alpha(self.alpha0.get(var,1.0))
            if self.chan_weight_lst is not None:
                self.subprob_dct[var].set_channel_weights(self.chan_weight_lst)

    def initialize(self):
        """
        initialize the estimated variable with
        the initial conditions stored in self.x0
        """
        for var in self.x0:
            self.x[var] = self.to_tensor(self.x0[var])
            # self.x[var] = torch.tensor(self.x0[var], device=self.device, dtype=self.dtype, requires_grad=True)

    def initialize_subproblems(self):
        """
        Perform last steps to initialize the sub problems 
        for optimization
        """
        for var in self.subprob_dct:
            if self.noise_model == 'poisson':
                self.subprob_dct[var].set_loss_poisson()
            elif self.noise_model == 'deadtime':
                self.subprob_dct[var].set_loss_deadtime(self.y_fit_act_lst)
            elif self.noise_model == 'gaussian':
                self.subprob_dct[var].set_loss_gaus()
            elif self.noise_model == 'gaussian_approx':
                self.subprob_dct[var].set_loss_gaus_approx()
            self.subprob_dct[var].set_fista(self.fista_ver_str,order=self.tv_order[var])

    def get_x(self)->Dict[str,np.ndarray]:
        """
        Return the estimated variable as a 
        dictionary of np.ndarrays
        """
        x_ret = {}
        for var in self.x:
            x_ret[var] = self.x[var].detach().cpu().numpy()

        return x_ret

    def output(self,str):
        """
        prints based on the verbose settings
        """
        if self.verbose:
            print(str)

    def solve(self,initialize=True)->str:
        """
        run the solver
        returns a status message

        if initialize is true, the initial conditions are
            loaded into the estimated variable and all of
            the subproblems are initialized
        """
        terminate_solve = False
        self.start_time = time.time()

        if initialize:
            self.initialize()
            self.load_subproblem_data()
            self.initialize_subproblems()
            self.step_size_lst = []
            self.loss_lst = []

        for idx in range(self.max_iterations):
            step_size = 0.0

            self.output("iteration %d"%idx)

            for var in self.estimate_lst:
                self.output("solving for "+var)
                self.subprob_dct[var].reset_iterations()
                self.subprob_dct[var].set_fit_param(self.x,model_scale=0.5)
                if len(self.estimate_lst) > 1:
                    # reset alpha if there are multiple variables in the
                    # optimization routine
                    self.subprob_dct[var].set_alpha(self.alpha0.get(var,1.0))

                # try:
                res_str = self.subprob_dct[var].solve_sparsa_subprob()
                # except RuntimeError:
                #     """
                #     nan values in gradient
                #     """
                #     terminate_solve = True
                #     self.output("nan values in the gradient")
                #     break

                self.x[var].data = self.subprob_dct[var].x.data  # copy the solution
                sub_prob_step_size = self.subprob_dct[var].total_rel_step
                step_size = np.maximum(step_size,sub_prob_step_size)
                self.output(f"Step: {sub_prob_step_size}")
                self.output(f"Loss: {self.fit_loss():e}")
                self.output(f"alpha: {self.subprob_dct[var].alpha.item()}")
                self.output(f"subproblem iterations: {self.subprob_dct[var].loop_iter}")
                self.output(res_str+"\n")

            if terminate_solve:
                term_str = "Nan in gradient"
                break
            
            self.step_size_lst.append(step_size)
            self.loss_lst.append(self.fit_loss())
            
            if step_size < self.spiral_eps and idx > self.min_iterations:
                term_str = f"Optimization Successful after {idx} iterations"
                break
            
            if time.time() - self.start_time > self.timeout:
                term_str = f"Exceeded maximum SPIRAL time of {self.timeout} seconds"
                break

        self.stop_time = time.time()
        
        if idx >= self.max_iterations-1:
            term_str = f"Exceeded maximum of {self.max_iterations} iterations"
        
        return term_str

    # TODO these should be general functions set based on the noise model
    def fit_loss(self)->float:
        """
        Calculate the Poisson NLL against
        the fit data
        """
        # TODO update so model call includes fit variable
        loss = 0.0
        for idx, model in enumerate(self.fwd_model_lst):
            if self.channel_mask_lst is not None:
                loss_mask = self.channel_mask_lst[idx]
            else:
                loss_mask = None
            loss += self.chan_weight_lst[idx]*self.nll(model(**self.x),self.y_fit_lst[idx],self.y_fit_act_lst[idx],mask=loss_mask).item()
        return loss

    def valid_loss(self)->float:
        """
        Calculate the Poisson NLL against
        the validation data
        """
        # TODO update so model call includes fit variable
        loss = 0.0
        for idx, model in enumerate(self.fwd_model_lst):
            if self.channel_mask_lst is not None:
                loss_mask = self.channel_mask_lst[idx]
            else:
                loss_mask = None
            loss += self.chan_weight_lst[idx]*self.nll(model(**self.x),self.y_val_lst[idx],self.y_val_act_lst[idx],mask=loss_mask).item()
        return loss

    def pois_nll(self,fwd_model,phot_counts,act_time,mask=None)->torch.tensor:
        self.output("Poisson Loss Call")
        if mask is None:
            return torch.sum(self.thin_factor*fwd_model-phot_counts*torch.log(self.thin_factor*fwd_model))
        else:
            return torch.sum(mask*(self.thin_factor*fwd_model-phot_counts*torch.log(self.thin_factor*fwd_model)))

    def deadtime_nll(self,fwd_model,phot_counts,act_time,mask=None)->torch.tensor:
        self.output("Deadtime Loss Call")
        if mask is None:
            return torch.sum(act_time*self.thin_factor*fwd_model-phot_counts*torch.log(self.thin_factor*fwd_model))
        else:
            return torch.sum(mask*(act_time*self.thin_factor*fwd_model-phot_counts*torch.log(self.thin_factor*fwd_model)))

    def gaus_nll(self,fwd_model,phot_counts,act_time,mask=None)->torch.tensor:
        self.output("Gaussian Loss Call")
        if mask is None:
            return torch.sum(0.5*torch.log(2*self.thin_factor*fwd_model*self.pi)+(self.thin_factor*fwd_model-phot_counts)**2/(2*self.thin_factorfwd_model))
        else:
            return torch.sum(mask*(0.5*torch.log(2*self.thin_factor*fwd_model*self.pi)+(self.thin_factor*fwd_model-phot_counts)**2/(2*self.thin_factor*fwd_model)))
    
    def gaus_approx_nll(self,fwd_model,phot_counts,act_time,mask=None)->torch.tensor:
        self.output("Gaussian Approx Loss Call")
        if mask is None:
            return torch.sum((self.thin_factor*fwd_model-phot_counts)**2)
        else:
            return torch.sum(mask*(self.thin_factor*fwd_model-phot_counts)**2)
