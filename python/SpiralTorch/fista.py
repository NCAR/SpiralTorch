"""
functions to implement FISTA
in PyTorch
"""

import torch
import numpy as np

from typing import Tuple

import copy

import pathlib

dir = str(pathlib.Path(__file__).parent.resolve())

def solve_FISTA_subproblem(b:torch.tensor,lam:float,max_iter:int = 10,
                           count_lim:int = 5,eps:float=1e-2,bnds:list=None)->torch.tensor:
    """
    Solves optimization subproblem by using FISTA [1].
    For most MLE applications 
    b = x_k - 1/alpha * grad(f)
        where x_k is the current state vector, 1/alpha is the step size,
        f - is the error function so grad(f) is the gradient of f with respect to x
        
    lam - TV penalty function is typically
        lam = tau/alpha, where tau is the TV penalty coefficient (often times also lambda)
    
    max_iter - maximum number of loop iterations to estimate the minimum of the subproblem
    
    count_lim - number of successful minimization steps needed in a row before exiting the subproblem
    
    eps - improvement in step needed to count a step as successful
    
    returns x - the solution to the subproblem
    
    Solving the sub problem should be done separately for each separable variable.
        for example, this function should be run once for backscatter coefficient, lidar ratio and depolarization each.
        For non-TV variables (gain, deadtime), don't use this function call.  Just run steepest descent:
        x_{k+1} = x_k - 1/alpha * grad(x)
        
    This function expects rectangular arrays.  Mapping non-rectangular spaces (e.g. altitude varying range data)
    needs to happen prior to calling this function.
    """

    print("The FISTA version called has not been corrected and will yield inaccurate results")
    if bnds is None:
        bnds = [-np.inf,np.inf]
    
    # initialize differential state variables with their history
    # k - this iteration, kp1=k+1, km1 = k-1
    r_k = torch.zeros((b.shape[0]-1,b.shape[1]))
    s_k = torch.zeros((b.shape[0],b.shape[1]-1))
    p_k = torch.zeros((b.shape[0]-1,b.shape[1]))
    q_k = torch.zeros((b.shape[0],b.shape[1]-1))
    
    p_km1 = torch.zeros((b.shape[0]-1,b.shape[1]))
    p_kp1 = torch.zeros((b.shape[0]-1,b.shape[1]))
    q_km1 = torch.zeros((b.shape[0],b.shape[1]-1))   
    q_kp1 = torch.zeros((b.shape[0],b.shape[1]-1))
    
    t_k = 1
    
    grad_a_km1 = torch.zeros(b.shape)
    
    cnt_iter = 0
    cnt_eps = 0
    
    # while(cnt_iter < max_iter and cnt_eps < count_lim):
    for _ in range(50):    
        grad_a = b - lam*map_d2x(r_k,s_k) # compute the FISTA gradient
        grad_a = torch.clamp(grad_a,min=bnds[0],max=bnds[1])
        grad_ad = map_x2d(grad_a)  # map the gradient to differential space
        
        # update estimates of differential variables
        p_kp1= p_k + 1.0/(8*lam)*grad_ad[0]
        q_kp1= q_k + 1.0/(8*lam)*grad_ad[1]
        
        t_kp1 = (1 + np.sqrt(1+4*t_k**2))/2.0
        
        # Willem performs this normalization step in his FISTA code but 
        # it does not appear in [1]
        # Preform the projection step
        p_kp1 = p_kp1 / torch.clamp (torch.abs (p_kp1),min= 1.0)
        q_kp1 = q_kp1 / torch.clamp (torch.abs (q_kp1),min= 1.0)
        
        # update normalized differential variables
        r_k = p_k+(t_k-1)/t_kp1*(p_k-p_km1)
        s_k = q_k+(t_k-1)/t_kp1*(q_k-q_km1)
        
        # update variable history before next iteration
        p_km1 = p_k.clone()
        p_k = p_kp1.clone()
        q_km1 = q_k.clone()
        q_k = q_kp1.clone()
        
        t_k = t_kp1
        
        # # check for exit criteria
        # # Compute the relative step size
        # rel_step_num = torch.linalg.norm (torch.ravel (grad_a) - torch.ravel (grad_a_km1))
        # rel_step_dem = torch.linalg.norm (torch.ravel (grad_a))
        
        # grad_a_km1 = grad_a.clone()
        
        # # re_flt = np.linalg.norm (D_mat - D_prev_mat, "fro") / np.linalg.norm (D_mat, "fro")
        # if rel_step_num < (eps * rel_step_dem):
        #     cnt_eps += 1
        # else:
        #     cnt_eps = 0
        # cnt_iter += 1
        
#        print('FISTA cnt_iter: %d'%cnt_iter)
#        print('FISTA cnt_eps:  %d'%cnt_eps)
#        print('FISTA numerator error: %e'%(rel_step_num))
#        print('FISTA denominator error: %e'%(rel_step_dem))

    return grad_a

def map_d2x(p:torch.tensor,q:torch.tensor):
    """
    Maps differential variables p and q (or r and s) to pixel space description
    of a 2D parameter space, x.
    This is script L in [1]
    """    
    
    x = torch.zeros((q.shape[0],p.shape[1]))
    x[:-1,:] = p
    x[:,:-1] = x[:,:-1] + q
    x[1:,:] = x[1:,:] - p
    x[:,1:] = x[:,1:] - q
    
    return x
    
    
def map_x2d(x:torch.tensor):
    """
    Maps pixel description of the image, x, to a differential description
    p and q (or r and s)
    """
    
    p = torch.zeros((x.shape[0]-1,x.shape[1]))
    q = torch.zeros((x.shape[0],x.shape[1]-1))
    
    p = x[:-1,:] - x[1:,:]
    q = x[:,:-1] - x[:,1:]
    
    return p,q


def map_d2x_jit(p:torch.Tensor,q:torch.Tensor,x:torch.Tensor): #x:torch.tensor):
    """
    Maps differential variables p and q (or r and s) to pixel space description
    of a 2D parameter space, x.
    This is script L in [1]
    """    
    
    x[:-1,:] = p
    x[:,:-1] = x[:,:-1] + q
    x[1:,:] = x[1:,:] - p
    x[:,1:] = x[:,1:] - q
    
    return x
    
    
def map_x2d_jit(x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]: #,q:torch.tensor,p:torch.tensor):
    """
    Maps pixel description of the image, x, to a differential description
    p and q (or r and s)
    """
    
#     p = torch.zeros((x.shape[0]-1,x.shape[1]),device=device)
#     q = torch.zeros((x.shape[0],x.shape[1]-1),device=device)
    
    p = x[:-1,:] - x[1:,:]
    q = x[:,:-1] - x[:,1:]
    
    return p,q

@torch.jit.script
def solve_FISTA_subproblem_jit(b:torch.Tensor,lam1:torch.Tensor,
                              lb:torch.Tensor,ub:torch.Tensor)->torch.Tensor:

    """
    Solves optimization subproblem by using FISTA [1].
    For most MLE applications 
    b = x_k - 1/alpha * grad(f)
        where x_k is the current state vector, 1/alpha is the step size,
        f - is the error function so grad(f) is the gradient of f with respect to x
        
    lam - TV penalty function is typically
        lam = 1/alpha or tau/alpha
    
    returns x - the solution to the subproblem
    
    Solving the sub problem should be done separately for each separable variable.
        for example, this function should be run once for backscatter coefficient, lidar ratio and depolarization each.
        For non-TV variables (gain, deadtime), don't use this function call.  Just run steepest descent:
        x_{k+1} = x_k - 1/alpha * grad(x)
        
    This function expects rectangular arrays.  Mapping non-rectangular spaces (e.g. altitude varying range data)
    needs to happen prior to calling this function.
    """

    device = b.device
    dtype = b.dtype
    
    # initialize differential state variables with their history
    # k - this iteration, kp1=k+1, km1 = k-1
    r_k = torch.zeros((b.shape[0]-1,b.shape[1]),device=device,dtype=dtype)
    s_k = torch.zeros((b.shape[0],b.shape[1]-1),device=device,dtype=dtype)
    p_k = torch.zeros((b.shape[0]-1,b.shape[1]),device=device,dtype=dtype)
    q_k = torch.zeros((b.shape[0],b.shape[1]-1),device=device,dtype=dtype)
    
    # p_km1 = torch.zeros((b.shape[0]-1,b.shape[1]),device=device,dtype=dtype)
    p_kp1 = torch.zeros((b.shape[0]-1,b.shape[1]),device=device,dtype=dtype)
    # q_km1 = torch.zeros((b.shape[0],b.shape[1]-1),device=device,dtype=dtype)   
    q_kp1 = torch.zeros((b.shape[0],b.shape[1]-1),device=device,dtype=dtype)
    
    t_k = torch.ones((),device=device,dtype=dtype) # t_k=1
    
    # grad_a_km1 = torch.zeros(b.shape,device=device)
    grad_a = torch.zeros(b.shape,device=device) # preallocate for map functions

    for _ in range(50):
        grad_a = b - lam1*map_d2x_jit(r_k,s_k,grad_a) # compute the FISTA gradient

        grad_a = torch.min(torch.max(grad_a,lb),ub)

        grad_ad = map_x2d_jit(grad_a)  # map the gradient to differential space
        
        # update estimates of differential variables
        p_kp1= r_k + 1.0/(8.0*lam1)*grad_ad[0]
        q_kp1= s_k + 1.0/(8.0*lam1)*grad_ad[1]
        # p_kp1= p_k + 1.0/(8*lam1)*grad_ad[0]
        # q_kp1= q_k + 1.0/(8*lam1)*grad_ad[1]

        # Willem performs this normalization step in his FISTA code but 
        # it does not appear in [1]
        # Preform the projection step
        p_kp1 = p_kp1 / torch.clamp (torch.abs (p_kp1),min= 1.0)
        q_kp1 = q_kp1 / torch.clamp (torch.abs (q_kp1),min= 1.0)

        t_kp1 = (1 + torch.sqrt(1+4*t_k**2))/2.0
        
        # update normalized differential variables
        r_k = p_kp1+(t_k-1.0)/t_kp1*(p_kp1-p_k)
        s_k = q_kp1+(t_k-1.0)/t_kp1*(q_kp1-q_k)
        # r_k = p_k+(t_k-1)/t_kp1*(p_k-p_km1)
        # s_k = q_k+(t_k-1)/t_kp1*(q_k-q_km1)
        
        # update variable history before next iteration
        # p_km1 = p_k.clone()
        p_k = p_kp1.clone()
        # p_k = copy.deepcopy(p_kp1) #.clone()
        # q_km1 = q_k.clone()
        q_k = q_kp1.clone()
        # q_k = copy.deepcopy(q_kp1) #.clone()
        
        t_k = t_kp1

    return grad_a


def map_d2x_2ndOrder_jit(p:torch.Tensor,q:torch.Tensor,x:torch.Tensor): #x:torch.tensor):
    """
    Maps differential variables p and q (or r and s) to pixel space description
    of a 2D parameter space, x.
    This is script L in [1]
    """    
    
    x[:-2,:] = p
    x[1:-1,:] = x[1:-1,:] - 2 * p
    x[2:,:] = x[2:,:] + p

    x[:,:-2] = x[:,:-2] + q
    x[:,1:-1] = x[:,1:-1] - 2 * q
    x[:,2:] = x[:,2:] + q
    
    return x
    
    
def map_x2d_2ndOrder_jit(x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]: #,q:torch.tensor,p:torch.tensor):
    """
    Maps pixel description of the image, x, to a differential description
    p and q (or r and s)
    """
    
#     p = torch.zeros((x.shape[0]-1,x.shape[1]),device=device)
#     q = torch.zeros((x.shape[0],x.shape[1]-1),device=device)
    
    p = torch.diff(torch.diff(x,dim=0),dim=0) 
    q = torch.diff(torch.diff(x,dim=1),dim=1) 
    
    # p = x[:-2,:] - 2 * x[1:-1,:] + x[2:,:]
    # q = x[:,:-2] - 2 * x[:,1:-1] + x[:,2:]

    return p,q

@torch.jit.script
def solve_FISTA_subproblem_2ndOrder_jit(b:torch.Tensor,lam1:torch.Tensor,
                              lb:torch.Tensor,ub:torch.Tensor)->torch.Tensor:

    """
    Solves optimization subproblem by using FISTA [1].
    For most MLE applications 
    b = x_k - 1/alpha * grad(f)
        where x_k is the current state vector, 1/alpha is the step size,
        f - is the error function so grad(f) is the gradient of f with respect to x
        
    lam - TV penalty function is typically
        lam = 1/alpha or tau/alpha
    
    returns x - the solution to the subproblem
    
    Solving the sub problem should be done separately for each separable variable.
        for example, this function should be run once for backscatter coefficient, lidar ratio and depolarization each.
        For non-TV variables (gain, deadtime), don't use this function call.  Just run steepest descent:
        x_{k+1} = x_k - 1/alpha * grad(x)
        
    This function expects rectangular arrays.  Mapping non-rectangular spaces (e.g. altitude varying range data)
    needs to happen prior to calling this function.
    """

    device = b.device
    dtype = b.dtype
    
    # initialize differential state variables with their history
    # k - this iteration, kp1=k+1, km1 = k-1
    r_k = torch.zeros((b.shape[0]-2,b.shape[1]),device=device,dtype=dtype)
    s_k = torch.zeros((b.shape[0],b.shape[1]-2),device=device,dtype=dtype)
    p_k = torch.zeros((b.shape[0]-2,b.shape[1]),device=device,dtype=dtype)
    q_k = torch.zeros((b.shape[0],b.shape[1]-2),device=device,dtype=dtype)
    
    # p_km1 = torch.zeros((b.shape[0]-1,b.shape[1]),device=device,dtype=dtype)
    p_kp1 = torch.zeros((b.shape[0]-2,b.shape[1]),device=device,dtype=dtype)
    # q_km1 = torch.zeros((b.shape[0],b.shape[1]-1),device=device,dtype=dtype)   
    q_kp1 = torch.zeros((b.shape[0],b.shape[1]-2),device=device,dtype=dtype)
    
    t_k = torch.ones((),device=device) # t_k=1
    
    # grad_a_km1 = torch.zeros(b.shape,device=device)
    grad_a = torch.zeros(b.shape,device=device) # preallocate for map functions

    for _ in range(50):
        grad_a = b - lam1*map_d2x_2ndOrder_jit(r_k,s_k,grad_a) # compute the FISTA gradient

        grad_a = torch.min(torch.max(grad_a,lb),ub)

        grad_ad = map_x2d_2ndOrder_jit(grad_a)  # map the gradient to differential space
        
        # update estimates of differential variables
        p_kp1= r_k + 1.0/(8*lam1)*grad_ad[0]
        q_kp1= s_k + 1.0/(8*lam1)*grad_ad[1]
        # p_kp1= p_k + 1.0/(8*lam1)*grad_ad[0]
        # q_kp1= q_k + 1.0/(8*lam1)*grad_ad[1]

        # Willem performs this normalization step in his FISTA code but 
        # it does not appear in [1]
        # Preform the projection step
        p_kp1 = p_kp1 / torch.clamp (torch.abs (p_kp1),min= 1.0)
        q_kp1 = q_kp1 / torch.clamp (torch.abs (q_kp1),min= 1.0)

        t_kp1 = (1 + torch.sqrt(1+4*t_k**2))/2.0
        
        # update normalized differential variables
        r_k = p_kp1+(t_k-1)/t_kp1*(p_kp1-p_k)
        s_k = q_kp1+(t_k-1)/t_kp1*(q_kp1-q_k)
        # r_k = p_k+(t_k-1)/t_kp1*(p_k-p_km1)
        # s_k = q_k+(t_k-1)/t_kp1*(q_k-q_km1)
        
        # update variable history before next iteration
        # p_km1 = p_k.clone()
        p_k = p_kp1.clone()
        # p_k = copy.deepcopy(p_kp1) #.clone()
        # q_km1 = q_k.clone()
        q_k = q_kp1.clone()
        # q_k = copy.deepcopy(q_kp1) #.clone()
        
        t_k = t_kp1

    return grad_a



def map_d2x_1st2ndOrder_jit(p:torch.Tensor,q:torch.Tensor,x:torch.Tensor): #x:torch.tensor):
    """
    Maps differential variables p and q (or r and s) to pixel space description
    of a 2D parameter space, x.
    This is script L in [1]
    """    
    
    x[:-1,:] = p
    x[1:,:] = x[1:,:] - p

    x[:,:-2] = x[:,:-2] + q
    x[:,1:-1] = x[:,1:-1] - 2 * q
    x[:,2:] = x[:,2:] + q
    
    return x
    
    
def map_x2d_1st2ndOrder_jit(x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]: #,q:torch.tensor,p:torch.tensor):
    """
    Maps pixel description of the image, x, to a differential description
    p and q (or r and s)
    """
    
#     p = torch.zeros((x.shape[0]-1,x.shape[1]),device=device)
#     q = torch.zeros((x.shape[0],x.shape[1]-1),device=device)
    
    p = torch.diff(x,dim=0) 
    q = torch.diff(torch.diff(x,dim=1),dim=1) 
    
    # p = x[:-2,:] - 2 * x[1:-1,:] + x[2:,:]
    # q = x[:,:-2] - 2 * x[:,1:-1] + x[:,2:]

    return p,q

@torch.jit.script
def solve_FISTA_subproblem_1st2ndOrder_jit(b:torch.Tensor,lam1:torch.Tensor,
                              lb:torch.Tensor,ub:torch.Tensor)->torch.Tensor:

    """
    Solves optimization subproblem by using FISTA [1].
    For most MLE applications 
    b = x_k - 1/alpha * grad(f)
        where x_k is the current state vector, 1/alpha is the step size,
        f - is the error function so grad(f) is the gradient of f with respect to x
        
    lam - TV penalty function is typically
        lam = 1/alpha or tau/alpha
    
    returns x - the solution to the subproblem
    
    Solving the sub problem should be done separately for each separable variable.
        for example, this function should be run once for backscatter coefficient, lidar ratio and depolarization each.
        For non-TV variables (gain, deadtime), don't use this function call.  Just run steepest descent:
        x_{k+1} = x_k - 1/alpha * grad(x)
        
    This function expects rectangular arrays.  Mapping non-rectangular spaces (e.g. altitude varying range data)
    needs to happen prior to calling this function.
    """

    device = b.device
    dtype = b.dtype
    
    # initialize differential state variables with their history
    # k - this iteration, kp1=k+1, km1 = k-1
    r_k = torch.zeros((b.shape[0]-1,b.shape[1]),device=device,dtype=dtype)
    s_k = torch.zeros((b.shape[0],b.shape[1]-2),device=device,dtype=dtype)
    p_k = torch.zeros((b.shape[0]-1,b.shape[1]),device=device,dtype=dtype)
    q_k = torch.zeros((b.shape[0],b.shape[1]-2),device=device,dtype=dtype)
    
    # p_km1 = torch.zeros((b.shape[0]-1,b.shape[1]),device=device,dtype=dtype)
    p_kp1 = torch.zeros((b.shape[0]-1,b.shape[1]),device=device,dtype=dtype)
    # q_km1 = torch.zeros((b.shape[0],b.shape[1]-1),device=device,dtype=dtype)   
    q_kp1 = torch.zeros((b.shape[0],b.shape[1]-2),device=device,dtype=dtype)
    
    t_k = torch.ones((),device=device) # t_k=1
    
    # grad_a_km1 = torch.zeros(b.shape,device=device)
    grad_a = torch.zeros(b.shape,device=device) # preallocate for map functions

    for _ in range(50):
        grad_a = b - lam1*map_d2x_1st2ndOrder_jit(r_k,s_k,grad_a) # compute the FISTA gradient

        grad_a = torch.min(torch.max(grad_a,lb),ub)

        grad_ad = map_x2d_1st2ndOrder_jit(grad_a)  # map the gradient to differential space
        
        # update estimates of differential variables
        p_kp1= r_k + 1.0/(8*lam1)*grad_ad[0]
        q_kp1= s_k + 1.0/(8*lam1)*grad_ad[1]
        # p_kp1= p_k + 1.0/(8*lam1)*grad_ad[0]
        # q_kp1= q_k + 1.0/(8*lam1)*grad_ad[1]

        # Willem performs this normalization step in his FISTA code but 
        # it does not appear in [1]
        # Preform the projection step
        p_kp1 = p_kp1 / torch.clamp (torch.abs (p_kp1),min= 1.0)
        q_kp1 = q_kp1 / torch.clamp (torch.abs (q_kp1),min= 1.0)

        t_kp1 = (1 + torch.sqrt(1+4*t_k**2))/2.0
        
        # update normalized differential variables
        r_k = p_kp1+(t_k-1)/t_kp1*(p_kp1-p_k)
        s_k = q_kp1+(t_k-1)/t_kp1*(q_kp1-q_k)
        # r_k = p_k+(t_k-1)/t_kp1*(p_k-p_km1)
        # s_k = q_k+(t_k-1)/t_kp1*(q_k-q_km1)
        
        # update variable history before next iteration
        # p_km1 = p_k.clone()
        p_k = p_kp1.clone()
        # p_k = copy.deepcopy(p_kp1) #.clone()
        # q_km1 = q_k.clone()
        q_k = q_kp1.clone()
        # q_k = copy.deepcopy(q_kp1) #.clone()
        
        t_k = t_kp1

    return grad_a