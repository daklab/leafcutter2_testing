import torch

import pyro
import pyro.distributions as dist
from torch.distributions import constraints

import numpy as np
import scipy.stats
import time
from sklearn import linear_model

from leafcutter.differential_splicing.optim import fit_with_lbfgs

from leafcutter.differential_splicing.dm_glm import brr_initialization, rr_initialization

from dataclasses import dataclass

#torch.set_num_threads(20)

@dataclass
class LeafcutterFit:
    """
    Stores the result of fitting the LeafCutter differential splicing model to one cluster. 
    """
    beta: torch.Tensor
    conc: torch.Tensor
    loss: float
    exit_status: str

class LeafCutterModel(pyro.nn.PyroModule):
    """
    The LeafCutter differential splicing model for one cluster. 
    """
    
    def __init__(self, P, J_list, eps = 1e-8, gamma_shape = 1.0001, gamma_rate = 1e-4, beta_scale = np.inf, multiconc = True): 
        """
        Initialize the LeafCutterModel.

        Args:
            P (iterable of ints): Number of covariates.
            J (iterable of ints): Number of junctions.
            eps (float): A small constant to prevent numerical issues.
            gamma_shape (float): Shape parameter for the gamma distribution used for the concentration prior. 
            gamma_rate (float): Rate parameter for the gamma distribution used for the concentration prior. 
            beta_scale (float): prior std for coefficients. 
            multiconc (bool): Indicates whether to use a separate concentration parameter for each junction.

        """
        super().__init__()
        self.P = P
        self.J = J_list
        self.eps = eps
        self.multinomial = gamma_shape is None
        if not self.multinomial: 
            conc_prior = dist.Gamma(gamma_shape,gamma_rate)
            self.conc_prior = [ (conc_prior.expand([J]).to_event(1) if multiconc else conc_prior) for J in J_list ]
        self.beta_scale = beta_scale
       
    def forward(self, x, y): 
        """
        Run the generative process. 

        Args:
            x (list of torch.Tensor): Input data representing covariates.
            y (list of torch.Tensor): Observed data, i.e. junction counts. 
        """

        factors = []

        for i in range(len(x)): 
            
            with pyro.plate(f"covariates_{i}", self.P[i]): # beta is P (covariates) x J (junctions)
                beta_dist = dist.Normal(0.0,self.beta_scale) if np.isfinite(self.beta_scale) else dist.ImproperUniform(constraints.real, (), ())
                b_param = pyro.sample(f"beta_{i}", beta_dist.expand([self.P[i], self.J[i]]).to_event(1)) 
    
            logits = x[i] @ b_param
            
            if self.multinomial: 
                logits_norm = logits - logits.logsumexp(1, keepdims = True)
                pyro.factor(f"multinomial_{i}", (y * logits_norm).sum())
            else:
                conc_param = pyro.sample(f"conc_{i}", self.conc_prior[i])
                a = logits.softmax(1) * conc_param + self.eps
                sum_a = a.sum(1)
                
                # manual implementation of the likelihood is faster since it avoids calculating normalization constants
                f = (sum_a.lgamma() + (a+y[i]).lgamma().sum(1) - (sum_a + y[i].sum(1)).lgamma() - a.lgamma().sum(1)).sum()
                pyro.factor(f"dm_{i}", f)

                factors.append(f.item())
            
                #y_dist = dist.DirichletMultinomial(a, total_count = y.sum(1))
                #with pyro.plate("data", y.shape[0]):
                #    return pyro.sample("obs", y_dist, obs=y)
        return factors

class BaseGuide():
    """
    Has common functionality for the different possible guides, shouldn't be used directly. 
    """
    
    def __init__(self, init_beta_list, multinomial = False, init_conc = None, multiconc = True, conc_max = 300.):
        self.multiconc = multiconc
        self.conc_max = conc_max
        (P,J_list) = zip(*[ init_beta.shape for init_beta in init_beta_list ])
        torch_types = { "device" : init_beta_list[0].device, "dtype" : init_beta_list[0].dtype }
        self.multinomial = multinomial
        if not multinomial: 
            default_init_conc = [ (torch.full([J],10.,**torch_types) if multiconc else torch.tensor(10.,**torch_types)) for J in J_list ]
            self.init_conc = default_init_conc if (init_conc is None) else init_conc

    @property
    def conc(self):
        if self.multinomial: return torch.inf
        return [ pyro.param(f"conc_loc_{i}", lambda: self.init_conc[i].clone().detach(), constraint=constraints.interval(0., self.conc_max)) for 
            i in range(len(self.init_conc)) ]
    
    @property
    def beta(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    def __call__(self, x, y): # ok i think this acutally incorporates the constraint
        beta_param = self.beta
        
        for i in range(len(x)): 
            if not self.multinomial: 
                conc_param = pyro.param(f"conc_loc_{i}", lambda: self.init_conc[i].clone().detach(), constraint=constraints.interval(0., self.conc_max))
                pyro.sample(f"conc_{i}", dist.Delta(self.conc[i], event_dim = 1 if self.multiconc else 0)) # do we need to adjust for the constraint manually here? 
            with pyro.plate(f"covariates_{i}", beta_param[i].shape[0]):
                pyro.sample(f"beta_{i}", dist.Delta(beta_param[i], event_dim = 1))

        
class SimpleGuide(BaseGuide):
    """
    Doesn't deal with the extra degree of freedom in beta. 
    """
    
    def __init__(self, init_beta, **kwargs):
        super().__init__(init_beta, **kwargs)
        self.init_beta = init_beta

    @property
    def beta(self):
        return [ pyro.param(f"beta_loc_{i}", lambda: self.init_beta.clone().detach()) for i in range(len(self.init_beta)) ]

class CleverGuide(BaseGuide):
    """
    This is the parametrization that the original LeafCutter R/Stan package used. 
    """
    
    def __init__(self, init_beta_list, **kwargs):
        super().__init__(init_beta_list, **kwargs)

        self.init_beta_raw = []
        self.init_beta_scale = []

        for i,init_beta in enumerate(init_beta_list): 
            (P,J) = init_beta.shape
    
            up = init_beta[ [range(P),init_beta.abs().max(1).indices] ] / (1.-1./J)
            down = init_beta[ [range(P),(-init_beta * up.sign()[:,None]).max(1).indices] ] / (1./J)
            beta_scale = up - down
            beta_raw = init_beta / beta_scale[:,None] + 1./J
            beta_raw[beta_scale == 0.,:] = 1./J # can this learn?
            assert(not beta_raw.isnan().any().item())
            beta_raw = beta_raw / beta_raw.sum(1, keepdim = True)
            assert(not beta_raw.isnan().any().item())
            
            self.init_beta_raw.append(beta_raw)
            self.init_beta_scale.append(beta_scale)

        #assert( (init_beta - (beta_raw - 1./J) * beta_scale[:,None]).abs().max().item() < 1e-5 )
        #assert( ((beta_raw.sum(1) - 1.).abs().mean() < 1e-6).item() )
        #beta_reconstructed = beta_scale[:,None] * (beta_raw - 1./J)
        #assert( ((beta_reconstructed - init_beta).abs().mean() < 1e-5).item() )

    @property
    def beta(self):
        beta = []
        for i in range(len(self.init_beta_raw)): 
            beta_raw_param = pyro.param(f"beta_raw_{i}", lambda: self.init_beta_raw[i].clone().detach(), constraint=constraints.simplex)
            beta_scale_param = pyro.param(f"beta_scale_{i}", lambda: self.init_beta_scale[i].clone().detach())
            beta.append( 
                beta_scale_param[:,None] * (beta_raw_param - 1./beta_raw_param.shape[1]) 
            )
        return beta

class DamCleverGuide(BaseGuide):
    """
    This is the more recently proposed approach in the Stan docs: https://mc-stan.org/docs/stan-users-guide/parameterizing-centered-vectors.html under `QR decomposition`. Originally proposed by Aaron J Goodman. 
    """

    def __init__(self, init_beta_list, **kwargs):
        super().__init__(init_beta_list, **kwargs)

        self.init_beta_raw = []
        self.A_qr = []

        for i,init_beta in enumerate(init_beta_list): 
            J = init_beta.shape[1]
            
            A = torch.eye(J, dtype = init_beta.dtype, device = init_beta.device)
            A[-1,:-1] = -1.
            A[-1,-1] = 0. 
            A_qr = torch.linalg.qr(A).Q[:,:-1] # [J x (J-1)]
            # v = torch.eye(J-1) / (1.-1./J)
            # A_qr @ v @ A_qr.t() # gives correct marginals! 
            # beta_trans = A_qr @ (torch.randn(J-1) / np.sqrt(1.-1./J)) # sums to 0! 
            self.init_beta_raw.append( 
                torch.linalg.solve(A_qr.t() @ A_qr, A_qr.t() @ init_beta.t()).t()
            )

            self.A_qr.append(A_qr)

    @property
    def beta(self):
        beta = []
        for i in range(len(self.init_beta_raw)): 
            beta_raw_param = pyro.param(f"beta_raw_{i}", lambda: self.init_beta_raw[i].clone().detach()) # note this transform does not require a Jacobian since it is constant/linear
            beta.append( beta_raw_param @ self.A_qr[i].t() )
        return beta

def fit_multinomial_glm(x_list, y_list, beta_init_list = None, fitter = fit_with_lbfgs, guide_type = DamCleverGuide): 
    """
    Try to get a good initialization for beta by moment matching. V slightly slower than BRR overall.
    """
    P_list = [ x.shape[1] for x in x_list ]
    J_list = [ y.shape[1] for y in y_list ]
    pyro.clear_param_store()
    
    if beta_init_list is None: 
        beta_init_list = [ torch.zeros(P, J, device = x.device, dtype = x.dtype) for P,J in zip(P_list, J_list) ]

    multinomial_model = LeafCutterModel(P_list, J_list, None, gamma_shape = None)
    guide = guide_type(beta_init, multinomial = True)
    losses = fitter(multinomial_model, guide, x, y)
    return guide.beta

def fit_dm_glm(x_list, y_list, beta_init, conc_max = 3000., concShape=1.0001, concRate=1e-4, beta_scale = np.inf, multiconc = True, fitter = fit_with_lbfgs, guide_type = DamCleverGuide, eps = 1.0e-8): 
    """
    Fits a Dirichlet Multinomial generalized linear model.

    Args:
        x (torch.Tensor): The input data with shape (N, P) where N is the number of observations and P is the number of predictors.
        y (torch.Tensor): The target data with shape (N, J) where J is the number of splice junctions.
        beta_init (torch.Tensor): The initial values for the coefficients, shape=(P,J)
        conc_max (float, optional): Maximum concentration parameter value. Defaults to 300.
        concShape (float, optional): Shape parameter for concentration prior. Defaults to 1.0001.
        concRate (float, optional): Rate parameter for concentration prior. Defaults to 1e-4.
        multiconc (bool, optional): Whether to use multiple concentration parameters. Defaults to True.
        fitter (function, optional): The fitting function to use. Defaults to fit_with_lbfgs.
        guide_type (class, optional): The type of guide to use. Defaults to DamCleverGuide.
        eps (float): small pseudocount added to DM parameter for numerical stability. 

    Returns:
        LeafcutterFit: An object containing fitted model parameters, losses, and exit status.
    """
    pyro.clear_param_store()

    P_list = [ x.shape[1] for x in x_list ]
    J_list = [ y.shape[1] for y in y_list ]
    
    model = LeafCutterModel(P_list, J_list, gamma_shape = concShape, gamma_rate = concRate, beta_scale = beta_scale, multiconc = multiconc, eps = eps)
    guide = guide_type(beta_init, multiconc = multiconc, conc_max = conc_max)
    losses, exit_status = fitter(model, guide, x_list, y_list)

    return LeafcutterFit(
            beta = [ g.clone().detach() for g in guide.beta ], 
            conc = [ g.clone().detach() for g in guide.conc ], 
            loss = model(x_list, y_list),
            exit_status = exit_status
        ) 

def dirichlet_multinomial_anova(x_full_list, x_null_list, y_list, init = "brr", **kwargs): 
    """
    Perform Dirichlet-Multinomial ANOVA analysis.

    Args:
        x_full_list (list of torch.Tensor): Full (i.e. alternative hypothesis) covariate data.
        x_null_list (list of torch.Tensor): Null (i.e. null hypothesis) covariate data.
        y_list (list of torch.Tensor): Observed junction counts
        init (str): initialization strategy. One of "brr" (Bayesian ridge regression), "rr" (ridge regression), "mult" (multinomial logistic regression) or "0" (set to 0). 
        concShape (float): Shape parameter for concentration priors (default: 1.0001).
        concRate (float): Rate parameter for concentration priors (default: 1e-4).
        multiconc (bool): Indicates whether to use separate concentration parameters for each junction (default: False).
        fitter (function): A function used to fit the model (default: fit_with_lbfgs).
        guide_type: one of BasicGuide, CleverGuide or DamCleverGuide

    Returns:
        tuple: A tuple containing the following elements:
            - loglr (float): Log-likelihood ratio statistic.
            - df (int): Degrees of freedom.
            - lrtp (float): Likelihood ratio test p-value.
            - null_fit (object): Result of fitting the null model.
            - full_fit (object): Result of fitting the full model.
            - refit_null_flag (bool): Flag indicating if the null model was refitted based on the full model.
    """

    P_full_list = [ x_full.shape[1] for x_full in x_full_list ]
    P_null_list = [ x_null.shape[1] for x_null in x_null_list ]
    J_list = [ y.shape[1] for y in y_list ]

    num_models = len(x_full_list) 
    
    torch_types = { "device" : y_list[0].device, "dtype" : y_list[0].dtype }
    
    def get_init(x,y): 
        if init == "brr":
            return brr_initialization(x,y)
        elif init == "rr": 
            return rr_initialization(x,y)
        elif init == "mult": # doesn't speed things up
            mult_kwargs = { k:v for k,v in kwargs.items() if k in ["fit_with_lbfgs", "guide_type"]}
            return fit_multinomial_glm(x_null, y, **mult_kwargs) 
        elif init == "0": 
            return torch.zeros(x.shape[1], y.shape[1], **torch_types)
        else: 
            raise ValueError(f"Unknown initialization strategy {init}")

    # fit null model
    start_time = time.time()
    beta_init_null = [ get_init(x, y) for x,y in zip(*[x_null_list, y_list]) ]
    null_fit = fit_dm_glm( x_null_list, y_list, beta_init_null, **kwargs )
    elapsed_time = time.time() - start_time
    #print(f"Elapsed time: {elapsed_time} seconds, {null_fit.loss}")

    # fit full model, initialized at null model solution
    beta_init_full = []
    for i in range(num_models): 
        beta_init_full.append( 
            torch.cat( (null_fit.beta[i], torch.zeros((P_full_list[i] - P_null_list[i], J_list[i]), **torch_types)) )
        )
        
    #beta_init_full -= beta_init_full.mean(1, keepdim = True) # shouldn't be necessary with the clever guides
    full_fit = fit_dm_glm( x_full_list, y_list, beta_init_full, **kwargs )
    
    # fit full model, initialized "smartly", and check if this gives a better fit
    beta_init_full = [ get_init(x, y) for x,y in zip(*[x_full_list, y_list]) ]

    full_fit_smart = fit_dm_glm( x_full_list, y_list, beta_init_full, **kwargs )

    df_list = []
    loglr_list = []
    lrtp_list = []
    for i in range(num_models): 
        if full_fit_smart.loss[i] < full_fit.loss[i]: # this initialization did better
            full_fit.beta[i] = full_fit_smart.beta[i]
            full_fit.loss[i] = full_fit_smart.loss[i]
            full_fit.conc[i] = full_fit_smart.conc[i]
        
        df=(P_full_list[i] - P_null_list[i])*(J_list[i]-1)
        
        refit_null_flag=False
        loglr = null_fit.loss[i] - full_fit.loss[i]
        lrtp = scipy.stats.chi2(df).sf(2.*loglr)

        df_list.append(df)
        loglr_list.append(loglr)
        lrtp_list.append(lrtp)

    # annoying to do
    if False: # lrtp < 0.001: # if result looks highly significant, check if we could improve fit initializing null based on full
        beta_init_null = full_fit.beta[:P_null,:]
        refit_null = fit_dm_glm(x_null, y, beta_init_null, **kwargs )
        if refit_null.loss < refit_null.loss: # if new fit is better
            null_fit = refit_null
            refit_null_flag = True
            loglr = null_fit.loss-full_fit.loss
            lrtp = scipy.stats.chi2(df = df).sf(2.*loglr)
    
    return loglr_list, df_list, lrtp_list, null_fit, full_fit, refit_null_flag
