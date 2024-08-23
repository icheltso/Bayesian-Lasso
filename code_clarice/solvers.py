import numpy as np

def soft(x,tau): 
    return np.sign(x)*np.maximum(np.abs(x)-tau,0)

# x = starting vector
# grad = gradient of negative log density (smooth part)
# lam = regularization for l1
# tau = stepsize
# gamma = moreau regularization
def prox_langevin(x, grad, tau, gamma, lam,beta=1):
    grad_F = lambda x,gamma:  grad(x) + (x - soft(x,lam*gamma))/gamma
    p = len(x)
    x_ = x - tau*grad_F(x,gamma) + np.sqrt(2*tau/beta)*np.random.randn(*x.shape)
    return x_



# x = starting vector
# grad = gradient of negative log density (smooth part)
# lam = regularization for l1
# tau = stepsize
def one_step_uv(x, grad, tau, lam,beta=1):
    p = len(x)//2
    ru = lambda x: x[:p]
    rv = lambda x: x[p:]
    prod = lambda x,g: np.concatenate((rv(x)*g ,ru(x)*g))
    Grd = lambda x: prod( x, grad( ru(x)*rv(x) ) )
    # S1 = lambda x: (x+np.sqrt(x**2 + 4*tau*(1+tau*lam)))/2

    def S1(x):
        x = np.abs(x)
        if x<1:
            return (x+np.sqrt(x**2 + 4*tau*(1+tau*lam)/beta))/2.
        else:
            return 0.5*x+0.5*x* np.sqrt(1+ (4/beta*tau*(1+tau*lam)/x)/x)
    S1 = np.vectorize(S1)
    z = x - tau*Grd(x) + np.random.randn(*x.shape)*np.sqrt(2*tau/beta)
    z[:p] = S1(z[:p])
    x_ = z/(1+tau*lam)
    return x_




from numpy.linalg import inv
from scipy.stats import invgamma, norm,invgauss


# Gibbs Sampler for Bayesian Lasso with sigma^2 = 1
def one_step_gibbs(x_eta, A,y, lam):
    
    n,p = A.shape

    eta = x_eta[p:]

    # Sample x | y, X, eta
    V_x = inv(A.T@A + np.diag(1/eta))
    m_x = V_x @ A.T @ y
    x = np.random.multivariate_normal(m_x, V_x)

    # Sample eta_j | x_j
    for j in range(p):
        eta[j]= 1/invgauss.rvs(mu=abs(1./(lam*x[j])), scale=(lam**2))

    
    return np.concatenate((x,eta))
    