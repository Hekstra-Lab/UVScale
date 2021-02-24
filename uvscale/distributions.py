import torch
import numpy as np
from torch._six import inf
from torch.distributions import constraints
from torch.distributions.transforms import ComposeTransform,ExpTransform,AffineTransform,PowerTransform,AbsTransform
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.distributions import Weibull,HalfNormal,FoldedDistribution,Normal
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.utils import broadcast_all

class Amoroso(torch.distributions.TransformedDistribution, TorchDistributionMixin):
    arg_constraints = {
        'a': constraints.real,
        'theta': constraints.real,
        'alpha': constraints.positive,
        'beta': constraints.real,
    }
    def __init__(self, a, theta, alpha, beta):
        """
        The Amoroso distribution is a very flexible 4 parameter distribution which 
        contains many important exponential families as special cases. 

        *PDF*
        ```
        Amoroso(x | a, θ, α, β) = 1/gamma(α) * abs(β/θ) * ((x - a)/θ)**(α*β-1) * exp(-((x - a)/θ)**β)
        for:
            x, a, θ, α, β \in reals, α > 0
        support:
            x >= a if θ > 0
            x <= a if θ < 0
        ```
        """
        self.a, self.theta, self.alpha, self.beta = broadcast_all(a, theta, alpha, beta)

        base_dist = Gamma(self.alpha, 1.)
        transform = ComposeTransform([
            AffineTransform(-self.a/self.theta, 1/self.theta),
            PowerTransform(self.beta),
        ]).inv
        super().__init__(base_dist, transform)

    def mean(self):
        """
        The mean of of the Amoroso distribution exists for `alpha + 1/beta >= 0`.
        It can be computed analytically by

        ```
        mean = a + theta * gamma(alpha + 1/beta) / gamma(alpha)
        ```
        """
        a,theta,alpha,beta = self.a,self.theta,self.alpha,self.beta
        return a + torch.exp(torch.log(theta)+torch.lgamma(alpha + torch.reciprocal(beta)) - torch.lgamma(alpha))

    def variance(self):
        """
        The variance of of the Amoroso distribution exists for `alpha + 2/beta >= 0`.
        It can be computed analytically by

        ```
        variance = theta**2 * (gamma(alpha + 2/beta) / gamma(alpha) - gamma(alpha + 1/beta)**2 / gamma(alpha)**2 )
        ```
        """
        theta,alpha,beta = self.theta,self.alpha,self.beta
        return theta**2. * (torch.exp(torch.lgamma(alpha + 2./beta) - torch.lgamma(alpha)) - torch.exp(2.*torch.lgamma(alpha + 1/beta) - 2.*torch.lgamma(alpha)))

    def stddev(self):
        """
        The variance of of the Amoroso distribution exists for `alpha + 2/beta >= 0`.
        It can be computed analytically by

        ```
        variance = theta**2 * (gamma(alpha + 2/beta) / gamma(alpha) - gamma(alpha + 1/beta)**2 / gamma(alpha)**2 )
        ```

        The standard deviation is computed by
        ```amoroso.stddev() = torch.sqrt(amoroso.variance())```
        """
        return torch.sqrt(self.variance())

    def log_prob(self, value):
        a,theta,value = broadcast_all(self.a, self.theta, value)
        log_prob = super().log_prob(value)
        log_prob[torch.logical_and(theta > 0, value < a)] = -inf
        log_prob[torch.logical_and(theta < 0, value > a)] = -inf
        return log_prob

class Stacy(Amoroso):
    """
    The Stacy distribution is a special case of the Amoroso distribution where the
    location parameter, a, equals zero. 

    *PDF*
    ```
    Stacy(x | θ, α, β) = 1/gamma(α) * abs(β/θ) * (x/θ)**(α*β-1) * exp(-(x/θ)**β)
                       = Amoroso(x | 0, θ, α, β)
    ```

    There is an alternative parameterization of the Stacy distribution in 
    """
    def __init__(self, theta, alpha, beta):
        return super().__init__(0., theta, alpha, beta)

    @classmethod
    def wilson_prior_f(cls, centric, epsilon, Sigma):
        """ 
        Construct a wilson prior over Structure Factors. 

        *Centric*
        ```
        P(F | Σ, ε) = (2*π*Σ*ε)**(-1/2) * exp(-F**2 /(2*Σ*ε))
                    = HalfNormal(F | sqrt(Σ*ε))
                    =    Amoroso(F | 0, sqrt(2*Σ*ε), 0.5, 2)
                    =      Stacy(F | sqrt(2*Σ*ε), 0.5, 2)
        ```
        *Acentric*
        ```
        P(F | Σ, ε) = (2/Σ/ε) * F * exp(-(F**2 / Σ / ε))
                    = Rayleigh(F | sqrt(Σ * ε / 2)) 
                    =  Amoroso(F | 0, 
        ```

        Parameters
        ------
        centric : array(bool)
            True for centric entries.
        epsilon : array(float)
            The multiplicity of each reflection.
        Sigma : array(float):
            The mean intensity. 
        """ 
        if not torch.is_tensor(centric):
            centric = torch.tensor(centric)
        if not torch.is_tensor(epsilon):
            centric = torch.tensor(epsilon)
        if not torch.is_tensor(Sigma):
            centric = torch.tensor(Sigma)

        theta = centric*torch.sqrt(2. * epsilon * Sigma) + (~centric)*torch.sqrt(Sigma*epsilon)
        alpha = centric*0.5 + (~centric)*1.
        beta = 2.
        return cls(theta, alpha, beta)

    @classmethod
    def wilson_prior_i(cls, centric, epsilon, Sigma):
        """ 
        A la French Wilson 1978, the Wilson distributions over intensities are

        *Centric*
        ```
        p_J(J) = (Σ**-1)*exp(-J/Σ)
        J >= 0

        # This is equivalent to
        Exponential(J | Σ) == Amoroso(J | 0, Σ, 1, 1) == Stacy(J | Σ, 1, 1)
        ```

        *Acentric*
        ```
        p_J(J) = (2πΣJ)**(-1/2)*exp(-J/2Σ)
        J >= 0

        # This is equivalent to 
        Gamma(J | 1/2, 2Σ) == Amoroso(J | 0, 2Σ, 1/2, 1) == Stacy(J | 2Σ, 1/2, 1)
        ```
 
        Construct a wilson prior.
        Parameters
        ------
        centric : array(bool)
            True for centric entries.
        epsilon : array(float)
            The multiplicity of each reflection.
        Sigma : array(float):
            The mean intensity. 
        """ 
        if not torch.is_tensor(centric):
            centric = torch.tensor(centric)
        if not torch.is_tensor(epsilon):
            centric = torch.tensor(epsilon)
        if not torch.is_tensor(Sigma):
            centric = torch.tensor(Sigma)

        theta = centric * epsilon * Sigma + (~centric)*2.*Sigma*epsilon
        alpha = centric*1.0 + (~centric)*0.5
        beta = 1.
        return cls(theta, alpha, beta)

    @staticmethod
    def _stacy_params(dist):
        if isinstance(dist, Stacy):
            params = (dist.theta, dist.alpha, dist.beta)
        elif isinstance(dist, Weibull):
            # Weibul(x; k, lambda) = Stacy(x; lambda, 1, k)
            k = dist.concentration
            lam = dist.scale
            params = (lam, 1., k)
        elif isinstance(dist, HalfNormal):
            #HalfNormal(x; scale) = Stacy(x; sqrt(2)*scale, 0.5, 2)
            scale = dist.scale
            params = (np.sqrt(2.) * scale, 0.5, 2.)
        else:
            raise TypeError(f"Equivalent Stacy parameters cannot be determined for distribution, {dist}. " 
                             "Only pyro.distributions.Weibull, pyro.distributions.HalfNormal, or Stacy "
                             "can be converted to Stacy parameterisation")
        return params

    @staticmethod
    def _bauckhage_params(dist):
        theta, alpha, beta = Stacy._stacy_params(dist)
        bauckhage_params = (theta, alpha*beta, beta)
        return bauckhage_params 

    def kl_divergence(self, other):
        """
        The Stacy distribution has an analytical KL div. 
        However, it isn't documented in the same parameterization as the Crooks Amoroso tome. 
        To avoid confusion, I will first translate the Stacy distributions from 
        Crooks's parameterization to the one in the KL div paper. 

        ```
        Stacy(x | a,d,p) = x**(d-1) * exp(-(x/a)**p) / Z
        ```
        where
        ```
        a = theta
        d = alpha * beta
        p = beta
        ```
        Then the KL div is
        ```
        log(p1*a2**d2*gamma(d2/p2)) - log(p2*a1**d1*gamma(d1/p1)) + 
        (digamma(d1/p1)/p1 + log(a1)) * (d1 - d2) + 
        gamma((d1 + p2)/p1) * (a1/a2)**p2 / gamma(d1/p1) - d1/p1
        ```

        See Bauckhage 2014 for derivation. 
        https://arxiv.org/pdf/1401.6853.pdf

        Parameters
        ----------
        other : Stacy or Weibull or HalfNormal
        """
        a1,d1,p1 = self._bauckhage_params(self)
        a2,d2,p2 = self._bauckhage_params(other)

        #The following numerics are easier to read if you alias this
        ln = torch.log

        kl = ln(p1) + d2*ln(a2) + torch.lgamma(d2/p2) - ln(p2) - d1*ln(a1) - torch.lgamma(d1/p1) + \
             (torch.digamma(d1/p1)/p1 + ln(a1))*(d1 - d2) +  \
             torch.exp(torch.lgamma((d1 + p2)/p1) - torch.lgamma(d1/p1) + p2*(ln(a1) - ln(a2))) \
             - d1/p1
        return kl

class FoldedNormal(FoldedDistribution):
    def __init__(self, *args, **kwargs):
        base_dist = Normal(*args, **kwargs)
        super().__init__(base_dist)

    def prob(self, value):
        p = self.base_dist.log_prob(value)
        return torch.exp(p)

    def mean(self):
        u = self.base_dist.loc
        s = self.base_dist.scale
        return s * np.sqrt(2/np.pi) * torch.exp(-0.5 * (u/s)**2.) + u * (1. - 2. * Normal(0., 1.).cdf(-u/s))

    def variance(self):
        u = self.base_dist.loc
        s = self.base_dist.scale
        return u**2. + s**2. - self.mean()**2.

    def stddev(self):
        return self.variance().sqrt()




if __name__=="__main__":
    from IPython import embed
    from matplotlib import pyplot as plt

#Figure 1 gogo
    X = torch.linspace(-3, 3, 1000)
    for beta in [1, 2, 3, 4, 5]:
        plt.plot(X, torch.exp(Amoroso(0, 1, 1, beta).log_prob(X)), label=f"beta={beta}")
    plt.legend()
    plt.title("Figure 1: Amoroso(x | 0, 1, 1, beta)")


#Figure 2 gogo
    plt.figure()
    X = torch.linspace(-3, 3, 1000)
    for beta in [1, 2, 3, 4]:
        plt.plot(X, torch.exp(Amoroso(0, 1, 2, beta).log_prob(X)), label=f"beta={beta}")
    plt.legend()
    plt.title("Figure 1: Amoroso(x | 0, 1, 2, beta)")


#Figure 6 gogo
    plt.figure()
    X = torch.linspace(-3, 3, 1000)
    for beta in [-1, -2, -3]:
        plt.plot(X, torch.exp(Amoroso(0, 1, 2, beta).log_prob(X)), label=f"beta={beta}")
    plt.legend()
    plt.title("Figure 6: Amoroso(x | 0, 1, 2, beta) [beta < 0]")


    epsilon = torch.tensor([1., 2., 3., 4., 6.])
    Sigma   = torch.tensor([1., 1., 1., 1., 1.])
    centric = True
    p = Stacy.wilson_prior_f(centric, epsilon, Sigma)
    plt.figure()
    plt.title("Centric")
    plt.plot(X, torch.exp(p.log_prob(X[:,None])))
    plt.legend(epsilon, title='Epsilon')

    centric = False
    p = Stacy.wilson_prior_f(centric, epsilon, Sigma)
    plt.figure()
    plt.title("Acentric")
    plt.plot(X, torch.exp(p.log_prob(X[:,None])))
    plt.legend(epsilon, title='Epsilon')
    
    
    embed(colors='Linux')
