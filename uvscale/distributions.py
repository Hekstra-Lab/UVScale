import torch
from torch._six import inf
from torch.distributions import constraints
from torch.distributions.transforms import ComposeTransform,ExpTransform,AffineTransform,PowerTransform
from pyro.distributions.torch_distribution import TorchDistributionMixin
from torch.distributions.gamma import Gamma
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
        self.a, self.theta, self.alpha, self.beta = broadcast_all(a, theta, alpha, beta)

        base_dist = Gamma(self.alpha, 1.)
        transform = ComposeTransform([
            AffineTransform(-self.a/self.theta, 1/self.theta),
            PowerTransform(self.beta),
        ]).inv
        super().__init__(base_dist, transform)

    @classmethod
    def wilson_prior(cls, centric, epsilon, Sigma):
        """ 
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

        theta = centric*torch.sqrt(2. * epsilon * Sigma) + (~centric)*torch.sqrt(Sigma*epsilon)
        alpha = centric*0.5 + (~centric)*1.
        beta = 2.
        return cls(0., theta, alpha, beta)

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
    p = Amoroso.wilson_prior(centric, epsilon, Sigma)
    plt.figure()
    plt.title("Centric")
    plt.plot(X, torch.exp(p.log_prob(X[:,None])))
    plt.legend(epsilon, title='Epsilon')

    centric = False
    p = Amoroso.wilson_prior(centric, epsilon, Sigma)
    plt.figure()
    plt.title("Acentric")
    plt.plot(X, torch.exp(p.log_prob(X[:,None])))
    plt.legend(epsilon, title='Epsilon')
    
    

    embed(colors='Linux')
