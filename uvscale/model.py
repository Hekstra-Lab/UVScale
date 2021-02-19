import pyro 
import numpy as np
from .distributions import Amoroso
import torch
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.distributions.util import broadcast_all


def sum_kernels(*args):
    kernel = args[0]
    for k in args[1:]:
        kernel = gp.kernels.Sum(kernel, k)
    return kernel

class FWilsonLikelihood(gp.likelihoods.Likelihood):
    def __init__(self, centric, epsilon, rec_func=None):
        super().__init__()
        self.centric = centric
        self.epsilon = epsilon
        if rec_func is None:
            self.rec_func = torch.nn.Softplus()
            #eps = 1e-5
            #self.rec_func = lambda x : x.clamp(min=eps)

    def forward(self, f_loc, f_var, y=None):
        Sigma = dist.Normal(f_loc, f_var.sqrt())()
        Sigma = self.rec_func(Sigma)
        idx = Sigma > 0
        Sigma = Sigma[idx]
        centric = self.centric[idx]
        epsilon = self.epsilon[idx]
        y_dist = Amoroso.wilson_prior(centric, epsilon, Sigma)
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-Sigma.dim()]).to_event(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)

class RBF(gp.kernels.RBF):
    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        scaled_X = X / self.lengthscale
        scaled_Z = Z / self.lengthscale
        r2 = ((scaled_X[:,None,...] - scaled_Z[None,:,...])**2).sum(-1)
        return r2.clamp(min=0)



#class Scale(gp.models.VariationalSparseGP):
class Scale(gp.models.VariationalSparseGP):
    @classmethod
    def from_resolution_and_structure_factors(cls, dHKL, centric, epsilon, F, num_inducing_points=100, jitter=1e-4):
        """
        Parameters
        ----------
        dHKL : tensor(float)
            Resolution of each reflection
        centric : tensor(bool)
            True for centrics reflections, False for acentrics
        epsilon : tensor(float)
            The multiplicities of each reflection
        F : tensor(float)
            Observed structure factor
        """
        if not torch.is_tensor(dHKL):
            dHKL = torch.tensor(dHKL)
        if not torch.is_tensor(centric):
            centric = torch.tensor(centric)
        if not torch.is_tensor(epsilon):
            epsilon = torch.tensor(epsilon)
        if not torch.is_tensor(F):
            F = torch.tensor(F)

        dHKL, centric, epsilon, F = broadcast_all(dHKL, centric, epsilon, F)
        likelihood = FWilsonLikelihood(centric, epsilon)
        X = dHKL[:,None]**-2.
        X = 1.*(X-X.mean())/X.std()
        y = F
        Xu = torch.linspace(torch.min(X), torch.max(X), num_inducing_points)[:,None]
        #Xu = X[np.random.choice(len(X), num_inducing_points, replace=False)]
        #kernel = gp.kernels.RationalQuadratic(input_dim=1)
        kernel = sum_kernels(
            gp.kernels.RBF(input_dim=1, lengthscale=torch.tensor(1.)),
            gp.kernels.RationalQuadratic(input_dim=1, lengthscale=torch.tensor(1.)),
            gp.kernels.Matern32(input_dim=1, lengthscale=torch.tensor(1.)),
            gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(1.)),
            gp.kernels.WhiteNoise(input_dim=1),
        )
        #kernel = RBF(input_dim=1) 
        return cls(X, y, kernel, Xu, likelihood, jitter=jitter, mean_function=lambda x: torch.mean(y))
#
#    def fit_model(self, num_steps, num_particles=1):
#        loss = pyro.infer.TraceMeanField_ELBO(num_particles).differentiable_loss
#        opt = torch.optim.Adam(self.parameters(), lr=0.001)
#        return gp.util.train(self, num_steps=num_steps, optimizer=opt, loss_fn=loss)

    def posterior(self):
        """
        Return the posterior over structure factors associated with self.X.
        This is not technically the "True" posterior but a decent stand in.
        Getting the "real" posterior would involve an integral over Sigma.
        Properly, this is Wilson(<Sigma>)

        Returns
        -------
        distributions.Amoroso

        """
        Sigma, _ = self(self.X)
        Sigma = self.likelihood.rec_func(Sigma)
        return Amoroso.wilson_prior(self.likelihood.centric, self.likelihood.epsilon, Sigma)

    def fit_model(self, num_steps, lr=0.01, num_particles=1, retain_graph=None):
        from tqdm import trange
        losses = []
        loss_fn = pyro.infer.TraceMeanField_ELBO(num_particles).differentiable_loss
        #loss_fn = pyro.infer.Trace_ELBO(num_particles).differentiable_loss
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        def closure():
            opt.zero_grad()
            loss = loss_fn(self.model, self.guide)
            if not(torch.isfinite(loss)):
                opt.zero_grad()
                print("Loss was not finite, zeroing gradients")
            pyro.infer.util.torch_backward(loss, retain_graph=retain_graph)
            return loss

        for i in trange(num_steps):
            loss = opt.step(closure)
            losses.append(loss.item())

        return losses
