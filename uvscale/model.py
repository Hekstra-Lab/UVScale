import pyro 
import numpy as np
import reciprocalspaceship as rs
from pyro.contrib.gp.util import conditional
from .distributions import Stacy
import torch
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_all
from pyro.nn.module import pyro_method


def sum_kernels(*args):
    kernel = args[0]
    for k in args[1:]:
        kernel = gp.kernels.Sum(kernel, k)
    return kernel

class FWilsonLikelihood(gp.likelihoods.Likelihood):
    def __init__(self, deg=50):
        """
        Parameters
        deg : int
            The degree of Chebyshev polynomial approximation being used.
        """
        super().__init__()
        self.deg = deg

    def forward(self, f_loc, f_var, y, centric, epsilon):
        """
        Parameters
        ----------
        f_loc : torch.Tensor
            The mean of the GP posterior. 
        f_var : torch.Tensor
            The variance of the GP posterior.
        y : torch.Tensor
            Experimental structure factor amplitude
        centric : torch.Tensor
            A tensor of booleans indicating which of the structure factors are centric.
        epsilon : torch.Tensor
            A tensor of floating point reflection multiplicities.
        """
        grid, weights = np.polynomial.chebyshev.chebgauss(self.deg)
        weights = torch.tensor(weights)
        loc,scale = f_loc,f_var.sqrt()

        #Change of interval
        width = scale*20. 
        lower = torch.maximum(loc - width/2., torch.tensor(0.))
        upper = lower + width
        prefactor = (upper - lower)/2.
        Sigma = (upper - lower)[None,:]*torch.tensor(grid)[:,None]/2. + (upper + lower)[None,:]/2.
        weights = weights*np.sqrt(1 - grid**2.)

        #Make the wilson distributions
        centric = self.centric if centric is None else centric
        epsilon = self.epsilon if epsilon is None else epsilon
        theta = centric*torch.sqrt(2. * epsilon) * Sigma + (~centric)*torch.sqrt(epsilon)*Sigma
        alpha = centric*0.5 + (~centric)*1.
        beta = 2.
        wilson_dist = Stacy(theta, alpha, beta)

        #Compute the <log P(I | X)> = int log P(I | Σ) * P(Σ | X) dΣ
        ll = wilson_dist.log_prob(y) * torch.exp(dist.Normal(loc, scale).log_prob(Sigma))
        expected_ll = prefactor * (weights@ll)

        y_dist = pyro.distributions.Delta(y, log_density=expected_ll)
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)

class Scale(gp.models.VariationalSparseGP):
    def __init__(self, X, y, kernel, Xu, likelihood, centric, epsilon, mean_function=None, num_data=None, whiten=False, jitter=1e-6):
        """
        Most users will find the `classmethod` decorated constructors more useful for constructing these `Scale` models.
        """
        super().__init__(X, y, kernel, Xu, likelihood, 
            mean_function=mean_function, num_data=num_data, whiten=whiten, jitter=jitter)
        self.centric = centric
        self.epsilon = epsilon

    @classmethod
    def _from_x_y_centric_epsilon(cls, X, y, centric, epsilon, kernel=None, num_inducing_points=100, **kwargs):
        if kernel is None:
            kernel = sum_kernels(
                gp.kernels.RBF(input_dim=1, lengthscale=torch.tensor(1.)),
                gp.kernels.RationalQuadratic(input_dim=1, lengthscale=torch.tensor(1.)),
                gp.kernels.Matern32(input_dim=1, lengthscale=torch.tensor(1.)),
                gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(1.)),
                gp.kernels.WhiteNoise(input_dim=1, variance=torch.tensor(0.01)),
            )

        Xu = X[np.random.choice(len(X), num_inducing_points, replace=False)]
        likelihood = FWilsonLikelihood()
        if 'mean_function' not in kwargs:
            mean = y.mean()
            kwargs['mean_function'] = lambda x: mean

        X,y,Xu,centric,epsilon = map(torch.tensor, (X,y,Xu,centric,epsilon))

        return cls(X, y, kernel, Xu, likelihood, centric, epsilon, **kwargs)
        

    @classmethod
    def isotropic_from_dataset(cls, ds, fkey=None, **kwargs):
        ds = ds.compute_dHKL().label_centrics().compute_multiplicity()
        X = ds.dHKL.to_numpy(np.float32)[:,None]**-2.

        if fkey is None:
            fkeys = ds.keys()[ds.dtypes=='F']
            if len(fkeys) == 0:
                raise ValueError("No fkey supplied and no columns with dtype=='F' in ds")
            fkey = fkeys[0]
        y = ds[fkey].to_numpy(np.float32)

        centric = ds.CENTRIC.to_numpy(bool)
        epsilon = ds.EPSILON.to_numpy(np.float32)

        return cls._from_x_y_centric_epsilon(X, y, centric, epsilon, **kwargs)

    @classmethod
    def anisotropic_from_dataset(cls, ds, fkey=None, **kwargs):
        ds = ds.stack_anomalous().compute_dHKL().label_centrics().compute_multiplicity()
        X = ds.get_hkls().astype(np.float32)

        if fkey is None:
            fkeys = ds.keys()[ds.dtypes=='F']
            if len(fkeys) == 0:
                raise ValueError("No fkey supplied and no columns with dtype=='F' in ds")
            fkey = fkeys[0]
        y = ds[fkey].to_numpy(np.float32)

        centric = ds.CENTRIC.to_numpy(bool)
        epsilon = ds.EPSILON.to_numpy(np.float32)

        return cls._from_x_y_centric_epsilon(X, y, centric, epsilon, **kwargs)

    @classmethod
    def isotropic_from_mtz(cls, mtz_file, **kwargs):
        ds = rs.read_mtz(mtz_file)
        return cls.isotropic_from_dataset(ds, **kwargs)

    @classmethod
    def anisotropic_from_mtz(cls, mtz_file, **kwargs):
        ds = rs.read_mtz(mtz_file)
        return cls.anisotropic_from_dataset(ds, **kwargs)

    @pyro_method
    def model(self):
        """
        This is verbatim the pyro VSPGP model method except that the call to
        the likelihood was modified to pass centric and epsilon. 
        """
        self.set_mode("model")

        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
        Luu = Kuu.cholesky()

        zero_loc = self.Xu.new_zeros(self.u_loc.shape)
        if self.whiten:
            identity = eye_like(self.Xu, M)
            pyro.sample(self._pyro_get_fullname("u"),
                        dist.MultivariateNormal(zero_loc, scale_tril=identity)
                            .to_event(zero_loc.dim() - 1))
        else:
            pyro.sample(self._pyro_get_fullname("u"),
                        dist.MultivariateNormal(zero_loc, scale_tril=Luu)
                            .to_event(zero_loc.dim() - 1))

        f_loc, f_var = conditional(self.X, self.Xu, self.kernel, self.u_loc, self.u_scale_tril,
                                   Luu, full_cov=False, whiten=self.whiten, jitter=self.jitter)

        f_loc = f_loc + self.mean_function(self.X)
        if self.y is None:
            return f_loc, f_var
        else:
            # we would like to load likelihood's parameters outside poutine.scale context
            self.likelihood._load_pyro_samples()
            with poutine.scale(scale=self.num_data / self.X.size(0)):
                return self.likelihood(f_loc, f_var, self.y, self.centric, self.epsilon)

    def set_data(self, X, y=None, centric=None, epsilon=None):
        super().set_data(X, y)
        self.centric = centric
        self.epsilon = epsilon

    def fit_model(self, batch_size, epochs, lr=0.001, betas=(0.9, 0.99), num_particles=1, retain_graph=None):
        from tqdm import tqdm
        splits = int(len(self.X) / batch_size)

        #Cache full data set
        X,y,centric,epsilon = self.X,self.y,self.centric,self.epsilon

        losses = []
        loss_fn = pyro.infer.TraceMeanField_ELBO(num_particles).differentiable_loss
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)

        for i in tqdm(range(epochs), desc="Overall", position=0, leave=True, total=epochs):
            for (
                    batch_X,
                    batch_y,
                    batch_centric,
                    batch_epsilon,
                ) in tqdm(zip(
                    X.split(batch_size),
                    y.split(batch_size),
                    centric.split(batch_size),
                    epsilon.split(batch_size),
                ), 
                desc=f"  Epoch", total=splits-1, position=1, leave=False):
                # Execute training step
                self.set_data(batch_X, batch_y, batch_centric, batch_epsilon)
                opt.zero_grad()
                loss = loss_fn(self.model, self.guide)
                if not(torch.isfinite(loss)):
                    opt.zero_grad()
                    print("Loss was not finite, zeroing gradients")
                pyro.infer.util.torch_backward(loss, retain_graph=retain_graph)
                losses.append(loss.item())
                opt.step()

        #Return to original data
        self.set_data(X, y, centric, epsilon)
        return losses
        

