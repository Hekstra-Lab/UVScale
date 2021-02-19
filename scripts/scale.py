import reciprocalspaceship as rs
import torch
import numpy as np
from uvscale.model import Scale

inFN = 'pyp_0.mtz'
ds = rs.read_mtz(inFN).compute_dHKL().compute_multiplicity().label_centrics()

dHKL = ds.dHKL.to_numpy(np.float32)
centric = ds.CENTRIC.to_numpy()
epsilon = ds.EPSILON.to_numpy(np.float32)
F = ds.F.to_numpy(np.float32)
#
model = Scale.from_resolution_and_structure_factors(dHKL, centric, epsilon, F)




num_steps = 1000
#losses = model.map_fit_model(num_steps)
losses = model.fit_model(num_steps)

from matplotlib import pyplot as plt

plt.figure()
plt.plot(losses)

plt.figure()
X = model.X.detach().numpy().flatten()
q = model.posterior()
loc = q.mean().detach().numpy()
scale = q.stddev().detach().numpy()

idx = np.argsort(X)
X,loc,scale = X[idx],loc[idx],scale[idx]
plt.fill_between(X, loc-scale, loc+scale, alpha=0.6)
plt.plot(X, loc, 'k-')

X = model.Xu.detach().numpy().flatten()
loc = model.likelihood.rec_func(model(model.Xu)[0])
from uvscale.distributions import Amoroso
loc  = Amoroso.wilson_prior(torch.tensor(False), torch.tensor(1.), loc).mean().detach().numpy().flatten()
plt.plot(X, loc, 'r.')
plt.xticks([])
plt.xlabel("$1 / D^2$")
plt.ylabel("$|F|$")

plt.plot(model.X.detach(), model.y.detach(), 'kx', alpha=0.2, zorder=0)
plt.xlim(model.X.min(), model.X.max())

from IPython import embed
embed()
