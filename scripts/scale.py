import reciprocalspaceship as rs
from uvscale.distributions import Amoroso
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




num_steps = 500
#losses = model.map_fit_model(num_steps)
losses = model.fit_model(num_steps)

from matplotlib import pyplot as plt

plt.figure()
plt.plot(losses)

plt.figure()
X = model.X.detach().numpy().flatten()
loc = model.likelihood.rec_func(model(model.X)[0])
q = Amoroso.wilson_prior(torch.tensor(False), torch.tensor(1.), loc)
loc = q.mean().detach().numpy().flatten()
scale = q.stddev().detach().numpy().flatten()

idx = np.argsort(X)
X,loc,scale = X[idx],loc[idx],scale[idx]
plt.fill_between(X, loc-scale, loc+scale, alpha=0.5)
plt.plot(X, loc, 'k-')

X = model.Xu.detach().numpy().flatten()
loc = model.likelihood.rec_func(model(model.Xu)[0])
loc  = Amoroso.wilson_prior(torch.tensor(False), torch.tensor(1.), loc).mean().detach().numpy().flatten()
plt.plot(X, loc, 'r.')

plt.plot(model.X.detach(), model.y.detach(), 'kx', alpha=0.2, zorder=0)

plt.xlim(model.X.min(), model.X.max())

plt.xlabel("$1  / D_h^{2}\ (\AA^{-2})$")
plt.ylabel("$|F|$")
plt.xticks([])
plt.yticks([])

from IPython import embed
embed()
