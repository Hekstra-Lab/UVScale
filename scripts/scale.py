import reciprocalspaceship as rs
import numpy as np
from uvscale.model import Scale

inFN = 'pyp_0.mtz'
ds = rs.read_mtz(inFN).compute_dHKL().compute_multiplicity().label_centrics()

dHKL = ds.dHKL.to_numpy(np.float32)
centric = ds.CENTRIC.to_numpy()
epsilon = ds.EPSILON.to_numpy(np.float32)
F = ds.F.to_numpy(np.float32)

model = Scale.from_resolution_and_structure_factors(dHKL, centric, epsilon, F)




num_steps = 500
losses = model.fit_model(num_steps)

from matplotlib import pyplot as plt

plt.figure()
plt.plot(losses)

plt.figure()
loc = model.likelihood.rec_func(model(model.X)[0]).detach().numpy()
plt.plot(model.X.numpy(), loc, 'k.')
X = model.Xu.detach().numpy().flatten()
loc = model.likelihood.rec_func(model(model.Xu)[0]).detach().numpy().flatten()
scale = model.likelihood.rec_func(model(model.Xu)[0]).detach().sqrt().numpy().flatten()
idx = np.argsort(X)
X,loc,scale = X[idx],loc[idx],scale[idx]
plt.errorbar(X, loc, yerr=scale)
plt.plot(model.X.detach(), model.y.detach(), 'kx')


from IPython import embed
embed()
