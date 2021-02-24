import reciprocalspaceship as rs
import torch
import numpy as np
from uvscale.model import Scale

inFN = 'pyp_0.mtz'
model = Scale.isotropic_from_mtz(inFN, num_inducing_points=100)


losses = model.fit_model(batch_size=100, epochs=100)

from matplotlib import pyplot as plt

plt.figure()
plt.plot(losses)

plt.figure(figsize=(7, 4))
loc,scale = model(model.X)
loc = loc.detach().numpy()
scale = scale.sqrt().detach().numpy()
X = model.X.detach().numpy().flatten()

idx = np.argsort(X)
X,loc,scale = X[idx],loc[idx],scale[idx]
plt.fill_between(X, loc-scale, loc+scale, alpha=0.5)
plt.plot(X, loc, 'k-')

loc,scale = model(model.Xu)
loc = loc.detach().numpy()
scale = scale.sqrt().detach().numpy()
X = model.Xu.detach().numpy().flatten()

idx = np.argsort(X)
X,loc,scale = X[idx],loc[idx],scale[idx]
plt.plot(X, loc, 'r.')

plt.plot(model.X.detach(), model.y.detach(), 'kx', alpha=0.2, zorder=0)

plt.xlim(model.X.min(), model.X.max())

plt.xlabel("$1  / D_h^{2}\ (\AA^{-2})$")
plt.ylabel("$|F|$")
plt.xticks([])
plt.yticks([])

from IPython import embed
embed()
