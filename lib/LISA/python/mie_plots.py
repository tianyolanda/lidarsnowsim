#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is used to generate the Mie theory vs experiment comparison plot
"""
import numpy as np
from lisa import LISA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mode = 'snow_1958'

augmenter = LISA(mode=mode)

title = '-'.join([s.capitalize() for s in augmenter.Nd.__name__.replace('_Nd', '').split('_')])

# Mie
D  = augmenter.D        # diameter (mm)
Qe = augmenter.qext     # extinction efficiency

# Alpha vs Rain rate
N=100
Rr = np.linspace(1, 50, N)

k1 = 1.45 * (Rr**.64)   # from: Ulbrich and Atlas, "Extinction of visible and Infrared Radiation in rain", Eq 8
k2 = np.zeros(k1.shape)

alpha = np.zeros(k1.shape)
beta = np.zeros(k1.shape)

for i in range(N):

    Nd_rain = augmenter.Nd(D, Rr[i])

    alpha[i] = augmenter.alpha(Nd_rain)
    beta[i] = augmenter.beta(Nd_rain)

    k2[i] = alpha[i] * 1000 * (10 / np.log(10))

Nd_rain_10 = augmenter.Nd(D, 10)
Nd_rain_20 = augmenter.Nd(D, 20)
Nd_rain_50 = augmenter.Nd(D, 50)

plt.rcParams.update({
    "text.usetex": True})

fig = plt.figure(constrained_layout=True,figsize=(6,6))
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[1.2,1])

ax1 = fig.add_subplot(spec[0, 0])
ax1.loglog(D, Qe, lw = 1)
ax1.set_xlim((1e-4,10))
ax1.set_ylim((1e-3,5))
ax1.set_ylabel('Extinction Efficiency', fontsize=12)

ax2 = fig.add_subplot(spec[1, 0])
ax2.loglog(D, Nd_rain_10, lw = 1)
ax2.loglog(D, Nd_rain_20, lw = 1)
ax2.loglog(D, Nd_rain_50, lw = 1)
ax2.set_ylabel('Raindrop Size Distribution $m^-3 \, mm^-1$' ,fontsize=12)
ax2.set_xlabel('Droplet Diameter $mm$' ,fontsize=12)
ax2.legend(('$Rr=10 \, mm/hr$', '$Rr=20 \, mm/hr$', '$Rr=50 \, mm/hr$'))

ax3 = fig.add_subplot(spec[:, 1])
ax3.plot(Rr, k1, lw = 1)
ax3.plot(Rr, k2, lw = 1)
ax3.set_ylabel('Extinction Coefficient $dB/km$', fontsize=12)
ax3.set_xlabel('Rain rate $mm/hr$' ,fontsize=12)
ax3.legend(('Ulbrich-Atlas', f'{title}'))

fig.savefig(f'imgs/{mode}.png',dpi=300)
