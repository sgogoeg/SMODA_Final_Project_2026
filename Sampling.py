import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline

## Defining the fonts before plotting:

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "monospace",  # Use monospace font
    "text.latex.preamble": r"\usepackage{courier}"  # Use Courier font
})
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({
    'font.family': 'monospace',  # monospace font
    "text.latex.preamble": r"\usepackage{courier}",
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20,
})

# Approximation of the error function

def erf(x):
    x0 = np.abs(x)
    sign = np.sign(x)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * x0)
    return (1.0 - ((a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5) * np.exp(-x0**2)))*sign

## From Abramowitz & Stegun (1964), formula 7.1.26, Handbook of Mathematical Functions: with Formulas, Graphs, and Mathematical Tables

### Truncated Gaussian distribution

# The limits for the distribution are:

xmin = 60
xmax = 160

# The CDF of a gaussian distribution:

def GCDF(x):
    return 1/2 * (1 + erf(x/np.sqrt(2))) 

# The PDF and CDF of the truncated gaussian distribution: 

def TruncatedGaussianPDF(x, mu, sigma, xmin, xmax):
    xi = (x - mu) / sigma
    alpha = (xmin - mu) / sigma
    beta = (xmax - mu) / sigma
    Z = GCDF(beta) - GCDF(alpha)
    return np.exp(-xi**2/2) / (sigma * np.sqrt(2 * np.pi) * Z)

def TruncatedGaussianCDF(x, mu, sigma, xmin, xmax):
    xi = (x - mu) / sigma
    alpha = (xmin - mu) / sigma
    beta = (xmax - mu) / sigma
    Z = GCDF(beta) - GCDF(alpha)
    return (GCDF(xi) - GCDF(alpha)) / Z 

from scipy.interpolate import CubicSpline

def SamplingTruncatedGaussian(mu, sigma, xmin, xmax, size = 1000, steps = 0.01):
    func = TruncatedGaussianCDF(np.arange(xmin, xmax, steps), mu, sigma, xmin, xmax)
    inter = CubicSpline(func, np.arange(xmin, xmax, steps))
    rand = np.random.rand(size)
    samples = inter(rand)
    return samples

samplesize = 50000

sample1 = SamplingTruncatedGaussian(91.2, 15, xmin, xmax, size=samplesize, steps=0.0001)
sample2 = SamplingTruncatedGaussian(125, 17, xmin, xmax, size=samplesize, steps=0.0001)

# Defining bin edges for a histogram with 20 equidistant bins between 60 and 160
bins = np.linspace(60, 160, 21)

counts1 = np.histogram(sample1, bins=bins)[0]

weights_sumBCKG = 500/samplesize*counts1

plt.figure(figsize=(12, 10))
plt.xlim(60, 160)

plt.hist(bins[:-1], bins=bins, density=True, weights=weights_sumBCKG, edgecolor='red', histtype='step', label='Background')
plt.legend()
plt.xlabel(r"$m$ (arbitrary units)")
plt.ylabel(r"Counts $N$")
plt.savefig("plots/Background.pdf", bbox_inches='tight')

plt.figure(figsize=(12, 10))
counts2 = np.histogram(sample2, bins=bins)[0]

weights_sumSGN = 30/samplesize*counts2

plt.hist(bins[:-1], bins, density=False, weights=weights_sumSGN, edgecolor='blue', histtype='step', label='Signal')
plt.xlim(60, 160)
plt.legend()
plt.xlabel(r"$m$ (arbitrary units)")
plt.ylabel(r"Counts $N$")
plt.savefig("plots/Signal.pdf", bbox_inches='tight')

plt.figure(figsize=(12, 10))
plt.hist(bins[:-1], bins=bins, density=False, weights=500/samplesize*counts1, edgecolor='red', histtype='step', label='Background')
plt.hist(bins[:-1], bins=bins, density=False, weights=30/samplesize*counts2, edgecolor='blue', histtype='step', label='Signal')
plt.xlim(60, 160)
plt.legend()
plt.xlabel(r"$m$ (arbitrary units)")
plt.ylabel(r"Counts $N$")
plt.savefig("plots/Back_and_Signal.pdf", bbox_inches='tight')

weights_sum = 500/samplesize *counts1 + 30/samplesize*counts2

plt.figure(figsize=(12, 10))
plt.hist(bins[:-1], bins=bins, density=False, weights=weights_sum, edgecolor='green', histtype='step', label='Background + Signal')
plt.hist(bins[:-1], bins=bins, density=False, weights=500/samplesize*counts1, edgecolor='red', histtype='step', label='Background')
plt.hist(bins[:-1], bins=bins, density=False, weights=30/samplesize*counts2, edgecolor='blue', histtype='step', label='Signal')
plt.xlim(60, 160)
plt.legend()
plt.xlabel(r"$m$ (arbitrary units)")
plt.ylabel(r"Counts $N$")
plt.savefig("plots/Back_plus_Signal.pdf", bbox_inches='tight')

# Generating the systematic uncertainties that require a new samplings:

# Systematic 1: Variation of the variance of the distributions.

sigmavar = 2

sampleSys111 = SamplingTruncatedGaussian(91.2, 15 + sigmavar, xmin, xmax, size=samplesize, steps=0.005)
sampleSys121 = SamplingTruncatedGaussian(125, 17 + sigmavar, xmin, xmax, size=samplesize, steps=0.005)

sampleSys112 = SamplingTruncatedGaussian(91.2, 15 - sigmavar, xmin, xmax, size=samplesize, steps=0.005)
sampleSys122 = SamplingTruncatedGaussian(125, 17 - sigmavar, xmin, xmax, size=samplesize, steps=0.005)

# Systematic 2: Variation of the means of the distributions.

muvar = 10

sampleSys211 = SamplingTruncatedGaussian(91.2 + muvar, 15, xmin, xmax, size=samplesize, steps=0.005)
sampleSys221 = SamplingTruncatedGaussian(125 + muvar, 17, xmin, xmax, size=samplesize, steps=0.005)

sampleSys212 = SamplingTruncatedGaussian(91.2 - muvar, 15, xmin, xmax, size=samplesize, steps=0.005)
sampleSys222 = SamplingTruncatedGaussian(125 - muvar, 17, xmin, xmax, size=samplesize, steps=0.005)

# Systematic 1: Variation of the variance of the distributions.

countsSys111 = np.histogram(sampleSys111, bins=bins)[0]
countsSys121 = np.histogram(sampleSys121, bins=bins)[0]
countsSys112 = np.histogram(sampleSys112, bins=bins)[0]
countsSys122 = np.histogram(sampleSys122, bins=bins)[0]

weights_sys1plusBCKG = 500/samplesize*countsSys111
weights_sys1plusSGN = 30/samplesize*countsSys121
weights_sys1minusBCKG = 500/samplesize*countsSys112
weights_sys1minusSGN = 30/samplesize*countsSys122

weights_sys1plus = weights_sys1plusBCKG + weights_sys1plusSGN
weights_sys1minus = weights_sys1minusBCKG + weights_sys1minusSGN


fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 10), sharex=False)

ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys1plus, edgecolor='blue', histtype='step', label='Systematic 1 +')
ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys1minus, edgecolor='red', histtype='step', label='Systematic 1 -')
ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sum, edgecolor='green', histtype='step', label='Baseline')
ax[0].set_xlim(60, 160)
ax[1].set_xlim(60, 160)
ax[0].legend()

ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys1plus / weights_sum, color='blue')
ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys1minus / weights_sum, color='red')


ax[1].set_xlim(60, 160)
ax[1].set_xlabel(r"$m$ (arbitrary units)")
ax[0].set_ylabel(r"Counts $N$")
ax[1].set_ylabel(r"$N^{\pm}_i / N_i$")

ax[1].grid()

plt.savefig("plots/Systematic_1.pdf", bbox_inches='tight')

# Systematic 2: Variation of the means of the distributions.

countsSys211 = np.histogram(sampleSys211, bins=bins)[0]
countsSys221 = np.histogram(sampleSys221, bins=bins)[0]
countsSys212 = np.histogram(sampleSys212, bins=bins)[0]
countsSys222 = np.histogram(sampleSys222, bins=bins)[0]

weights_sys2plusBCKG = 500/samplesize*countsSys211
weights_sys2plusSGN = 30/samplesize*countsSys221
weights_sys2minusBCKG = 500/samplesize*countsSys212
weights_sys2minusSGN = 30/samplesize*countsSys222

weights_sys2plus = weights_sys2plusBCKG + weights_sys2plusSGN
weights_sys2minus = weights_sys2minusBCKG + weights_sys2minusSGN

fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 10), sharex=False)

ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys2plus, edgecolor='blue', histtype='step', label='Systematic 2 +')
ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys2minus, edgecolor='red', histtype='step', label='Systematic 2 -')
ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sum, edgecolor='green', histtype='step', label='Baseline')

ax[0].set_xlim(60, 160)
ax[0].legend()

ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys2plus / weights_sum, color='blue')
ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys2minus / weights_sum, color='red')
ax[1].set_xlim(60, 160)
ax[1].set_xlabel(r"$m$ (arbitrary units)")
ax[0].set_ylabel(r"Counts $N$")
ax[1].set_ylabel(r"$N^{\pm}_i / N_i$")

ax[1].grid()

plt.savefig("plots/Systematic_2.pdf", bbox_inches='tight')

# Systematic 3: Adding a factor to the counts.

count_scale = 1.2

weights_sys3plusBCKG = count_scale * weights_sumBCKG
weights_sys3plusSGN = count_scale * weights_sumSGN
weights_sys3minusBCKG = weights_sumBCKG / count_scale
weights_sys3minusSGN = weights_sumSGN / count_scale

weights_sys3plus = count_scale * weights_sum
weights_sys3minus = weights_sum / count_scale

fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 10), sharex=False)


ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys3plus, edgecolor='blue', histtype='step', label='Systematic 3 +')
ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys3minus, edgecolor='red', histtype='step', label='Systematic 3 -')
ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sum, edgecolor='green', histtype='step', label='Baseline')
ax[0].set_xlim(60, 160)
ax[0].legend()

ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys3plus / weights_sum, color='blue')
ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys3minus / weights_sum, color='red')
ax[1].set_xlabel(r"$m$ (arbitrary units)")
ax[0].set_ylabel(r"Counts $N$")
ax[1].set_ylabel(r"$N^{\pm}_i / N_i$")
ax[1].set_xlim(60, 160)
ax[1].grid()

plt.savefig("plots/Systematic_3.pdf", bbox_inches='tight')

# Systematic 4: Assymetric factor that depends m, decreasing or increasing with m.

count_scale = 0.9*(1/(bins[-2] - bins[0])*bins[:-1]) 
invcount_scale = 0.9*(-1/(bins[-2] - bins[0])*bins[:-1]+2)

weights_sys4plusBCKG = count_scale*weights_sumBCKG
weights_sys4minusBCKG = invcount_scale*weights_sumBCKG
weights_sys4plusSGN = count_scale*weights_sumSGN
weights_sys4minusSGN = invcount_scale*weights_sumSGN

weights_sys4plus = weights_sys4plusBCKG + weights_sys4plusSGN
weights_sys4minus = weights_sys4minusBCKG + weights_sys4minusSGN

fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 10), sharex=False)

ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys4plus, edgecolor='blue', histtype='step', label='Systematic 4 +')
ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys4minus, edgecolor='red', histtype='step', label='Systematic 4 -')
ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sum, edgecolor='green', histtype='step', label='Baseline')
ax[0].set_xlim(60, 160)
ax[0].legend()

ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys4plus / weights_sum, color='blue')
ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys4minus / weights_sum, color='red')
ax[1].set_xlim(60, 160)
ax[1].set_xlabel(r"$m$ (arbitrary units)")
ax[0].set_ylabel(r"Counts $N$")
ax[1].set_ylabel(r"$N^{\pm}_i / N_i$")
ax[1].grid()

plt.savefig("plots/Systematic_4.pdf", bbox_inches='tight')

np.savez("weights.npz", bins = bins, weights_sumBCKG=weights_sumBCKG, weights_sumSGN=weights_sumSGN,
         weights_sys1plusBCKG=weights_sys1plusBCKG, weights_sys1plusSGN=weights_sys1plusSGN, weights_sys1minusBCKG=weights_sys1minusBCKG, weights_sys1minusSGN=weights_sys1minusSGN,
         weights_sys2plusBCKG=weights_sys2plusBCKG, weights_sys2minusBCKG=weights_sys2minusBCKG, weights_sys2plusSGN=weights_sys2plusSGN, weights_sys2minusSGN=weights_sys2minusSGN,
         weights_sys3plusBCKG=weights_sys3plusBCKG, weights_sys3minusBCKG=weights_sys3minusBCKG, weights_sys3plusSGN=weights_sys3plusSGN, weights_sys3minusSGN=weights_sys3minusSGN,
         weights_sys4plusBCKG=weights_sys4plusBCKG, weights_sys4minusBCKG=weights_sys4minusBCKG, weights_sys4plusSGN=weights_sys4plusSGN, weights_sys4minusSGN=weights_sys4minusSGN)