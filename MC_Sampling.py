import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats

# Defining parameters for plotting:

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "monospace",  # Use monospace font
    "text.latex.preamble": r"\usepackage{courier}"  # Use Courier font
})
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({
    'font.family': 'monospace',  # monospace font
    "text.latex.preamble": r"\usepackage{courier}",
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 22,
})

def SamplingTruncatedGaussian(mu, sigma, a, b, size):
    """
    Random sample of a truncated normal distribution.
    """
    a, b = (a - mu) / sigma, (b - mu) / sigma
    return stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)

def SampleSignalAndBackground(samplesize, muBCKG, sigmaBCKG, muSGN, sigmaSGN, xmin, xmax):
    """
    Generate a sample for a signal and a background distributions, both following a truncated Gaussian distribution.
    """
    sample1 = SamplingTruncatedGaussian(muBCKG, sigmaBCKG, xmin, xmax, size=samplesize)
    sample2 = SamplingTruncatedGaussian(muSGN, sigmaSGN, xmin, xmax, size=samplesize)
    return sample1, sample2

def Histogram_Nominal(samplesize, sampleBCKG, sampleSGN, bins, yields_BCKG, yields_SGN):
    """
    Generate weights for the nominal histogram of signal and background, given the samples, yields and bins.
    """
    countsBCKG = np.histogram(sampleBCKG, bins=bins)[0]
    countsSGN = np.histogram(sampleSGN, bins=bins)[0]

    weights_sumBCKG = yields_BCKG/samplesize*countsBCKG
    weights_sumSGN = yields_SGN/samplesize*countsSGN

    return weights_sumBCKG, weights_sumSGN, weights_sumBCKG + weights_sumSGN

def Plot_Hist_Nominal(bins, weights_sumBCKG, weights_sumSGN):
    """
    Plot the nominal histogram of signal and background, given the weights and bins.
    """
    plt.figure(figsize=(12, 10))
    plt.hist(
        [bins[:-1], bins[:-1]],
        bins=bins,
        weights=[weights_sumBCKG, weights_sumSGN],
        histtype='barstacked',
        label=['Background', 'Signal'],
        color=['xkcd:sky blue', 'orange'])
    
    plt.xlim(60, 160)
    plt.xlabel(r"$m$ (arbitrary units)")
    plt.ylabel(r"Counts $N$")
    plt.legend()
    plt.savefig("plots/BCKG_plus_SGN.pdf", bbox_inches='tight')
    plt.close()

def SystematicVariation1(persigmavar, samplesize, muBCKG, sigmaBCKG, muSGN, sigmaSGN, xmin, xmax, yields_BCKG, yields_SGN, bins):
    """
    Generate samples for the systematic variation of the standard deviation of the distributions, given their parameters and bins, and return the corresponding weights.
    The variation is calculated as a percent of the corresponding standard deviation.
    """
    sampleSys111 = SamplingTruncatedGaussian(muBCKG, (1+persigmavar)*sigmaBCKG , xmin, xmax, size=samplesize)
    sampleSys121 = SamplingTruncatedGaussian(muSGN, (1+persigmavar)*sigmaSGN, xmin, xmax, size=samplesize)

    sampleSys112 = SamplingTruncatedGaussian(muBCKG, (1-persigmavar)*sigmaBCKG , xmin, xmax, size=samplesize)   
    sampleSys122 = SamplingTruncatedGaussian(muSGN, (1-persigmavar)*sigmaSGN, xmin, xmax, size=samplesize)

    countsSys111 = np.histogram(sampleSys111, bins=bins)[0]
    countsSys121 = np.histogram(sampleSys121, bins=bins)[0]
    countsSys112 = np.histogram(sampleSys112, bins=bins)[0]
    countsSys122 = np.histogram(sampleSys122, bins=bins)[0]

    weights_sys1plusBCKG = yields_BCKG/samplesize*countsSys111
    weights_sys1plusSGN = yields_SGN/samplesize*countsSys121
    weights_sys1minusBCKG = yields_BCKG/samplesize*countsSys112
    weights_sys1minusSGN = yields_SGN/samplesize*countsSys122

    return weights_sys1plusBCKG, weights_sys1plusSGN, weights_sys1minusBCKG, weights_sys1minusSGN

def SystematicVariation2(permuvar, samplesize, muBCKG, sigmaBCKG, muSGN, sigmaSGN, xmin, xmax, yields_BCKG, yields_SGN, bins):
    """
    Generate samples for the systematic variation of the mean of the distributions, given their parameters and bins, and return the corresponding weights.
    The variation is calculated as a percent of the corresponding mean.
    """
    sampleSys211 = SamplingTruncatedGaussian((1+permuvar)*muBCKG, sigmaBCKG, xmin, xmax, size=samplesize)
    sampleSys221 = SamplingTruncatedGaussian((1+permuvar)*muSGN, sigmaSGN, xmin, xmax, size=samplesize)

    sampleSys212 = SamplingTruncatedGaussian((1-permuvar)*muBCKG, sigmaBCKG, xmin, xmax, size=samplesize)
    sampleSys222 = SamplingTruncatedGaussian((1-permuvar)*muSGN, sigmaSGN, xmin, xmax, size=samplesize)

    countsSys211 = np.histogram(sampleSys211, bins=bins)[0]
    countsSys221 = np.histogram(sampleSys221, bins=bins)[0]
    countsSys212 = np.histogram(sampleSys212, bins=bins)[0]
    countsSys222 = np.histogram(sampleSys222, bins=bins)[0]

    weights_sys2plusBCKG = yields_BCKG/samplesize*countsSys211
    weights_sys2plusSGN = yields_SGN/samplesize*countsSys221
    weights_sys2minusBCKG = yields_BCKG/samplesize*countsSys212
    weights_sys2minusSGN = yields_SGN/samplesize*countsSys222

    return weights_sys2plusBCKG, weights_sys2plusSGN, weights_sys2minusBCKG, weights_sys2minusSGN


def SystematicVariation3(count_scale, weights_sumBCKG, weights_sumSGN):
    """
    Generate samples for the systematic variation of the counts of the distributions by a total scale factor, and return the corresponding weights.
    For the positiva variation, the scaling factor is greater than the unit, while for the negative variation, the scaling factor is smaller than the unit.
    """
    weights_sys3plusBCKG = count_scale * weights_sumBCKG
    weights_sys3plusSGN = count_scale * weights_sumSGN
    weights_sys3minusBCKG = weights_sumBCKG / count_scale
    weights_sys3minusSGN = weights_sumSGN / count_scale

    return weights_sys3plusBCKG, weights_sys3plusSGN, weights_sys3minusBCKG, weights_sys3minusSGN

def SystematicVariation4(weights_sumBCKG, weights_sumSGN, bins):
    """
    Generate the variation for a linear systematic effect on the random variable m, and return the corresponding weights.
    """
    count_scale = 0.9*(1/(bins[-2] - bins[0])*bins[:-1])
    invcount_scale = 0.9*(-1/(bins[-2] - bins[0])*bins[:-1]+2)

    weights_sys4plusBCKG = count_scale*weights_sumBCKG
    weights_sys4minusBCKG = invcount_scale*weights_sumBCKG
    weights_sys4plusSGN = count_scale*weights_sumSGN
    weights_sys4minusSGN = invcount_scale*weights_sumSGN


    return weights_sys4plusBCKG, weights_sys4plusSGN, weights_sys4minusBCKG, weights_sys4minusSGN


def Plot_Syst_1(weights_sys1plusBCKG, weights_sys1plusSGN, weights_sys1minusBCKG, weights_sys1minusSGN, bins, weights_sum):
    """
    Plot the histogram of the systematic variation 1 alongside the ratio for variated and nominal counts, given the weights and bins.
    """

    weights_sys1plus = weights_sys1plusBCKG + weights_sys1plusSGN
    weights_sys1minus = weights_sys1minusBCKG + weights_sys1minusSGN

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 10), sharex=False)
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys1plus, edgecolor='blue', histtype='step', label='Systematic 1 +')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys1minus, edgecolor='red', histtype='step', label='Systematic 1 -')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sum, edgecolor='green', histtype='step', label='Nominal')
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
    plt.close()

def Plot_Syst_2(weights_sys2plusBCKG, weights_sys2plusSGN, weights_sys2minusBCKG, weights_sys2minusSGN, bins, weights_sum):
    """
    Plot the histogram of the systematic variation 2 alongside the ratio for variated and nominal counts, given the weights and bins.
    """

    weights_sys2plus = weights_sys2plusBCKG + weights_sys2plusSGN
    weights_sys2minus = weights_sys2minusBCKG + weights_sys2minusSGN

    weights_sys2plus = weights_sys2plusBCKG + weights_sys2plusSGN
    weights_sys2minus = weights_sys2minusBCKG + weights_sys2minusSGN

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 10), sharex=False)
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys2plus, edgecolor='blue', histtype='step', label='Systematic 2 +')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys2minus, edgecolor='red', histtype='step', label='Systematic 2 -')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sum, edgecolor='green', histtype='step', label='Nominal')
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
    plt.close()


def Plot_Syst_3(weights_sys3plusBCKG, weights_sys3plusSGN, weights_sys3minusBCKG, weights_sys3minusSGN, bins, weights_sum):
    """
    Plot the histogram of the systematic variation 3 alongside the ratio for variated and nominal counts, given the weights and bins.
    """
    
    weights_sys3plus = weights_sys3plusBCKG + weights_sys3plusSGN
    weights_sys3minus = weights_sys3minusBCKG + weights_sys3minusSGN

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 10), sharex=False)
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys3plus, edgecolor='blue', histtype='step', label='Systematic 3 +')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys3minus, edgecolor='red', histtype='step', label='Systematic 3 -')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sum, edgecolor='green', histtype='step', label='Nominal')
    ax[0].set_xlim(60, 160)
    ax[0].legend()
    ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys3plus / weights_sum, color='blue')
    ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, weights_sys3minus / weights_sum, color='red')
    ax[1].set_xlim(60, 160)
    ax[1].set_xlabel(r"$m$ (arbitrary units)")
    ax[0].set_ylabel(r"Counts $N$")
    ax[1].set_ylabel(r"$N^{\pm}_i / N_i$")
    ax[1].grid()
    plt.savefig("plots/Systematic_3.pdf", bbox_inches='tight')
    plt.close()

def Plot_Syst_4(weights_sys4plusBCKG, weights_sys4plusSGN, weights_sys4minusBCKG, weights_sys4minusSGN, bins, weights_sum):
    """
    Plot the histogram of the systematic variation 4 alongside the ratio for variated and nominal counts, given the weights and bins.
    """

    weights_sys4plus = weights_sys4plusBCKG + weights_sys4plusSGN
    weights_sys4minus = weights_sys4minusBCKG + weights_sys4minusSGN

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 10), sharex=False)
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys4plus, edgecolor='blue', histtype='step', label='Systematic 4 +')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sys4minus, edgecolor='red', histtype='step', label='Systematic 4 -')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=weights_sum, edgecolor='green', histtype='step', label='Nominal')
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
    plt.close()

def main():
    """
    Produces a Montecarlo sampling for the signal and background distributions of interest, alongside 4 systematic variations, and plots the corresponding histograms and ratios for variated and nominal counts.
    """

    ## Relevant Parameters

    # For the distributions and sampling:

    xmin = 60
    xmax = 160

    samplesize = 50000

    muBCKG = 91.2
    muSGN = 125

    sigmaBCKG = 15
    sigmaSGN = 17

    yields_BCKG = 500
    yields_SGN = 30

    # For the systematic variations:

    persigmavar = 0.05
    permuvar = 0.03
    count_scale = 1.2

    ## Taking the bins for the histrograms as 20 equally spaced bins between 60 and 160:

    bins = np.linspace(60, 160, 21)

    # Calculated the nominal samples and weights:

    sample1, sample2 = SampleSignalAndBackground(samplesize = 50000, muBCKG = 91.2, sigmaBCKG= 15, muSGN=125, sigmaSGN=17, xmin=60, xmax=160)
    weights_sumBCKG, weights_sumSGN, weights_sum = Histogram_Nominal(samplesize, sample1, sample2, bins, yields_BCKG, yields_SGN)

    # Plotting the histogram for the nominal case:

    Plot_Hist_Nominal(bins, weights_sumBCKG, weights_sumSGN)

    ## Calculating the systematic variations and the corresponding weights:

    # Systematic 1: Variation of the standard deviations of the distributions.

    weights_sys1plusBCKG, weights_sys1plusSGN, weights_sys1minusBCKG, weights_sys1minusSGN = SystematicVariation1(persigmavar, samplesize, muBCKG, sigmaBCKG, muSGN, sigmaSGN, xmin, xmax, yields_BCKG, yields_SGN, bins)

    # Systematic 2: Variation of the means of the distributions.

    weights_sys2plusBCKG, weights_sys2plusSGN, weights_sys2minusBCKG, weights_sys2minusSGN = SystematicVariation2(permuvar, samplesize, muBCKG, sigmaBCKG, muSGN, sigmaSGN, xmin, xmax, yields_BCKG, yields_SGN, bins)

    # Systematic 3: Adding a scaling factor to the counts.

    weights_sys3plusBCKG, weights_sys3plusSGN, weights_sys3minusBCKG, weights_sys3minusSGN = SystematicVariation3(count_scale, weights_sumBCKG, weights_sumSGN)

    # Systematic 4: Adding a linear factor to the counts.

    weights_sys4plusBCKG, weights_sys4plusSGN, weights_sys4minusBCKG, weights_sys4minusSGN = SystematicVariation4(weights_sumBCKG, weights_sumSGN, bins)

    # Generating the plots for the systematic variations:

    Plot_Syst_1(weights_sys1plusBCKG, weights_sys1plusSGN, weights_sys1minusBCKG, weights_sys1minusSGN, bins, weights_sum)

    Plot_Syst_2(weights_sys2plusBCKG, weights_sys2plusSGN, weights_sys2minusBCKG, weights_sys2minusSGN, bins, weights_sum)

    Plot_Syst_3(weights_sys3plusBCKG, weights_sys3plusSGN, weights_sys3minusBCKG, weights_sys3minusSGN, bins, weights_sum)

    Plot_Syst_4(weights_sys4plusBCKG, weights_sys4plusSGN, weights_sys4minusBCKG, weights_sys4minusSGN, bins, weights_sum)

    # Saving in a new file for future use:

    np.savez("weights.npz", bins = bins, weights_sumBCKG=weights_sumBCKG, weights_sumSGN=weights_sumSGN,
         weights_sys1plusBCKG=weights_sys1plusBCKG, weights_sys1plusSGN=weights_sys1plusSGN, weights_sys1minusBCKG=weights_sys1minusBCKG, weights_sys1minusSGN=weights_sys1minusSGN,
         weights_sys2plusBCKG=weights_sys2plusBCKG, weights_sys2minusBCKG=weights_sys2minusBCKG, weights_sys2plusSGN=weights_sys2plusSGN, weights_sys2minusSGN=weights_sys2minusSGN,
         weights_sys3plusBCKG=weights_sys3plusBCKG, weights_sys3minusBCKG=weights_sys3minusBCKG, weights_sys3plusSGN=weights_sys3plusSGN, weights_sys3minusSGN=weights_sys3minusSGN,
         weights_sys4plusBCKG=weights_sys4plusBCKG, weights_sys4minusBCKG=weights_sys4minusBCKG, weights_sys4plusSGN=weights_sys4plusSGN, weights_sys4minusSGN=weights_sys4minusSGN)

if __name__ == "__main__":
    main()