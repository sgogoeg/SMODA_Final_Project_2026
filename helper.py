import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from iminuit import Minuit
import pandas as pd
from pandas import plotting as pdplt


def sample_truncated_gaussian(mu, sigma, a, b, size=1):
    """
    Random sample of a truncated normal distribution.
    """
    a, b = (a - mu) / sigma, (b - mu) / sigma
    return stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)


def generate_hists_nominal(n_mc, mu_sig, sigma_sig, mu_bkg, sigma_bkg, yield_sig, yield_bkg, bins=np.linspace(60, 160, 21)):
    """
    Generate nominal histograms for signal and background.
    """
    a, b = bins[0], bins[-1]

    # sample signal and background
    sig_sample = sample_truncated_gaussian(mu_sig, sigma_sig, a, b, size=n_mc)
    bkg_sample = sample_truncated_gaussian(mu_bkg, sigma_bkg, a, b, size=n_mc)

    # create weights to match yields
    sig_weights = np.ones_like(sig_sample) * (yield_sig / n_mc)
    bkg_weights = np.ones_like(bkg_sample) * (yield_bkg / n_mc)

    # calculate histograms
    sig_hist = np.histogram(sig_sample, bins=bins, weights=sig_weights)[0]
    bkg_hist = np.histogram(bkg_sample, bins=bins, weights=bkg_weights)[0]

    return sig_hist, bkg_hist


def generate_hists_variation_1(rel_sigma_var, n_mc, mu_sig, sigma_sig, mu_bkg, sigma_bkg, yield_sig, yield_bkg, bins=np.linspace(60, 160, 21)):
    """
    Generate samples for the systematic variation of the standard deviation of the distributions, 
    given their parameters and bins, and return the corresponding histograms.
    The variation is calculated as a percent of the corresponding standard deviation.
    """
    h_sig_up_1, h_bkg_up_1 = generate_hists_nominal(n_mc, mu_sig, sigma_sig * (1 + rel_sigma_var), mu_bkg, sigma_bkg * (1 + rel_sigma_var), yield_sig, yield_bkg, bins)
    h_sig_down_1, h_bkg_down_1 = generate_hists_nominal(n_mc, mu_sig, sigma_sig * (1 - rel_sigma_var), mu_bkg, sigma_bkg * (1 - rel_sigma_var), yield_sig, yield_bkg, bins)

    return h_sig_up_1, h_bkg_up_1, h_sig_down_1, h_bkg_down_1


def generate_hists_variation_2(rel_mu_var, n_mc, mu_sig, sigma_sig, mu_bkg, sigma_bkg, yield_sig, yield_bkg, bins=np.linspace(60, 160, 21)):
    """
    Generate samples for the systematic variation of the mean of the distributions, 
    given their parameters and bins, and return the corresponding weights.
    The variation is calculated as a percent of the corresponding mean.
    """
    h_sig_up_2, h_bkg_up_2 = generate_hists_nominal(n_mc, mu_sig * (1 + rel_mu_var), sigma_sig, mu_bkg * (1 + rel_mu_var), sigma_bkg, yield_sig, yield_bkg, bins)
    h_sig_down_2, h_bkg_down_2 = generate_hists_nominal(n_mc, mu_sig * (1 - rel_mu_var), sigma_sig, mu_bkg * (1 - rel_mu_var), sigma_bkg, yield_sig, yield_bkg, bins)

    return h_sig_up_2, h_bkg_up_2, h_sig_down_2, h_bkg_down_2


def generate_hists_variation_3(yield_scale, h_sig_nominal, h_bkg_nominal):
    """
    Generate samples for the systematic variation of the counts of the distributions by a total scale factor, 
    and return the corresponding weights.
    For the positiva variation, the scaling factor is greater than the unit, 
    while for the negative variation, the scaling factor is smaller than the unit.

    To save time, the histograms are not re-calculated, but the nominal histograms are scaled by the given factor.
    """
    h_sig_up_3 = h_sig_nominal * yield_scale
    h_bkg_up_3 = h_bkg_nominal * yield_scale
    h_sig_down_3 = h_sig_nominal / yield_scale
    h_bkg_down_3 = h_bkg_nominal / yield_scale

    return h_sig_up_3, h_bkg_up_3, h_sig_down_3, h_bkg_down_3


def generate_hists_variation_4(h_sig_nominal, h_bkg_nominal, bins):
    """
    Generate the variation for a linear systematic effect on the random variable m, and return the corresponding weights.

    To save time, the histograms are not re-calculated, but the nominal histograms are scaled by the given factor.
    """
    count_scale = 0.9*(1/(bins[-2] - bins[0])*bins[:-1])
    invcount_scale = 0.9*(-1/(bins[-2] - bins[0])*bins[:-1]+2)

    h_sig_up_4 = h_sig_nominal * count_scale
    h_bkg_up_4 = h_bkg_nominal * count_scale
    h_sig_down_4 = h_sig_nominal * invcount_scale
    h_bkg_down_4 = h_bkg_nominal * invcount_scale

    return h_sig_up_4, h_bkg_up_4, h_sig_down_4, h_bkg_down_4


def generate_all_mc_variations(
    n_mc, mu_sig, sigma_sig, mu_bkg, sigma_bkg, yield_sig, yield_bkg,
    rel_sigma_var, rel_mu_var, yield_scale, bins=np.linspace(60, 160, 21),
    save_path=None
):
    h_sig_nom, h_bkg_nom = generate_hists_nominal(n_mc, mu_sig, sigma_sig, mu_bkg, sigma_bkg, yield_sig, yield_bkg, bins)
    h_sig_up_1, h_bkg_up_1, h_sig_down_1, h_bkg_down_1 = generate_hists_variation_1(rel_sigma_var, n_mc, mu_sig, sigma_sig, mu_bkg, sigma_bkg, yield_sig, yield_bkg, bins)
    h_sig_up_2, h_bkg_up_2, h_sig_down_2, h_bkg_down_2 = generate_hists_variation_2(rel_mu_var, n_mc, mu_sig, sigma_sig, mu_bkg, sigma_bkg, yield_sig, yield_bkg, bins)
    h_sig_up_3, h_bkg_up_3, h_sig_down_3, h_bkg_down_3 = generate_hists_variation_3(yield_scale, h_sig_nom, h_bkg_nom)
    h_sig_up_4, h_bkg_up_4, h_sig_down_4, h_bkg_down_4 = generate_hists_variation_4(h_sig_nom, h_bkg_nom, bins)
    hists = {
        'h_sig_nom': h_sig_nom, 'h_bkg_nom': h_bkg_nom,
        'h_sig_up_1': h_sig_up_1, 'h_bkg_up_1': h_bkg_up_1, 'h_sig_down_1': h_sig_down_1, 'h_bkg_down_1': h_bkg_down_1,
        'h_sig_up_2': h_sig_up_2, 'h_bkg_up_2': h_bkg_up_2, 'h_sig_down_2': h_sig_down_2, 'h_bkg_down_2': h_bkg_down_2,
        'h_sig_up_3': h_sig_up_3, 'h_bkg_up_3': h_bkg_up_3, 'h_sig_down_3': h_sig_down_3, 'h_bkg_down_3': h_bkg_down_3,
        'h_sig_up_4': h_sig_up_4, 'h_bkg_up_4': h_bkg_up_4, 'h_sig_down_4': h_sig_down_4, 'h_bkg_down_4': h_bkg_down_4,
    }
    if save_path is not None:
        np.savez(save_path, **hists)
    return tuple(hists.values())


def plot_systematic(h_sig_nom, h_bkg_nom, h_sig_up, h_bkg_up, h_sig_down, h_bkg_down, bins=np.linspace(60, 160, 21), name='Systematic Variation', save_name='Systematic_Variation'):
    """
    Plot the nominal and systematic variation histograms for the sum of sig and bkg.
    """
    h = h_sig_nom + h_bkg_nom
    h_up = h_sig_up + h_bkg_up
    h_down = h_sig_down + h_bkg_down

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05}, figsize=(12, 10), sharex=True)
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=h_up, edgecolor='blue', histtype='step', label='Systematic 1 +')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=h_down, edgecolor='red', histtype='step', label='Systematic 1 -')
    ax[0].hist(bins[:-1], bins=bins, density=False, weights=h, edgecolor='green', histtype='step', label='Nominal')
    ax[0].set_xlim(bins[0], bins[-1])
    ax[0].legend()
    ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, h_up / h, color='blue')
    ax[1].scatter(bins[:-1] + (bins[1]-bins[0])/2, h_down / h, color='red')
    ax[1].set_xlabel(r"$m$ (arbitrary units)")
    ax[0].set_ylabel(r"Counts")
    ax[1].set_ylabel(r"$N^{\pm}_i / N_i$")
    ax[1].set_ylim(0.5, 1.5)
    ax[1].grid()
    plt.savefig(f"plots/{save_name}.pdf", bbox_inches='tight')
    plt.close()


def likelihood_building(data, h_sig_nom, h_bg_nom, h_sig_up, h_sig_down, h_bg_up, h_bg_down): 
    """
    Building the negative log likelihood from the given data and histograms.
    """

    def V_total(theta):
        """
        Calculate expected count in each bin.
        First part is signal and background contribution without systematic uncertainties.
        Second part is shift by systematic uncertainties.
        """
        # signal and background contribution
        V_part01 = theta[0]*h_bg_nom + theta[1]*h_sig_nom

        # select nuisance parameteres
        theta_3to6 = theta[2:6]
        
        # if theta < 0, the down direction of sys contributes
        coeff_minus = np.maximum(0, -theta_3to6)
        # if theta > 0, the up
        coeff_plus = np.maximum(0, theta_3to6)

        # difference for background
        delta_bg_up = h_bg_up - h_bg_nom
        delta_bg_down = h_bg_down - h_bg_nom

        # difference for signal
        delta_sig_up = h_sig_up - h_sig_nom
        delta_sig_down = h_sig_down - h_sig_nom

        # second part with the nuisance parameters
        V_part02 = np.sum(coeff_minus.reshape(-1,1) * (delta_bg_down + delta_sig_down)
                    + coeff_plus.reshape(-1,1) * (delta_bg_up + delta_sig_up), axis=0)
        
        return V_part01 + V_part02

    def nll_poisson(theta):
        """ 
        -2 log likelihood for the binned poisson part
        """
        # the V_total is the expecte value for the poisson
        mu = V_total(theta)
        return 2 * np.sum(mu - data * np.log(mu))

    def nll_gaussian(theta):
        """
        -2 log likelihood for the gaussians
        """
        # only for the nuisance parameters
        theta_nuis = theta[2:]
        return np.sum(theta_nuis**2)

    def negative_log_likelihood(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6):
        """"
        final  likelihood
        """
        theta = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
        
        nll = nll_poisson(theta) + nll_gaussian(theta)
        return nll

    return negative_log_likelihood


def do_fit(data_hist, h_sig_nom, h_bg_nom, h_sig_up, h_sig_down, h_bg_up, h_bg_down):
    """
    Perform the fit by minimizing the negative log likelihood.
    """
    likelihood = likelihood_building(data_hist, h_sig_nom, h_bg_nom, h_sig_up, h_sig_down, h_bg_up, h_bg_down)
    m = Minuit(likelihood, theta_1=1.0, theta_2=1.0, theta_3=0.0, theta_4=0.0, theta_5=0.0, theta_6=0.0)
    m.migrad()
    m.hesse()

    return m


def plot_fit_results(
    data_hist, h_sig_nom, h_bg_nom, h_sig_up, h_sig_down, h_bg_up, h_bg_down,
    minuit_obj, bins=np.linspace(60, 160, 21), save_name='Fit_Results'
):
    # Get fit results
    theta_values = np.array(minuit_obj.values)
    theta_nuis = theta_values[2:]
    cov_matrix = np.array(minuit_obj.covariance)

    # Extract covariance for nuisance parameters
    cov_syst = cov_matrix[2:, 2:]

    #data uncertainty from poisson distribution
    data_err = np.sqrt(data_hist)

    delta_bg_up = h_bg_up - h_bg_nom
    delta_bg_down = h_bg_down - h_bg_nom
    delta_sig_up = h_sig_up - h_sig_nom
    delta_sig_down = h_sig_down - h_sig_nom

    # Derivatives of expected counts with respect to nuisance params
    derivatives = np.where(
        theta_nuis.reshape(-1, 1) < 0,
        delta_bg_up + delta_sig_up,
        -(delta_bg_down + delta_sig_down)
    ).T

    # error propagation
    variance = np.sum(derivatives @ cov_syst * derivatives, axis=1)
    system_err = np.sqrt(variance)

    # expected counts after fit
    h_sig_nom = theta_values[1] * h_sig_nom
    h_bg_nom = theta_values[0] * h_bg_nom
    coeff_minus = np.maximum(0, -theta_nuis)
    coeff_plus = np.maximum(0, theta_nuis)
    h_sig_shift = np.sum(coeff_minus.reshape(-1,1) * delta_sig_down + coeff_plus.reshape(-1,1) * delta_sig_up, axis=0)
    h_bg_shift = np.sum(coeff_minus.reshape(-1,1) * delta_bg_down + coeff_plus.reshape(-1,1) * delta_bg_up, axis=0)
    h_sig_fit = h_sig_nom + h_sig_shift
    h_bg_fit = h_bg_nom + h_bg_shift
    h_total_nom = h_sig_fit + h_bg_fit    

    # Plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = np.diff(bins)
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(12, 8), 
                                        gridspec_kw={'height_ratios': [3, 1],
                                                    'hspace': 0.05})

    ax.bar(bin_centers, h_bg_nom, width=bin_width, 
        label='Background', color='xkcd:sky blue', alpha=0.5, 
        edgecolor='black', linewidth=0.5, align='center')

    ax.bar(bin_centers, h_sig_nom, width=bin_width, bottom=h_bg_nom,
        label='Signal', color='orange', alpha=0.5, 
        edgecolor='black', linewidth=0.5, align='center')

    x_plot = np.repeat(bins, 2)[1:-1]  
    y_upper_plot = np.repeat(h_total_nom + system_err, 2)
    y_lower_plot = np.repeat(h_total_nom - system_err, 2)

    ax.fill_between(x_plot, y_lower_plot, y_upper_plot,
                    facecolor='none',  
                    hatch='///', 
                    edgecolor='gray',  
                    linewidth=0,       
                    alpha=1.0,         
                    label='Syst. uncertainty')

    ax.errorbar(bin_centers, data_hist, yerr=data_err, 
                fmt='o', color='black', markersize=6, 
                capsize=2, capthick=1, linewidth=1.5,
                label='Data')

    ax.set_ylabel('Events / 5 GeV', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(bins[0], bins[-1])
    ax.tick_params(axis='x', which='both', labelbottom=False)



    # Ratio plot (Data/MC)
    ratio = data_hist / h_total_nom
    ratio_err = data_err / h_total_nom

    ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_err, 
                    fmt='o', color='black', markersize=6,
                    capsize=2, capthick=1, linewidth=1.5)

    ratio_syst_high = (h_total_nom + system_err) / h_total_nom
    ratio_syst_low = (h_total_nom - system_err) / h_total_nom
    ax_ratio.fill_between(bin_centers, 
                        ratio_syst_low, ratio_syst_high,
                        step='mid', color='gray', alpha=0.3, hatch='///')

    ax_ratio.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax_ratio.set_xlabel('m (GeV)', fontsize=12)
    ax_ratio.set_ylabel('Data / MC', fontsize=12)
    ax_ratio.set_xlim(bins[0], bins[-1])
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, alpha=0.3, axis='y')

    plt.savefig(f"plots/{save_name}.pdf", bbox_inches='tight')
    plt.clf()


def plot_fit_values(minuit_obj, save_name='Fit_Values'):
    Results = pd.DataFrame({
        'Parameter': minuit_obj.parameters,
        'Value': [round(minuit_obj.values[p], 3) for p in minuit_obj.parameters],
        'Error': [round(minuit_obj.errors[p], 3) for p in minuit_obj.parameters]
    })
    Results['Parameter'] = [rf'$\theta_{i+1}$' for i in range(len(Results))]

    colours = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    fig, ax_plot = plt.subplots(figsize=(16, 8))

    for i in range(len(Results.values)):
        ax_plot.errorbar(
            y=i,
            x=Results['Value'].iloc[i],
            xerr=Results['Error'].iloc[i],
            fmt='o',
            capsize=5,
            color=colours[i],
            label=f'theta_{i+1}'
        )

    plt.xlim(-3.5, 7.5)  
    plt.ylim(-1, len(Results.values) + 1)
    plt.tick_params(axis='both', labelsize=16)  
    y_labels = [fr'$\theta_{i+1}$' for i in range(len(Results.values))]
    ax_plot.set_yticks(np.arange(len(Results.values)))
    ax_plot.set_yticklabels(y_labels)
    ax_plot.set_xlabel('Value')
    ax_plot.grid(True, alpha=0.3)
    ax_plot.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    Results_reversed = Results.iloc[::-1].reset_index(drop=True)
    colours_reversed = colours[::-1]

    #table positioning
    xlim = ax_plot.get_xlim()
    table_x_start = xlim[1] - 3.5

    #table 
    for idx, row in Results_reversed.iterrows():
        y_pos = len(Results.values) - 1 - idx  # Reverse order to match plot
        ax_plot.text(table_x_start, y_pos, row['Parameter'],
                    color=colours_reversed[idx], fontsize=16, fontweight='bold')
        ax_plot.text(table_x_start + 1.2, y_pos, row['Value'],
                    color='black', fontsize=12)
        ax_plot.text(table_x_start + 2.4, y_pos, '±' + str(row['Error']),
                    color='black', fontsize=12)

    ax_plot.text(table_x_start, len(Results.values), 'Parameter', fontweight='bold', fontsize=12)
    ax_plot.text(table_x_start + 1.2, len(Results.values), 'Value', fontweight='bold', fontsize=12)
    ax_plot.text(table_x_start + 2.4, len(Results.values), 'Error', fontweight='bold', fontsize=12)

    plt.savefig(f"plots/{save_name}.pdf", bbox_inches='tight')
    plt.clf()
