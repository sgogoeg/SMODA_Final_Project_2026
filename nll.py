import numpy as np

def likelihood_building(data, h_sig_nom, h_bg_nom, h_sig_up, h_sig_down, h_bg_up, h_bg_down): 
    """ 
    Building the negative log likelihood
    """

    def V_total(theta): 
        V_part01 = theta[0]*h_bg_nom + theta[1]*h_sig_nom

        theta_3to6 = theta[2:6]

        coeff_minus = np.maximum(0, -theta_3to6)
        coeff_plus = np.maximum(0, theta_3to6)

        delta_bg_up = h_bg_up - h_bg_nom
        delta_bg_down = h_bg_down - h_bg_nom
    
        delta_sig_up = h_sig_up - h_sig_nom
        delta_sig_down = h_sig_down - h_sig_nom

        V_part02 = np.sum(coeff_minus.reshape(-1,1) * (delta_bg_down + delta_sig_down) 
                    + coeff_plus.reshape(-1,1) * (delta_bg_up + delta_sig_up), axis = 0)
        return V_part01 + V_part02

    def nll_poisson (theta): 
        mu = V_total(theta)
        return 2 * np.sum(mu - data * np.log(mu))

    def nll_gaussian(theta):
        theta_nuis = theta[2:]
        return np.sum(theta_nuis**2)

    def negative_log_likelihood(theta_1,theta_2, theta_3, theta_4, theta_5, theta_6): 
        theta = np.array([theta_1, theta_2, theta_3,theta_4,theta_5,theta_6])
        nll = nll_poisson(theta) + nll_gaussian(theta)
        return nll

    return negative_log_likelihood
