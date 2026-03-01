import numpy as np 
import matplotlib.pyplot as plt
import uproot 
from iminuit import Minuit 
from nll import likelihood_building

# loading monte carlo 
MC = np.load('/home/isadora-galvao/Documents/Uni/1_Semester/SMDA/Final_Project/weigths.npz')

# nominal histograms, no systematics
h_bg_nom = MC["weights_sumBCKG"]
h_sig_nom = MC["weights_sumSGN"]

# systematics up background
h_bg_up = np.stack([
    MC["weights_sys1plusBCKG"],
    MC["weights_sys2plusBCKG"],
    MC["weights_sys3plusBCKG"],
    MC["weights_sys4plusBCKG"],
    ])
# systematics down background
h_bg_down = np.stack([
    MC["weights_sys1minusBCKG"],
    MC["weights_sys2minusBCKG"],
    MC["weights_sys3minusBCKG"],
    MC["weights_sys4minusBCKG"]
    ])
# systematics up signal
h_sig_up = np.stack([
    MC["weights_sys1plusSGN"],
    MC["weights_sys2plusSGN"],
    MC["weights_sys3plusSGN"],
    MC["weights_sys4plusSGN"]
    ])
# systematics down signal
h_sig_down = np.stack([
    MC["weights_sys1minusSGN"],
    MC["weights_sys2minusSGN"],
    MC["weights_sys3minusSGN"],
    MC["weights_sys4minusSGN"]
    ])

bins = MC["bins"]

# loading the data 
with uproot.open('~/Documents/Uni/1_Semester/SMDA/Final_Project/codes/data.root') as f: 
    data02 = f['events;2'].arrays(['m'], library = "np")['m']

data_binned = np.histogram(data02, bins = bins)[0]

likelihood = likelihood_building(data_binned, h_sig_nom, h_bg_nom, h_sig_up, h_sig_down, h_bg_up, h_bg_down)

m = Minuit(likelihood, theta_1 = 1.0, theta_2 = 1.0, theta_3 = 0.0, theta_4 = 0.0, theta_5 = 0.0, theta_6 = 0.0)

m.migrad()
m.hesse() 

theta_values = np.array(m.values)

print(theta_values)

h_bg_posfit= theta_values[0] * h_bg_nom
h_sig_posfit= theta_values[1] * h_sig_nom 

plt.figure(figsize=(8,6))
plt.hist(
    [bins[:-1], bins[:-1]],
    bins=bins,
    weights=[h_bg_posfit, h_sig_posfit],
    histtype='barstacked',
    label=['Background', 'Signal']
)
plt.xlabel('m')
plt.ylabel('Events')
plt.legend()
plt.savefig('histo1.png')



