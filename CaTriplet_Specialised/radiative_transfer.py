import random
import numpy as np
import os
from numpy.random import choice,shuffle

# Constants
m_p = 1.67*10**-24 # Proton mass [g]
k_b = 1.38*10**-16 # Boltzmann constant [J/k]
c = 3*10**5        # Speed of light [km/s]

# Below some requirements for sampling a Maxwellian velocity distribution (to create a Gaussian line profile)
r_step = 10**-2
r_range = np.arange(10**-3, 1, r_step)

mu, sigma = 0, 1/3.5 # 1/3.5 assures that the chance of r > 1 is << 0.001
part1 = 1 * (r_range**2)/sigma**3
part2 = np.exp( -r_range**2 / (2*sigma**2) )

P = part1*part2
CDF = np.cumsum(P) # * r_step to get a normalised CDF
CDF_maxwellian_normalised = CDF/CDF[-1]


def do_escape(distouter, wl, r, mu, sin_theta, v_out, scattered, hist, histscatt, binarr):

    dr = distouter
    
    r_nextpos = 1. # Again r is in normalised coordinates, so that 1 is the photosphere
    
    sin_theta = r/r_nextpos*sin_theta

    mu = np.sqrt(1.-sin_theta**2) # mu positive reaching surface..

    gamma = 1./np.sqrt(1.-(dr*v_out/c)**2)

    wl =  wl*(1 + v_out/c*dr)*gamma # Transformation to new CMF..

    # In observer frame:

    gamma = 1./np.sqrt(1.-(v_out/c)**2)  

    wl_escaped = wl*(1. - v_out/c*mu)*gamma  # Rybicki Lightman eq 4.11
    
    wl_index = np.argmin(abs(wl_escaped - binarr))
    
    hist[wl_index] += 1 # Add the photon packet
    
    if scattered == True:
        histscatt[wl_index] += 1
    
    escaped = True
    
    return escaped, hist, histscatt


def check_interaction(distscatt, wl, mu, r, sin_theta, v_out, T, tau0vec, destrprob, index_of_this_line, destroyed, scattered):
    
    dr = distscatt
            
    r_nextpos = np.sqrt(r**2 + dr**2 - 2*r*dr*(-mu)) # The next radial position

    gamma = 1./np.sqrt(1.-(dr*v_out/c)**2) # Gamma factor..

    wl = wl*(1 + v_out/c*dr)*gamma    # New CMF wavelength (Rybicki Lightman eq 4.11 with cos_theta = -1 (homologous flow)).

    # Thermal reshuffling
    xi = random.random()
    vthermal = np.sqrt(k_b*T/m_p)/1e5 # km/s
    dwl_thermal = wl*vthermal/c*(1.-2*xi) # TEMP
    wl = wl + dwl_thermal
    
    # TEMP rewrite this so that tau_accum >= tau_this has been checked already..
    xi = random.random()

    if (xi < 1. - np.exp(-tau0vec[index_of_this_line])): # Interaction happens..

        mu = 1. - 2*random.random()  # Isotropic scattering --> random new mu

        sin_theta = np.sqrt(1.-mu**2) # Update sin(theta).. 

        xi = random.random()

        if (xi < destrprob[index_of_this_line]): # Random draw if photon is thermalized..

            destroyed = True # TEMP                    

        else:

            scattered = True

    else: # Interaction does not happen.. , just change angles

        sin_theta = r/r_nextpos*sin_theta

        if (mu < 0 and dr < abs(r*mu)):

            mu = -np.sqrt(1.-sin_theta**2)

        else:

            mu = np.sqrt(1.-sin_theta**2)
            
    r = r_nextpos # Update the position
    
    return wl, r, mu, sin_theta, destroyed, scattered
    
    
    
def draw_random_photon(wllines, line_shape, el_probabilities, wlmin, wlmax, rin):
    
    #======================================================================================================================
    # Pick the initial position
    if line_shape == 'parabola':
        r = random.random() ** (1/3)
        
    elif line_shape == 'gaussian':
        
        x = random.random()
        
        r = r_range[np.where(x <= CDF_maxwellian_normalised)[0][0]]
        
        
    elif line_shape == 'lorentzian':
        pass
    #======================================================================================================================

    #======================================================================================================================
    # Pick the initial wavelength
    
    # The total probability, assuring P_tot = 1
    P_CaII, P_CI, P_OI, P_MgI = el_probabilities
    P_norm = (P_CaII + P_CI + P_OI + P_MgI)
    
    # Get the individual line probabilities
    
    # For CaII, we have a separate formula
    E2, E3, E4, E5 = 1.692408, 1.699932, 3.123349, 3.150984
    k_b_eV, T_typical = 8.6173 * 10**-5, 5000 # [eV/K, K]
    
    n5_over_n4 = 2 * np.exp( - (E5-E4) / (k_b_eV * T_typical) ) # Take statistical weights corrected for Boltzmann factor
    n3_over_n2 = 1.5 * np.exp( - (E3-E2) / (k_b_eV * T_typical) )
    
    partial_8498, partial_8542, partial_8662 = get_CaII_P(n3_over_n2, n5_over_n4)
    
    
    P_8498 = (P_CaII/P_norm) * partial_8498
    P_8542 = (P_CaII/P_norm) * partial_8542
    P_8662 = (P_CaII/P_norm) * partial_8662
    P_8727 = (P_CI/P_norm)
    P_8446 = (P_OI/P_norm)
    P_8806 = (P_MgI/P_norm)
    
    probability_list = [0,
                        P_8446,
                        P_8446 + P_8498,
                        P_8446 + P_8498 + P_8542,
                        P_8446 + P_8498 + P_8542 + P_8662,
                        P_8446 + P_8498 + P_8542 + P_8662 + P_8727,
                        P_8446 + P_8498 + P_8542 + P_8662 + P_8727 + P_8806]
    emitting_line_wl = [8446,
                        8498,
                        8542,
                        8662,
                        8727,
                        8806]
    
    wl = 0
    
    xi = np.random.random()
    for i in range(len(probability_list)-1): # We now pick the line wavelength                  
        if xi >= probability_list[i] and xi <= probability_list[i+1]:
            wl = emitting_line_wl[i]
    
    mu = 1 - 2 * random.random() # Isotropic
    
    return wl, mu, r
    

    
    
    
    
    
#=============================================================================================================================
#Below is a function that gives the ratio between the three calcium triplet lines when it comes to their own emission.

def get_CaII_P(n3_over_n2, n5_over_n4):
    
    A8662 = 1.06 * 10**7
    A8542 = 9.9 * 10**6
    A8498 = 1.11 * 10**6

    g2, g3 = 4, 6
    g4, g5 = 2, 4
    
    n2, n3 = 1, n3_over_n2
    n4, n5 = 1, n5_over_n4
    
    beta8662 = g2/g4 * (1/A8662) * (1/8662**3) * (1/n2)
    beta8542 = g3/g5 * (1/A8542) * (1/8542**3) * (1/n3)
    beta8498 = g2/g5 * (1/A8498) * (1/8498**3) * (1/n2)
    
    #These are unscaled luminosities
    P_8662 = n4 * A8662 * beta8662 * 8662 
    P_8542 = n5 * A8542 * beta8542 * 8542
    P_8498 = n5 * A8498 * beta8498 * 8498
    
    P_tot = (P_8662 + P_8542 + P_8498)
    
    return P_8498/P_tot, P_8542/P_tot, P_8662/P_tot