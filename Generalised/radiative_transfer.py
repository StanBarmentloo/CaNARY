import random
import numpy as np
import os
from numpy.random import choice,shuffle

# Constants
m_p = 1.67262*10**-24 # Proton mass [g]
k_b = 1.38065*10**-16 # Boltzmann constant [J/k]
c = 2.99792*10**5     # Speed of light [km/s]

# Below some requirements for sampling a Maxwellian velocity distribution (to create a Gaussian line profile)
#=============================================================================================================
r_step = 10**-2
r_range = np.arange(10**-3, 1, r_step)

mu, sigma = 0, 1/3.5 # 1/3.5 assures that the chance of r > 1 is << 0.001
part1 = 1 * (r_range**2)/sigma**3
part2 = np.exp( -r_range**2 / (2*sigma**2) )

P = part1*part2
CDF = np.cumsum(P) # * r_step to get a normalised CDF
CDF_maxwellian_normalised = CDF/CDF[-1]
#=============================================================================================================

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


def check_interaction(distscatt, wl, mu, r, sin_theta, v_out, T, tau_vec, destrprob, index_of_this_line, destroyed, scattered):
    
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

    if (xi < 1. - np.exp(-tau_vec[index_of_this_line])): # Interaction happens..

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
    
    
    
def draw_random_photon(wllines, line_shape, lum_vec, wlmin, wlmax):
    
    #======================================================================================================================
    # Pick the initial position
    if line_shape == 'parabola':
        r = random.random() ** (1/3)
        
    elif line_shape == 'gaussian':
        
        x = random.random()
        
        r = r_range[np.where(x <= CDF_maxwellian_normalised)[0][0]]
    #======================================================================================================================

    #======================================================================================================================
    # Pick the initial wavelength
    
    P_norm = np.sum(lum_vec) # The total probability, assuring P_tot = 1
    probability_bounds = np.zeros(len(lum_vec)+1) # a bounds array to find which photon wl to pick
    
    for i in range(len(lum_vec)):
        probability_bounds[i+1] = np.sum(lum_vec[:i+1])/P_norm # fill i+1, as the zeroth element of probability bounds should be 0 
    
    xi = np.random.random() # Now select the wl by drawing a random number between 0 and 1
    wl = wllines[np.where((xi > probability_bounds))[0][-1]] # The selected wl is the last element where xi > bound.
    
    # Pick the initial travel direction
    mu = 1 - 2 * random.random() # Isotropic
    
    return wl, mu, r