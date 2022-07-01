import os
import numpy as np
from scipy.special import erf
from scipy.integrate import simpson

#import matplotlib.pyplot as plt
import time

from QEdark_constants import *
from DM_halo_dist import *


"""
set directories
"""
dataDir = os.getcwd()
FigDir = os.getcwd() + '/figs'

rho_X = 0.4e9 # eV/cm^3
#v0 = 230e5 # cm/s
#vE = 240e5
#vesc = 600e5

"""
rho_X = 0.3e9 # eV/cm^3
v0 = 220e5 # cm/s
vE = 232e5
vesc = 544e5
"""




def vmin(EE,qin,mX):
    q = qin * alpha *me_eV
    return (EE/q+q/(2*mX))*c_light*1e-3 # to convert to km/s


dQ = .02*alpha*me_eV #eV
dE = 0.1 # eV
wk = 2/137

## import QEdark data
nq = 900
nE = 500

fcrys = {'Si': np.transpose(np.resize(np.loadtxt(dataDir+'/Si_f2.txt',skiprows=1),(nE,nq))),
         'Ge': np.transpose(np.resize(np.loadtxt(dataDir+'/Ge_f2.txt',skiprows=1),(nE,nq)))}

"""
    materials = {name: [Mcell #eV, Eprefactor, Egap #eV, epsilon #eV, fcrys]}
    N.B. If you generate your own fcrys from QEdark, please remove the factor of "wk/4" below.
"""
#materials = {'Si': [2*28.0855*amu2kg, 2.0, 1.2, 3.8,wk/4*fcrys['Si']], \
#             'Ge': [2*72.64*amu2kg, 1.8, 0.7, 2.8,wk/4*fcrys['Ge']]}

materials = {'Si': [2*28.0855*amu2kg, 2.0, 1.11, 3.6, wk/4*fcrys['Si']], \
             'Ge': [2*72.64*amu2kg, 1.8, 0.7, 2.8,wk/4*fcrys['Ge']]}


def FDM(q_eV,n):
    """
    DM form factor
    n = 0: FDM=1, heavy mediator
    n = 1: FDM~1/q, electric dipole
    n = 2: FDM~1/q^2, light mediator
    """
    return (alpha*me_eV/q_eV)**n

def mu_Xe(mX):
    """
    DM-electron reduced mass
    """
    return mX*me_eV/(mX+me_eV)


"""
#---------------------------------------------------------
# Velocity integral eta
#def calcEta(vmin, vlag=230.0, sigmav=156.0,vesc=544.0):
def calcEta(vmin, vlag=vE, sigmav=v0/np.sqrt(2),vesc=vesc):
    aplus = np.minimum((vmin+vlag), vmin*0.0 + vesc)/(np.sqrt(2)*sigmav)
    aminus = np.minimum((vmin-vlag), vmin*0.0 + vesc)/(np.sqrt(2)*sigmav)
    aesc = vesc/(np.sqrt(2)*sigmav)
    
    vel_integral = 0
     
    N = 1.0/(erf(aesc) - np.sqrt(2.0/np.pi)*(vesc/sigmav)*np.exp(-0.5*(vesc/sigmav)**2))
     
    vel_integral = (0.5/vlag)*(erf(aplus) - erf(aminus))
    vel_integral -= (1.0/(np.sqrt(np.pi)*vlag))*(aplus - aminus)*np.exp(-0.5*(vesc/sigmav)**2)
     
    vel_integral = np.clip(vel_integral, 0, 1e30)
 
    return N*vel_integral
"""

def eta1(vmin,vE,v0,vesc):
    aesc = vesc/v0
    K = v0**3*np.pi*( np.sqrt(np.pi)*erf(aesc) -2*aesc*np.exp(-aesc**2) )
    A = v0**2*np.pi/(2*vE*K)
    diff = erf((vmin+vE)/v0) - erf((vmin-vE)/v0)  
    return A*(-4*np.exp(-aesc**2)*vE+np.sqrt(np.pi)*v0*diff)

def eta2(vmin,vE,v0,vesc):
    aesc = vesc/v0
    K = v0**3*np.pi*( np.sqrt(np.pi)*erf(aesc) -2*aesc*np.exp(-(aesc)**2) )
    A = v0**2*np.pi/(2*vE*K)
    diff = erf((vmin+vE)/v0) - erf((vmin-vE)/v0)  
    return A*(-2*np.exp(-aesc**2)*(vesc-vmin+vE)+np.sqrt(np.pi)*v0*diff)


def calcEta(vmin, vE, v0, vesc):
    if hasattr(vmin, "__len__"):
        eta = np.zeros_like(vmin)
        eta[vmin <= vesc-vE] = eta1(vmin[vmin < vesc-vE], vE,v0,vesc)
        eta[(vesc-vE < vmin) & (vmin < vesc+vE)] = eta2(vmin[(vesc-vE < vmin) & (vmin < vesc+vE)], vE,v0,vesc)
        eta[vmin >= vesc+vE] = 0
        return eta
    else:
        if vmin <= vesc-vE:
            return eta1(vmin, vE,v0,vesc)
        elif vesc-vE < vmin and vmin < vesc+vE:
            return eta2(vmin, vE,v0,vesc)
        elif vmin >= vesc+vE:
            return 0

"""
def speed_dist(vmin,vE,v0,vesc):
    aesc = vesc/v0
    K = erf(aesc) - (2/np.sqrt(np.pi)*aesc*np.exp(-aesc**2))
    cosmax = (vesc**2-vmin**2-vE**2)/(2*vmin*vE)
    cmax = np.minimum(1, cosmax)

    f = vmin/(np.sqrt(np.pi)*v0*vE*K)*( np.exp(-(vmin-vE)**2/v0**2) - np.exp(-(vmin**2+vE+2*vmin*vE*cmax)/v0**2) )

    if hasattr(vmin, "__len__"):
        f[vmin>vE+vesc] = 0
    else:
        if u>vE+vEsc: f=0

    return f/vmin

def calcEta(vmin, vE, v0, vesc):
    vmax = vE+vesc
    vmin_matrix = []
    for vm in vmin:
        vmin_matrix.append(np.geomspace(vm,vmax,100).tolist())

    vmin_matrix = np.array(vmin_matrix)

    if hasattr(vmin, "__len__"):
        eta = speed_dist(vmin_matrix, vE,v0,vesc)
        return simpson(eta,vmin_matrix)
    else:
        eta = speed_dist(vmin_matrix, vE,v0,vesc)
        if u>vE+vEsc: return simpson(eta,vmin_matrix).astype(float)
"""



def dRdE(material, mX, Ee, FDMn, halo, params):
    """
    differential scattering rate for sigma_e = 1 cm^2
    at a fixed electron energy Ee
    given a DM mass, FDM, halo model
    returns dR/dE [events/kg/year]
    """
    n = FDMn
    vesc, vE = params[2], params[0]
    if (Ee < materials[material][2]) or (Ee>50.0): # check if less than Egap
        return 0
    else:
        qunit = dQ
        Eunit = dE
        Mcell = materials[material][0]
        Eprefactor = materials[material][1]
        Ei = int(np.floor(Ee*10)) # eV
        #print("Ei: ", Ei)
        prefactor = ccms**2*sec2year*rho_X/mX*1/Mcell*alpha*me_eV**2 / mu_Xe(mX)**2
        array_ = np.zeros(nq)
        qi = np.arange(1,nq+1)
        q = qi*qunit
        vmin = (q/(2*mX)+Ee/q)*ccms
        #vmin_where = np.where(vmin > (vesc+vE)*1.1)[0]
        vmin_where = np.where(vmin > (vesc+vE))[0]
        #eta = calcEta(vmin*1e-5, params[0]*1e-5, params[1]*1e-5, params[2]*1e-5)*1e-5
        eta = calcEta(vmin, params[0], params[1], params[2])
        #print([qi-1], [Ei-1])
        array_[qi-1] = Eprefactor*(1/q)*eta*FDM(q,n)**2*materials[material][4][qi-1, Ei-1]
        array_[vmin_where-1] = 0
        return prefactor*np.sum(array_, axis=0) # [(kg-year)^-1]


def dRdE_slow(material, mX, Ee, FDMn, halo, params):
    """
    differential scattering rate for sigma_e = 1 cm^2
    at a fixed electron energy Ee
    given a DM mass, FDM, halo model
    returns dR/dE [events/kg/year]
    """

    vesc, vE = params[2], params[0]
    n = FDMn
    if Ee < materials[material][2]: # check if less than Egap
        return 0
    else:
        qunit = dQ
        Eunit = dE
        Mcell = materials[material][0]
        Eprefactor = materials[material][1]
        Ei = int(np.floor(Ee*10)) # eV
        prefactor = ccms**2*sec2year*rho_X/mX*1/Mcell*alpha*me_eV**2 / mu_Xe(mX)**2
        array_ = np.zeros(nq)
        for qi in range(1,nq+1):
            q = qi*qunit
            vmin = (q/(2*mX)+Ee/q)*ccms
            if vmin > (vesc+vE)*1.1: # rough estimate for kinematicaly allowed regions
                array_[qi-1] = 0
            else:
                #eta = calcEta(vmin, params[0],params[1],params[2])
                #define array
                #array_[qi-1] = Eprefactor*(1/q)*eta*FDM(q,n)**2*materials[material][4][qi-1, Ei-1]
                if halo == 'shm':
                    eta = etaSHM(vmin,params) # (cm/s)^-1
                elif halo == 'tsa':
                    eta = etaTsa(vmin,params)
                elif halo == 'dpl':
                    eta = etaDPL(vmin,params)
                elif halo == 'msw':
                    eta = etaMSW(vmin,params)
                elif halo == 'debris':
                    eta = etaDF(vmin,params)
                else:
                    print("Undefined halo parameter. Options are ['shm','tsa','dpl','msw','debris']")
                array_[qi-1] = Eprefactor*(1/q)*eta*FDM(q,n)**2*materials[material][4][qi-1, Ei-1]
        return prefactor*np.sum(array_, axis=0) # [(kg-year)^-1]


def dRdne(sigmae, mX, ne, FDMn, halo, params,rhoDM=rho_X,Ebin=materials["Si"][3],material = "Si", slow=False):
    """
    calculates rate for sigmae = 1 cm^2 in the ne bin,
    assuming fiducial values for binsize
    Si: 3.8 eV, Ge: 2.9 eV
    return dRdne [events/kg/year]
    """
    #Ebin = materials[material][3]
    ## check if ne is defined

    if ne*Ebin > dE*nE:
        print(ne,sigmae)
        print('$n_e$ is out of range, pick a smaller value')
        return 0
    elif ne == 0:
        print('$n_e$ must be > 0')
        return 0
    else:
        tmpEbin = int(np.floor(Ebin/dE))+1
        tmpdRdE = np.zeros(tmpEbin)

        if slow:
            #start_time = time.time()
            for i in range(tmpEbin):
                ## add up in bins of [1.2,4.9],[5,8.7],...
                tmpdRdE[i] = dRdE_slow(material, mX, (ne-1)*Ebin+dE*i+materials[material][2], FDMn, halo, params)
            dRdne_slow = np.sum(tmpdRdE, axis = 0)
            dRdne = dRdne_slow
            #print("Time Slow: ", (time.time() - start_time)/60, ' min')
            #print("Slow: ", dRdne_slow)
        else:
            #start_time = time.time()
            for i in range(tmpEbin):
                ## add up in bins of [1.2,4.9],[5,8.7],...
                #print(ne-1, Ebin, dE, i, materials[material][2])
                tmpdRdE[i] = dRdE(material, mX, (ne-1)*Ebin+dE*i+materials[material][2], FDMn, halo, params)
                #tmpdRdE[i] = dRdE(material, mX, (ne-1)*Ebin+dE*i+1.11, FDMn, halo, params)
            dRdne = np.sum(tmpdRdE, axis = 0)
            #print("Time Fast: ", (time.time() - start_time)/60, ' min')
            #print("Fast: ", dRdne)
        return sigmae*dRdne/365

def dRdnearray(material, mX, Ebin, FDMn, halo, params):
    """
    calculates rate for sigmae = 1 cm^2, binned in Ebin,
    return binlist [eV], dRdne [events/kg/year/Ebin]
    """
    numbins = int(np.floor(nE*dE/Ebin))# calculate number of bins of size Ebin
    binlist = [materials[material][2]+ii*Ebin for ii in range(numbins)]
    tmpEbin = int(np.floor(Ebin/dE))
    tmpdRdE = np.zeros(tmpEbin)
    array_ = np.zeros(numbins)

    for ne in range(numbins-1):
        for i in range(tmpEbin):
            tmpdRdE[i] = dRdE(material, mX, ne*Ebin+dE*i+materials[material][2], FDMn, halo, params)
        array_[ne] = np.sum(tmpdRdE, axis = 0)

    return binlist, array_

def sigmae(mXlist, material, nevents, exposure, ne, FDMn, halo, params):
    """
    cross-section assuming nevents (3 for bkgdfree),
    exposure is in kg-years,
    takes in array of mX in MeV
    returns mX [MeV], sigma_e [cm^2]
    """
#    nevents = 3 # for bkgd free experiment
    sigmae = np.zeros(len(mXlist))
    for iimX in range(len(mXlist)):
        mX = mXlist[iimX]*1e6
        if dRdne(material, mX, ne, FDMn, halo, params) == 0:
            sigmae[iimX] = np.nan
        else:
            sigmae[iimX] = nevents/(dRdne(material, mX, ne, FDMn, halo, params)*exposure)
    return mXlist, sigmae


