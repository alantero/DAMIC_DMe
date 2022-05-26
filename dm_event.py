import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
import scipy.special
from scipy.stats import norm
from scipy.stats import poisson

from WIMpy import DMUtils as DMU
#from lindhard import *
from lindhard import lindhard_transform 
from differential_rate import *
from simulate_events import *
from likelihood import *
from differential_rate_electronDM import *


class dm_event(object):
    def __init__(self, mass_dm, cross_section, q, mass_det, t_exp, noise, nx = 4000, ny= 1000, nccd= 2, tread=2, ccd_mass = 0.02,xmin=-2,xmax=5, dRdE_name="Si"):
        self.q, self.mass_dm, self.cross_section = q, mass_dm, cross_section
        self.mass_det, self.t_exp = mass_det, t_exp
        self.nx, self.ny, self.nccd = nx, ny, nccd
        self.n_image = int(np.round(self.t_exp/(tread/24)))
        self.pix_mass = self.nx*self.ny*self.mass_det/ccd_mass
        #self.npix = self.nccd*self.nx*self.ny*self.n_image
        self.npix = int(np.round(self.pix_mass*self.n_image))
        self.xmin, self.xmax = xmin, xmax
        
        ### dRdE model
        self.dRdE_name = dRdE_name
        self.dRdE_model = dRdne#getattr(DMU, self.dRdE_name)
        
        ### Signal Efficiency resolution
        self.noise = noise
        self.mu = 0
        self.gain = 1

        self.normalization_signal()


    def normalization_signal(self):
        """ Calculates the normalization of the signal PDF.
        """

        ne_i = 1 # Number of electrons produced
        self.dRdne = []
        self.ne = []

        while True:
            se = dRdne(self.cross_section,self.mass_dm,ne_i,self.q,"shm",[220e5,232e5,544e5])
            if se == 0.0:
                break
            self.dRdne.append(se)
            self.ne.append(ne_i)
            ne_i += 1
        
        self.dRdne = np.array(self.dRdne)
        self.C_sig = np.sum(self.dRdne)
        self.s = self.C_sig*self.t_exp*self.mass_det
        self.n_s_det = np.random.poisson(self.s)
        self.fs = np.array(self.dRdne)/self.C_sig


    def dRdne_calc(self, xs, mx):
        """ Calculates the normalization of the signal PDF.
        """
        
        ne_i = 1 # Number of electrons produced
        dRdne_l = []
        ne = []

        while True:
            se = dRdne(xs,mx,ne_i,self.q,"shm",[220e5,232e5,544e5])
            if se == 0.0:
                break
            dRdne_l.append(se)
            ne.append(ne_i)
            ne_i += 1
        
        dRdne_l = np.array(dRdne_l)
        C_sig = np.sum(dRdne_l)
        s = C_sig*self.t_exp*self.mass_det
        fs = dRdne_l/C_sig
        return s,fs


    def simul_ev(self, bkg_pars, **kwargs):
        """ Simulates n_s_det events for signal.
            bkg_pars contains the dark current.
        """
        self.noise = kwargs["noise"] if "noise" in kwargs else self.noise
        xmin = kwargs["xmin"] if "xmin" in kwargs else self.xmin
        xmax = kwargs["xmax"] if "xmax" in kwargs else self.xmax

        if bkg_pars:
            #### Add background events
            lamb = bkg_pars*self.t_exp 
            self.simul_bkg(bkg_pars)
            self.simul_dm()
            self.events = self.signal_ev + self.bkg_ev
            ### Add readout noise
            self.events = self.events + np.random.normal(0,self.noise, self.npix)
            self.events = self.events.tolist()
            plt.hist(self.events, bins=int((self.xmax-self.xmin)/0.1))
            plt.yscale("log")
            plt.show()
            self.signal_generated = True
            return self.events
        else:
            self.simul_dm()
            self.events = self.signal_ev
            self.events = np.array(self.events) + np.random.normal(0,self.noise, self.npix)
            self.events = self.events.tolist()
            self.signal_generated = True
            return self.signal_ev

    def simul_dm(self):
        nx = np.random.randint(0,self.npix,self.n_s_det)
        self.signal_ev = np.zeros([self.npix])
        dm_ev = np.random.choice(self.ne, self.n_s_det, p=self.fs)
        self.signal_ev[nx] = dm_ev

    def simul_bkg(self, darkC):
        self.darkC = darkC
        self.lamb = darkC*self.t_exp
        print("Lamb: ", self.lamb)
        ### Events in Eee
        self.bkg_ev = np.random.poisson(self.lamb, self.npix)

    def joint_probability(self, **kwargs):

        x0 = kwargs["x0"] if "x0" in kwargs else [self.npix/0.1, 0, 1.6, 1, 1e-4, np.log10(self.cross_section)]
        fix_pars = kwargs["fix_pars"] if "fix_pars" in kwargs else []
        pars_lims = kwargs["pars_lims"] if "pars_lims" in kwargs else []
        self.cross_section_original = self.cross_section

        def prob(x,p):
            #Npix,mu,noise,gain,dc,xs = p[0],p[1],p[2],p[3],p[4],p[5]
            Npix,mu,noise,gain,dc,xs = p[0],p[1],p[2],p[3],p[4]*self.t_exp,p[5]
            self.cross_section = 10**xs
            self.normalization_signal()
            """
            s,fs = self.dRdne_calc(xs,self.mass_dm)
            ds = s*fs
            S = [self.npix-s]+ds.tolist()
            S = np.array(S)/(self.npix)
            """
            #S = np.array([self.npix-self.s] + (self.fs*self.s).tolist())/self.npix
            S = np.array([1-np.sum(self.fs)] + self.fs.tolist())
            pdf = Npix*S[0]*r.TMath.Poisson(0,dc)*r.TMath.Gaus(x[0],mu*gain,noise*gain,r.kTRUE)
            nPeaks = np.floor(np.max(self.events)+1).astype(int)
            for ntot in range(1,nPeaks):
                for j in range(0,ntot):
                    pdf += Npix*S[j]*r.TMath.Poisson(ntot-j,dc)*r.TMath.Gaus(x[0],(mu+ntot)*gain,noise*gain,1)
            return pdf

        self.simul_ev(1e-3)
        r.gStyle.SetOptFit()
        nbins = int((self.xmax-self.xmin)/0.1)
        hist = r.TH1F('hist', 'DM PCD', nbins, self.xmin, self.xmax)
        for ev in self.events:
            hist.Fill(ev)
        fitfunc = r.TF1("fitfunc",prob,self.xmin,self.xmax, 6)
        #fitfunc.SetParameters(self.npix/0.1, 0, 0.1, 1, 1e-4*self.t_exp, np.log10(self.cross_section))
        fitfunc.SetParameters(x0[0],x0[1],x0[2],x0[3],x0[4],x0[5])
        fitfunc.SetParNames("Norm", "#mu_{0}", "noise", "#Omega","#lambda", "#sigma_{DM-e}")

        if len(pars_lims)>0:
            for i,lim in enumerate(pars_lims):
                fitfunc.SetParLimits(i,lim[0],lim[1])

        if len(fix_pars)>0:
            for p_i in fix_pars:
                print("Parameter Fixed: ", fitfunc.GetParName(p_i))
                fitfunc.FixParameter(p_i,x0[p_i])

        f0 = hist.Fit(fitfunc, "QSL")
        c = r.TCanvas()
        hist.Draw()
        c.SetLogy()


    def likelihood(self, **kwargs):

        x0 = kwargs["x0"] if "x0" in kwargs else [self.npix/0.1, 0, 1.6, 1, 1e-4, np.log10(self.cross_section)]
        fix_pars = kwargs["fix_pars"] if "fix_pars" in kwargs else []
        pars_lims = kwargs["pars_lims"] if "pars_lims" in kwargs else []
        self.cross_section_original = self.cross_section

        def prob(p,x):
            Npix,mu,noise,gain,dc,xs = p[0],p[1],p[2],p[3],p[4],p[5]
            #mu,noise,gain,dc,xs = p[0],p[1],p[2],p[3]*self.t_exp,p[4]
            print(Npix,mu,noise,gain,dc,xs)
            self.cross_section = 10**xs
            self.normalization_signal()
            """
            s,fs = self.dRdne_calc(xs,self.mass_dm)
            ds = s*fs
            S = [self.npix-s]+ds.tolist()
            S = np.array(S)/(self.npix)
            """
            S = np.array([self.npix-self.s] + (self.fs*self.s).tolist())/self.npix
            pdf = Npix*S[0]*poisson.pmf(0,dc)*norm.pdf(x,mu*gain,noise*gain)
            nPeaks = np.floor(np.max(self.events)+1).astype(int)
            for ntot in range(1,nPeaks):
                for j in range(0,ntot):
                    pdf += Npix*S[j]*poisson.pmf(ntot-j,dc)*norm.pdf(x,(mu+ntot)*gain,noise*gain)
            return pdf

        self.simul_ev(1e-3)
 
        nbins = int((self.xmax-self.xmin)/0.1)
        hist = plt.hist(self.events, bins=nbins)#, density=True)
        n_data, dx = hist[0],hist[1]
        n = np.sum(n_data)

        def log_like(theta,n_data,dx):
            dx_m = (dx[:-1] + dx[1:])/2
            n_theo = prob(theta,dx_m)*np.diff(dx)
            print(np.sum(n_theo), theta[0])
            #plt.plot(dx_m,n_data,'o')
            #plt.plot(dx_m,n_theo)
            #plt.yscale("log")
            #plt.ylim(0.1, None)
            #plt.show()
            #lnL = -np.log(scipy.special.factorial(n)) -np.sum(n_data*np.log(n_theo)) + np.sum(scipy.special.factorial(n_data))
            lnL = theta[0] - np.sum(n_data*np.log(n_theo)) 
            return lnL

        theta_max = minimize(log_like, x0, bounds=pars_lims,method='Powell', args = (n_data,dx)).x
        print(theta_max)


    def verbose(self):
        print("------------ Using {} model -----------".format(self.dRdE_name))
        print("Number of Total Pixels: ", self.npix)
        print("Number of Signal events: ", self.n_s_det)
        if hasattr(self, "theta"): print("Likelihood parameters: ", self.theta)
        if hasattr(self, "cross_section_95"): print("Upper Limit Cross Section: ", self.cross_section_95)
        if hasattr(self, "theta_confidence"): print("Theta Confidence intervals: ", self.confidence)

    def plot_var(self,var_name, step = 100, **kwargs):
        """ Make different plots of the variables. 
        """
        if "xmin" in kwargs:
            xmin = kwargs["xmin"]
        else:
            xmin = self.xmin
        if "xmax" in kwargs:
            xmax = kwargs["xmax"]
        else:
            xmax = self.xmax

        label = kwargs["label"] if "label" in kwargs else None
        n_std = kwargs["n_std"] if "n_std" in kwargs else 5

        if var_name == "pcd":
                if hasattr(self, "events"):
                    plt.hist(self.events, bins = int((self.x_max-self.min)/0.1))
                plt.xlabel(r"Charge [e$^-$]")

