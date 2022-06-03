import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
import scipy.special
from scipy.stats import norm
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.optimize import bisect 

from WIMpy import DMUtils as DMU
from differential_rate_electronDM import *


class dm_event(object):
    def __init__(self, mass_dm, cross_section, q, mass_det, t_exp, noise, nx = 4000, ny= 1000, nccd= 2, tread=2, ccd_mass = 0.02,xmin=-2,xmax=5, dRdE_name="Si"):
        self.q, self.mass_dm, self.cross_section = q, mass_dm, cross_section
        self.mass_det, self.t_exp = mass_det, t_exp
        self.nx, self.ny, self.nccd = nx, ny, nccd
        self.tread = tread/24
        #self.n_image = int(np.round(self.t_exp/self.tread))
        self.n_image = self.mass_det/ccd_mass
        self.pix_mass = self.nx*self.ny/ccd_mass#*self.mass_det/ccd_mass
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
            try:
                se = dRdne(self.cross_section,self.mass_dm,ne_i,self.q,"shm",[220e5,232e5,544e5])
                if se == 0.0:
                    break
                self.dRdne.append(se)
                self.ne.append(ne_i)
                ne_i += 1
            except IndexError:
                ### It may reach the last index of the crystal form factor.
                ### We avoid this
                break
        self.dRdne = np.array(self.dRdne)
        self.C_sig = np.sum(self.dRdne)
        self.s = self.C_sig*self.tread*self.mass_det
        self.n_s_det = np.random.poisson(self.s)
        self.fs = np.array(self.dRdne)/self.C_sig


    def simul_ev(self, bkg_pars, **kwargs):
        """ Simulates n_s_det events for signal.
            bkg_pars contains the dark current.
        """
        self.noise = kwargs["noise"] if "noise" in kwargs else self.noise
        #xmin = kwargs["xmin"] if "xmin" in kwargs else self.xmin
        #xmax = kwargs["xmax"] if "xmax" in kwargs else self.xmax

        if bkg_pars:
            #### Add background events
            self.simul_bkg(bkg_pars)
            self.simul_dm()
            self.events = self.signal_ev + self.bkg_ev
            ### Add readout noise
            self.events = self.events + np.random.normal(0,self.noise, self.npix)
            self.events = self.events.tolist()
            """
            plt.hist(self.events, bins=int((self.xmax-self.xmin)/0.1))
            plt.yscale("log")
            plt.show()
            """
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
        self.lamb = darkC*self.tread
        print("Lamb: ", self.lamb)
        ### Events in Eee
        self.bkg_ev = np.random.poisson(self.lamb, self.npix)


    """
    def joint_probability(self, **kwargs):

        x0 = kwargs["x0"] if "x0" in kwargs else [self.npix/0.1, 0, 1.6, 1, 1e-4, np.log10(self.cross_section)]
        fix_pars = kwargs["fix_pars"] if "fix_pars" in kwargs else []
        pars_lims = kwargs["pars_lims"] if "pars_lims" in kwargs else []
        self.cross_section_original = self.cross_section

        self.iter = 0

        def prob(x,p):
            #Npix,mu,noise,gain,dc,xs = p[0],p[1],p[2],p[3],p[4],p[5]
            Npix,mu,noise,gain,dc,xs = p[0],p[1],p[2],p[3],p[4]*self.t_exp,p[5]
            self.iter += 1
            if self.iter % 100 == 0:
                print(Npix,mu,noise,gain,dc,xs)
            self.cross_section = 10**xs
            self.normalization_signal()
            
            S = np.array([self.npix-self.s] + (self.fs*self.s).tolist())/self.npix
            #S = np.array([1-np.sum(self.fs)] + self.fs.tolist())
            pdf = Npix*S[0]*r.TMath.Poisson(0,dc)*r.TMath.Gaus(x[0],mu*gain,noise*gain,r.kTRUE)
            nPeaks = np.floor(np.max(self.events)+1).astype(int)
            for ntot in range(1,nPeaks):
                for j in range(0,ntot):
                    pdf += Npix*S[j]*r.TMath.Poisson(ntot-j,dc)*r.TMath.Gaus(x[0],(mu+ntot)*gain,noise*gain,1)
            return pdf

        self.simul_ev(1e-3)

        nbins = int((self.xmax-self.xmin)/0.1)
        hist = plt.hist(self.events, bins=nbins)#, density=True)
        n_data, dx = hist[0],hist[1]
        n_theo = prob(dx, x0)
        plt.plot(dx, n_theo)
        plt.yscale("log")
        plt.ylim(0.1,None)
        plt.show()
        #n = np.sum(n_data)

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
    """

    def likelihood(self, **kwargs):
        """ Minimizes the log likelihood of signal+background model.
            It can also calculate the upper limits given a value for dLL.
            It can simulate the signal and background if it havent already simulated.
        """
        x0 = kwargs["x0"] if "x0" in kwargs else [self.npix/0.1, 0, 1.6, 1, 1e-4, np.log10(self.cross_section)]
        fix_pars = kwargs["fix_pars"] if "fix_pars" in kwargs else []
        pars_lims = kwargs["pars_lims"] if "pars_lims" in kwargs else []
        upper_limit = kwargs["upper_limit"] if "upper_limit" in kwargs else None
        simulate = kwargs["simulate"] if "simulate" in kwargs else None
        verbose = kwargs["verbose"] if "verbose" in kwargs else False

        self.cross_section_original = self.cross_section

        def prob(p,x):
            Npix,mu,noise,gain,dc,xs = p[0],p[1],p[2],p[3],p[4]*self.tread,p[5]
            #print(Npix,mu,noise,gain,dc,xs)
            self.cross_section = 10**xs
            self.normalization_signal()
 
            #f_str_num = []
            #f_str = []
            
            ### Adds the 0 electron to peak to DM interaction probability 
            S = np.array([self.npix-self.s] + (self.fs*self.s).tolist())/self.npix
            ### To match the number of peaks
            if self.nPeaks > len(S):
                S_extended = np.zeros([self.nPeaks])
                S_extended[:len(S)] = S
                S = S_extended
            pdf = 0 
            
            for ntot in range(0,self.nPeaks):
                for j in range(0,ntot+1):
                    #f_str_num.append("Npix*S({})*P({})*G({})".format(np.round(S[j],2),np.round(poisson.pmf(ntot-j,dc),2),ntot))
                    #f_str.append("Npix*S({})*P({})*G({})".format(j,ntot-j,ntot))
                    pdf += Npix*S[j]*poisson.pmf(ntot-j,dc)*norm.pdf(x,(mu+ntot)*gain,noise*gain)
            #print("+".join(f_str))
            #print("+".join(f_str_num))
            return pdf


        if upper_limit and simulate:
            deltaLL, bkg_only = upper_limit
            dc = simulate 
            if bkg_only:
                self.simul_bkg(dc)
                self.events = self.bkg_ev
                self.events = self.events + np.random.normal(0,self.noise, self.npix)
                self.events = self.events.tolist()
            else:
                self.simul_ev(dc)
            ### Add readout noise
        if upper_limit and not simulate:
            deltaLL = upper_limit
        elif simulate and not upper_limit:
            dc = simulate 
            self.simul_ev(dc)
        elif not simulate and not upper_limit:
            if hasattr(self, "events"):
                pass
            else:
                print("Data Not yet simulated. Goodbye!!")


        ### Sets the number of electron peaks to fit
        self.nPeaks = kwargs["nPeaks"] if "nPeaks" in kwargs else int(np.round(np.max(self.events)))+1
        print("Number of electron peaks: ", self.nPeaks)

        ### Sets the number of bins and the bin content
        bin_size = kwargs["bin_size"] if "bin_size" in kwargs else 0.1
        nbins = int((self.xmax-self.xmin)/bin_size)

        ### Creates and histogram to get the bin content
        hist = np.histogram(self.events, nbins)
        n_data, dx = hist[0],hist[1]

        if verbose:
            ### Plot the events histogram (Useful to see the number of peaks)
            dx_m = (dx[:-1] + dx[1:])/2
            dx_m = dx[1:]#.flip(dx)))
            n_theo = prob(x0,dx_m)*np.diff(dx)
            #plt.plot(dx_m, n_theo, color = "r", label = "Initial Guess")
            plt.hist(self.events, nbins)
            plt.yscale("log")
            plt.ylim(0.1,None)
            plt.show() 
            plt.clf()

        def log_like(theta,n_data,dx):
            ### Uses the bin center approximation for binning the likelihood
            dx_m = (dx[:-1] + dx[1:])/2
            n_theo = prob(theta,dx_m)*np.diff(dx)
            #lnL = theta[0] - np.sum(n_data*np.log(n_theo)) 
            lnL = np.sum(n_theo) - np.sum(n_data*np.log(n_theo)) 
            return lnL

        ### Minimizes the loglikelihood
        if len(fix_pars) != 0:
            ### Fix the given parameter numbers
            p = np.zeros_like(x0)+np.inf
            x0 = np.array(x0)
            p[fix_pars] = x0[fix_pars]
            not_fix_pars = np.where(x0!=p)[0]
            pars_lims_nofix = np.array(pars_lims)[not_fix_pars]
            x0_nofix = x0[not_fix_pars]

            def log_like_fix(theta, n_data, dx):
                """ Function with fixed parameters
                """
                p[not_fix_pars] = theta
                return log_like(p, n_data, dx)

            ### Minimizes log-likelihood
            theta_max = minimize(log_like_fix, x0_nofix, bounds=pars_lims_nofix,method='Powell', args = (n_data,dx)).x

            ### Returns fitted+fixed parameters
            p[not_fix_pars] = theta_max
            theta_max = p
        else:
            ### Not fixed parameters
            theta_max = minimize(log_like, x0_nofix, bounds=pars_lims,method='Powell', args = (n_data,dx)).x

        print("----------- Fit Result -----------")
        pars_name = [r"$Norm$", r"$\mu_{0}$", r"$noise$", r"$\Omega$",r"$\lambda$", r"$\sigma_{DM-e}$"]
        fit_leg = []
        for p, name in zip(theta_max, pars_name):
            fit_leg.append(name+" = "+str(p))
        textstr = "\n".join(fit_leg)
        print(textstr) 
        print("----------------------------------")


        if verbose:
            ### Plot fit and histogram
            plt.clf()
            fig, ax = plt.subplots()
            nbins = int((self.xmax-self.xmin)/bin_size)
            ax.hist(self.events, bins=nbins)#, density=True)
            n_data, dx = hist[0],hist[1]

            dx_m = (dx[:-1] + dx[1:])/2
            dx_m = dx[1:]#.flip(dx)))
            n_theo = prob(theta_max,dx_m)*np.diff(dx)
            plt.plot(dx_m, n_theo, color = "r", label = "Best Fit")
            plt.yscale("log")
            plt.ylim(0.1,None)
            plt.legend(loc="best")

            pars_name = [r"$Norm$", r"$\mu_{0}$", r"$noise$", r"$\Omega$",r"$\lambda$", r"$\sigma_{DM-e}$"]
            fit_leg = []
            for p, name in zip(theta_max, pars_name):
                fit_leg.append(name+" = "+str(p))
            
            ### Legend text
            textstr = "\n".join(fit_leg)
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            plt.show()
            plt.clf()


        ### Calculates the upper limit
        if upper_limit:
            ### Value of the likelihood at its maximum
            lnL_max = -log_like(theta_max, n_data, dx)

            def lnL_dLL(xs_range, n_data, dx):
                """ Function to find the value where ln_max- ln_confidence = deltaLL
                """
                theta = np.array(theta_max[:-1]).tolist() + [xs_range]
                return -log_like(theta, n_data, dx)-lnL_max+deltaLL

            if verbose:
                xs_range = np.linspace(pars_lims[-1][0], pars_lims[-1][1], 100)
                dLL_range = [lnL_dLL(xs_i,n_data,dx) for xs_i in xs_range]
                plt.plot(10**xs_range, dLL_range)
                plt.xscale("log")
                plt.ylim(0,deltaLL*1.5)
                plt.ylabel(r"$d\mathcal{LL}$")
                plt.xlabel(r"$\sigma$ [cm$^{-2}$]")
                plt.show()

            ### Upper limit cross section
            self.cross_section_dLL = 10**bisect(lnL_dLL, pars_lims[-1][0], pars_lims[-1][1], args=(n_data,dx),rtol=1e-3)
            print(self.cross_section_dLL)



    def verbose(self):
        print("------------ Using {} model -----------".format(self.dRdE_name))
        print("Number of Total Pixels: ", self.npix)
        print("Number of Signal events: ", self.n_s_det)
        if hasattr(self, "theta"): print("Likelihood parameters: ", self.theta)
        if hasattr(self, "cross_section_dLL"): print("Upper Limit Cross Section: ", self.cross_section_dLL)
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

