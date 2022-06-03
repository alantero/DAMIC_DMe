import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.optimize import bisect
from matplotlib import colors 
from scipy.stats import multivariate_normal
import pickle

from differential_rate_electronDM import *


class dm_event2D(object):
    def __init__(self, mass_dm, cross_section, q, mass_det, t_exp, noise, nx = 4000, ny=1000, nccd= 2, tread=2, amps=2, ccd_mass = 0.02,xmin=-2,xmax=5, dRdE_name="Si", nx_bin=1, ny_bin=1):
        self.q, self.mass_dm, self.cross_section = q, mass_dm, cross_section
        self.mass_det, self.t_exp = mass_det, t_exp
        self.nx, self.ny, self.nccd, self.amps = nx, ny, nccd, amps
        self.nx_bin, self.ny_bin = nx_bin, ny_bin
        self.namp = [i for i in range(self.amps)]
        self.n_image = int(np.round(self.t_exp/(tread/24)))
        self.tread = tread/24 
        self.pix_mass = self.nx*self.ny*self.mass_det/ccd_mass
        self.ccd_mass = ccd_mass
        #self.npix = self.nccd*self.nx*self.ny*self.n_image
        self.npix = int(np.round(self.pix_mass*self.n_image))
        self.xmin, self.xmax = xmin, xmax
        
        ### dRdE model
        self.dRdE_name = dRdE_name
        self.dRdE_model = dRdne#getattr(DMU, self.dRdE_name)
        
        ### Signal Efficiency resolution
        self.noise = noise
        self.mu = [0,0]
        self.gain = [1,1]

        self.xs_nosignal()
        self.normalization_signal()

    def xs_nosignal(self):
        original_xs = self.cross_section
        i = 0
        while True:
            self.cross_section = np.power(10,np.log10(self.cross_section)-i)
            self.normalization_signal()
            if self.s>1:
                i += 1
                continue
            else:
                break
        self.lowest_cross_section = self.cross_section
        self.cross_section = original_xs
        print("Lowest Cross Section: ", self.lowest_cross_section)


    def normalization_signal(self):
        """ Calculates the normalization of the signal PDF.
        """

        ne_i = 1 # Number of electrons produced
        self.dRdne = []
        self.ne = []


        if hasattr(self, "lowest_cross_section") and self.cross_section <= self.lowest_cross_section:
            self.dRdne = np.array([0])
            self.s = 0 
            self.C_sig = 0
            self.n_s_det = [0,0]
            self.fs = np.array([0])

        else:
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
            self.n_s_det = []
            self.C_sig = np.sum(self.dRdne)
            self.s = self.C_sig*self.t_exp*self.mass_det
            self.fs = np.array(self.dRdne)/self.C_sig

            for amp in self.namp:
                self.n_s_det.append( np.random.poisson(self.s) )


    def cov_matrix(self,sigma1,sigma2,rho):
        return np.array([[sigma1**2,rho*sigma1*sigma2],[rho*sigma1*sigma2,sigma2**2]]) 


    def simul_ev(self, darkC, **kwargs):
        """ Simulates n_s_det events for signal.
            bkg_pars contains the dark current.
        """
        self.noise = kwargs["noise"] if "noise" in kwargs else self.noise
        rho = kwargs["rho"] if "rho" in kwargs else 0
        self.mu = kwargs["mu"] if "mu" in kwargs else self.mu
        bin_size = kwargs["bin_size"] if "bin_size" in kwargs else 0.1
        nbins = int((self.xmax-self.xmin)/bin_size)
        #xmin = kwargs["xmin"] if "xmin" in kwargs else self.xmin
        #xmax = kwargs["xmax"] if "xmax" in kwargs else self.xmax


        if darkC:
            self.events = []
            #### Add background events
            self.simul_bkg(darkC)
            self.simul_dm()
            for amp in self.namp:
                self.events.append(self.signal_ev[amp] + self.bkg_ev[amp])

            ### Creates 2D histogram to simulate covariate readout noise
            #self.hist, xedge, yedge = np.histogram2d(self.events[0], self.events[1], bins=nbins)
            #self.plot_hist2D(bin_size=bin_size)
            cov = self.cov_matrix(self.noise[0], self.noise[1], rho)
            read_multi = multivariate_normal.rvs(self.mu, cov=cov, size=self.npix) 
            for amp in self.namp:
                self.events.append(self.signal_ev[amp] + self.bkg_ev[amp])
                ### Add readout noise
                self.events[amp] = self.events[amp] + read_multi[:,amp]
            self.signal_generated = True
            return self.events
        else:
            self.simul_dm()
            self.events = self.signal_ev
            self.events = np.array(self.events) + np.random.normal(self.mu[amp],self.noise[amp], self.npix)
            self.events = self.events.tolist()
            self.signal_generated = True
            return self.signal_ev

    def simul_dm(self):
        self.nx, self.signal_ev = [], []
        for amp in self.namp:
            if self.cross_section > self.lowest_cross_section:
                signal_ev_amp = np.zeros([self.npix])
                nx_amp = np.random.randint(0,self.npix,self.n_s_det[amp])
                dm_ev_amp = np.random.choice(self.ne, self.n_s_det[amp], p=self.fs)
                signal_ev_amp[nx_amp] = dm_ev_amp
                self.signal_ev.append(signal_ev_amp)
            else:
                self.signal_ev = [np.zeros(self.npix), np.zeros(self.npix)]

    def simul_bkg(self, darkC):
        self.lamb, self.bkg_ev = [], []
        for amp in self.namp:
            self.lamb.append( darkC[amp]*self.t_exp )
            ### Events in Eee
            self.bkg_ev.append( np.random.poisson(self.lamb[amp], self.npix) )
            print("Lamb Amp {}: ".format(amp), self.lamb[amp])


    def plot_hist2D(self, bin_size=0.1, xmin=-10, xmax=10):
        nbins = int((xmax-xmin)/bin_size)
        plt.hist2d(self.events[0], self.events[1], bins=nbins, cmap = "Greens", norm = colors.LogNorm())
        #plt.imshow(self.hist, aspect="auto", cmap = "Greens", norm = colors.LogNorm())
        plt.xlabel("Amp. {}".format(self.namp[0]))
        plt.ylabel("Amp. {}".format(self.namp[1]))
        plt.show()


    def likelihood(self, **kwargs):
        """ Minimizes the log likelihood of signal+background model.
            It can also calculate the upper limits given a value for dLL.
            It can simulate the signal and background if it havent already simulated.
        """
        x0 = kwargs["x0"] if "x0" in kwargs else [self.npix**2, 0, 0, 0.1,0.1, 0.4, 1, 1, 1e-4, 1e-4, np.log10(self.cross_section)]
        fix_pars = kwargs["fix_pars"] if "fix_pars" in kwargs else []
        pars_lims = kwargs["pars_lims"] if "pars_lims" in kwargs else [(1e3,1e14),(-0.1,0.1),(-0.1,0.1),(0.07,2),(0.07,2),(-1,1),(0.8,1.2),(0.8,1.2),(1e-5,0.1),(1e-5,0.1),(-40,-24)]
        upper_limit = kwargs["upper_limit"] if "upper_limit" in kwargs else None
        simulate = kwargs["simulate"] if "simulate" in kwargs else None
        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        method = kwargs["method"] if "method" in kwargs else "Powell" 

        ### Parameters Names 
        pars_name = [r"$Norm$", r"$\mu_{1}$", r"$\mu_{2}$", r"$noise_{1}$", r"$noise_{2}$", r"$\rho$", r"$\Omega_{1}$", r"$\Omega_{2}$",r"$\lambda_{1}$", r"$\lambda_{2}$",r"$\sigma_{DM-e}$"]

        self.cross_section_original = self.cross_section

        def prob(p,data):
            #Npix,mu1,mu2,noise1,noise2,rho,gain1,gain2,dc1,dc2,xs = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]*self.t_exp*self.nx_bin*self.ny_bin,p[9]*self.t_exp*self.nx_bin*self.ny_bin,p[10]
            Npix,mu1,mu2,noise1,noise2,rho,gain1,gain2,dc1,dc2,xs = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]*self.t_exp,p[9]*self.t_exp,p[10]
            #print(pars_name)    
            #print(Npix,mu1,mu2,noise1,noise2,rho,gain1,gain2,dc1,dc2,xs)
            #print(Npix,mu,noise,gain,dc,xs)
            self.cross_section = 10**xs
            self.normalization_signal()
 
            #f_str_num = []
            f_str = []
            
            ### Adds the 0 electron to peak to DM interaction probability 
            S = np.array([self.npix-self.s] + (self.fs*self.s).tolist())/self.npix
            ### To match the number of peaks
            if self.nPeaks > len(S):
                S_extended = np.zeros([self.nPeaks])
                S_extended[:len(S)] = S
                S = S_extended
            pdf = 0 
            ### Gaussian parameters
            ### FIXME if noise1,noise2 and rho are not fixed. Matrix be not positive defined. Then crashes
            cov = self.cov_matrix(gain1*noise1, gain2*noise2, rho)
            #z = [xy for xy in zip(x,y)]

            for ntotx in range(0, self.nPeaks):
                for ntoty in range(0,self.nPeaks):
                    mu = [gain1*mu1+ntotx,gain2*mu2+ntoty]
                    try:
                        gauss = multivariate_normal.pdf(data,mean = mu, cov=cov)
                    except ValueError:
                        print("Not Positive Defined")
                        print(rho,noise1,noise2)
                        print(cov)
                        return -np.inf
                    for jx in range(0,ntotx+1):
                        for jy in range(0,ntoty+1):
                            #f_str_num.append("Npix*S({})*Px({})Py({})**G({})".format(np.round(S[j],2),np.round(poisson.pmf(ntotx-j,dc),2),ntot))
                            #f_str.append("Npix*Sx({})*Sy({})*Px({})*Py({})*G({})".format(jx,jy,ntotx-jx,ntoty-jy,[ntotx,ntoty]))
                            pdf += Npix*S[jx]*S[jy]*poisson.pmf(ntotx-jx,dc1)*poisson.pmf(ntoty-jy,dc2)*gauss

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

        ### Creates 2D histogram to simulate covariate readout noise
        n_data, dx, dy = np.histogram2d(self.events[0], self.events[1], bins=nbins)

        def log_like(theta,n_data,dx,dy):
            ### Uses the bin center approximation for binning the likelihood
            dx_m = (dx[:-1] + dx[1:])/2
            dy_m = (dy[:-1] + dy[1:])/2
            DX, DY = np.meshgrid(np.diff(dx), np.diff(dy)) 
            X,Y = np.meshgrid(dx_m,dy_m)
            data = np.dstack((X, Y))
            n_theo = prob(theta, data)*np.diff(dx)*np.diff(dy)
 
            #n_theo = prob(theta,dx_m,dy_m)*np.diff(dx)*np.diff(dy)
            #lnL = theta[0] - np.sum(n_data*np.log(n_theo)) 
            lnL = np.sum(n_theo) - np.sum(n_data*np.log(n_theo)) 
            return lnL

        ### Minimizes the loglikelihood
        if len(fix_pars) != 0:
            print("----------- Fixed Parameters -----------")
            for p_fix in fix_pars:
                print(pars_name[p_fix])
            print("----------------------------------")

            ### Fix the given parameter numbers
            p = np.zeros_like(x0)+np.inf
            x0 = np.array(x0)
            p[fix_pars] = x0[fix_pars]
            not_fix_pars = np.where(x0!=p)[0]
            pars_lims_nofix = np.array(pars_lims)[not_fix_pars]
            x0_nofix = x0[not_fix_pars]

            def log_like_fix(theta, n_data, dx,dy):
                """ Function with fixed parameters
                """
                p[not_fix_pars] = theta
                return log_like(p, n_data, dx,dy)

            ### Minimizes log-likelihood
            theta_max = minimize(log_like_fix, x0_nofix, bounds=pars_lims_nofix,method=method, args = (n_data,dx,dy)).x

            ### Returns fitted+fixed parameters
            p[not_fix_pars] = theta_max
            theta_max = p
        else:
            ### Not fixed parameters
            theta_max = minimize(log_like, x0, bounds=pars_lims,method=method, args = (n_data,dx,dy)).x

        print("----------- Fit Result -----------")
        fit_leg = []
        for p, name in zip(theta_max, pars_name):
            fit_leg.append(name+" = "+str(p))
        textstr = "\n".join(fit_leg)
        print(textstr) 
        print("----------------------------------")


        if verbose:
            plt.clf()
            ### Plot fit and histogram

            nbins = int((self.xmax-self.xmin)/bin_size)
            n_,dx_,dy_,image = plt.hist2d(self.events[0], self.events[1], bins=nbins,cmap = plt.cm.rainbow)
            plt.clf()
            fig, ax = plt.subplots()
            ax.set_xlabel("Amp. {}".format(self.namp[0]))
            ax.set_ylabel("Amp. {}".format(self.namp[1]))
 
            ax.contour(np.log10(1+n_.transpose()),extent=[dx_[0],dx_[-1],dy_[0],dy_[-1]],linewidths=3, cmap = "Greens", levels = verbose)


            #n_,dx,dy = hist[0],hist[1],hist[2]

            dx_m = (dx_[:-1] + dx_[1:])/2
            dx_m = dx_[1:]
            dy_m = (dy_[:-1] + dy_[1:])/2
            dy_m = dy_[1:]

            X,Y = np.meshgrid(dx_m,dy_m)
            data = np.dstack((X, Y))
            Z = prob(theta_max, data)*np.diff(dx_)*np.diff(dy_)
            
            #ax.contour(X, Y, Z, verbose,colors='black', label = "Best Fit",alpha=0.6, norm = colors.LogNorm())
            ax.contour(X, Y, np.log10(Z+1), verbose,colors='black',alpha=0.6)#, norm = colors.LogNorm())
            #plt.ylim(0.1,None)

            #pars_name = [r"$Norm$", r"$\mu_{0}$", r"$noise$", r"$\Omega$",r"$\lambda$", r"$\sigma_{DM-e}$"]
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

            ax.legend(loc="best")

            fig2, ax2 = plt.subplots()
            hist = ax2.hist2d(self.events[0], self.events[1], bins=nbins, cmap = "Greens")
            ax2.set_xlabel("Amp. {}".format(self.namp[0]))
            ax2.set_ylabel("Amp. {}".format(self.namp[1]))
            ax2.contour(X, Y, Z, verbose,colors='black', label = "Best Fit",alpha=0.6)

            plt.show()
            plt.clf()


        ### Calculates the upper limit
        if upper_limit:
            ### Value of the likelihood at its maximum
            lnL_max = -log_like(theta_max, n_data, dx, dy)

            def lnL_dLL(xs_range, n_data, dx, dy):
                """ Function to find the value where ln_max- ln_confidence = deltaLL
                """
                theta = np.array(theta_max[:-1]).tolist() + [xs_range]
                return -log_like(theta, n_data, dx, dy)-lnL_max+deltaLL

            if verbose:
                xs_range = np.linspace(pars_lims[-1][0], pars_lims[-1][1], 100)
                dLL_range = np.array([lnL_dLL(xs_i,n_data,dx, dy) for xs_i in xs_range])
                dLL_range = dLL_range[dLL_range != np.inf]
                xs_range = xs_range[dLL_range != np.inf]
                plt.plot(10**xs_range, dLL_range)
                plt.xscale("log")
                plt.ylim(0,deltaLL*1.5)
                plt.ylabel(r"$d\mathcal{LL}$")
                plt.xlabel(r"$\sigma$ [cm$^{-2}$]")
                plt.show()

            ### Upper limit cross section
            self.cross_section_dLL = 10**bisect(lnL_dLL, pars_lims[-1][0], pars_lims[-1][1], args=(n_data,dx,dy),rtol=1e-3)
            print(self.cross_section_dLL)


    def import_events(self, filename):
        pkl_data = pickle.load(open(filename,"rb"))
        self.events, total_pix, self.texp = pkl_data["Pixels"], pkl_data["TotalPix"], 1/0.26#pkl_data["texp"]*5
        self.npix = len(self.events[0])
        self.mass_det=self.ccd_mass*self.npix/total_pix
        print("{} g*days".format(self.mass_det*self.texp*1000))


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

