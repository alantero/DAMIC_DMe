import numpy as np
import matplotlib.pyplot as plt
import ROOT as r
from scipy.stats import norm
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.optimize import bisect 
from scipy.optimize import brentq
import emcee
import corner
from scipy.stats import chi2
import iminuit
from memoization import cached , CachingAlgorithmFlag
import pandas as pd
import pickle
import glob


from WIMpy import DMUtils as DMU
from differential_rate_electronDM import *


class dm_event(object):
    def __init__(self, mass_dm, cross_section, q, mass_det, t_exp, noise, nx = 4000, ny= 1000, nccd= 2, tread=2, ccd_mass = 0.02,xmin=-2,xmax=5,nxbin=1,nybin=1,n_image=38,mask_frac=0,dRdE_name="Si",rhoDM=0.3,Eeh=3.77,vpars=[250e5,238e5,544e5]):
        self.q, self.mass_dm, self.cross_section = q, mass_dm, cross_section

        ### All the variables should be now arrays for each ccd
        self.mass_det, self.t_exp = np.array(mass_det), np.array(t_exp)
        self.nx, self.ny = np.array(nx), np.array(ny)
        self.nxbin, self.nybin = np.array(nxbin), np.array(nybin)
        self.mask_frac = 1-np.array(mask_frac)
        self.tread = np.array(tread)/24
        #self.n_image = int(np.round(self.t_exp/self.tread))
        self.nccd = nccd
        self.n_image = np.array(n_image)#self.mass_det/ccd_mass
        self.pix_mass = self.nx/self.nxbin*self.ny/self.nybin*ccd_mass#*self.mass_det/ccd_mass
        self.npix = np.round(self.mask_frac*self.nx/self.nxbin*self.ny/self.nybin*self.n_image).astype(int)
        #self.npix = int(np.round(self.pix_mass*self.n_image))
        self.xmin, self.xmax = xmin, xmax
        self.vpars, self.rhoDM, self.Eeh = vpars, rhoDM, Eeh
        ### dRdE model
        self.dRdE_name = dRdE_name
        self.dRdE_model = dRdne#getattr(DMU, self.dRdE_name)
        
        ### Signal Efficiency resolution
        self.noise = np.array(noise)
        self.mu = np.array( [0] * self.nccd )
        self.gain = np.array([1] * self.nccd ) 

        self.normalization_signal()
        self.xs_nosignal()
        self.normalization_signal()
        self.diffusion = False


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
        #print("Lowest Cross Section: ", self.lowest_cross_section)

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
            self.n_s_det = 0
            self.fs = np.array([0])
        else:
            while True:
                try:
                    se = dRdne(self.cross_section,self.mass_dm,ne_i,self.q,"shm",self.vpars,rhoDM=self.rhoDM,Ebin=self.Eeh)
                    #print(ne_i,se,self.cross_section)
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
            self.s = self.C_sig*self.t_exp*self.mass_det
            #print("s",self.s)
            if not hasattr(self,"signal_simulated"):
                self.n_s_det = np.random.poisson(self.s, self.nccd)
                self.signal_simulated = True
            self.fs = np.array(self.dRdne)/self.C_sig


    def simul_ev(self, bkg_pars, **kwargs):
        """ Simulates n_s_det events for signal.
            bkg_pars contains the dark current.
        """

        if bkg_pars:
            #### Add background events
            mu0,noise,dc = bkg_pars 
            self.simul_bkg(dc)
            self.simul_dm()
            self.events = self.signal_ev + self.bkg_ev
            self.nPeaks = int(np.max(self.events))
            ### Add readout noise
            for i in range(self.nccd):
                self.events[i] = self.events[i] + np.random.normal(mu0[i],noise[i], self.npix[i])
            self.events = self.events
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
            for i in range(self.nccd):
                self.events[i] = self.events[i] + np.random.normal(mu0[i],noise[i], self.npix[i])
            self.events = self.events
            self.signal_generated = True
            return self.events

    def simul_dm(self):
        self.signal_ev = []
        for i in range(self.nccd):
            nx = np.random.randint(0,self.npix,self.n_s_det)
            signal_ev_i = np.zeros([self.npix])
            dm_ev = np.random.choice(self.ne, self.n_s_det[i], p=self.fs)
            signal_ev_i[nx] = dm_ev
            self.signal_ev.append(signal_ev_i)
        self.signal_ev = np.array(self.signal_ev)

    def simul_bkg(self, darkC):
        self.lamb = darkC
        self.bkg_ev = []
        for i in range(self.nccd):
            self.bkg_ev.append( np.random.poisson(self.lamb[i], self.npix[i]) )
        self.bkg_ev = np.array(self.bkg_ev)

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
            Npix,mu,noise,gain,dc,xs = p[0],p[1],p[2],p[3],p[4],p[5]
            #print(Npix,mu,noise,gain,dc,xs)
            self.cross_section = 10**xs
            if self.diffusion:
                self.normalization_signal_diffusion() 
            else:
                self.normalization_signal()
 
            #f_str_num = []
            #f_str = []
            
            ### Adds the 0 electron to peak to DM interaction probability 
            S = np.array([self.npix-self.s] + (self.fs*self.s).tolist())/self.npix_i
            ### If the number of signals electrons is less than background the number of peaks is extended
            if self.nPeaks > len(S):
                S_extended = np.zeros([self.nPeaks])
                S_extended[:len(S)] = S
                S = S_extended
            #if xs <= np.log10(self.lowest_cross_section):
            #    print(S)
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
            cf, bkg_only = upper_limit
            deltaLL = chi2.ppf(cf,df=1)/2
            mu0,noise,dc = simulate 
            #print(mu0, noise, dc)
            if bkg_only:
                self.simul_bkg(dc)
                self.events = self.bkg_ev
                self.nPeaks = int(np.max(self.events))
                for i in range(self.nccd):
                    self.events[i] = self.events[i] + np.random.normal(mu0[i],noise[i],self.npix[i])
            else:
                self.simul_ev(simulate)
                self.nPeaks = int(np.max(self.events))
            ### Add readout noise
        if upper_limit and not simulate:
            cf, _ = upper_limit
            deltaLL = chi2.ppf(cf,df=1)/2
            self.nPeaks = int(np.round(np.max(self.events)))
            if hasattr(self,"events"):
                pass
        elif simulate and not upper_limit:
            self.simul_ev(simulate)
        elif not simulate and not upper_limit:
            if hasattr(self, "events"):
                pass
            else:
                print("Data Not yet simulated. Goodbye!!")


        ### Sets the number of electron peaks to fit
        self.nPeaks = kwargs["nPeaks"] if "nPeaks" in kwargs else self.nPeaks 
        #print("Number of electron peaks: ", self.nPeaks)

        ### Sets the number of bins and the bin content
        bin_size = kwargs["bin_size"] if "bin_size" in kwargs else 0.1
        nbins = int((self.xmax-self.xmin)/bin_size)

        ### Creates and histogram to get the bin content
        n_data, dx = [], []
        for i in range(self.nccd):
            self.events[i] = self.events[(self.events[i]>=self.xmin) & (self.events[i]<=self.xmax)]
            hist = np.histogram(self.events[i], nbins)
            n_data.append(hist[0])
            dx.append(hist[1])

        
        def log_like(theta,data,x):
            ### Uses the bin center approximation for binning the likelihood
            dx_m = (x[:-1] + x[1:])/2
            n_theo = prob(theta,dx_m)*np.diff(x)
            n_theo = n_theo[data>0]
            data = data[data>0]
            #lnL = theta[0] - np.sum(n_data*np.log(n_theo)) 
            #lnL = np.sum(n_theo) - np.sum(n_data*np.log(n_theo)) 
            #n_data = n_data.astype(int)
            LL = data*np.log(n_theo)-n_theo+data-data*np.log(data)-0.5*np.log(2*np.pi*data)
            lnL = -np.sum(LL)
            #lnL = -np.sum(n_data*np.log(n_theo)-n_theo-(n_data*np.log(n_data)+n_data+np.log(2*np.pi*n_data))) 
            #lnL = np.sum(n_theo) - np.sum(n_data*np.log(n_theo)) 
            #print(lnL)
            return lnL

        pars_name = [r"$Norm$", r"$\mu_{0}$", r"$noise$", r"$\Omega$",r"$\lambda$"]#, r"$\sigma_{DM-e}$"]
        pars_name_dict = ["Norm", "mu", "noise", "gain","lamb"]#, "sigma"]
        ### Minimizes the loglikelihood
        if len(fix_pars) != 0:
            ### Fix the given parameter numbers
            pa_ = np.zeros_like(x0)+np.inf
            x0 = np.array(x0)
            pa_[fix_pars] = x0[fix_pars]
            not_fix_pars = np.where(x0!=pa_)[0]
            pars_lims_nofix = np.array(pars_lims)[not_fix_pars]
            x0_nofix = x0[not_fix_pars]

            @cached(algorithm=CachingAlgorithmFlag.LFU)
            def log_like_minuit(theta, data, x, ccd_i=0):#(Norm,mu,noise,gain,lamb,sigma):
                #theta = [Norm,mu,noise,gain,lamb,sigma]
                Norm,mu,noise,gain,lamb,sigma = theta
                ln_penalty = 0
                if 2 in fix_pars: # Readout gaussian penalty
                    ln_penalty += (x0[2+int(len(pars_name)*ccd_i)]-noise)**2/np.diff(pars_lims[2+int(len(pars_name)*ccd_i)])

                if 1 in fix_pars: # mu gaussian penalty
                    ln_penalty += (x0[1+int(len(pars_name)*ccd_i)]-mu)**2/np.diff(pars_lims[1+int(len(pars_name)*ccd_i)])

                return ln_penalty + log_like(theta, data, x)

            def log_like_amps(*p):
                LL = 0
                for ccd_i in range(self.nccd):
                    p_i = p[int(ccd_i*len(pars_name)):int(len(pars_name)+ccd_i*len(pars_name))]
                    ### DM cross section
                    p_i.append(p[-1])
                    self.npix_i = self.npix[ccd_i]
                    LL += log_like_minuit(p_i, n_data[i], dx[i], ccd_i=ccd_i)
                return LL

            ### Minimizing likelihood
            x0_dict = {}
            for ccd_i in range(self.nccd):
                for i in range(len(x0)):
                    x0_dict[pars_name_dict[i]+str(ccd_i)] = x0[i]
            x0_dict["sigma"] = x0[-1]
            #m = iminuit.Minuit(log_like_minuit,**x0_dict)
            m = iminuit.Minuit(log_like_amps,**x0_dict)
            m.errors[x0_dict.keys()] = 1e-5
            m.errordef = m.LIKELIHOOD
            m.strategy=0
            m.limits[x0_dict.keys()] = pars_lims
            for ccd_i in range(self.nccd):
                for i in fix_pars:
                    if i != 2 or i != 1:
                        m.fixed[int(ccd_i*len(pars_name_dict)+pars_name_dict[i])] = True
            ### Cross section
            if len(x0)-1 in fix_pars:
                m.fixed[len(x0)-1] = True
            m.migrad()
            theta_max = m.values[x0_dict.keys()]
            #print(theta_max)
            #if verbose:
            #    m.draw_contour("lamb","sigma")
            ### Returns fitted+fixed parameters
            #pa_[not_fix_pars] = theta_max
            #theta_max = pa_
        else:
            ### Not fixed parameters
            theta_max = minimize(log_like, x0, bounds=pars_lims,method='Powell', args = (n_data,dx)).x

        fit_leg = []
        for ccd_i in range(self.nccd):
            for p, name in zip(theta_max[int(ccd_i*len(pars_name)):int(len(pars_name)+ccd_i*len(pars_name))], pars_name):
                fit_leg.append(name+str(ccd_i)+" = "+str(p))
        fit_leg.append(r"\sigma_DM = " +str(theta_max[-1]) )
        textstr = "\n".join(fit_leg)
        
        #print(textstr) 
        #print("----------------------------------")


        if verbose:
            ### Plot fit and histogram
            for i in range(self.nccd):
                plt.clf()
                fig, ax = plt.subplots()
                nbins = int((self.xmax-self.xmin)/bin_size)
                ax.hist(self.events[i], bins=nbins)#, density=True)
                n_data_, dx_ = hist[0],hist[1]

                dx_m = (dx_[:-1] + dx_[1:])/2
                dx_m = dx_[1:]#.flip(dx)))
                n_theo_ = prob(theta_max,dx_m)*np.diff(dx_)
                ax.plot(dx_m, n_theo_, color = "r", label = "Best Fit")
                ax.set_yscale("log")
                ax.set_ylim(0.1,None)
                plt.legend(loc="best")

                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                # place a text box in upper left in axes coords
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                plt.title("CCD {}".format(i))
                plt.show()
                plt.clf()


        ### Calculates the upper limit
        if upper_limit:
            ### Value of the likelihood at its maximum
            lnL_max = -log_like_amps(theta_max)#, n_data, dx)

            def lnL_dLL(xs_range, n_data, dx):
                """ Function to find the value where ln_max- ln_confidence = deltaLL
                """
                theta = np.array(theta_max[:-1]).tolist() + [xs_range]
                return -log_like_amps(theta)-lnL_max+deltaLL

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
            #self.cross_section_dLL = 10**bisect(lnL_dLL, pars_lims[-1][0], pars_lims[-1][1], args=(n_data,dx),rtol=1e-3)
            #try:
            upper = bisect(lnL_dLL, a=pars_lims[-1][0], b=pars_lims[-1][1], args=(n_data,dx),rtol=1e-3)

            #print("Cross 90CF: ", upper)
            self.cross_section_dLL = 10**upper #, full_output=True, disp=True)
            #except:
            #print("The minimum is not well determined")

        """
        def log_prior(theta, *bounds):
            non_inf = 0
            #for p,bounds in zip(theta,p_lims):
            for i in range(len(theta)):
                if bounds[i][0] < theta[i] < bounds[i][1]:
                    non_inf += 0.0
                else:
                    #print(theta[i], bounds[i])
                    return np.inf      
            if len(fix_pars) != 0:
                return -log_like_fix(theta, n_data, dx)
            else:
                return -log_like(theta, n_data, dx)
 
        if len(fix_pars) != 0:
            ### Fix the given parameter numbers
            theta_max = np.array(theta_max)[not_fix_pars].tolist()
            labels = np.array(pars_name)[not_fix_pars].tolist()
            pos = theta_max + 1e-4 * np.random.randn(32, len(not_fix_pars))
        else:
            labels = pars_name
            pos = theta_max + 1e-4 * np.random.randn(32, len(theta_max))

        ### emcee calculations
        nwalkers, ndim = pos.shape

        if len(fix_pars) != 0:
            print(pars_lims_nofix)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prior, args=(pars_lims_nofix.tolist()))
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prior, args=(pars_lims))
        sampler.run_mcmc(pos, 5000, progress=True)

        samples = sampler.get_chain()

        tau = sampler.get_autocorr_time() 
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        inds = np.random.randint(len(flat_samples), size=100)

        if verbose:
            fig = corner.corner(flat_samples, labels=labels, truths=theta_max)

        theta_err = []
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [5, 50, 90])
            q = np.diff(mcmc)
            #txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            #txt = txt.format(mcmc[1], q[0], q[1], "\sigma")
            #display(Math(txt))
            if pars_name[i] == r"$\sigma_{DM-e}$":
                print("90th percentil: ", np.power(10,mcmc[1]+q[1]))
                print("Central vlue: ", np.power(10,mcmc[1]))
                print("5th percentile: ", np.power(10,mcmc[1]-q[0]))
                theta_err.append(np.power(10,[mcmc[1]-q[0],mcmc[1],mcmc[1]+q[1]]).tolist())
            else:
                print("90th percentil: ", mcmc[1]+q[1])
                print("Central vlue: ", mcmc[1])
                print("5th percentile: ", mcmc[1]-q[0])
        """


    def import_events(self, filename):
        self.events, self.texp, self.npix, self.texp, self.mass_det = [],[],[],[],[]
        for f in filename:
            df = pd.read_csv(filename, header=None)
            self.events.append( df[0] )
            self.npix.append( len(self.events) )
            self.texp.append( 22583.2/86400) #days
            self.mass_det.append( self.npix*3.507e-9 ) #kg
        self.events, self.texp, self.npix, self.texp, self.mass_det = np.array(self.events), np.array(self.texp), np.array(self.npix), np.array(self.texp), np.array(self.mass_det)
            print("{} g*days".format(self.mass_det*self.texp*1000))


    def infer_diffusion_model_file(self, diffusion_dir, diffusion_files):

        df_diffusion = pd.read_csv(diffusion_dir + "dR_list_MeV.csv")
        print(self.mass_dm, df_diffusion["mx"])
        filename = diffusion_files + df_diffusion[df_diffusion["mx"]==self.mass_dm]["name"].values[0].split("/")[-1] + ".pkl"
        print(filename)
        self.cross_section_ref = df_diffusion[df_diffusion["mx"]==self.mass_dm]["xs"].values[0]
        print(self.cross_section_ref)


        ### Calculates number of Reference expected events
        self.xs_0 = self.cross_section
        self.cross_section = self.cross_section_ref
        self.normalization_signal()
        self.Nref = self.s

        ### Back to original cross section
        self.cross_section = self.xs_0
        self.normalization_signal()

        self.import_diffusion_model(filename)


    def import_diffusion_model(self, filename):
        pkl_dict = pickle.load(open(filename,"rb"))
        self.diffusion = True
        self.fs, self.ne = [],[]
        print(pkl_dict)
        #for key in pkl_dict:
        #    self.fs.append( pkl_dict[key])
        #    self.ne.append(int(key))
        #self.fs = np.array(self.fs)
        self.fs = pkl_dict["fs"]
        self.ne = pkl_dict["ne"].astype(int)
 
        ### Calculates the cross section simulated for te simulated number of events
        self.Nsim = pkl_dict["Nev"]
        self.cross_section_sim = self.cross_section_ref*self.Nsim*self.t_exp*self.mass_det/self.Nref
        print(self.cross_section_ref,self.cross_section_sim)
        print(self.ne)

        self.normalization_signal_diffusion()
 
    def normalization_signal_diffusion(self):
        """ Calculates the normalization of the signal PDF.
        """

        signal_strength = self.cross_section/self.cross_section_sim
        self.s = signal_strength*self.Nsim*self.t_exp*self.mass_det
        self.dRdne = np.array(self.fs)*self.s
        #plt.plot(self.ne,self.dRdne,'o')
        #plt.yscale("log")
        #plt.show()
        #self.n_s_det = np.random.poisson(self.s)


    def verbose(self):
        print("------------ Using {} model -----------".format(self.dRdE_name))
        print("Exposure [g*days]: ", self.mass_det*1e3*self.t_exp)
        print("Number of Total Pixels: ", self.npix)
        print("Number of Signal events: ", self.n_s_det)
        if hasattr(self, "theta"): print("Likelihood parameters: ", self.theta_max)
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

