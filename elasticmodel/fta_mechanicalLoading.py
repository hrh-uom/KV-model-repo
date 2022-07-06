import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
import pandas as pd
import sympy as sp
import glob
import os
import csv
from scipy.optimize import curve_fit
from scipy.integrate import simps
plt.style.use('~/dbox/4-Thesis/stylesheet.mplstyle')

#----------------------------------------------------------------------------
#........................... ELASTIC LOADING...............................
#------------------------------------------------------------------------
atom=False
datanames  =       ['9am-1R', '9am-2L', '9am-4R'] ; datanames_no_hyphen = ['9am1R', '9am2L', '9am4R']
directories=    [   f'/Users/user/dbox/2-mechanics-model/MODEL/FTA/csf-output/{i}/final/*/' for i in datanames  ]
dir_output=         '/Users/user/dbox/2-mechanics-model/MODEL/FTA/csf-output/mechanical-all/'

def remove_nan_convert_numpy(dataframe):
    """
    convert a dataframe with nan values to a np array without nan values
    """
    return dataframe[~np.isnan(dataframe.to_numpy())].to_numpy()

def create_FTA_results_dataframe():
    def load_FTA_data(DSN):
        """
        Loads data from fibril tracking. If path is set above this should work automatically, no need to adjust file names
        """
        d=directories[DSN] ; err=0
        try:
            e_c=np.load(glob.glob(d+f'stats/critical*npy')[0])-1
        except:
            print(f'{datanames[DSN]} No critical strain file found'); err=1
        try:
            scaledlengths=np.load(glob.glob(d+f'stats/scaled*npy')[0]) #um
        except:
            print(f'{datanames[DSN]} No scaled lengths file found'); err=1
        try:
            MFDs=np.load(glob.glob(d+'mfds*')[0]) #nm
        except:
            print(f'{datanames[DSN]} No MFD file found') ; err=1
        try:
            areas=np.load(glob.glob(d+f'area*')[0]) #nm^2
        except:
            print( f'{datanames[DSN]} No areas file found') ; err=1
        try:
            VF=pd.read_csv(glob.glob(d+f'stats/VF.csv')[0]).VF_raw.to_numpy()
        except:
            print (f'{datanames[DSN]} No  volume Fraction found'); err=1
        if err==0:
            return  e_c,scaledlengths, MFDs, areas,VF
    d0=pd.DataFrame(load_FTA_data(0)).T
    d1=pd.DataFrame(load_FTA_data(1)).T
    d2=pd.DataFrame(load_FTA_data(2)).T
    d=pd.concat([d0, d1, d2], axis=1).set_axis(['ec0', 'sl0','mfd0', 'a0','VF0', 'ec1', 'sl1','mfd1', 'a1','VF1','ec2', 'sl2','mfd2', 'a2' , 'VF2' ,], axis=1)
    return d

d=create_FTA_results_dataframe()
#%% Plot FTA data

def plot_3_mfds():
    fig, ax = plt.subplots(1,3 , figsize=(20, 4))

    n, bins, patches = ax[0].hist(d.mfd0, 50, density=False, facecolor='g', alpha=0.75)
    n, bins, patches = ax[1].hist(d.mfd1, 50, density=False, facecolor='g', alpha=0.75)
    n, bins, patches = ax[2].hist(d.mfd2, 50, density=False, facecolor='g', alpha=0.75)

    [ax[i].set_title(datanames[i]) for i in range (3)]
    [ax[i].set_xlabel('MFD') for i in range(3)]
    [ax[i].set_ylabel('Number') for i in range (3)]
    fig.tight_layout()
    if atom:
        plt.show()
# plot_3_mfds()
#%%------------------Criticalstrain
def plot_3_csd():
    fig, ax = plt.subplots(1,3 , figsize=(24, 5))
    for i in range(3):
        ec=remove_nan_convert_numpy(d[f'ec{i}'])
        n, bins, patches = ax[i].hist(100*d[f'ec{i}'], 30, density=True, weights=d[f'a{i}'], histtype='bar', edgecolor='black')

    [ax[i].set_xlabel('Area weighted critical strain (%)') for i in range(3)]
    [ax[0].set_ylabel('Probability density') for i in range (3)]

    [ax[i].set_title(datanames[i]) for i in range (3)]
    if atom:
        plt.show()
# plot_3_csd()
def plot_3_lens():

    fig, ax = plt.subplots(1,3 , figsize=(20, 5))

    n, bins, patches = ax[0].hist(d.sl0, 50, density=True, facecolor='g', alpha=0.75)
    n, bins, patches = ax[1].hist(d.sl1, 50, density=True, facecolor='g', alpha=0.75)
    n, bins, patches = ax[2].hist(d.sl2, 50, density=True, facecolor='g', alpha=0.75)

    [ax[i].set_xlabel('Scaled Lengths(um)') for i in range(3)]
    [ax[0].set_ylabel('Probability density') for i in range (3)]
    [ax[i].set_title(datanames[i]) for i in range (3)]
    if atom:
        plt.show()
# plot_3_lens()

#%%--------------------------- STRAIN CONDITIONS----------------------
max_strain=6 /100 #Percent
c1, c2, c3=max_strain/2, -max_strain/2, np.pi/(60*2) ; T=2*np.pi/c3 #Period
elin=np.linspace(0.0,max_strain,1001)  #global strain values
tlin=np.linspace(0, 15*T, 1001)

def e_t(c1, c2, c3, t):
    """
    Oscillating strain function, t is given in minutes
    """
    return  c1 + c2 * np.cos(c3 * t )
def de_dt(c1, c2, c3, t):
    """
    Derivative of strain function, t is given in minutes
    """
    return  -c2 * c3*np.sin(c3 * t)
def plot_strain_conditions():
    fig, ax2=plt.subplots()
    ax2.plot(tlin/60,e_t(c1, c2, c3,tlin)*100)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Strain (%)")
    plt.savefig(dir_output+'straintime')
    if atom:
        plt.show()
plot_strain_conditions()

#%%-------------ELASTIC RESPONSE---------------------------------------
def plot_elastic_response():
    """
    E= Youngs / Elastic modulus
    """
    def calculate_fascicle_stress_with_ec(e_c, areas, VF, E=350):

        """
        Calculates the fibril stress given
        elin= strain range
        e_c = critical strain of fibrils
        areas = areas of fibrils
        E=youngs modulus  = 350 #MPa VanDerRijt2006
        """
        VF=remove_nan_convert_numpy(VF)[0]
        e_c=remove_nan_convert_numpy(e_c)
        areas=remove_nan_convert_numpy(areas)
        nfibs=e_c.size

        def F_fib(i):
            """
            Returns the force in a particular fibril  i at each value of global strain elin
            Units are nm^2 * MPa = (10^9)^2 m^2 x 10^6 Pa = 10^-12 N = pN
            """
            f_i=E*areas[i]*(elin-e_c[i])/(1+e_c[i]) #WHATARETHEUNITS
            f_i=f_i*(f_i>0).astype('float') #Values below zero are when fibrils not contibuting
            return f_i

        F=np.zeros(elin.shape) #Force in whole fascicle
        for i in range (nfibs): #Sums over the individual fibrils i
            F+=F_fib(i)
        return VF*F/np.sum(areas)
    sig_s0=calculate_fascicle_stress_with_ec( d.ec0,d.a0, d.VF0)
    sig_s1=calculate_fascicle_stress_with_ec( d.ec1,d.a1, d.VF1)
    sig_s2=calculate_fascicle_stress_with_ec( d.ec2,d.a2, d.VF2)
    mean  = np.mean([sig_s0, sig_s1, sig_s2], axis=0)
    sd =np.std([sig_s0, sig_s1, sig_s2], axis=0)

    fig, ax3=plt.subplots()
    ax3.plot(100*elin,sig_s0, '-r', label=datanames[0])
    ax3.plot(100*elin,sig_s1, '-b', label=datanames[1])
    ax3.plot(100*elin,sig_s2, '-g', label=datanames[2])
    ax3.plot(100*elin,mean, '--k', label='mean')
    ax3.fill_between(100*elin, mean-sd, mean+sd, label = 'standard deviation', color='k', alpha=0.2)

    ax3.set_xlabel('Tensile strain (%)') ; ax3.set_ylabel('Tensile stress (MPa)')
    ax3.legend() ; plt.savefig(dir_output+'elastic-response-E350MPa');
    if atom:
        plt.show()
    elas_model_results_E350MPa_df=pd.DataFrame(np.array([elin, mean, sd, sig_s0, sig_s1, sig_s2]).T, columns=['strain_nodims', 'mean_stress_all_MPa', 'sd_MPa', f'stress_{datanames_no_hyphen[0]}_MPa', f'stress_{datanames_no_hyphen[1]}_MPa', f'stress_{datanames_no_hyphen[2]}_MPa'])
    elas_model_results_E350MPa_df.to_csv(dir_output+'elasticresponsevalues_E_350MPa.csv')
    return elas_model_results_E350MPa_df
elas_model_results_E350MPa_df=plot_elastic_response()
#%%----------------------IMPORT EXPERIMENTAL RESULTS

#EXPERIMENTAL RESULTS
expres_folder='~/dbox/2-mechanics-model/EXPERIMENT/plots-instron-results/mechanics-paper-results/'
expRes_df=pd.read_csv(expres_folder+'mean_loading_stress_all.csv', index_col=0)
expMeta_df=pd.read_csv(expres_folder+'eloss-mean-values-5-10.csv', index_col=0)

#%%----------------------Plotting experimental mean curves besides recruitment model prediction
print('Plotting experimental mean curves besides recruitment model prediction ')

labels_=['AT 1' ,'AT 2', 'AT 3']
indices= [expMeta_df.loc[(expMeta_df['Time'] == '9am') & (expMeta_df['Mouse'] == 1) & (expMeta_df['L/R'] == 'R')].index[0],
            expMeta_df.loc[(expMeta_df['Time'] == '9am') & (expMeta_df['Mouse'] == 2) & (expMeta_df['L/R'] == 'L')].index[0],
            expMeta_df.loc[(expMeta_df['Time'] == '9am') & (expMeta_df['Mouse'] == 4) & (expMeta_df['L/R'] == 'R')].index[0]] #Which rows are they in the dataframe

def plot_experiment_beside_model():
    # ==========EXPERIMENT=========
    fig1,ax1=plt.subplots()
    for i in range(3):
        exp_stress=np.mean(expRes_df[[f'mean_stress_L_{indices[i]}',f'mean_stress_UL_{indices[i]}']], axis=1)
        ax1.plot(expRes_df.meanStrain, exp_stress, label=labels_[i])
    units = '(MPa)'

    ax1.set_ylabel(f'Stress {units}') ; ax1.set_xlabel('Strain (%)') ; ax1.set_title('Experiment')
    ax1.legend()
    plt.savefig(dir_output+'/experiment_mean_loadunlod')

    if atom:
        plt.show()

    # ==========MODEL=========
    # fig2, ax2=plt.subplots()
    fig2,ax2=plt.subplots()
    for i in range(3):
        ax2.plot(100*elas_model_results_E350MPa_df.strain_nodims, elas_model_results_E350MPa_df[f'stress_{datanames_no_hyphen[i]}_MPa'], label=labels_[i])
    ax2.set_ylabel(f'Stress {units}') ; ax2.set_xlabel('Strain (%)'); ax2.set_title('Model')
    ax2.legend()
    plt.savefig(dir_output+'/recruitment_model_E350');
    if atom:
        plt.show()

plot_experiment_beside_model()

#%% FITTING EXPERIMENTAL DATA WITH L and E as free parameters
print('Fitting mean elastic experimental data using E and fas len as a free  parameters')

def fit_recruitment_model_to_exp():
    def calculate_fas_stress_with_L_for_fitting(elin, fas_len, E):
        F=np.zeros(elin.shape) #Force in whole fascicle
        e_c=lens/fas_len -1
        for i in range (nfibs): #Sums over the individual fibrils i
            f_i=E*areas[i]*(elin-e_c[i])/(1+e_c[i]) #WHATARETHEUNITS
            f_i=f_i*(f_i>0).astype('float') #Values below zero are when fibrils not contibuting
            F+=f_i
        return VF*F/np.sum(areas)
    fig, ax=plt.subplots()
    expstrain=expRes_df.meanStrain
    fitparams=np.zeros((3, 2))

    for DSN in range(3):
        print(datanames[DSN])
        lens=remove_nan_convert_numpy(d[f'sl{DSN}'])
        areas=remove_nan_convert_numpy(d[f'a{DSN}'])
        VF=remove_nan_convert_numpy(d[f'VF{DSN}'])[0]
        nfibs=lens.size
        expstress=np.mean(expRes_df[[f'mean_stress_L_{indices[DSN]}',f'mean_stress_UL_{indices[DSN]}']], axis=1)

        #Fit to experimental data
        pars, cov=curve_fit(calculate_fas_stress_with_L_for_fitting, expstrain/100, expstress, bounds=([40, 1], [200,1000]))
        fit=[elin, calculate_fas_stress_with_L_for_fitting(elin, pars[0], pars[1])]
        calculate_fas_stress_with_L_for_fitting(elin, pars[0], pars[1])
        ax.set_xlabel('Strain (%)') ;ax.set_ylabel('Stress (MPa)')
        ax.plot(expstrain, expstress, color=plt.get_cmap("tab10")(DSN), label=['AT 1', 'AT 2', 'AT 3'][DSN]+ ' Experiment')
        ax.plot(fit[0]*100, fit[1], '--', color=plt.get_cmap("tab20")(1+2*DSN), label=f' Model')
        fitparams[DSN]=pars

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = [labels[0], labels[2], labels[4], labels[1], labels[3], labels[5]], [handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]]
    ax.legend(handles, labels,ncol=2)
    plt.savefig(dir_output+'recruitmentmodelfit_E_L')
    if atom:
        plt.show()
    dd=pd.DataFrame(fitparams, columns=['FasLen_um', 'YM_Mpa'])
    dd['Name']=datanames
    dd.to_csv(dir_output+'fittingparameters.csv')
    return dd

fitting_df=fit_recruitment_model_to_exp()


#%%# FITTING EXPERIMENTAL DATA WITH ONLY L  as free parameters
fitting_df=pd.read_csv(dir_output+'fittingparameters.csv')
print ('fitting parameters')
print(fitting_df)

def fit_recruitment_model_to_exp_only_FL():
    print(f'Fitting mean elastic experimental data again using fas len as a free  parameters and a mean YM {np.mean(fitting_df.YM_Mpa)} MPa')

    def calculate_fas_stress_with_L_E(elin, fas_len):
        E=np.mean(fitting_df.YM_Mpa)
        F=np.zeros(elin.shape) #Force in whole fascicle
        e_c=lens/fas_len -1
        for i in range (nfibs): #Sums over the individual fibrils i
            f_i=E*areas[i]*(elin-e_c[i])/(1+e_c[i]) #WHATARETHEUNITS
            f_i=f_i*(f_i>0).astype('float') #Values below zero are when fibrils not contibuting
            F+=f_i
        return VF*F/np.sum(areas)
    fig, ax=plt.subplots()
    expstrain=expRes_df.meanStrain
    fitparams=np.zeros(3)

    for DSN in range(3):
        print(datanames[DSN])
        lens=remove_nan_convert_numpy(d[f'sl{DSN}'])
        areas=remove_nan_convert_numpy(d[f'a{DSN}'])
        VF=remove_nan_convert_numpy(d[f'VF{DSN}'])[0]
        nfibs=lens.size
        expstress=np.mean(expRes_df[[f'mean_stress_L_{indices[DSN]}',f'mean_stress_UL_{indices[DSN]}']], axis=1)

        #Fit to experimental data
        pars, cov=curve_fit(calculate_fas_stress_with_L_E, expstrain/100, expstress, bounds=(20, 100))
        fit=[elin, calculate_fas_stress_with_L_E(elin, pars[0])]
        ax.set_xlabel('Strain (%)') ;ax.set_ylabel('Stress (MPa)')
        ax.plot(expstrain, expstress, color=plt.get_cmap("tab10")(DSN), label=['AT 1', 'AT 2', 'AT 3'][DSN]+ ' Experimental mean')
        ax.plot(fit[0]*100, fit[1], '--', color=plt.get_cmap("tab20")(1+2*DSN), label=['AT 1', 'AT 2', 'AT 3'][DSN]+' Model Fit 2')
        fitparams[DSN]=pars

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = [labels[0], labels[2], labels[4], labels[1], labels[3], labels[5]], [handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]]
    ax.legend(handles, labels,ncol=2)
    plt.savefig(dir_output+f'recruitmentmodelfit_L.png')
    if atom:
        plt.show()
    fitting_df['FasLen_second_fitting_using_mean_YM_um']=fitparams
    fitting_df.to_csv(dir_output+'fittingparameters.csv')

# fit_recruitment_model_to_exp_only_FL()

#%%-------------------------HYSTERESIS /VISSCOELAASTIC CURVE--------------
 # Fitting experimental hysteresis curves to model using viscosity as a free parameter
def find_timepoints_of_elin_in_et():
    t=sp.symbols('t')
    def e_t_scipy(c1, c2, c3, t):
        """
        Oscillating strain function, t is given in minutes
        """
        return  c1 + c2 * sp.cos(c3 * t )
    timepoints=[solve(strain - e_t_scipy(c1, c2, c3, t), t) for strain in elin]

    def find_timepoints_when_singular(timepoints):
        pad = len(max(timepoints, key=len))
        timepoints=np.array([i + [0]*(pad-len(i)) for i in timepoints])
        meetwhere=np.argwhere(timepoints[:,1]==0)[0,0]
        timepoints[meetwhere,1]=timepoints[meetwhere,0]
        return timepoints

    if np.any(np.array([len(xi) for xi in timepoints])==1): #Singular
        timepoints=find_timepoints_when_singular(timepoints) #timepoints[:,0] = load and timepoints[:, 1]=unload
    return np.array(timepoints).astype('float')
timepoints=find_timepoints_of_elin_in_et()
#%%
def fit_hys_curves_to_predict_viscosity():
    print('Fitting experimental hysteresis curves to model using viscosity as a free parameter')

    def calculate_viscoelastic_stress_fitting(elin, mu=1e-3):
        # print(f"DSN {DSN}, loading {loading}")
        def calculate_sig_fibrils():
            """
            Returns stress in MPa
            """

            areas=remove_nan_convert_numpy(d[f'a{DSN}'])#In nm**2 but you dont need to change units as they cancel
            lens=remove_nan_convert_numpy(d[f'sl{DSN}']) #In um but you dont need to change units as they cancel
            fas_len=fitting_df.FasLen_um.iloc[DSN] #In um but you dont need to change units as they cancel
            E=fitting_df.YM_Mpa.iloc[DSN] #As these values are in MPa
            F=np.zeros(elin.shape) #Force in whole fascicle
            e_c=lens/fas_len -1
            for i in range (areas.size): #Sums over the individual fibrils i
                f_i=E*areas[i]*(elin-e_c[i])/(1+e_c[i]) #units = YM MPa x Nm^2 = 10^-12 N
                f_i=f_i*(f_i>0).astype('float') #Values below zero are when fibrils not contibuting
                F+=f_i
            return np.array(F/np.sum(areas))


        def calclulate_sig_fluid(mu):
            """
            Returns in MPa assuming p.alpha is in Mpa
            """

            def find_nu(p_dot_alpha=22536529939087, A_f=38965893.12, mu_=1.0016e-3):
                """
                The constant term to nultiply the strain rate function
                by in order to get the viscous stress contribution
                Pulled from fluid model
                used like,
                mu=viscosity in Pa * s
                sig_f = nu * d(epsilon)/dt
                """
                return (mu / A_f)*(p_dot_alpha + 2 *A_f)
            # nu=543.974 #MPA OUTPUT FROM BENS ANALYSIS
            nu=find_nu(mu_=mu)
            # sig_f=np.array([nu*sp.N(deps_dt(c2, c3, c4, tt) )for tt in timepoints[:,which_half]]) #fluid stress
            sig_f_load=np.array(nu*de_dt(c1, c2, c3, timepoints[:, 0]))
            sig_f_unload=np.array(nu*de_dt(c1, c2, c3, timepoints[:, 1]))
            return sig_f_load, sig_f_unload

        VF=remove_nan_convert_numpy(d[f'VF{DSN}'])[0]
        sig_s=calculate_sig_fibrils()
        sig_f_load, sig_f_unload=calclulate_sig_fluid(mu)
        return np.concatenate(((VF*sig_s+(1-VF)*sig_f_load), (VF*sig_s+(1-VF)*sig_f_unload)))
        # return sig_f
    fitparams=np.zeros(3) #3 datasets
    for DSN in range(3):
        print(f"fitting hysteresis {datanames[DSN]}")
        expstrain=np.array(expRes_df.meanStrain)/100
        expstressL=np.array(expRes_df[f'mean_stress_L_{indices[DSN]}'])
        expstressUL=np.array(expRes_df[f'mean_stress_UL_{indices[DSN]}'])
        expstress=np.concatenate((expstressL, expstressUL))
        pars, cov=curve_fit(calculate_viscoelastic_stress_fitting, expstrain, expstress, bounds=(1e-6, 1e-2)) ; fitparams[DSN]=pars
        pd.DataFrame(fitparams).to_csv(dir_output+'viscosity.csv')

fit_hys_curves_to_predict_viscosity()

viscosity=pd.read_csv(dir_output+'viscosity.csv', index_col=0).to_numpy()
def calculate_viscoelastic_stress(DSN, mu):
    """
    returns ve stress on either loading or unloading curve in MPA
    given a certain dataset number
    Given a viscosity

    """
    def calculate_sig_fibrils():
        """
        Returns stress in MPa
        """

        areas=remove_nan_convert_numpy(d[f'a{DSN}'])#In nm**2 but you dont need to change units as they cancel
        lens=remove_nan_convert_numpy(d[f'sl{DSN}']) #In um but you dont need to change units as they cancel
        fas_len=fitting_df.FasLen_um.iloc[DSN] #In um but you dont need to change units as they cancel
        E=fitting_df.YM_Mpa.iloc[DSN] #As these values are in MPa
        F=np.zeros(elin.shape) #Force in whole fascicle
        e_c=lens/fas_len -1
        for i in range (areas.size): #Sums over the individual fibrils i
            f_i=E*areas[i]*(elin-e_c[i])/(1+e_c[i]) #units = YM MPa x Nm^2 = 10^-12 N
            f_i=f_i*(f_i>0).astype('float') #Values below zero are when fibrils not contibuting
            F+=f_i
        return F/np.sum(areas) #Units of MPa as areas cancel
    def calclulate_sig_fluid(mu):
        """
        Returns in MPa procided p.alpha is in MPa
        """

        def find_nu(p_dot_alpha=22536529939087, A_f=38965893.12, mu_=1.0016e-3):
            """
            The constant term to nultiply the strain rate function
            by in order to get the viscous stress contribution
            Pulled from fluid model
            used like,
            mu=viscosity in Pa * s
            sig_f = nu * d(epsilon)/dt
            """
            return (mu / A_f)*(p_dot_alpha + 2 *A_f)
        # nu=543.974 #MPA OUTPUT FROM BENS ANALYSIS
        nu=find_nu(mu_=mu)
        sig_f_load=nu*de_dt(c1, c2, c3, timepoints[:, 0])
        sig_f_unload=nu*de_dt(c1, c2, c3, timepoints[:, 1])
        return sig_f_load, sig_f_unload

    VF=remove_nan_convert_numpy(d[f'VF{DSN}'])[0]
    sig_s=calculate_sig_fibrils()
    sig_f_load, sig_f_unload=calclulate_sig_fluid(mu)
    return ((VF*sig_s+(1-VF)*sig_f_load), (VF*sig_s+(1-VF)*sig_f_unload))

def plot_hys_curves_with_predicted_viscosities():
    fig, ax=plt.subplots(dpi=100)
    for DSN in range(3):
        expstrain=expRes_df.meanStrain
        expstressL=np.array(expRes_df[f'mean_stress_L_{indices[DSN]}'])
        expstressUL=np.array(expRes_df[f'mean_stress_UL_{indices[DSN]}'])
        modelstress=calculate_viscoelastic_stress(DSN, viscosity[DSN])
        modelstress
        ax.set_xlabel('Strain (%)') ;ax.set_ylabel('Stress (MPa)')
        ax.plot(expstrain, expstressL, color=plt.get_cmap("tab10")(DSN), label=['AT 1', 'AT 2', 'AT 3'][DSN]+ ' Experiment')
        ax.plot(expstrain, expstressUL, color=plt.get_cmap("tab10")(DSN))
        ax.plot(elin*100, modelstress[0],'--',color=plt.get_cmap("tab20")(1+2*DSN), label=' Model')
        ax.plot(elin*100, modelstress[1],'--',color=plt.get_cmap("tab20")(1+2*DSN))


    handles, labels = ax.get_legend_handles_labels()
    labels, handles = [labels[0], labels[2], labels[4], labels[1], labels[3], labels[5]], [handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]]
    ax.legend(handles, labels,ncol=2)
    plt.savefig(dir_output+f'KV-model-prediction-individual-viscosities.png')
plot_hys_curves_with_predicted_viscosities()



#%%
def OLD_fit_hys_curves_to_predict_viscosity():

    def calculate_viscoelastic_stress_fitting(elin, mu=1e-3):
        # print(f"DSN {DSN}, loading {loading}")
        def calculate_sig_fibrils():
            """
            Returns stress in MPa
            """

            areas=remove_nan_convert_numpy(d[f'a{DSN}'])#In nm**2 but you dont need to change units as they cancel
            lens=remove_nan_convert_numpy(d[f'sl{DSN}']) #In um but you dont need to change units as they cancel
            fas_len=fitting_df.FasLen_um.iloc[DSN] #In um but you dont need to change units as they cancel
            E=fitting_df.YM_Mpa.iloc[DSN] #As these values are in MPa
            F=np.zeros(elin.shape) #Force in whole fascicle
            e_c=lens/fas_len -1
            for i in range (areas.size): #Sums over the individual fibrils i
                f_i=E*areas[i]*(elin-e_c[i])/(1+e_c[i]) #units = YM MPa x Nm^2 = 10^-12 N
                f_i=f_i*(f_i>0).astype('float') #Values below zero are when fibrils not contibuting
                F+=f_i
            return F/np.sum(areas)


        def calclulate_sig_fluid(mu):
            """
            Returns in MPa assuming p.alpha is in Mpa
            """

            def find_nu(p_dot_alpha=22536529939087, A_f=38965893.12, mu_=1.0016e-3):
                """
                The constant term to nultiply the strain rate function
                by in order to get the viscous stress contribution
                Pulled from fluid model
                used like,
                mu=viscosity in Pa * s
                sig_f = nu * d(epsilon)/dt
                """
                return (mu / A_f)*(p_dot_alpha + 2 *A_f)
            # nu=543.974 #MPA OUTPUT FROM BENS ANALYSIS
            nu=find_nu(mu_=mu)
            which_half=0 if loading else 1
            # sig_f=np.array([nu*sp.N(deps_dt(c2, c3, c4, tt) )for tt in timepoints[:,which_half]]) #fluid stress
            sig_f=nu*de_dt(c1, c2, c3, timepoints[:, which_half])
            return sig_f

        VF=remove_nan_convert_numpy(d[f'VF{DSN}'])[0]
        sig_s=calculate_sig_fibrils()
        sig_f=calclulate_sig_fluid(mu)
        return (VF*sig_s+(1-VF)*sig_f)
        # return sig_f

    fitparams=np.zeros((3, 2)) #3 datasets, one loading one unloading

    for DSN in range(0):

        print(f"fitting hysteresis {datanames[DSN]}")
        expstrain=expRes_df.meanStrain
        expstressL=expRes_df[f'mean_stress_L_{indices[DSN]}']
        expstressUL=expRes_df[f'mean_stress_UL_{indices[DSN]}']
        loading = True ; print(f'DSN {DSN} loading {loading} ')
        pars, cov=curve_fit(calculate_viscoelastic_stress_fitting, expstrain/100, expstressL, bounds=(1e-6, 1e-3)) ; fitparams[DSN, 0]=pars
        loading = False ; print(f'DSN {DSN} loading {loading} ')
        pars, cov=curve_fit(calculate_viscoelastic_stress_fitting, expstrain/100, expstressUL, bounds=(1e-6, 1e-3)) ; fitparams[DSN, 1]=pars
        pd.DataFrame(fitparams).to_csv(dir_output+'viscosity.csv')

#%% Plotting hysteresis curves with individual viscosities


def OLD_calculate_viscoelastic_stress(DSN, loading, mu):

    """
    returns ve stress on either loading or unloading curve in MPA
    given a certain dataset number
    Given a viscosity

    """

    # print(f"DSN {DSN}, loading {loading}")
    def calculate_sig_fibrils():
        """
        Returns stress in MPa
        """

        areas=remove_nan_convert_numpy(d[f'a{DSN}'])#In nm**2 but you dont need to change units as they cancel
        lens=remove_nan_convert_numpy(d[f'sl{DSN}']) #In um but you dont need to change units as they cancel
        fas_len=fitting_df.FasLen_um.iloc[DSN] #In um but you dont need to change units as they cancel
        E=fitting_df.YM_Mpa.iloc[DSN] #As these values are in MPa
        F=np.zeros(elin.shape) #Force in whole fascicle
        e_c=lens/fas_len -1
        for i in range (areas.size): #Sums over the individual fibrils i
            f_i=E*areas[i]*(elin-e_c[i])/(1+e_c[i]) #units = YM MPa x Nm^2 = 10^-12 N
            f_i=f_i*(f_i>0).astype('float') #Values below zero are when fibrils not contibuting
            F+=f_i
        return F/np.sum(areas) #Units of MPa as areas cancel
    def calclulate_sig_fluid(mu):
        """
        Returns in MPa procided p.alpha is in MPa
        """

        def find_nu(p_dot_alpha=22536529939087, A_f=38965893.12, mu_=1.0016e-3):
            """
            The constant term to nultiply the strain rate function
            by in order to get the viscous stress contribution
            Pulled from fluid model
            used like,
            mu=viscosity in Pa * s
            sig_f = nu * d(epsilon)/dt
            """
            return (mu / A_f)*(p_dot_alpha + 2 *A_f)
        # nu=543.974 #MPA OUTPUT FROM BENS ANALYSIS
        nu=find_nu(mu_=mu)
        which_half=0 if loading else 1
        sig_f=nu*de_dt(c1, c2, c3, timepoints[:, which_half])
        return sig_f

    VF=remove_nan_convert_numpy(d[f'VF{DSN}'])[0]
    sig_s=calculate_sig_fibrils()
    sig_f=calclulate_sig_fluid(mu)
    return (VF*sig_s+(1-VF)*sig_f)
viscosity=pd.read_csv(dir_output+'viscosity.csv', index_col=0).to_numpy()


def OLD_plot_hys_curves_with_predicted_viscosities():
    print('Plotting hysteresis curves with KV model using individual viscosities')

    fig, ax=plt.subplots()
    for DSN in range(3):
        expstrain=expRes_df.meanStrain
        expstressL=expRes_df[f'mean_stress_L_{indices[DSN]}']
        expstressUL=expRes_df[f'mean_stress_UL_{indices[DSN]}']
        modelstressL=calculate_viscoelastic_stress(DSN, True, viscosity[DSN, 0])
        modelstressUL=calculate_viscoelastic_stress(DSN, False, viscosity[DSN, 1])
        # modelstressL=calculate_viscoelastic_stress(DSN, True, 1e-3)
        # modelstressUL=calculate_viscoelastic_stress(DSN, False, 1e-3)
        ax.set_xlabel('Strain (%)') ;ax.set_ylabel('Stress (MPa)')
        ax.plot(expstrain, expstressL, color=plt.get_cmap("tab10")(DSN), label=['AT 1', 'AT 2', 'AT 3'][DSN]+ ' Experiment')
        ax.plot(expstrain, expstressUL, color=plt.get_cmap("tab10")(DSN))
        ax.plot(elin*100, modelstressL, '--',color=plt.get_cmap("tab20")(1+2*DSN), label=['AT 1', 'AT 2', 'AT 3'][DSN]+ f' Model -- viscosity = {viscosity[DSN,0]:2.2e} {viscosity[DSN,1]:2.2e}')
        ax.plot(elin*100, modelstressUL,'--', color=plt.get_cmap("tab20")(1+2*DSN))


    handles, labels = ax.get_legend_handles_labels()
    labels, handles = [labels[0], labels[2], labels[4], labels[1], labels[3], labels[5]], [handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]]
    ax.legend(handles, labels,ncol=2)
    plt.savefig(dir_output+f'KV-model-prediction-various-viscosities.png')

#%% Plotting more hysteresis curves to model using mean viscosity

print('Plotting hysteresis curves with KV model using mean viscosity')

def predict_VE_with_viscosity_fit():
    fig, ax=plt.subplots()
    for DSN in range(3):
        expstrain=expRes_df.meanStrain
        expstressL=expRes_df[f'mean_stress_L_{indices[DSN]}']
        expstressUL=expRes_df[f'mean_stress_UL_{indices[DSN]}']
        ax.plot(expstrain, expstressL ,color=plt.get_cmap("tab10")(DSN), label=['AT 1', 'AT 2', 'AT 3'][DSN]+ ' Experiment')
        ax.plot(expstrain, expstressUL ,color=plt.get_cmap("tab10")(DSN))

        # mu=1e4
        ax.plot(elin*100, calculate_viscoelastic_stress(DSN, True, mean_viscosity), '--',color=plt.get_cmap("tab20")(1+2*DSN),  label=['AT 1', 'AT 2', 'AT 3'][DSN]+' Model')
        ax.plot(elin*100, calculate_viscoelastic_stress(DSN, False, mean_viscosity), '--',color=plt.get_cmap("tab20")(1+2*DSN))

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = [labels[0], labels[2], labels[4], labels[1], labels[3], labels[5]], [handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]]
    ax.legend(handles, labels,ncol=2)
    ax.set_xlabel('Strain (%)') ; ax.set_ylabel('Stress (MPa)')
    # ax.text(1, 0.3, 'DRAFT', color='gray', fontsize=100)
    plt.savefig(dir_output+f'KV-model-prediction-mean-viscosity.png')

    if atom:
        plt.show()

# predict_VE_with_viscosity_fit()

#%% EXPORT
def calculate_work_done(DSN):
    """
    IN the units of viscoelastic_prediction which are MPa -> MJ
    """
    load_work=float(simps(calculate_viscoelastic_stress(DSN, viscosity[DSN]),dx=elin[1])[0])
    unload_work=float(simps(calculate_viscoelastic_stress(DSN, viscosity[DSN]),dx=elin[1])[1])
    delta_work=load_work-unload_work
    frac=delta_work/load_work
    return load_work, unload_work, delta_work, frac
cols=['dataset',"totalwork_loading_MJm-3", "totalwork_unloading_MJm-3", "deltawork MJm-3", "Frac" ]
dexport=pd.DataFrame(columns=cols)
for DSN in range(3):
    load_work, unload_work, delta_work, frac=calculate_work_done(DSN)
    # _, _,  nu, A_f, mu_, E, faslen, vol_frac = viscoelastic_prediction(DSN)
    dtemp=pd.DataFrame([[datanames[DSN], load_work, unload_work, delta_work, frac]], columns=cols)
    dexport=pd.concat((dexport, dtemp))
dexport.to_csv(dir_output+'model_loading_params.csv')
