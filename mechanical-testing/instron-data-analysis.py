import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import pandas as pd
import itertools
import re
import textwrap
from ast import literal_eval
import scipy.stats as stats
from scipy.signal import argrelextrema
plt.style.use('~/dbox/4-Thesis/stylesheet.mplstyle')
atom=False
#WHERE IS RAW DATA
headerline=18 #where is the line that begins 'Time, Extension'
rawdatafile_strs = glob.glob( '../labwork/circadian-time-series/*.csv'); rawdatafile_strs.sort()

#WHERE TO SAVE PLOTS AND RESULTS
dir_M_paper='../plots-instron-results/mechanics-paper-results/'
dir_supp_NCB='../plots-instron-results/circadian-data-for-thesis/'


#--------------PROCESS RAW DATA--------------------------------------#
li=[]
for file in rawdatafile_strs:
    meta=pd.read_csv(file, header=None, nrows=headerline-2,index_col=0,names=['a', 'b']).T
    li.append(meta)
mdf = pd.concat(li, axis=0, ignore_index=True) #MetaDataFrame

#----Renaming Columns to something more helpful------#
short_cols=['Date', 'Area', 'L0','Time','Mouse','L/R','Sex','TenType','Age','EM', 'Spec Notes','Test Obsv','T','Who','SR','Exc']
mdf.columns=short_cols
#----------------------------------------------------#
def make_df(whichtest):
    return pd.read_csv (rawdatafile_strs[whichtest], header=headerline-2, skiprows=[headerline])
def make_df_list(whichtests):
    df_lis=[]
    for i in whichtests:
        df_lis.append(make_df(i))
    return df_lis
#-----Some boolian dfs for later-----#
excludedlist=[1,7,12,14] ; mdf.Exc[excludedlist]=True; included_bool=(mdf.Exc=='False');
inclist=np.where(included_bool)[0]
am9_bool=(mdf.Time=='9am')
pm9_bool=(mdf.Time=='9pm')
m_bool=(mdf.Sex=='M')
f_bool=(mdf.Sex=='F')
L_bool=(mdf.Sex=='F')
R_bool=(mdf.Sex=='F')
#--------------------------------------#

units=pd.read_csv (rawdatafile_strs[0], header=headerline-2, nrows=1)
reorder_cols=['Date','Time', 'Mouse','L/R','SR','Sex', 'Area', 'L0','Spec Notes', 'Test Obsv']

#-----Some calcs for later-----#
max_strain=6.0 #percent
ntests=len(mdf)
mechanics_paper_repeats=[mdf[(mdf.Time=='9am') & (mdf.Mouse=='1') & (mdf['L/R']=='R')].index[0], mdf[(mdf.Time=='9am') & (mdf.Mouse=='2') & (mdf['L/R']=='L')].index[0], mdf[(mdf.Time=='9am') & (mdf.Mouse=='4') & (mdf['L/R']=='R')].index[0]]
ncycles=np.max(make_df(0)['Total Cycle Count'])
strain_rate=float(mdf['SR'].iloc[0])
period=2*max_strain/ strain_rate  #in seconds
#----------------------#

mdf.iloc[(mechanics_paper_repeats)]
#%%All tests
def whole_test_comparison():
    whichtests=np.arange(ntests)
    which_variables=[['Time', 'Tensile stress'],['Tensile strain', 'Tensile stress']]
    fig, ax=plt.subplots(ntests,3, figsize=(20, 80),gridspec_kw={'width_ratios': [1,4,4]})
    for j in range(len(whichtests)):
        whichtest=whichtests[j];df=make_df(whichtest);roll=15
        ax[j,0].set_title(f"Test {j} ({j+1}/{ntests}) ") #{mdf['Date'].loc[whichtest]} {' '.join(mdf[['Time', 'Mouse', 'L/R']].loc[whichtest].values)} ")
        ax[j,0].set_axis_off()

        for i in range(1,len(which_variables)+1):
            x_=which_variables[i-1][0];y_=which_variables[i-1][1]
            x=[1/60 if x_=='Time' else 1]*df[:][x_].rolling(roll).mean() ;  y=df[:][y_].rolling(roll).mean()
            ax[j,i].plot(x,y)
            ax[j,i].set_xlabel(x_+' '+[units[x_].to_string(index=False) if x_!='Time' else '(min)'][0])

            ax[j,i].set_ylabel(y_+units[y_].to_string(index=False))
        # ax[j,len(which_variables)].set_title(f"Test {j+1}/{ntests}") #{mdf['Date'].loc[whichtest]} {' '.join(mdf[['Time', 'Mouse', 'L/R']].loc[whichtest].values)} ")
        # ax[j,len(which_variables)].set_axis_off()
        value =mdf[['Date','Time', 'Mouse', 'L/R','Sex','Spec Notes', 'Test Obsv']].iloc[whichtest].to_string(index=False)
        value=" ".join(value.split())
        value=value.replace("NaN", "")
        exc_str='\nExcluded' if np.isin(j,  excludedlist) else '\nIncluded'
        string = textwrap.TextWrapper(width=20).fill(text=value+exc_str)
        ax[j, 0].text(.5,.5,string, fontsize=14,  ha='center', va='center')
    plt.savefig(dir_supp_NCB + 'all-tests')
    plt.tight_layout();
    if atom:
        plt.show()
whole_test_comparison()

#%%OVERLAP COMPARISON
def overlap_comparison(whichtests, lo=5,hi=10):
    fig1, ax1=plt.subplots(figsize=(8, 9))
    for i in whichtests:
        df=make_df(i); roll=15

        ax1.set_title(f"Cycles {lo}-{hi}/{ncycles}") #{mdf['Date'].loc[whichtest]} {' '.join(mdf[['Time', 'Mouse', 'L/R']].loc[whichtest].values)} ")

        x_='Tensile strain';y_='Tensile stress'
        chosen_cycles=(df['Total Cycle Count']>lo) & (df['Total Cycle Count']<hi)
        shift=-df[chosen_cycles][y_].iloc[0] #This essentially 'tares' the stress,  by subtracting residual stress at the start of the loading

        x=[1/60 if x_=='Time' else 1]*df[chosen_cycles][x_].rolling(roll).mean() ;  y=shift+df[chosen_cycles][y_].rolling(roll).mean()
        style='--' if mdf["Time"].iloc[i]=='9am' else '-'
        exc_str=' Excluded' if np.isin(i,  excludedlist) else ''

        label_=f'Test {i} ,{mdf["Time"].iloc[i]}{mdf["Mouse"].iloc[i]}{mdf["Sex"].iloc[i]}'+exc_str
        ax1.plot(x,y,style, label=label_)
        ax1.set_xlabel(x_+' '+[units[x_].to_string(index=False) if x_!='Time' else '(min)'][0])
        ax1.set_ylabel(y_+units[y_].to_string(index=False))

    plt.tight_layout()
    plt.legend()
    plt.savefig(dir_supp_NCB + f'included-hysteresis_{lo}_{hi}')
    if atom:
        plt.show()

overlap_comparison(whichtests=mechanics_paper_repeats)
def overlap_comparison_load(whichtests):
    fig1, ax1=plt.subplots()
    lo,hi=5,10
    for i in range(len(whichtests)):
        whichtest=whichtests[i];    df=make_df(whichtest); roll=15

        ax1.set_title(f"Cycles {lo}-{hi}/{ncycles}") #{mdf['Date'].loc[whichtest]} {' '.join(mdf[['Time', 'Mouse', 'L/R']].loc[whichtest].values)} ")

        x_='Tensile strain';y_='Load'
        chosen_cycles=(df['Total Cycle Count']>lo) & (df['Total Cycle Count']<hi)
        shift=-df[chosen_cycles][y_].iloc[0] #This essentially 'tares' the stress,  by subtracting residual stress at the start of the loading

        x=[1/60 if x_=='Time' else 1]*df[chosen_cycles][x_].rolling(roll).mean() ;  y=shift+df[chosen_cycles][y_].rolling(roll).mean()
        style='--' if mdf["Time"].iloc[whichtest]=='9am' else '-'
        label_=f'{mdf["Time"].iloc[whichtest]} {mdf["Mouse"].iloc[whichtest]} {mdf["Sex"].iloc[whichtest]}'
        ax1.plot(x,y,style, label=label_)
        ax1.set_xlabel(x_+' '+[units[x_].to_string(index=False) if x_!='Time' else '(min)'][0])
        ax1.set_ylabel(y_+units[y_].to_string(index=False))

    plt.tight_layout();plt.legend();
    if atom:
        plt.show()
# overlap_comparison_load(mechanics_paper_repeats)
#%% HYSTERESIS

#%%PEAK FINDING
#Find the peaks
def peak_times_arr():
    """
    Generates an array of size Ntests X 2*Ncycles+1
    Each point [i,j] corresponds to the jth turning point on the ith test
    this includes the start and end of the run
    """
    allpeakslis=[];ilocs=[] ; roll=7
    for i in range(ntests):
        df=make_df(i);
        lo, hi=1, ncycles
        start=df[:].Time.iloc[0] ; end=df[:].Time.iloc[-1]
        peak_time_list=[start, end] #list of times which correspond to peaks /troughs in data
        for n in np.arange(lo, hi+1): #Find times where strain plot peaks        # n=10
            search_window=(df.Time>=start + (n-1) * period)& (df.Time <= start+ n * period)
            searchdf=df[search_window]['Tensile strain'].rolling(roll).mean()
            index=int(np.median(df[search_window][searchdf==np.max(searchdf)].index.values))
            time=df.Time.iloc[index]
            peak_time_list.append(time)
        for n in np.arange(lo, hi): #Find times where strain plot is a minimum
            search_window=(df.Time>=start + (n-0.5) * period)& (df.Time <= start+ (n+0.5) * period)
            searchdf=df[search_window]['Tensile strain'].rolling(roll).mean()
            index=int(np.median(df[search_window][searchdf==np.min(searchdf)].index.values))
            time=df.Time.iloc[index]
            peak_time_list.append(time)
        peak_time_list.sort()

        #Error checking
        if len(peak_time_list)!= ncycles*2 +1:
            print('Wrong number of peaks detected')
            return 0
        else:
            allpeakslis.append(peak_time_list)
            ilocs.append(df[df.Time.isin(peak_time_list)].index.values) #indices of peaks (and troughs)

    return np.array(allpeakslis),np.array(ilocs)
peak_times, peak_ilocs=peak_times_arr()

#%% FINDING MEAN LOADING AND UNLOADING CURVES FOR EACH TEST AND FOR ALL TESTS
linelist=['solid', 'dotted', 'dashed', 'dashdot','solid','dotted', 'dashed']
def meanloading(i, lo, hi, roll, strainAxInterp, x_='Tensile strain', y_='Tensile stress'):
    xs=[];ys=[];
    # https://stackoverflow.com/questions/51933785/getting-a-mean-curve-of-several-curves-with-x-values-not-being-the-same
    df=make_df(i)
    for j in np.arange(lo, hi):
        chosen_cycles=df.iloc[peak_ilocs[i, 2*j-2]:peak_ilocs[i, 2*j-1]]
        shift=-chosen_cycles[y_].iloc[0] #This essentially 'tares' the stress,  by subtracting residual stress at the start of the loading
        x=[1/60 if x_=='Time' else 1]*chosen_cycles[x_].rolling(roll).mean().to_numpy()
        y=shift+chosen_cycles[y_].rolling(roll).mean().to_numpy()
        xs.append(x[~np.isnan(x)])
        ys.append(y[~np.isnan(y)])
    stressInterp = [np.interp(strainAxInterp, xs[i], ys[i]) for i in range(len(xs))]
    meanStressInterp = np.mean(stressInterp, axis=0)
    error=np.std(np.array(stressInterp), axis=0)
    return meanStressInterp, error
def meanunloading(i, lo, hi, roll, strainAxInterp, x_='Tensile strain', y_='Tensile stress'):
    xs=[];ys=[];
    # https://stackoverflow.com/questions/51933785/getting-a-mean-curve-of-several-curves-with-x-values-not-being-the-same
    df=make_df(i)
    for j in range (lo, hi):
        chosen_cycles=df.iloc[peak_ilocs[i, 2*j-1]:peak_ilocs[i, 2*j]]
        shift=-chosen_cycles[y_].iloc[-1] #This essentially 'tares' the stress,  by subtracting residual stress at the start of the loading
        x=[1/60 if x_=='Time' else 1]*chosen_cycles[x_].rolling(roll).mean().to_numpy()
        y=shift+chosen_cycles[y_].rolling(roll).mean().to_numpy()
        xs.append(np.flip(x[~np.isnan(x)]))
        ys.append(np.flip(y[~np.isnan(y)]))
        exc_str='\nExcluded' if np.isin(i,  excludedlist) else '\nIncluded'
        label_=f'Test {i} ,{mdf["Time"].iloc[i]}{mdf["Mouse"].iloc[i]}{mdf["Sex"].iloc[i]}'+f' cycle {j//2} '+exc_str
        # ax1.plot(x,y, label=label_)
    stressInterp = [np.interp(strainAxInterp, xs[i], ys[i]) for i in range(len(xs))]
    meanStressInterp = np.mean(stressInterp, axis=0)
    error=np.std(np.array(stressInterp), axis=0)
    return meanStressInterp, error
def mean_hysteresis_each_test(lo=5, hi=10, nx=1001):
    roll=15
    linelist=['solid', 'dotted', 'dashed', 'dashdot','solid','dotted', 'dashed']

    fig1, ax1=plt.subplots()
    x_='Tensile strain';y_='Tensile stress'
    strainAxInterp = np.linspace(0, max_strain, nx)
    df_export=pd.DataFrame(strainAxInterp, columns=['meanStrain'])

    allMeanStressLArr=np.zeros((len(inclist), nx))
    allMeanStressULArr=np.zeros((len(inclist), nx))
    ii=0 ; labellis=[0]

    for i in inclist:
        stressMeanL, error_L=meanloading(i, lo, hi, roll, strainAxInterp)
        stressMeanUL, error_UL=meanunloading(i, lo, hi, roll, strainAxInterp)
        allMeanStressLArr[ii]   =stressMeanL
        allMeanStressULArr[ii]  =stressMeanUL

        style= '--' if mdf["Sex"].iloc[i]=='M' else '-'
        mdf["Time"].iloc[i]
        colour='k' if mdf["Time"].iloc[i] =='9am' else 'r'
        ZT='ZT3' if mdf["Time"].iloc[i] == '9am' else 'ZT15'
        label_=ZT+' '+ mdf["Sex"].iloc[i]

        ax1.plot(strainAxInterp, stressMeanL,label=label_ if label_ not in labellis else None, linestyle=style, color=colour )
        ax1.plot(strainAxInterp, stressMeanUL,linestyle=style,color=colour)

        labellis.append(label_)
        cols=[f'mean_stress_L_{i}', f'e_stress_L_{i}', f'mean_stress_UL_{i}', f'e_stress_UL_{i}']
        dfjoin=pd.DataFrame(np.array([stressMeanL, error_L, stressMeanUL, error_UL]).T, columns=cols)
        df_export=pd.concat([df_export,dfjoin], axis="columns")

        ii+=1

    dfjoin=pd.DataFrame(np.array([np.mean(allMeanStressLArr, axis=0),np.std(allMeanStressLArr, axis=0),np.mean(allMeanStressULArr, axis=0), np.std(allMeanStressULArr, axis=0)]).T, columns=['mean_stress_L_all','error_L_all','mean_stress_UL_all', 'error_UL_all'])
    df_export=pd.concat([df_export,dfjoin], axis="columns")

    ax1.set_xlabel(x_+' '+[units[x_].to_string(index=False) if x_!='Time' else '(min)'][0])
    ax1.set_ylabel(y_+units[y_].to_string(index=False))
    plt.tight_layout();

    handles, labels = ax1.get_legend_handles_labels()
    #   sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax1.legend(reversed(handles), reversed(labels))
    plt.savefig(dir_supp_NCB+f'included_mean_hysteresis_cycles_{lo}_{hi}');
    if atom:
        plt.show()
    return df_export
mean_stress_df=mean_hysteresis_each_test()


#%%HYSTERESIS CALCLATIONS
def write_hyster_df():
    """
    Retunrs a dataframe which stores info pertaining to hysteresis on every cycle for every test
    """
    cols=['testnumber', 'cycle number','load work MJ', 'unload work MJ']
    hyster_df=pd.DataFrame(columns=cols);rows=0

    for i in range(ntests):
        df=make_df(i)
        energy_change=[] #stores energy loss at each half-cycle
        for j in range (2*ncycles): #counts through the gaps between peaks /troughs, of which there are 2*ncycles
            x='Tensile strain' ; y='Tensile stress'

            #Adjustments to y axis - A FEW OPTIONS
            shift=0

            cyclestarts_ilocs=np.ndarray.flatten(np.dstack((peak_ilocs[i, 0::2],peak_ilocs[i, 0::2])))
            shift=df[y].iloc[cyclestarts_ilocs[j]] #This essentially 'tares' the stress,  by subtracting residual stress at the start of the loading on THIS cycle

            shift=df[y].iloc[cyclestarts_ilocs[j+2]] #This essentially 'tares' the stress,  by subtracting residual stress at the END of the loading on THIS cycle

            # shift=df[y].iloc[peak_ilocs[i, 2]] #This essentially 'tares' the stress,  by subtracting residual stress at the start of the second cycle

            X=0.01*df[x].iloc[peak_ilocs[i, j]:peak_ilocs[i, j+1]].values #comes off the instron as a % rather than a decimal, so divide through by 100
            Y=df[y].iloc[peak_ilocs[i, j]:peak_ilocs[i, j+1]].values-shift #remember stress is in MPa! So the energy is in MJ
            energy_change.append(np.trapz(Y,X)) #Intgrating

        energy_change=np.reshape(energy_change,(ncycles, 2)) #two columns, one load one unload

        #Now fill in hyster_df
        for k in range(ncycles):
            hyster_df.at[rows+k, 'testnumber']=int(i)
            hyster_df.at[rows+k, 'cycle number']=k+1
            hyster_df.at[rows+k, 'load work MJ']=energy_change[k, 0]
            hyster_df.at[rows+k, 'unload work MJ']=energy_change[k, 1]
        rows=rows+ncycles #for the hysteresis dataframe, to prevent overwriting
    return hyster_df
hyster_df=write_hyster_df()
hyster_df.loc[hyster_df['testnumber'].isin(inclist)].to_csv(dir_M_paper+'eloss_cyclewise.csv')


#%%HYSTERESIS PER CYCLE PLOTTING
def energy_loss_per_cycle():
    fig3, ax3 =plt.subplots();
    i=0 ; labellis=[]

    for i in inclist:
        df=hyster_df[hyster_df['testnumber']==i]
        style= '--' if mdf["Sex"].iloc[i]=='M' else '-'
        colour='k' if mdf["Time"].iloc[i] =='9am' else 'r'
        ZT='ZT3' if mdf["Time"].iloc[i] == '9am' else 'ZT15'
        label_=ZT+' '+ mdf["Sex"].iloc[i]
        # if label_ not in labellis:
        ax3.plot(df['cycle number'], abs(df[f'unload work MJ']-df[f'load work MJ']),style,color=colour, label=label_ if label_ not in labellis else None)
        labellis.append(label_)
    ax3.set_xlim(0.8, 15.2)
    ax3.set_xlabel('Cycle Number')
    ax3.set_ylabel('Energy loss per unit volume (J${^3}$m$^{-3}$)')
    handles, labels = ax3.get_legend_handles_labels()
    #   sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax3.legend(reversed(handles), reversed(labels), ncol=2)
    plt.subplots_adjust(wspace=.0)
    plt.savefig(dir_supp_NCB + 'energy-loss-percycle.png')
    if atom:
        plt.show()
energy_loss_per_cycle()


#%% ENERGY LOSS

def calculate_energy_losses(lo=5, hi=10):
    mean_EL=np.zeros(ntests)#Hysteresis PUV
    mean_REL=np.zeros(ntests)#Hysteresis PUV as a fraction of loading energy
    for i in np.arange(ntests):
        df=hyster_df[hyster_df['testnumber']==i][lo-1:hi-1]
        #Energy loss PUV
        mean_EL[i]=1000*((df['load work MJ']+df['unload work MJ'])).mean(axis=0) #mean energy loss per cycle as a fraction of loading energy #kJ conversion
        #Relative EL PUV
        mean_REL[i]=((df['load work MJ']+df['unload work MJ'])/df['load work MJ']).mean(axis=0) #mean energy loss per cycle as a fraction of loading energy
    return mean_EL, mean_REL
mean_EL, mean_REL=calculate_energy_losses()

def plot_eloss(lo=5, hi=10):
    fig, (ax1, ax2)=plt.subplots(1,2, figsize=(20, 10))
    ax1.boxplot(mean_EL[included_bool])
    ax1.set_ylabel('Mean energy Loss per unit volume (kJ/m$^{3}$)')
    ax2.boxplot(mean_REL[ included_bool])
    ax2.set_ylabel('Mean relative energy loss per unit volume (no units)')
    plt.suptitle(f"Cycles {lo} to {hi} of {ncycles} n={len(inclist)}")
    plt.savefig(dir_supp_NCB+f'energyloss_hist_{lo}_{hi}.png');
    if atom:
        plt.show()
plot_eloss()


#-----------EXPORT--------------#
export_Df=pd.DataFrame(np.vstack([mean_EL,mean_REL]).T, columns=['Energy Loss per unit Vol(kJ/m^3)', 'Relative Energy Loss'])
pd.concat([mdf, export_Df], axis=1).to_csv(dir_M_paper+'eloss-mean-values-5-10.csv')


#%%===================================== 9am 9pm stuff below

#%%Question: Is there a relationship between Temperature and maximum tensile stress?

x=mdf['T'].to_numpy(dtype='float')
y=mean_EL

    # y[i]=(np.max(make_df(i)['Tensile stress']))
fig2,ax2=plt.subplots()
ax2.set_xlabel('Temperature ($\degree$C)')
ax2.set_ylabel('Mean energy loss (per unit volume) (kJ/m$^{3}$)', fontsize=17)
# ax2.set_title(f'Question: Is there a relationship between RT and maximum tensile stress? N={np.count_nonzero(included_bool)}')
ax2.scatter(x[ am9_bool& included_bool],y[ am9_bool& included_bool], color='k', label='ZT3')
ax2.scatter(x[ pm9_bool& included_bool],y[ pm9_bool& included_bool], color='r', label='ZT15')
ax2.legend(); plt.tight_layout();plt.savefig(dir_supp_NCB+'temptest')
if atom:
    plt.show()

#%%Question: Is there a relationship between CSA and length?
def csa_length():
    x=np.zeros(ntests);y=x.copy()
    for i in inclist:
        df=make_df(i); roll=15
        x[i]=(float(mdf.Area.iloc[i]))
        y[i]=(float(mdf['L0'].iloc[i]))


    x1=np.array(mdf.Area[included_bool&am9_bool]).astype(float)
    x2=np.array(mdf.Area[included_bool&pm9_bool]).astype(float)
    y1=np.array(mdf.L0[included_bool&am9_bool]).astype(float)
    y2=np.array(mdf.L0[included_bool&pm9_bool]).astype(float)
    fig2,ax2=plt.subplots()
    ax2.set_xlabel('Cross sectional area (mm$^2$)')
    ax2.set_ylabel('Grip to grip length (mm)')
    # ax2.set_title(f'Question: Is there a relationship between CSA and L0? N={len(inclist)}')
    ax2.scatter(x1, y1, color='k', label='ZT3')
    ax2.scatter(x2, y2, color='r', label='ZT15')
    ax2.legend(); plt.tight_layout()
    plt.savefig(dir_supp_NCB + 'csa-l0')
    if atom:
        plt.show()



#%%Question: Is there a relationship between CSA and sex? Is there sexual dimorphism?
def csa_sex():
    fig2,ax2=plt.subplots(figsize=(8, 9), dpi=80)
    ax2.set_ylabel('Cross sectional area (mm$^2$)')
    ax2.set_xlabel('Sex')

    xm=np.array(mdf.Area[m_bool]).astype(float)
    xf=np.array(mdf.Area[f_bool]).astype(float)
    ttest=stats.ttest_ind(xm, xf)
    result="reject" if ttest[1]<0.05 else "accept"
    resultstr=f"H0_equalmeans_{result}_{100*ttest[1]:.0f}"
    ax2.boxplot([xm, xf]) #Dont necessarily care if it was included or not!!!
    plt.tight_layout()
    plt.xticks([1, 2], ['M','F'])
    plt.savefig(dir_supp_NCB + 'csa-sex-'+resultstr)
    print(resultstr)
    if atom:
        plt.show()
csa_sex()




#%%Question: Is there a relationship between CSA and maximum tensile stress?
x=np.zeros(ntests);y=x.copy()
for i in inclist:
    df=make_df(i); roll=15
    x[i]=(float(mdf.Area.iloc[i]))
    y[i]=(np.max(make_df(i)['Tensile stress']))
fig2,ax2=plt.subplots()
ax2.set_xlabel('Cross sectional area (mm$^2$)')
ax2.set_ylabel('Maximum tensile stress (MPa)')
# ax2.set_title(f'Question: Is there a relationship between CSA and maximum tensile stress? N={np.count_nonzero(included_bool)}')
ax2.scatter(x[ am9_bool& included_bool],y[ am9_bool& included_bool], color='k', label='ZT3')
ax2.scatter(x[ pm9_bool& included_bool],y[ pm9_bool& included_bool], color='r', label='ZT15')
ax2.legend(); plt.tight_layout()
plt.savefig(dir_supp_NCB + 'csa-max-tensile-stress')
if atom:
    plt.show()


#%% SEX
plt.figure(figsize=(8, 9), dpi=80)
plt.boxplot([mean_EL[ m_bool& included_bool],mean_EL[ f_bool& included_bool]])
plt.xlabel('Sex')
plt.xticks([1, 2], ['M','F'])
plt.ylabel('Mean energy Loss per unit volume (kJ/m$^{3}$)')
ttest=stats.ttest_ind(mean_EL[ m_bool& included_bool],mean_EL[ f_bool& included_bool])
result="reject" if ttest[1]<0.05 else "accept"
resultstr=f"H0_equalmeans_{result}_{100*ttest[1]:.0f}"
plt.tight_layout()
# plt.title(f'Mean energy loss over {lo} to {hi} cycles of {ncycles} per unit vol \n N={np.count_nonzero(included_bool)} of 16 \n {resultstr}')
plt.savefig(dir_supp_NCB + 'energyloss_hist_sex_'+resultstr)
if atom:
    plt.show()
#%% ---------------HYSTERESIS ENERGY LOSS 9AM /9PM----------

fig1,ax1=plt.subplots(figsize=(8, 9), dpi=80)
ax1.boxplot([mean_EL[ am9_bool&included_bool],mean_EL[ pm9_bool& included_bool]])
ax1.set_xlabel('Time')
ax1.set_xticks([1,2])
ax1.set_xticklabels(['ZT3','ZT15'])
ax1.set_ylabel('Mean energy Loss per unit volume (kJ/m$^{3}$)')
ttest=stats.ttest_ind(mean_EL[ am9_bool&included_bool],mean_EL[ pm9_bool& included_bool])
result="reject" if ttest[1]<0.05 else "accept"
resultstr=f"H0_equalmeans_{result}_{100*ttest[1]:.0f}"
plt.tight_layout()
plt.savefig(dir_supp_NCB+'9am9pm-energyloss_hist_'+resultstr)
if atom:
    plt.show()


fig2,ax2=plt.subplots(figsize=(8, 9), dpi=80)
ax2.boxplot([mean_REL[ am9_bool&included_bool],mean_REL[ pm9_bool& included_bool]])
ax2.set_xlabel('Time')
ax2.set_xticks([1,2])
ax2.set_xticklabels(['ZT3','ZT15'])
ax2.set_ylabel('Mean relative energy loss per unit volume (no units)')
ttest=stats.ttest_ind(mean_REL[ am9_bool&included_bool],mean_REL[ pm9_bool& included_bool])
result="reject" if ttest[1]<0.05 else "accept"
resultstr=f"H0_equalmeans_{result}_{100*ttest[1]:.0f}"
plt.tight_layout()
plt.savefig(dir_supp_NCB+'9am9pm-relative-energyloss_hist_'+resultstr)
if atom:
    plt.show()


#%%PLOT MEAN LOADING CURVES 9AM 9PM

def plot_mean_loading_curve(lo=5, hi=10):
    fig, ax=plt.subplots()
    x_='Tensile strain';y_='Tensile stress'
    x=mean_stress_df.meanStrain

    am9indices=np.where(am9_bool&included_bool)[0]
    pm9indices=np.where(pm9_bool&included_bool)[0]
    y1=np.mean(np.array([(np.array(mean_stress_df[f'mean_stress_L_{i}'])) for i in am9indices]), axis=0)
    y2=np.mean(np.array([(np.array(mean_stress_df[f'mean_stress_L_{i}'])) for i in pm9indices]), axis=0)
    y3=np.mean(np.array([(np.array(mean_stress_df[f'mean_stress_UL_{i}'])) for i in am9indices]), axis=0)
    y4=np.mean(np.array([(np.array(mean_stress_df[f'mean_stress_UL_{i}'])) for i in pm9indices]), axis=0)

    ax.plot(x, y1, '-k', label='ZT3')
    ax.plot(x, y2, '-r', label='ZT15')
    ax.plot(x, y3, '-k', )
    ax.plot(x, y4, '-r', )
    ax.set_xlabel(x_+' '+[units[x_].to_string(index=False) if x_!='Time' else '(min)'][0])
    ax.set_ylabel(y_+units[y_].to_string(index=False))
    plt.tight_layout();plt.legend();
    plt.savefig(dir_supp_NCB+f'mean_loading_curve_9am9pm_{lo}_{hi}');
    if atom:
        plt.show()
plot_mean_loading_curve()

mean_stress_df.to_csv(dir_M_paper+'mean_loading_stress_all.csv')
mean_stress_df.to_csv(dir_supp_NCB+'mean_loading_stress_9am9pm.csv')

#%% MECHANICS PAPER

def mean_hysteresis_mechanics_paper(lo=5, hi=10, nx=1001):
    roll=15
    linelist=['solid', 'dotted', 'dashed', 'dashdot','solid','dotted', 'dashed']

    fig1, ax1=plt.subplots()
    x_='Tensile strain';y_='Tensile stress'
    strainAxInterp = np.linspace(0, max_strain, nx)
    allMeanStressLArr=np.zeros((len(inclist), nx))
    allMeanStressULArr=np.zeros((len(inclist), nx))
    ii=0 ; labellis=[0]

    for i in mechanics_paper_repeats:
        scale=[0.833, 0.73, 0.812]
        stressMeanL, error_L=meanloading(i, lo, hi, roll, strainAxInterp)
        stressMeanUL, error_UL=meanunloading(i, lo, hi, roll, strainAxInterp)
        allMeanStressLArr[ii]   =stressMeanL
        allMeanStressULArr[ii]  =stressMeanUL
        # colour=plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        # plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        cmap = plt.get_cmap("tab10")
        label_=['AT 1', 'AT 2', 'AT 3'][ii]
        # label_=mdf.Time.iloc[i]+mdf.Mouse.iloc[i]+mdf['L/R'].iloc[0]
        ax1.plot(strainAxInterp, stressMeanL,color=cmap(ii), label= label_)
        ax1.plot(strainAxInterp, stressMeanUL,color=cmap(ii) )
        ii+=1
    ax1.set_xlabel(x_+' '+[units[x_].to_string(index=False) if x_!='Time' else '(min)'][0])
    ax1.set_ylabel(y_+units[y_].to_string(index=False))
    ax1.legend();
    plt.savefig(dir_M_paper+'hysteresis_experimental');
    if atom:
        plt.show()

mean_hysteresis_mechanics_paper()
