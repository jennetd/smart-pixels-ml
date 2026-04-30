import os, subprocess
import random
from datetime import datetime
import time

from scipy.optimize import curve_fit

import pandas as pd
from tqdm import tqdm
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.keys())
markers = ['o','s','v','^']

fontsize=16
plt.rc('font', size=fontsize-4)

from models import *

minval=1e-9
maxval=1e9

def grep(string, filename):

    out_list = []

    cmd = "grep '" + string + "' " + filename
    #print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Get output data
    cmd_out = result.stdout.splitlines()
    for line in cmd_out:
        #print(line)

        if '#' in line:
            print("Skipping line " + line)
            continue
        out_list += [line.split(':')[-1].strip()]

    return out_list

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def inverse_cot(cota):
    a = np.array(np.arctan(1.0/cota))
    a[np.where(a<0)] = a[np.where(a<0)] + np.pi
    return a 

def sigmas_in_deg(df):
    for a in ['A','B']:
        df['upsigma'+a] = abs(inverse_cot(df['cot'+a] + df['sigmacot'+a])*180/np.pi - df[a])
        df['downsigma'+a] = abs(inverse_cot(df['cot'+a] - df['sigmacot'+a])*180/np.pi - df[a])             
    return df

def rescale(df, scale_factor):

    # Only rescales means and sigmas, not correlations
    for v in ['x','y','cotA','cotB']:
        if v in df.columns:
            df[v] = scale_factor[v]*df[v]
            df[v+'true'] = scale_factor[v]*df[v+'true']
        if 'sigma'+v in df.columns:
            df['sigma'+v] = scale_factor[v]*df['sigma'+v]

    return df

def construct_df(model, test_generator, scale_factor):

    # predicts test data
    p_test = model.predict(test_generator)

    # Jennet: TBH not sure why this is needed but ok
    complete_truth = None
    for _, y in tqdm(test_generator):
        if complete_truth is None:
            complete_truth = y
        else:
            complete_truth = np.concatenate((complete_truth, y), axis=0)

    # creates df with all predicted values and matrix elements - 4 predictions, all 10 unique matrix elements
    df = pd.DataFrame(p_test,columns=['x','M11','y','M22','cotA','M33','cotB','M44','M21','M31','M32','M41','M42','M43'])

    # stores all true values in same matrix as xtrue, ytrue, etc.
    df['xtrue'] = complete_truth[:,0]
    print(np.mean(df['xtrue']),np.min(df['xtrue']),np.max(df['xtrue']))
    df['ytrue'] = complete_truth[:,1]
    df['cotAtrue'] = complete_truth[:,2]
    df['cotBtrue'] = complete_truth[:,3]
    df['M11'] = minval+tf.math.maximum(df['M11'], 0)
    df['M22'] = minval+tf.math.maximum(df['M22'], 0)
    df['M33'] = minval+tf.math.maximum(df['M33'], 0)
    df['M44'] = minval+tf.math.maximum(df['M44'], 0)

    df['sigmax'] = abs(df['M11'])
    df['sigmay'] = np.sqrt(df['M21']**2 + df['M22']**2)
    df['sigmacotA'] = np.sqrt(df['M31']**2+df['M32']**2+df['M33']**2)
    df['sigmacotB'] = np.sqrt(df['M41']**2+df['M42']**2+df['M43']**2+df['M44']**2)

    extra_columns = ['M11','M22','M33','M44','M21','M31','M32', 'M41', 'M42', 'M43']
    df.drop(labels=extra_columns, axis=1, inplace=True)

    df = rescale(df, scale_factor)

    # calculates residuals for x, y, cotA, cotB
    for v in ['x','y','cotA','cotB']:
        if v in df.columns:
            df['residual'+v] = (df[v] - df[v+'true'])
            if 'sigma'+v in df.columns:
                df['pull'+v] = (df[v] - df[v+'true'])/df['sigma'+v]

    for v in ['cotA','cotB']:
        if v in df.columns:
            df[v[-1]] = inverse_cot(df[v])*180/np.pi
            df[v[-1]+'true'] = inverse_cot(df[v+'true'])*180/np.pi
            df['residual'+v[-1]] = (df[v[-1]] - df[v[-1]+'true'])

    return df

def pull_plot(thisdf, ax, var, name):
    
    h = ax.hist(thisdf[var],bins=np.linspace(-5,5,50),histtype='step')
    ax.set_xlabel(name)
    ax.set_yscale('log')

    ydata = h[0]
    xdata = h[1][:-1]+3/50.

    pars, cov = curve_fit(gauss,xdata,ydata)

    xbins = np.linspace(-5,5,100)
    ax.plot(xbins,gauss(xbins,pars[0],pars[1],pars[2]),color='black')
    ax.set_ylim(0.5,100000)

    print('Mean',pars[1])
    print('Sigma',pars[2])
    
    ax.text(-5,2000,"$\mu$="+str(round(pars[1],2)))
    ax.text(-5,1000,"$\sigma$="+str(round(abs(pars[2]),2)))

def history(base_dir):

    training_cp_path = os.path.join(base_dir, 'training_log.csv')
    training_history = pd.read_csv(training_cp_path)

    plt.scatter(training_history['epoch'], training_history['loss'])
    plt.scatter(training_history['epoch'], training_history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('NLL loss')
    plt.tight_layout() 

    plt.savefig(os.path.join(base_dir,'training_hist.png'))
    plt.show()

def threshold_evolution(base_dir):
    '''Threshold migration and loss optimization for one training'''
    SQ_log = pd.read_csv(os.path.join(base_dir, 'soft_quantizer_state_log.csv'))
    print(SQ_log.columns)

    th_keys = [f'threshold_{i}' for i in range(3)]

    for i, th_key in enumerate(th_keys):
        epochs = SQ_log['epoch']
        th_charge = SQ_log[th_key]
        plt.scatter(epochs, th_charge, s=3)

    plt.grid()
    plt.legend(th_keys)
    plt.xlabel('Epochs', fontsize =14)
    plt.ylabel('Charge threshold [e]', fontsize = 14)
    plt.title('SQ threshold migration', fontsize = 16)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,'threshold_evolution.png'))
    plt.show()

def residual_plot(ax, thisdf, var1, var2, name):

    scaling = 1.0
    
    nbins = 15
    
    var1_scaled = thisdf[var1] * scaling
    var2_scaled = thisdf[var2] * scaling
    residual_scaled = var1_scaled - var2_scaled
    
    xmin = np.min(var1_scaled)
    xmax = np.max(var1_scaled)
    
    step = 1.0*(xmax-xmin)/nbins
    
    x = sns.regplot(x=var1_scaled, y=residual_scaled, x_bins=np.linspace(xmin,xmax,nbins), fit_reg=None, marker='.', ax=ax)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    
    thisdf['residual'+var2] = residual_scaled
    print(var1)
    
    means = []
    upbar = []
    downbar = []
    for i in range(0,nbins):
        means += [np.mean(thisdf['residual'+var2][(var1_scaled>xmin + i*step) & (var1_scaled<xmin + (i+1)*step)])]
        upbar += [means[i] + np.mean(thisdf['sigma'+var2][(var1_scaled>xmin + i*step) & (var1_scaled<xmin + (i+1)*step)] * scaling)]
        downbar += [means[i] - np.mean(thisdf['sigma'+var2][(var1_scaled>xmin + i*step) & (var1_scaled<xmin + (i+1)*step)] * scaling)]
    ax.fill_between(x=np.linspace(xmin,xmax,nbins),y1=upbar,y2=downbar, alpha=0.2)

def residual_plot_deg(ax, thisdf, var1, var2, name, scaling=1.0):
    # positions
    if 'cot' not in var1:
        residual_plot(ax, thisdf, var1, var2, name, scaling=scaling)
        return

    thisdf['angle'] = inverse_cot(thisdf[var2].values * scaling)*180/np.pi
    
    thisdf['angleup'] = abs(inverse_cot((thisdf[var2].values + thisdf['sigma'+var2].values) * scaling)*180/np.pi - thisdf['angle'])
    thisdf['angledown'] = abs(inverse_cot((thisdf[var2].values - thisdf['sigma'+var2].values) * scaling)*180/np.pi - thisdf['angle'])
    thisdf['angletrue'] = inverse_cot(thisdf[var1].values * scaling)*180/np.pi
        
    var1 = 'angletrue'
    var2 = 'angle'
    
    nbins = 15
    xmin = np.min(thisdf[var1])
    xmax = np.max(thisdf[var1])
    
    step = 1.0*(xmax-xmin)/nbins
        
    x = sns.regplot(x=thisdf[var1], y=(thisdf[var1]-thisdf[var2]), x_bins=np.linspace(xmin,xmax,nbins), fit_reg=None, marker='.', ax=ax)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    
    thisdf['residual'+var2] = (thisdf[var1]-thisdf[var2])
    print(var1)
    
    means = []    
    upbar = []
    downbar = []
    for i in range(0,nbins):
        means += [np.mean(thisdf['residual'+var2][(thisdf[var1]>xmin + i*step) & (thisdf[var1]<xmin + (i+1)*step)])]
        upbar += [means[i] + np.mean(thisdf['angleup'][(thisdf[var1]>xmin + i*step) & (thisdf[var1]<xmin + (i+1)*step)])]
        downbar += [means[i] - np.mean(thisdf['angledown'][(thisdf[var1]>xmin + i*step) & (thisdf[var1]<xmin + (i+1)*step)])]
    #ax.scatter(x=np.linspace(xmin,xmax,nbins),y=means)
    ax.fill_between(x=np.linspace(xmin,xmax,nbins),y1=upbar,y2=downbar, alpha=0.2)

# From Sofi
def shortest_interval_68(data, center_type='mean'):
    """
    Compute the shortest interval that contains 68% of the data.
    """
    data = np.sort(data)
    n = len(data)
    ci_size = int(np.floor(0.68 * n))
    min_width = float("inf")
    min_i = 0

    for i in range(n - ci_size):
        width = data[i + ci_size] - data[i]
        if width < min_width:
            min_width = width
            min_i = i

    low = data[min_i]
    high = data[min_i + ci_size]

    if center_type == 'mean':
        center = np.mean(data)
    else:
        center = np.median(data)

    return {
        "center": center,
        "error_low": float(center - low),
        "error_high": float(high - center),
        "center_type": center_type
    }

# Helper functions for plotting comparisons
def draw_one_vbl_slim(vbl, x, d, names, ax_ii):
    
    x_array = [-1*x+0.3 - 0.2*i for i in range(len(names))]
    y_array = [d[k]['mean_'+vbl] for k in names]
    yerr_array = [[d[k]['down68_'+vbl] for k in names],[d[k]['up68_'+vbl] for k in names]]
    
    alpha = [int(num > 0) for num in yerr_array[0]]
    
    line = ax_ii.errorbar(x = y_array, y = x_array, xerr=yerr_array, color = colors[x+4], linestyle='')
    
    dot1 = ax_ii.scatter(y_array[0], x_array[0], marker='o', color = colors[x+4], alpha = alpha[0])
    if len(y_array) > 1:
        dot2 = ax_ii.scatter(y_array[1], x_array[1], marker='s', color = colors[x+4], alpha = alpha[1])
    if len(y_array) > 2:
        dot3 = ax_ii.scatter(y_array[2], x_array[2], marker='v', color = colors[x+4], alpha = alpha[2])
    if len(y_array) > 3:
        dot4 = ax_ii.scatter(y_array[3], x_array[3], marker='^', color = colors[x+4], alpha = alpha[3])

    dot1 = ax_ii.scatter([0], [-99], marker='o', color = 'black')
    dot2 = ax_ii.scatter([0], [-99], marker='s', color = 'black')
    dot3 = ax_ii.scatter([0], [-99], marker='v', color = 'black')
    dot4 = ax_ii.scatter([0], [-99], marker='^', color = 'black')
    ax_ii.set_xlabel(vbl_names[vbl],fontsize=fontsize)
    ax_ii.set_yticks([])

    return [line], [dot1,dot2,dot3,dot4]

def draw_one_vbl(vbl, x, d, ax_ii, vbl_names):
    
    x_array = [-1*x+0.3]
    y_array = [d['mean_'+vbl]]
    yerr_array = [[d['down68_'+vbl]],[d['up68_'+vbl]]]
    yerr2_array = [[d['mean_downsigma'+vbl]],[d['mean_upsigma'+vbl]]]
    
    
    line1 = ax_ii.errorbar(x = y_array, y = x_array, xerr=yerr_array, color = colors[x], linestyle='')
    line2 = ax_ii.errorbar(x = y_array, y = x_array, xerr=yerr2_array, elinewidth=10, alpha=0.2, color = colors[x], linestyle='')
    
    dot1 = ax_ii.scatter(y_array, x_array, marker='o', color = colors[x])
    
    ax_ii.set_xlabel(vbl_names[vbl],fontsize=fontsize)
    ax_ii.set_yticks([])

    return [line1,line2], [dot1]
    
def draw_one_model(x, d, names, ax, vbl_names, slim = False):

    if slim:     
        line, dots = draw_one_vbl_slim('x', x, d, ax[1][0], vbl_names)
        line, dots = draw_one_vbl_slim('y', x, d, ax[1][1], vbl_names)
        line, dots = draw_one_vbl_slim('B', x, d, ax[1][2], vbl_names)
        
    else:
        line, dots = draw_one_vbl('x', x, d, ax[1][0], vbl_names)
        line, dots = draw_one_vbl('y', x, d, ax[1][1], vbl_names)
        line, dots = draw_one_vbl('A', x, d, ax[1][2], vbl_names)
        line, dots = draw_one_vbl('B', x, d, ax[1][3], vbl_names)

    return line, dots



