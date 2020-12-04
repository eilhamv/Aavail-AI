#!/usr/bin/env python

import re
import time
import os
import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
plt.style.use('seaborn')

## import functions to load the data
from data_ingestion import load_feature_matrix
from data_ingestion import DEV

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title


IMAGE_DIR = os.path.join(".","images")

def save_fig(fig_id, tight_layout=True, image_path=IMAGE_DIR):
    """
    save the image as png file in the image directory
    """
    
    ## Check the data directory
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    path = os.path.join(image_path, fig_id + ".png")
    print("...saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def create_plots(df):
    """
    create plots for data visualization
    """
    print("Creating Plots")
    
    ## Analyze Total revenues over time
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    table = df[df.country=="total"].set_index("invoice_date")[["revenue"]].resample("MS").sum()
    table.plot(ax=ax1)
    ax1.set_xlabel("months")
    ax1.set_ylabel("revenues")
    ax1.title.set_text("Monthly Revenue")
    save_fig("monthly_revenues")
    
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    df["year"] = df.invoice_date.dt.year
    table = df[df.country=="total"].set_index("invoice_date")[["revenue"]].resample("AS").sum()
    table.plot(kind='bar', stacked=False, ax=ax1, rot=0)
    ax1.set_xlabel("year")
    ax1.set_ylabel("revenue")
    ax1.title.set_text("Total revenue per Year")
    save_fig("yearly_revenues")
    
    ## Analyze revenues per country
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    table = pd.pivot_table(df[df.country!="total"], index = ['country'], columns=["year"],values = 'revenue', aggfunc="sum")
    table.plot(kind='bar', stacked=False, ax=ax1, rot=90)
    ax1.set_xlabel("country")
    ax1.set_ylabel("revenues")
    ax1.title.set_text("Revenues in the top 10 markets")
    save_fig("revenues_per_market")
    
    ## Analyze Revenues in UK compared to the Total
    mask = (df.country=="united_kingdom") | (df.country=="total")
    df_filter = df[mask].copy()
    df_filter.drop(["invoice_date", "year"], axis=1, inplace=True)
    
    # revenue correlation matrix
    fig = plt.figure(figsize=(8,8))
    num_features = ["purchases","unique_invoices","unique_streams","total_views","revenue"]
    corrmat = df_filter.corr()
    k = len(num_features) #number of variables for heatmap
    cols = corrmat.nlargest(k, 'revenue')['revenue'].index
    cm = np.corrcoef(df_filter[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    save_fig("correlations")

if __name__ == "__main__":
    
    run_start = time.time()
    
    df = load_feature_matrix(dev=DEV, clean=False)
    create_plots(df)
    
    print("METADATA")
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
