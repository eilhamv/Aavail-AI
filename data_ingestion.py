#!/usr/bin/env python

import os
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime

## Project directories
PROJECT_ROOT_DIR = "."
DATA_DIR = os.path.join("data")
DEV = True
   
def _ingest_data(dev=DEV, verbose=True):
    """
    load, join and clean json invoice data
    """
    
    if dev:
        data_dir = os.path.join(PROJECT_ROOT_DIR,DATA_DIR,"train")
    else:
        data_dir = os.path.join(PROJECT_ROOT_DIR,DATA_DIR,"production")
    
    ## Check the data directory
    if not os.path.exists(data_dir):
        raise Exception("specified data dir does not exist")
    
    if not len(os.listdir(data_dir))>=1:
        raise Exception("specified data dir does not contain any files")
        
    ## create a list with all json files in the data directory
    if verbose:
        print("...reading from json files")
    json_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search("\.json",f)]
    
    ## read json file from the data directory
    total = len(json_files)
    if verbose:
        print("...{} json files in the '{}' directory".format(total, data_dir))
    aavail_data = {}
    for iteration,json_file in enumerate(json_files):
        if verbose:
            end = "" if (iteration+1) < total else "\n"
            print("\r...reading file: {}/{}".format(iteration+1,total), end=end)
        key = os.path.split(json_file)[-1]
        df = pd.read_json(json_file)
        aavail_data[key] = df
    
    ## default columns names
    column_names = ['country', 'customer_id', 'day', 'invoice', 'month', 'price', 'stream_id', 'times_viewed', 'year']
    
    ## rename columns to uniformly named features
    for invoice, df in aavail_data.items():
        if "StreamID" in df.columns.values:
            df.rename(columns={'StreamID':'stream_id'},inplace=True)
        if "TimesViewed" in df.columns.values:
            df.rename(columns={'TimesViewed':'times_viewed'},inplace=True)
        if 'total_price' in df.columns.values:
            df.rename(columns={'total_price':'price'},inplace=True)
        if set(df.columns.values) != set(column_names):
            raise Exception("column names for {} could not be matched to correct columns".format(invoice))
    
    ## concatenate the data to one dataframe
    df = pd.concat(aavail_data, ignore_index=True)
    print("...concatenated to dataset of {} rows and {} columns".format(df.shape[0], df.shape[1]))
    
    ## Extract date from year, month and day
    df["invoice_date"] = pd.to_datetime(df[["year","month","day"]])

    ## invoice IDs should not contain any letters
    df["invoice_id"] = df["invoice"].str.replace(pat="\D+", repl="", regex=True).values.astype(np.int)
    df.drop(["invoice"], axis=1, inplace=True)
    
    return df

def _convert_to_ts(df, country=None):
    """
    convert the original clean dataframe to a time series
    by aggregating over each day for the given country
    """
    
    ## define the columns to group
    if country:
        cols = ["invoice_date", "country"]
    else:
        cols = ["invoice_date"]
        
    ## group the data    
    grouped = df.groupby(cols)
    
    ## count the purchases
    df_count = grouped[["price"]].count()
    df_count.rename(columns={'price':'purchases'},inplace=True)
    
    ## apply aggregation functions
    df_aggr = grouped.agg({"invoice_id":"nunique","stream_id": "nunique","times_viewed":"sum","price":"sum"})
    df_aggr.rename(columns={"invoice_id":"unique_invoices","stream_id":"unique_streams","times_viewed":"total_views","price":"revenue"}, inplace=True)
    
    ## merge the counts and the results from aggregation
    df_final = df_count.merge(df_aggr, left_index=True, right_index=True).reset_index()
    
    if country:
        mask = df_final.country == country
        return df_final[mask].drop("country", axis=1).set_index("invoice_date")
    else:
        return df_final.set_index("invoice_date")
    
def ingest_ts(clean=True, dev=DEV, verbose=True):
    """
    fetch timeseries data
    if the csv files exist, it reads from the file directory,
    otherwise it creates them from the original data and saves the files
    """
    
    if verbose:
        print("Ingesting Data")
    
    if dev:
        ts_dir = os.path.join(PROJECT_ROOT_DIR,DATA_DIR,"train-ts")
    else:
        ts_dir = os.path.join(PROJECT_ROOT_DIR,DATA_DIR,"production-ts")
        
    ## Check the data directory
    if not os.path.exists(ts_dir):
        os.makedirs(ts_dir)
        
    if (len(os.listdir(ts_dir)) > 0) & (clean == False):
        if verbose:
            print("...loading timeseries data from files")
        ts_files = [os.path.join(ts_dir,f) for f in os.listdir(ts_dir) if re.search("\.csv",f)]
        return {re.sub("\.csv","",os.path.split(f)[-1]):pd.read_csv(f, index_col=0) for f in ts_files}
        
    ## Load original data
    df = _ingest_data(dev=dev, verbose=verbose)
    

    ## Find the top ten countries by revenue
    top_10_countries = df.groupby("country")[["price"]].sum().sort_values(by="price",ascending=False)[:10].index.values.tolist()
    
    ## Add total to the countries
    countries = top_10_countries
    countries.append("Total")
    
    ## Convert the data to timeseries
    ts = {}
    if verbose:
        print("...converting data to timeseries")
    for country in countries:
        
        key = re.sub("\s+","_",country.lower())
        if key == "total":
            ts[key] = _convert_to_ts(df)
        else:
            ts[key] = _convert_to_ts(df, country=country)
        
        ## write file
        csv_path = os.path.join(ts_dir,"{}.csv".format(key))
        ts[key].to_csv(csv_path, index=True)
        
    return ts

def load_feature_matrix(clean=False, dev=DEV, verbose=True):
    """
    load the clean dataset after ingestion
    """
    
    ## load the dataset
    ts = ingest_ts(clean=clean, dev=dev, verbose=verbose)
    
    if verbose:
        print("Creating Feature Matrix")
    
    df = pd.concat(ts, keys=ts.keys(), names=["country"]).reset_index()

    ## check the type of the invoice_date
    if df.dtypes["invoice_date"] != "datetime64[ns]":
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        
    return df

if __name__ == "__main__":
    
    run_start = time.time()
  
    ## ingest data
    ts = ingest_ts(dev=DEV)
    
    print("METADATA")
    for key, item in ts.items():
        print("...{} {}".format(key, item.shape))
    
    ## metadata
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
