#!/usr/bin/env python

import time,os,re,csv,sys,uuid,joblib
import pandas as pd
from datetime import date

PROJECT_DIR = "."
LOG_DIR = os.path.join("logs")


## import mode
from data_ingestion import DEV

def _update_train_log(tag,algorithm,score,runtime,model_version,model_note,dev=DEV, verbose=True):
    """
    update train log file
    """
    if verbose:
        print("...updating train log")
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if dev:
        logfile = "{}-train-{}-{}.log".format("test",today.year, today.month)
    else:
        logfile = "{}-train-{}-{}.log".format("prod",today.year, today.month)
        
    
    ## write the data to a csv file
    logpath = os.path.join(LOG_DIR, logfile)
    
    ## write the data to a csv file    
    header = ["unique_id","timestamp",'tag','score',"runtime",'model_version','model_note']
    write_header = False
    if not os.path.exists(logpath):
        write_header = True
    with open(logpath,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),tag,algorithm,score,runtime,model_version,model_note])
        writer.writerow(to_write)
        
def _update_predict_log(tag,y_pred,target_date,runtime,model_version,model_note,dev=DEV, verbose=True):
    """
    update predict log file
    """
    
    if verbose:
        print("...update predict log")
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if dev:
        logfile = "{}-predict-{}-{}.log".format("test",today.year, today.month)
    else:
        logfile = "{}-predict-{}-{}.log".format("prod",today.year, today.month)
        
    
    ## write the data to a csv file
    logpath = os.path.join(LOG_DIR, logfile)
    
    ## write the data to a csv file    
    header = ["unique_id","timestamp",'tag','y_pred',"target_date","runtime",'model_version','model_note']
    write_header = False
    if not os.path.exists(logpath):
        write_header = True
    with open(logpath,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),tag,y_pred,target_date,runtime,model_version,model_note])
        writer.writerow(to_write)
        
def log_load(tag,year,month,env,verbose=True):
    """
    load requested log file
    """
    logfile = "{}-{}-{}-{}.log".format(env,tag,year,month)
    
    if verbose:
        print(logfile)
    return logfile
    
