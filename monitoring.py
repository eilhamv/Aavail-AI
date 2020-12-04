#!/usr/bin/env python

import time
import numpy as np
import pandas as pd

from data_engineering import engineer_features
from modelling import _model_train, model_predict
from sklearn.metrics import mean_squared_error
from sklearn.covariance import EllipticEnvelope
from scipy.stats import wasserstein_distance
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

## switch to production
DEV = False

def simulate_samples(n_samples, X, y, dates):
    """
    simulate new samples (via bootstrap)
    """
    
    indices = np.arange(y.size)
    new_indices = np.random.choice(indices, n_samples, replace=True)
    
    X_new = X[new_indices,:]
    y_new = y[new_indices]
    dates_new = dates[new_indices]
    return X_new, y_new, dates_new


def model_monitor(country="total", dev=DEV, training=True):
    """
    performance monitoring
    """
    print("Monitor Model")
    
    ## import data
    datasets = engineer_features(training=training, dev=dev)
    X, y, dates, labels = datasets[country]
    dates = pd.to_datetime(dates)
    print(X.shape)
    
    ## train the model
    if training:
        _model_train(X, y, labels, tag=country, dev=dev)
    
    ## monitor RMSE
    samples = [10, 20, 30, 50, 60]

    for n in samples:
        X_new, y_new, dates_new = simulate_samples(n, X, y, dates)
        queries = [(str(d.year), str(d.month), str(d.day), country) for d in dates_new]
        y_pred = [model_predict(year=query[0], month=query[1], day=query[2], country=query[3],verbose=False, dev=dev)["y_pred"][0].round(2) for query in queries]
        rmse = np.sqrt(mean_squared_error(y_new.tolist(),y_pred))
        print("sample size: {}, RSME: {}".format(n, rmse.round(2)))
        
    ## monitor performance
    ## scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    samples = [25, 50, 75, 90]

    clf_y = EllipticEnvelope(random_state=0,contamination=0.01)
    clf_X = EllipticEnvelope(random_state=0,contamination=0.01)

    clf_X.fit(X)
    clf_y.fit(y.reshape(y.size,1))

    results = defaultdict(list)
    for n in samples:
        X_new, y_new, dates_new = simulate_samples(n,X,y, dates)
        results["sample_size"].append(n)
        results['wasserstein_X'].append(np.round(wasserstein_distance(X.flatten(),X_new.flatten()),2))
        results['wasserstein_y'].append(np.round(wasserstein_distance(y,y_new),2))
        test1 = clf_X.predict(X_new)
        test2 = clf_y.predict(y_new.reshape(y_new.size,1))
        results["outlier_percent_X"].append(np.round(1.0 - (test1[test1==1].size / test1.size),2))
        results["outlier_percent_y"].append(np.round(1.0 - (test2[test2==1].size / test2.size),2))
    
    return pd.DataFrame(results)
    
    
if __name__ == "__main__":
    
    run_start = time.time()
  
    ## monitor model
    result = model_monitor(dev=DEV)
    
    print(result)
    
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...running time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
    
