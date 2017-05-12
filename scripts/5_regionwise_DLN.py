import pandas as pd, numpy as np

df_regionwise = pd.read_pickle('/home/raid3/oligschlager/workspace/tractdist/data/df_regionwise.pkl')
df_pairwise = pd.read_pickle('/home/raid3/oligschlager/workspace/tractdist/data/df_pairwise.pkl')


df_regionwise['incoming'] = np.nan
df_regionwise['incoming_mindist'] = np.nan
df_regionwise['incoming_eucl'] = np.nan

for target in df_regionwise.area[~df_regionwise['injection sites'].isnull()]:
    sources = df_pairwise.source[df_pairwise.target == target]
    
    temp = df_pairwise[(df_pairwise.target == target) &
                   (df_pairwise.source.isin(sources))]
    
    dist = np.sum(temp.NEURONS * temp['GDIST'])/temp.NEURONS.sum()
    df_regionwise['incoming'][df_regionwise.area == target] = dist
    
    dist = np.sum(temp.NEURONS * temp['EuclDIST'])/temp.NEURONS.sum()
    df_regionwise['incoming_eucl'][df_regionwise.area == target] = dist
    
    dist = np.sum(temp.NEURONS * temp['GDIST min'])/temp.NEURONS.sum()
    df_regionwise['incoming_mindist'][df_regionwise.area == target] = dist

    

df_regionwise['outgoing'] = np.nan
df_regionwise['outgoing_eucl'] = np.nan
df_regionwise['outgoing_mindist'] = np.nan

for source in df_regionwise.area:    
    targets = df_pairwise.target[df_pairwise.source == source]
    
    temp = df_pairwise[(df_pairwise.source == source) &
                       (df_pairwise.target.isin(targets))]
    
    dist = np.sum(temp.NEURONS * temp['GDIST'])/temp.NEURONS.sum()
    df_regionwise['outgoing'][df_regionwise.area == source] = dist
    
    dist = np.sum(temp.NEURONS * temp['EuclDIST'])/temp.NEURONS.sum()
    df_regionwise['outgoing_eucl'][df_regionwise.area == source] = dist
    
    dist = np.sum(temp.NEURONS * temp['GDIST min'])/temp.NEURONS.sum()
    df_regionwise['outgoing_mindist'][df_regionwise.area == source] = dist
    

# save
df_regionwise.to_pickle('/home/raid3/oligschlager/workspace/tractdist/data/df_regionwise.pkl')