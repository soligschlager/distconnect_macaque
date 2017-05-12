import pandas as pd, numpy as np

df_regionwise = pd.read_pickle('/home/raid3/oligschlager/workspace/tractdist/data/df_regionwise.pkl')
df_pairwise = pd.read_pickle('/home/raid3/oligschlager/workspace/tractdist/data/df_pairwise.pkl')

df_pairwise['GDIST'] = np.nan
df_pairwise['EuclDIST'] = np.nan
df_pairwise['GDIST min'] = np.nan


# gdist
dist_mat = np.load('/nobackup/hunte1/sabine/data/tractdist/yerkes19_gdist_node-by-node.npy')

for target in df_regionwise.area[~df_regionwise['injection sites'].isnull()]:
    target_nodes = list(df_regionwise['injection sites'][df_regionwise.area == target])[0]
    
    for source in df_pairwise.source[df_pairwise.target == target]:  
        source_nodes = list(df_regionwise['vertices'][df_regionwise.area == source])[0]
        
        dist_mean = dist_mat[target_nodes,:][:,source_nodes].astype('float32').mean()
        dist_min = dist_mat[target_nodes,:][:,source_nodes].astype('float32').min()
        
        df_pairwise['GDIST'][(df_pairwise.source == source) &
                                  (df_pairwise.target == target)] = dist_mean
        df_pairwise['GDIST min'][(df_pairwise.source == source) &
                                  (df_pairwise.target == target)] = dist_min
          

# eucldist
dist_mat = np.load('/nobackup/hunte1/sabine/data/tractdist/yerkes19_eucldist_node-by-node.npy')

for target in df_regionwise.area[~df_regionwise['injection sites'].isnull()]:
    target_nodes = list(df_regionwise['injection sites'][df_regionwise.area == target])[0]
    
    for source in df_pairwise.source[df_pairwise.target == target]:  
        source_nodes = list(df_regionwise['vertices'][df_regionwise.area == source])[0]
        
        dist_mean = dist_mat[target_nodes,:][:,source_nodes].astype('float32').mean()
        
        df_pairwise['EuclDIST'][(df_pairwise.source == source) &
                                  (df_pairwise.target == target)] = dist_mean  
        
        
        
df_pairwise.to_pickle('/home/raid3/oligschlager/workspace/tractdist/data/df_pairwise.pkl')