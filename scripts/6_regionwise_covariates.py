import pandas as pd, numpy as np

df_r = pd.read_pickle('/home/raid3/oligschlager/workspace/tractdist/data/df_regionwise.pkl')
df_p = pd.read_pickle('/home/raid3/oligschlager/workspace/tractdist/data/df_pairwise.pkl')


# region size
df_r['region_size'] = np.nan

for area in df_r.area:
    n = len(list(df_r['vertices'][df_r.area == area])[0])
    df_r['region_size'][df_r.area == area] = n


    
# location of injection site within the cortex
yerkes_dir ='/nobackup/hunte1/sabine/data/macaque/Yerkes19'
dist_mat = np.load('/nobackup/hunte1/sabine/data/tractdist/yerkes19_gdist_node-by-node.npy')
mwall = np.load('%s/medial_wall.lh.label.npy' % yerkes_dir)
cort = np.array([node for node in range(32492) if node not in mwall])

df_r['location'] = np.nan
for i in df_r.index:   
    inj_nodes = df_r['injection sites'].iloc[i]
    if type(inj_nodes) == list:
        df_r['location'].iloc[i] = dist_mat[inj_nodes][:,cort].astype('float32').mean()

        

# location of injection site within the cortex
gdist_mat = np.load('/nobackup/hunte1/sabine/data/tractdist/yerkes19_eucldist_node-by-node.npy')

df_r['location_eucl'] = np.nan
for i in df_r.index:   
    inj_nodes = df_r['injection sites'].iloc[i]
    if type(inj_nodes) == list:
        df_r['location_eucl'].iloc[i] = dist_mat[inj_nodes][:,cort].astype('float32').mean()        
del dist_mat        
        
      
        
# region's average geodesic distance from injection sites (covariate for outgoing)
df_r['temp'] = [list() for _ in df_r.index]

for i in df_r.index:
    src = df_r.area.iloc[i]
    df_r['temp'].iloc[i] = list(df_p['GDIST'][(df_p.source == src) &
                                              (~df_p['GDIST'].isnull())])
df_r['gdist_from_injs'] = np.nan
for i in df_r.index:
    df_r['gdist_from_injs'].iloc[i] = np.mean(df_r['temp'].iloc[i]) 
    
df_r.drop('temp', axis=1, inplace=True)



# region's average euclidean distance from injection sites (covariate for outgoing)
df_r['temp'] = [list() for _ in df_r.index]

for i in df_r.index:
    src = df_r.area.iloc[i]
    df_r['temp'].iloc[i] = list(df_p['EuclDIST'][(df_p.source == src) &
                                              (~df_p['EuclDIST'].isnull())])
df_r['edist_from_injs'] = np.nan
for i in df_r.index:
    df_r['edist_from_injs'].iloc[i] = np.mean(df_r['temp'].iloc[i]) 
    
df_r.drop('temp', axis=1, inplace=True)
    
    
    
# save
df_r.to_pickle('/home/raid3/oligschlager/workspace/tractdist/data/df_regionwise.pkl')