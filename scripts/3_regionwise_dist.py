import pandas as pd, numpy as np, nibabel as nib
import xml.etree.ElementTree as ET

data_dir = '/home/raid3/oligschlager/workspace/tractdist/data'
yerkes_dir = '/nobackup/hunte1/sabine/data/macaque/Yerkes19'

# data
df = pd.read_pickle('%s/df_pairwise.pkl' % data_dir)
areas_annot = pd.read_csv('%s/downloads/M132LH/areas.csv' % data_dir, header=None).values[:,1]
areas_key = pd.read_csv('%s/downloads/M132LH/key.csv' % data_dir, header=None).values[:,1]
cort = np.array([n for n, val in enumerate(areas_annot) if val != np.where(areas_key == 'MedialWall')[0]])


# dataframe with measure per region
df_regionwise = pd.DataFrame.from_dict(data = {'area': df.source.unique()})


# vertices of injections sites
f = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/data/M132.L.InjSites_20150701.32k_fs.wb.foci' % yerkes_dir

tree = ET.parse(f)
root = tree.getroot()

areas = list()
nodes = list()

for t in root.findall('Focus'):
    areas.append(t[5].text[:t[5].text.index('_')])
    #nodes.append(t[16][3][1].text.split())
    nodes.append([int(n) for n in t[16][3][1].text.split()])
    
df_inj = pd.DataFrame({'area' : areas, 'injection sites' : nodes})


for area in set(df_inj.area[df_inj.area.duplicated()]):
    
    l = list()
    for nodes in df_inj['injection sites'][df_inj.area == area]:
        l.extend(nodes)
    df_inj['injection sites'].iloc[df_inj.index[df_inj.area == area][0]] = list(set(l))
    
df_inj.drop(df_inj.index[df_inj.area.duplicated()], inplace=True)
df_inj.replace({'area': {'9-46d': '9_46d',
                         '9-46v': '9_46v'}}, inplace=True)

df_inj.set_index([range(len(df_inj.index))], inplace=True)

df_regionwise = pd.merge(df_regionwise, df_inj, how='left')
del df_inj



# Yerkes19 surface
mwall = np.load('%s/medial_wall.lh.label.npy' % yerkes_dir)
cort = np.array([node for node in range(32492) if node not in mwall])
gdist_mat = np.load('/nobackup/hunte1/sabine/data/tractdist/yerkes19_gdist_node-by-node.npy')

f_M132 = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/data/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.nii' % yerkes_dir
areas_annot = nib.load(f_M132).get_data().squeeze()[:28261]
areas_key_yerkes = np.load('/nobackup/hunte1/sabine/data/macaque/Yerkes19/M132_key.npy')



# vertices of each region
df_regionwise['vertices'] = np.nan
vertices = list()

for area in df_regionwise.area:
    area_idx = np.where(areas_key_yerkes == area)[0][0]
    area_nodes = cort[np.where(areas_annot == area_idx)[0]]
    vertices.append(list(area_nodes))
df_regionwise.vertices = vertices



# distance from pimary
primary_regions = list(df_regionwise.area[df_regionwise.area.isin(['V1', 'F1', 
                                                                   'Core', '3'])])

# minimum distance of injection from nearest primary cortex
df_regionwise['nearest primary region from injection site'] = np.nan
df_regionwise['injection distance from nearest primary region'] = np.nan

for idx in df_regionwise[~df_regionwise['injection sites'].isnull()].index:
    inj_site = df_regionwise['injection sites'].iloc[idx]
    dists = list()
    for primary_nodes in df_regionwise.vertices[df_regionwise.area.isin(primary_regions)]:
        dists.append(gdist_mat[inj_site,:][:,primary_nodes].min(axis=1).mean())

    df_regionwise['nearest primary region from injection site'].iloc[idx] = primary_regions[np.argmin(np.array(dists))]
    df_regionwise['injection distance from nearest primary region'].iloc[idx] = np.min(np.array(dists))
    

# average minimum distance from all vertices within region to closest primary cortex
df_regionwise['average nearest primary region'] = np.nan
df_regionwise['average distance from nearest primary region'] = np.nan

for idx in df_regionwise[~df_regionwise['vertices'].isnull()].index:
    vertices = df_regionwise['vertices'].iloc[idx]
    dists = list()
    for primary_nodes in df_regionwise.vertices[df_regionwise.area.isin(primary_regions)]:
        dists.append(gdist_mat[vertices,:][:,primary_nodes].min(axis=1).mean())
        
    df_regionwise['average nearest primary region'].iloc[idx] = primary_regions[np.argmin(np.array(dists))]
    df_regionwise['average distance from nearest primary region'].iloc[idx] = np.min(np.array(dists))

       
        
# distance from limbic

limbic_regions = list(df_regionwise.area[df_regionwise.area.isin(['Ento', 'Peri', 
                                                                  'TEMPORAL-POLE',
                                                                  'Pi', '24a', '25',
                                                                  'Sub', '29_30',
                                                                  '32', 'Pro.'])])

# minimum distance of injection from nearest limbic cortex
df_regionwise['nearest limbic region from injection site'] = np.nan
df_regionwise['injection distance from nearest limbic region'] = np.nan

for idx in df_regionwise[~df_regionwise['injection sites'].isnull()].index:
    inj_site = df_regionwise['injection sites'].iloc[idx]
    dists = list()
    for limbic_nodes in df_regionwise.vertices[df_regionwise.area.isin(limbic_regions)]:
        dists.append(gdist_mat[inj_site,:][:,limbic_nodes].min(axis=1).mean())

    df_regionwise['nearest limbic region from injection site'].iloc[idx] = limbic_regions[np.argmin(np.array(dists))]
    df_regionwise['injection distance from nearest limbic region'].iloc[idx] = np.min(np.array(dists))

    
# average minimum distance from all vertices within region to closest limbic cortex
df_regionwise['average nearest limbic region'] = np.nan
df_regionwise['average distance from nearest limbic region'] = np.nan

for idx in df_regionwise[~df_regionwise['vertices'].isnull()].index:
    vertices = df_regionwise['vertices'].iloc[idx]
    dists = list()
    for limbic_nodes in df_regionwise.vertices[df_regionwise.area.isin(limbic_regions)]:
        dists.append(gdist_mat[vertices,:][:,limbic_nodes].min(axis=1).mean())
        
    df_regionwise['average nearest limbic region'].iloc[idx] = limbic_regions[np.argmin(np.array(dists))]
    df_regionwise['average distance from nearest limbic region'].iloc[idx] = np.min(np.array(dists))
        


# save
df_regionwise.to_pickle('%s/df_regionwise.pkl' % data_dir)