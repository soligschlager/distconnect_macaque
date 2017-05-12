def plot_inj_yerkes(df, var, vmin=None, vmax=None):
    
    import pandas as pd, numpy as np, nibabel as nib
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotting_lighterbg as plot
    
    sns.set_style('white')
    #sns.set_context('notebook')
    
    yerkes_dir ='/nobackup/hunte1/sabine/data/macaque/Yerkes19'

    # macaque surface
    f_surf = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/spec/MacaqueYerkes19.L.very_inflated.32k_fs_LR.surf.gii' % yerkes_dir
    coords = nib.load(f_surf).darrays[0].data
    faces = nib.load(f_surf).darrays[1].data
    surf = tuple((coords, faces))
    f_sulc = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/spec/MacaqueYerkes19.sulc.32k_fs_LR.dscalar.nii' % yerkes_dir
    sulc = nib.load(f_sulc).get_data().squeeze()[:32492]

    # labels
    mwall = np.load('%s/medial_wall.lh.label.npy' % yerkes_dir)
    cort = np.array([node for node in range(32492) if node not in mwall])
    areas_f = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/data/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.nii' % yerkes_dir
    areas_annot = nib.load(areas_f).get_data().squeeze()[:28261]
    areas_key = np.load('/nobackup/hunte1/sabine/data/macaque/Yerkes19/M132_key.npy')
    
    data = np.zeros((32492))

    for i in df.index:
        try:

            nodes = list()

            for node in df['injection sites'].iloc[i]:
                neighbors = np.unique(faces[np.where(np.in1d(faces.ravel(), 
                                                         [node]).reshape(faces.shape))[0]])
                nodes.extend(neighbors)
            l = list(set(nodes))

            for node in l:
                neighbors = np.unique(faces[np.where(np.in1d(faces.ravel(), 
                                                         [node]).reshape(faces.shape))[0]])
                nodes.extend(neighbors)
            nodes = np.unique(nodes)

            data[nodes] = df[var].iloc[i]
        except:
            None

    for azim in [180,0]:
        img = plot.plot_surf_stat_map(coords, faces, data, 
                                      bg_map=sulc, bg_on_stat=True,
                                      mask=np.where(data>0)[0], 
                                      azim=azim,
                                      cmap='inferno',
                                      vmin=vmin, vmax=vmax)
        plt.show()




def plot_yerkes(df, var, vmin=None, vmax=None, mask=None, cmap=None):
    
    import pandas as pd, numpy as np, nibabel as nib
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotting_lighterbg as plot
    
    sns.set_style('white')
    #sns.set_context('notebook')
    
    yerkes_dir ='/nobackup/hunte1/sabine/data/macaque/Yerkes19'

    # macaque surface
    f_surf = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/spec/MacaqueYerkes19.L.very_inflated.32k_fs_LR.surf.gii' % yerkes_dir
    coords = nib.load(f_surf).darrays[0].data
    faces = nib.load(f_surf).darrays[1].data
    surf = tuple((coords, faces))
    f_sulc = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/spec/MacaqueYerkes19.sulc.32k_fs_LR.dscalar.nii' % yerkes_dir
    sulc = nib.load(f_sulc).get_data().squeeze()[:32492]

    # labels
    mwall = np.load('%s/medial_wall.lh.label.npy' % yerkes_dir)
    cort = np.array([node for node in range(32492) if node not in mwall])
    areas_f = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/data/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.nii' % yerkes_dir
    areas_annot = nib.load(areas_f).get_data().squeeze()[:28261]
    areas_key = np.load('/nobackup/hunte1/sabine/data/macaque/Yerkes19/M132_key.npy')
    
    data = np.zeros((32492))
    
    for n in range(92):
        
        area = areas_key[n]
        if area in list(df.area):
            if ~np.isnan(df.iloc[df[df.area == area].index[0]][var]):
                
                data[cort[np.where(areas_annot == n)[0]]] = df[var][df.area == area]
    
    if vmin == None:
        vmin = data[cort][np.nonzero(data[cort])].min()
    if vmax == None:
        vmax = data.max()
        
        
    if mask == None:    
        mask = cort.copy()
        for area in df['area'][df[var].isnull()]:
            mask = np.delete(mask,
                             np.searchsorted(mask, cort[np.where(areas_annot == np.where(areas_key == area)[0][0])[0]]))
            
    if cmap == None:
        cmap = 'inferno'
        
    for azim in [180,0]:
        img = plot.plot_surf_stat_map(coords, faces, data, 
                                      bg_map=sulc, bg_on_stat=True,
                                      mask=mask, 
                                      azim=azim,
                                      cmap=cmap,
                                      vmin=vmin, vmax=vmax)
        plt.show()


        
        

def plot_M132(df, var, inflated=True, vmin=None, vmax=None):
    
    import pandas as pd, numpy as np, nibabel as nib
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotting_lighterbg as plot
    
    sns.set_style('white')
    #sns.set_context('notebook')
    
    data_dir = '/home/raid3/oligschlager/workspace/tractdist/data'

    # macaque surface
    if inflated:
        surf = nib.freesurfer.read_geometry('%s/downloads/M132LH/Core-Nets_M132LH.inflated' % data_dir)
        coords = nib.freesurfer.read_geometry('%s/downloads/M132LH/Core-Nets_M132LH.inflated' % data_dir)[0]
    else:
        surf = nib.freesurfer.read_geometry('%s/downloads/M132LH/Core-Nets_M132LH.surf' % data_dir)
        coords = nib.freesurfer.read_geometry('%s/downloads/M132LH/Core-Nets_M132LH.surf' % data_dir)[0]
    faces = nib.freesurfer.read_geometry('%s/downloads/M132LH/Core-Nets_M132LH.surf' % data_dir)[1]
    sulc = nib.freesurfer.read_morph_data('%s/downloads/M132LH/Core-Nets_M132LH.sulc' % data_dir)

    # labels
    areas_annot = pd.read_csv('%s/downloads/M132LH/areas.csv' % data_dir, header=None).values[:,1]
    areas_key = pd.read_csv('%s/downloads/M132LH/key.csv' % data_dir, header=None).values[:,1]
    cort = np.array([n for n, val in enumerate(areas_annot) if val != np.where(areas_key == 'MedialWall')[0]])
    

    data = np.zeros((coords.shape[0]))
    
    for n in range(92):
        area = areas_key[n]
        data[np.where(areas_annot == n)[0]] = df[var][df.area == area]

    if vmin == None:
        vmin = data[cort][np.nonzero(data[cort])].min()
    if vmax == None:
        vmax = data.max()

    for azim in [0, 180]:
        img = plot.plot_surf_stat_map(coords, faces, data, 
                                      bg_map=sulc, bg_on_stat=True,
                                      mask=cort, 
                                      azim=azim,
                                      cmap='inferno',
                                      vmin=vmin, vmax=vmax)
        plt.show()

