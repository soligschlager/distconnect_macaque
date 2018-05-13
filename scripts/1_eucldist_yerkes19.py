import numpy as np, nibabel as nib
from scipy.spatial import distance_matrix

yerkes_dir ='/nobackup/hunte1/sabine/data/macaque/Yerkes19'


# macaque surface
f_surf = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/spec/MacaqueYerkes19.L.midthickness.32k_fs_LR.surf.gii' % yerkes_dir
coords = nib.load(f_surf).darrays[0].data

mat = distance_matrix(coords, coords)

np.save('/nobackup/hunte1/sabine/data/tractdist/yerkes19_eucldist_node-by-node.npy', mat)