import numpy as np, nibabel as nib
from surfdist import utils
import gdist
import multiprocessing



data_dir = '/home/raid3/oligschlager/workspace/tractdist/data'
yerkes_dir ='/nobackup/hunte1/sabine/data/macaque/Yerkes19'


# macaque surface
f_surf = '%s/Donahue_et_al_2016_Journal_of_Neuroscience_W336/spec/MacaqueYerkes19.L.midthickness.32k_fs_LR.surf.gii' % yerkes_dir
coords = nib.load(f_surf).darrays[0].data
faces = nib.load(f_surf).darrays[1].data
surf = tuple((coords, faces))

mwall = np.load('%s/medial_wall.lh.label.npy' % yerkes_dir)
cort = np.array([node for node in range(32492) if node not in mwall])

# coords and faces indexed by cortex nodes only
vertices, triangles = utils.surf_keep_cortex(surf, cort)
np.save('%s/yerkes_vertices_in_cort.npy' % data_dir, vertices)
np.save('%s/yerkes_triangles_in_cort.npy' % data_dir, triangles)

# geodesic distance for one row of the node-by-node matrix 
# filling only what's right wrt the source node (upper triangle of matrix)
def faster_gdists(node, cort, vertices, triangles, surf, return_dict):
    idx = np.where(cort == node)[0][0]
    src = utils.translate_src(node, cort)
    trgt = utils.translate_src(cort[idx:], cort)
    temp = np.zeros(len(cort))
    temp[idx:] = gdist.compute_gdist(vertices, triangles, source_indices=src, target_indices=trgt)
    return_dict[node] = utils.recort(temp, surf, cort)  
    

# chunking the the list of cortex nodes into groups of 40 for parallelizing
# reshaping tricky because slicing needs each number twice
# e.g. [0:40], [40:80]
count = 0
counter = list()

while count < len(cort):
    
    if (count+40) >= len(cort):
        counter.append(tuple((count, len(cort))))
        count +=40
    else:
        counter.append(tuple((count, count+40)))
        count += 40
        
        
# computing distances for 40 nodes at a time
# this is necessary so that after each subprocess the memory is freed up again
# if all nodes would be passed to multiprocessing at once, memory builds up
manager = multiprocessing.Manager() 
return_dict = manager.dict()
jobs = []

for bunch in counter:
    for n, node in enumerate(cort[bunch[0]:bunch[1]]):
        
        p = multiprocessing.Process(target = faster_gdists, 
                                    args = (node, cort, vertices, triangles, surf, return_dict))
        jobs.append(p)
        p.start()
    p.join()
    
    
# converting results from dictionary format to array
gdists_mat = np.zeros(shape=(len(coords), len(coords)), dtype=np.float16)

for node in cort:
    gdists_mat[node] = return_dict[node]

# filling the empty lower triangle
gdists_mat = gdists_mat + gdists_mat.T

# saving
# not in workspace due to size
np.save('/nobackup/hunte1/sabine/data/tractdist/yerkes19_gdist_node-by-node.npy', gdists_mat)
np.savez_compressed('/nobackup/hunte1/sabine/data/tractdist/yerkes19_gdist_node-by-node.npz', gdists_mat)
