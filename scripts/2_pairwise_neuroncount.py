import pandas as pd, numpy as np
import itertools

data_dir = '/home/raid3/oligschlager/workspace/tractdist/data' 


############################## import - data ###################################

# creating df for each pair of source/target combination from original data 
# (multiple source/target cases)

df_raw = pd.read_excel('%s/downloads/PNAS_2013.xls' % data_dir)
df_raw.SOURCE = df_raw.SOURCE.apply(str)
df_raw.TARGET = df_raw.TARGET.apply(str)

# df with all possible source-target pairs
c = np.array(list(itertools.product(df_raw.SOURCE.unique(), df_raw.SOURCE.unique()))).T
df = pd.DataFrame({'SOURCE': c[0], 'TARGET': c[1]})

# if multiple entries for a source-target pair, average
# if no entry, cell will be nan
for col in ['NEURONS', 'DISTANCE (mm)']:
    df[col] = np.nan
    for i in range(len(df)):
        df[col].iloc[i] = df_raw[col].loc[(df_raw.SOURCE == df.SOURCE.iloc[i]) & 
                                          (df_raw.TARGET == df.TARGET.iloc[i])].mean()
del df_raw


#################################### labels ######################################

# modifying labelling to match that of surface labels

area_labels = pd.read_csv('%s/downloads/M132LH/key.csv' % data_dir, header=None).values[:,1]

matches = [('PERI', 'Peri'), ('TEa/ma', 'TEa_m-a'), ('TEa/mp', 'TEa_m-p'),
           ('ENTO', 'Ento'), ('INS', 'Ins'), ('9/46d', '9_46d'), ('9/46v', '9_46v'),
           ('Pro.St.', 'Pro.'), ('SII', 'S2'), ('29/30', '29_30'), ('OPRO', 'Opro'),
           ('SUB', 'Sub'), ('PIR', 'Pir'), ('POLE', 'TEMPORAL-POLE')]

# new columns
df['source'] = np.nan
df['target'] = np.nan

for label in area_labels:
    
    # if surface label not found in df, find matching df label and assign surface label to new col  
    if len(df.source.loc[df.SOURCE == label]) == 0:
        try:
            df_area_name = [pair[0] for pair in matches if pair[1] == label][0] 
            df.source.loc[df.SOURCE == df_area_name] = label
        except:
            print label # neither found in df nor in matches
    # if surface label found in df, keep it
    else:
        df.source.loc[df.SOURCE == label] = label
        
    # same for targets
    if len(df.target.loc[df.TARGET == label]) == 0:
        try:
            df_area_name = [pair[0] for pair in matches if pair[1]==label][0] 
            df.target.loc[df.TARGET == df_area_name] = label
        except:
            ''
    else:
        df.target.loc[df.TARGET == label] = label

# retain labelling that matches surface labels        
df.drop(labels=['SOURCE','TARGET'], axis=1, inplace=True)

df.to_pickle('%s/df_pairwise.pkl' % data_dir)