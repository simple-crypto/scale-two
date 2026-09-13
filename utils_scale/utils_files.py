import numpy as np
from pathlib import Path


# Version 2026 scale2
RELATIVE_DIR_FILES="./scale2-ds"

DS_CFG={
    "sw-aes_training": f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1/training0/data.npz", 
    "sw-aes_atcks": [f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1/validation{e}/data.npz" for e in range(5)],
    "hw-aes_training": f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-8MHz-HW-AVG1/training0/data.npz", 
    "hw-aes_atcks": [f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-8MHz-HW-AVG1/validation{e}/data.npz" for e in range(5)], 
    "hw-aes-avg10_training": f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-8MHz-HW-AVG10/training0/data.npz", 
    "hw-aes-avg10_atcks": [f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-8MHz-HW-AVG10/validation{e}/data.npz" for e in range(5)], 
    "hw-aes-avg100_training": f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-8MHz-HW-AVG100/training0/data.npz", 
    "hw-aes-avg100_atcks": [f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-8MHz-HW-AVG100/validation{e}/data.npz" for e in range(5)], 
    "sw-aes-RIoff_training": f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-sept-RIno-Os-RI/training0/data.npz",
    "sw-aes-RIoff_atcks": [f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-sept-RIno-Os-RI/validation{e}/data.npz" for e in range(5)],
    "sw-aes-RIon_training": f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-sept-RIon-Os-RI/training0/data.npz",
    "sw-aes-RIon_atcks": [f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-sept-RIon-Os-RI/validation{e}/data.npz" for e in range(5)],
    "sw-aes-SHon_training": f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-sept-Os-FP-RI/training0/data.npz",
    "sw-aes-SHon_atcks": [f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-sept-Os-FP-RI/validation{e}/data.npz" for e in range(5)],
    "sw-aes-MSK_training": f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-sept-Os-pAKSB-MSK/training0/data.npz",
    "sw-aes-MSK_atcks": [f"{RELATIVE_DIR_FILES}/scale2-dataset-26-dev-CT1-sept-Os-pAKSB-MSK/validation{e}/data.npz" for e in range(5)],

}

def I2keep(traces):
    tmpm = np.mean(traces,axis=1)
    mtmpm = np.mean(tmpm)
    th = np.std(tmpm)
    dst = np.abs(tmpm-mtmpm)
    hig = dst>4*th
    kept = np.where(np.logical_not(hig))[0]
    return np.array(kept)

def load_npy(fp, out_dic, out_key, dtype=None):
    with open(fp, "rb") as f:
        if dtype is not None:
            out_dic[out_key] =  np.load(f).astype(dtype)
        else:
            out_dic[out_key] = np.load(f)

def load_dataset(datafile_path, traces_dtype=np.float32, cropping=None, seed_shuffle=None, remove_first=False, load_rnd=False, size=None, extracted_dir="extracted_data", load_msk=False):
    """
    Open to file and load all the data fields from it. 
    """
    ds = dict()
    extracted_ds = Path(datafile_path).name == extracted_dir
    if not(extracted_ds):
        with np.load(datafile_path, mmap_mode="r") as coll:
            ds['traces'] = coll["traces"].astype(traces_dtype)
            ds['pts'] = coll["pts"]
            ds['ks'] = coll["ks"]
            ds['cts'] = coll["cts"]
            if load_rnd:
                ds['rand'] = coll["rand"]
            if load_msk:
                ds["msk_pSB"] = coll["msk_pSB"]
                ds["msk_pAK"] = coll["msk_pAK"]
            del coll
    else:
        load_npy(f"{datafile_path}/traces.npy",ds, 'traces', dtype=traces_dtype)
        load_npy(f"{datafile_path}/pts.npy",ds, 'pts')
        load_npy(f"{datafile_path}/ks.npy",ds, 'ks')
        load_npy(f"{datafile_path}/cts.npy",ds, 'cts')
        if load_rnd:
            load_npy(f"{datafile_path}/rand.npy",ds, 'rand')
        if load_msk:
            load_npy(f"{datafile_path}/msk_pSB.npy",ds, 'msk_pSB')
            load_npy(f"{datafile_path}/msk_pAK.npy",ds, 'msk_pAK')

    if cropping is not None:
        [start, end] = cropping
        ds['traces'] = ds['traces'][:,start:end]
    if remove_first:
        idx = I2keep(ds['traces'])
        ds['traces']=ds['traces'][idx]
        ds['pts']=ds['pts'][idx]
        ds['ks']=ds['ks'][idx]
        ds['cts']=ds['cts'][idx]
        if load_rnd:
            ds['rand']=ds['rand'][idx]
        if load_msk:
            ds['msk_pSB']=ds['msk_pSB'][idx]
            ds['msk_pAK']=ds['msk_pAK'][idx]
    if seed_shuffle is not None:
        apply_permutation_dataset_inplace(ds, seed=seed_shuffle)
    if size is not None:
        for k in ds.keys():
            ds[k] = ds[k][:size,:]
    return ds

def apply_permutation_dataset_inplace(dataset, seed=0):
    N = dataset['traces'].shape[0]
    np.random.seed(seed)
    rp = np.random.permutation(np.arange(N))
    np.random.seed(None)
    for f in dataset.keys():
        dataset[f][:,:] = dataset[f][rp]

def assert_file_exists(f):
    fp = Path(f)
    exp_path = "{}/{}".format(Path.cwd(), f)
    assert fp.is_file(), f"The file '{exp_path}' does not exist. Please verify the location of the dataset." 

#############
def load_datasets_profiled_setting(fp_train, fp_validation):
    # Open the training file and load all the data fields from it. 
    with open(fp_train, 'rb') as f:
        coll = np.load(f, allow_pickle=True)
        training_traces = coll["traces"].astype(np.single)
        training_pts = coll["pts"]
        training_ks = coll["ks"]
    
    # Open the validation file and load all the data fields from it. 
    with open(fp_validation, 'rb') as f:
        coll = np.load(f, allow_pickle=True)
        validation_traces = coll["traces"].astype(np.single)
        validation_pts = coll["pts"]
        validation_ks = coll["ks"]

    # Return the data
    return training_traces, training_pts, training_ks, validation_traces, validation_pts, validation_ks

def filter_cst_vs_random(data, reference):
    Is0 = []
    Is1 = []
    for ri,r in enumerate(data):
        if (r==reference).all():
            Is0.append(ri)
        else:
            Is1.append(ri)
    # Return set of indexes
    return Is0, Is1

def load_ttest_dataset(filepath):
    with open(filepath,"rb") as f:
        coll = np.load(f, allow_pickle=True)
        traces_tt = coll["traces"]
        pts_tt = coll["pts"]
        ks_tt = coll["ks"]
    # Re-identify the class of each traces
    # Look for proper indexes
    vp, cp = np.unique(pts_tt,axis=0, return_counts=True)
    vk, ck = np.unique(ks_tt, axis=0, return_counts=True)
    
    # Switch based on the max value
    if max(cp)>max(ck):
        # filter based on the key
        Is0, Is1 = filter_cst_vs_random(ks_tt, vk[np.argmax(ck)])
    elif max(cp)<max(ck):
        # filter based on the plaintext
        Is0, Is1 = filter_cst_vs_random(pts_tt, vp[np.argmax(cp)])
    else:
        raise ValueError("Not able to distinguish filtering argument")

    labels = np.zeros(pts_tt.shape[0], dtype=np.uint16)
    for i in Is1:
        labels[i]=1

    return traces_tt, pts_tt, ks_tt, labels

def apply_permutation_dataset(traces, pts, ks, seed=0):
    N = traces.shape[0]
    np.random.seed(seed)
    rp = np.random.permutation(np.arange(N))
    np.random.seed(None)
    traces[:,:] = traces[rp]
    pts[:,:] = pts[rp]
    ks[:,:] = ks[rp]
