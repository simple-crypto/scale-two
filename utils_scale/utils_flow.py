from utils_scale import utils_files, utils_aes, utils_plot, utils_IT, utils_ta
import numpy as np
from scalib.metrics import SNR

def full_PI_flow_perm(q_t, q_as, enpois, endims, ptrain, chunk_size=10000, show_heatmap=False, cropping=None):
    # Load training dataset 
    ds = utils_files.load_dataset(ptrain, seed_shuffle=0, remove_first=True, load_rnd=True, cropping=cropping)

    # Compute labels
    labels = (ds["rand"][:,0] & 0xf)[:,np.newaxis]

    # identify best params 
    explos_params = utils_IT.explore_params_LDA(ds['traces'], labels, 2048, enpois, endims, q_t=q_t, chunk_size=chunk_size, nclasses=16)
    if show_heatmap:
        # Display the resulting heatmaps.
        utils_plot.make_heatmap(explos_params)

    # Identify the best param set from the exploration results
    best_param_sets = utils_IT.identify_best_params(explos_params)


def full_TA_flow_pSB(q_t, q_as, enpois, endims, ptrain, patck, chunk_size=10000, show_heatmap=False, bidx=range(16)):
    # Load training dataset 
    ds = utils_files.load_dataset(ptrain, seed_shuffle=0, remove_first=True)

    # Compute labels
    labels = utils_aes.Sbox[ds['pts'] ^ ds['ks']][:,bidx]

    # identify best params 
    explos_params = utils_IT.explore_params_LDA(ds['traces'], labels, 2048, enpois, endims, q_t=q_t, chunk_size=chunk_size)
    if show_heatmap:
        # Display the resulting heatmaps.
        utils_plot.make_heatmap(explos_params)

    # Identify the best param set from the exploration results
    best_param_sets = utils_IT.identify_best_params(explos_params)

    # Delete ds, since no more useful
    del ds

    # Perform the attacks
    atcks_res = utils_ta.explore_TA_multivariate_best(
        ptrain,
        patck,
        q_t,
        q_as,
        best_param_sets,
    )

    # Bytes to consider for ranks computation.
    utils_plot.display_ranks_full_key([atcks_res], bidx)

