import numpy as np
from numpy._core.numeric import normalize_axis_index
from utils_scale import utils_files, utils_aes
import matplotlib.pyplot as plt
from scalib.modeling import Lda, LdaAcc
from scalib.metrics import SNR

from scalib.postprocessing import rank_accuracy

def gaussian_pdf(xs, mean, std):
    coef = 1/(std*np.sqrt((2*np.pi)))
    return coef * np.exp(-(xs-mean)**2 / (2*(std**2)))

def ref_snr_scalib(traces, classes, nclasses):    
    snrobj = SNR(nc=nclasses)
    snrobj.fit_u(
        traces.astype(np.int16),
        classes.astype(np.uint16),
    )
    return snrobj.get_snr()

def POI_selection_SNR(traces, classes, nclasses):
    snrs = ref_snr_scalib(traces, classes, nclasses)
    return np.flip(np.argsort(snrs,axis=1),axis=1)

def univariate_gaussian_models(traces, classes, pois):
    # Allocate return value
    us = np.zeros([classes.shape[1],256])
    ss = np.zeros([classes.shape[1],256])
    
    # Iterate over every variables
    for si in range(classes.shape[1]):
        # Allocate memory squared traces accumulator (used for pooled variance computation)
        sum_sq_center_traces = 0

        # Iterate over all the possible bytes values
        for b in range(256):
            # Compute class mean
            us[si,b] = np.mean(traces[classes[:,si] == b, pois[si]])
            
            # Compute the centered traces
            ctraces = traces[classes[:,si] == b, pois[si]] - us[si,b]
            
            # Accumulate the squared centered traces
            sum_sq_center_traces += np.sum(np.square(ctraces),axis=0)
            
        # Second, compute the pooled variance
        ss[si,:] = np.sqrt((sum_sq_center_traces / (traces.shape[0]-1)))
    # Return
    return (us, ss)
    ###ANSWER_STOP

def log2Pr_class(traces, models):
    # Allocate log-probabilities matrix
    probas = np.zeros([models[0].shape[0],traces.shape[0],models[0].shape[1]])
    for vi, (vtrs, vus, vss) in enumerate(zip(traces.T, models[0], models[1])):
        # Compute the raw density values
        for ci in range(models[0].shape[1]):
            probas[vi, :, ci] = gaussian_pdf(vtrs, vus[ci], vss[ci])
        # Normalize
        probas[vi] = probas[vi] / np.sum(probas[vi],axis=1)[:,np.newaxis]
    return np.log2(probas)

def maximum_likelihood(pts, log2pr_sb):
    lprobas = np.zeros([pts.shape[1], log2pr_sb.shape[2]])
    for vi, (vpt, logpr) in enumerate(zip(pts.T, log2pr_sb)):
        for ki in range(log2pr_sb.shape[2]):
            ist = utils_aes.Sbox[vpt ^ ki]
            lprobas[vi, ki] = np.sum(logpr[np.arange(pts.shape[0]), ist])
    # Normalization
    # Scaling to avoid numerical instabilities
    max_log = np.max(lprobas,axis=1)[:,np.newaxis]
    lprobas = lprobas - max_log 
    # Compute the sum of probas for each key guess
    sum_probas = np.sum(np.exp2(lprobas),axis=1)[:,np.newaxis]
    # Compute the normalized probabilities
    return np.exp2(lprobas - np.log2(sum_probas))

def univariate_TA(traces, pts, pois, models):
    traces_poi = traces[:, pois]
    log2pr_sb = log2Pr_class(traces_poi, models)
    lprobas = maximum_likelihood(pts, log2pr_sb)
    return lprobas

def explore_TA_univariate(dspath_train, dspaths_valid, qp, qas, clean_dataset=False, fn_prof=None):
    ### TRAINING phase
    # First train using the training dataset 
    ds = utils_files.load_dataset(dspath_train, seed_shuffle=0, remove_first=clean_dataset)
    # Fetch all dataset if no training complexity provided
    if qp is None:
        qpu = ds['traces'].shape[0]
    else:
        qpu = qp
    # Compute intermediate states 
    classes = utils_aes.Sbox[ds["pts"][:qpu] ^ ds["ks"][:qpu]]
    # Compute the POIs based on your function
    pois = POI_selection_SNR(ds['traces'][:qpu], classes, 256)

    # Compute the models
    if fn_prof is None:
        models = univariate_gaussian_models(ds['traces'][:qpu], classes[:qpu], pois[:,0])
    else:
        models = fn_prof(ds['traces'][:qpu], classes[:qpu], pois[:,0])
    
    ### ONLINE Phase
    # Allocate memory for the results
    corrprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        ])
    allprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        256
        ])
    correct_kbytes = np.zeros([len(dspaths_valid),ds["pts"].shape[1]],dtype=np.uint8)

    for dsi, dsp in enumerate(dspaths_valid):
        # Load the dataset
        ds = utils_files.load_dataset(dsp, seed_shuffle=0, remove_first=clean_dataset)
        correct_kbytes[dsi] = ds['ks'][0]
        for qavi, q_a in enumerate(qas):
            # Performs the template
            probas = univariate_TA(ds['traces'][:q_a], ds['pts'][:q_a], np.array([pois[:,0]]), models)
            allprobs[dsi, qavi] = probas.copy()
            for vi, kc in enumerate(correct_kbytes[dsi]):
                corrprobs[dsi, qavi, vi] = probas[vi,kc]

    return (allprobs, corrprobs, qpu, qas, correct_kbytes)

def resolve_chunk_size(n, chunk_size):
    if chunk_size is None:
        return [n]
    elif (n % chunk_size) == 0:
        return (n // chunk_size) * [chunk_size]
    else:
        nfull = n // chunk_size
        remaining = n - (nfull * chunk_size)
        sizes = nfull * [chunk_size]
        return sizes + [remaining]

def multivariate_gaussian_models(traces, classes, pois, ndim, nclasses=256, chunk_size=None):
    lda_acc = LdaAcc(nc=nclasses, pois=pois.tolist())
    if chunk_size is None:
        lda_acc.fit_u(np.round(traces).astype(np.int16), classes.astype(np.uint16))
    else:
        csizes = resolve_chunk_size(traces.shape[0], chunk_size)
        off = 0
        for csize in csizes:
            lda_acc.fit_u(
                np.round(traces[off:off+csize]).astype(np.int16), 
                classes[off:off+csize].astype(np.uint16)
            )
            off += csize
    return Lda(lda_acc, p=ndim)

def multivariate_LDA_TA(traces, pts, models):
    pr_sb = models.predict_proba(np.round(traces).astype(np.int16))
    # Finally, perform the ML
    lprobas = maximum_likelihood(pts, np.log2(pr_sb))
    # return lprobas
    return lprobas

def explore_TA_multivariate(dspath_train, dspaths_valid, qp, qas, npois, ndim, pois=None, fn_prof=None, fn_TA=None, clean_dataset=False):
    # TODO: 
    ### TRAINING phase
    # First train using the training dataset 
    ds = utils_files.load_dataset(dspath_train, seed_shuffle=0, remove_first=clean_dataset)
    # Fetch all dataset if no training complexity provided
    if qp is None:
        qpu = ds['traces'].shape[0]
    else:
        qpu = qp
    # Compute intermediate states 
    classes = utils_aes.Sbox[ds["pts"][:qpu] ^ ds["ks"][:qpu]]
    # Compute the POIs based on your function
    if pois is None:
        poisu = POI_selection_SNR(ds['traces'][:qpu], classes, 256)[:,:npois]
    else:
        poisu = pois
    # Compute the models
    if fn_prof is None:
        models = multivariate_gaussian_models(ds['traces'][:qpu], classes[:qpu], poisu, ndim)
    else:
        models = fn_prof(ds['traces'][:qpu], classes[:qpu], poisu, ndim)

    
    ### ONLINE Phase
    # Allocate memory for the results
    corrprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        ])
    allprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        256
        ])
    correct_kbytes = np.zeros([len(dspaths_valid),ds["pts"].shape[1]],dtype=np.uint8)

    for dsi, dsp in enumerate(dspaths_valid):
        # Load the dataset
        ds = utils_files.load_dataset(dsp, seed_shuffle=0, remove_first=clean_dataset)
        correct_kbytes[dsi] = ds['ks'][0]
        for qavi, q_a in enumerate(qas):
            # Performs the template
            if fn_TA is None:
                probas = multivariate_LDA_TA(ds['traces'][:q_a], ds['pts'][:q_a], models )
            else:
                probas = fn_TA(ds['traces'][:q_a], ds['pts'][:q_a], models )
                
            allprobs[dsi, qavi, :, :] = probas
            for vi, kc in enumerate(correct_kbytes[dsi]):
                corrprobs[dsi, qavi, vi] = probas[vi,kc]

    return (allprobs, corrprobs, qpu, qas, correct_kbytes)

def explore_TA_multivariate_best(dspath_train, dspaths_valid, qp, qas, param_sets, clean_dataset=False, cropping=None, chunk_size=1000):
    ### TRAINING phase
    # First train using the training dataset 
    ds = utils_files.load_dataset(dspath_train, seed_shuffle=0, remove_first=clean_dataset, cropping=cropping,size=qp)
    # Fetch all dataset if no training complexity provided
    if qp is None:
        qpu = ds['traces'].shape[0]
    else:
        qpu = qp
    # Compute intermediate states 
    classes = utils_aes.Sbox[ds["pts"][:qpu] ^ ds["ks"][:qpu]]
    # Compute the POIs based on your function
    poisu = POI_selection_SNR(ds['traces'][:qpu], classes, 256)

    # First create the mapping of parameters set to enable efficient grouping of same params
    psets = {}
    for (bi, npois, ndim) in param_sets:
        if (npois, ndim) in psets:
            psets[(npois, ndim)] += [bi]
        else:
            psets[(npois, ndim)] = [bi]

    # Compute the models
    models = {}
    for (npois, ndim), bis in psets.items():
        print(f"Fit for {npois} POIs ; {ndim} dim (bytes {bis})")
        model = multivariate_gaussian_models(ds['traces'][:qpu], classes[:qpu,bis], poisu[bis,:npois], ndim, chunk_size=chunk_size)
        models[tuple(bis)] = model
    
    allbis = [e for (e, _, _) in param_sets]

    ### ONLINE Phase
    # Allocate memory for the results
    corrprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        ])
    allprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        256
        ])
    correct_kbytes = np.zeros([len(dspaths_valid),ds["pts"].shape[1]],dtype=np.uint8)

    probas = np.zeros([ds['pts'].shape[1],256])
    for dsi, dsp in enumerate(dspaths_valid):
        # Load the dataset
        ds = utils_files.load_dataset(dsp, seed_shuffle=0, remove_first=clean_dataset, cropping=cropping)
        correct_kbytes[dsi] = ds['ks'][0]
        for qavi, q_a in enumerate(qas):
            # Performs the template
            allbis =[]
            for bis, model in models.items():
                probas[list(bis),:] = multivariate_LDA_TA(ds['traces'][:q_a], ds['pts'][:q_a,list(bis)], model )
                allbis += list(bis)
                
            allprobs[dsi, qavi, allbis, :] = probas[allbis, :]
            for vi, kc in enumerate(correct_kbytes[dsi]):
                corrprobs[dsi, qavi, vi] = probas[vi,kc]

    return (allprobs, corrprobs, qpu, qas, correct_kbytes)

MAP_INDEX=np.array(
    [[(e + idx)%16 for e in range(16)] for idx in range(16)],dtype=int)

MAP_INDEX_REVERT=np.array(
    [[(i-e)%16 for i in range(16)]for e in range(16)],dtype=int
)

def RI_labels_cycles_inplace(pSB, rndi):
    for i, idx in enumerate(rndi):
        pSB[i,:] = pSB[i, MAP_INDEX[idx]]
    

def RI_probas_intermediates(prs_operations, prs_perm):
    """
    prs_operations: 16 x qa x 256 ; probas of the intermediate of the 16 Sbox operations
    prs_perm: qa x 16 ; probas of the random index used 

    out: the recombined probabilities associated to each Sbox output, re-ordered in natural byte order. 
    """
    prs = np.zeros([16, prs_perm.shape[0], 256])
    for qi in range(prs_perm.shape[0]):
        for pi in range(16):
            prs[:, qi, :] += prs_perm[qi,pi] * prs_operations[MAP_INDEX_REVERT[pi], qi, :]
    return prs

def explore_TA_multivariate_best_RI(dspath_train, dspaths_valid, qp, qas, param_sets_cycle, param_set_perm, clean_dataset=False, chunk_size=1000):
    ### TRAINING phase
    # First train using the training dataset 
    ds = utils_files.load_dataset(dspath_train, seed_shuffle=0, remove_first=clean_dataset, load_rnd=True, size=qp)
    # Fetch all dataset if no training complexity provided
    if qp is None:
        qpu = ds['traces'].shape[0]
    else:
        qpu = qp
    ## Compute intermediate states 
    classes_cycles = utils_aes.Sbox[ds["pts"][:qpu] ^ ds["ks"][:qpu]]
    ridxes = ds['rand'][:qpu,[0]] % 16
    RI_labels_cycles_inplace(classes_cycles, ridxes)

    # Compute the POIs 
    pois_cycles = POI_selection_SNR(ds['traces'][:qpu], classes_cycles, 256)
    pois_perm = POI_selection_SNR(ds['traces'][:qpu], ridxes, 16)

    # First create the mapping of parameters set to enable efficient grouping of same params
    psets = {}
    for (ci, npois, ndim) in param_sets_cycle:
        if (npois, ndim) in psets:
            psets[(npois, ndim)] += [ci]
        else:
            psets[(npois, ndim)] = [ci]

    # Compute the models of the variable for each cycle
    models_cycles = {}
    for (npois, ndim), cis in psets.items():
        print(f"Fit for {npois} POIs ; {ndim} dim (cycle {cis})")
        model = multivariate_gaussian_models(ds['traces'][:qpu], classes_cycles[:qpu,cis], pois_cycles[cis,:npois], ndim, nclasses=256, chunk_size=chunk_size)
        models_cycles[tuple(cis)] = model
    # Compute the model for the permutation
    (_, npois_perm, ndims_perm) = param_set_perm[0]
    print(f"Fit for {npois_perm} POIs ; {ndims_perm} dim (Permutation)")
    model_perm = multivariate_gaussian_models(ds['traces'][:qpu], ridxes[:qpu,[0]], pois_perm[[0],:npois_perm], ndims_perm, nclasses=16, chunk_size=chunk_size) 
    
    ### ONLINE Phase
    # Allocate memory for the results
    corrprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        ])
    allprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        256
        ])
    correct_kbytes = np.zeros([len(dspaths_valid),ds["pts"].shape[1]],dtype=np.uint8)

    for dsi, dsp in enumerate(dspaths_valid):
        # Load the dataset
        ds = utils_files.load_dataset(dsp, seed_shuffle=0, remove_first=clean_dataset)
        correct_kbytes[dsi] = ds['ks'][0]
        for qavi, q_a in enumerate(qas):
            probas_operations = np.zeros([16,q_a,256])
            # Compute the probas for the variables at each operation
            for cis, model in models_cycles.items():
                probas_operations[list(cis),:] = model.predict_proba(ds['traces'][:q_a].astype(np.int16))
            # Compute the probability of the permutation value 
            probas_perms = model_perm.predict_proba(ds['traces'][:q_a].astype(np.int16))
            # Compute the probas associated to each intermediate state 
            probas_interms = RI_probas_intermediates(probas_operations, probas_perms[0])
            probas = maximum_likelihood(ds['pts'][:q_a,:],np.log2(probas_interms))
            allprobs[dsi, qavi, :, :] = probas
            for vi, kc in enumerate(correct_kbytes[dsi]):
                corrprobs[dsi, qavi, vi] = probas[vi,kc]

    return (allprobs, corrprobs, qpu, qas, correct_kbytes)


#########

MY_COLORS = [
         "xkcd:blue",
         "xkcd:green",
         "xkcd:red",
         "xkcd:orange",
         "xkcd:pink",
         ]

def tipping_point(prs, correct):
    runner = prs.shape[0]-1
    while np.argmax(prs[runner])==correct:
        if runner==0:
            break
        else:
            runner -= 1
    return runner+1

def display_explore_TA_univariate_result(res, use_colors=False):
    # Unpack
    (allprobs, corrprobs, qp, qas, correct_kbytes) = res

    # Print some stats for the first results
    for vi in range(allprobs.shape[2]):
        tp = tipping_point(allprobs[0,:,vi,:],correct_kbytes[0,vi])
        print(f'Byte {vi}: {tp} traces required')

    # Plot the res
    scale=0.6
    ax_sx_inch = 5*scale
    ax_sy_inch = 3*scale
    figsize=(4*ax_sx_inch, 4*ax_sy_inch)
    f = plt.figure(figsize=figsize)

    amval = corrprobs.shape[0] 
    nc = allprobs.shape[3]
    axes=[]
    print(allprobs.shape)
    for i in range(corrprobs.shape[2]):
        axes.append(f.add_subplot(4,4,i+1))
        # Plot the wrong guess only for the first to easu visualization
        for dsvi in range(1):
            for b in range(nc): 
                axes[i].plot(qas, allprobs[dsvi, :, i, b], color="xkcd:light grey")
        # Plot the valid guess
        for dsvi in range(amval):
            udsvi = amval-1-dsvi
            if use_colors:
                color_c = MY_COLORS[udsvi % len(MY_COLORS)]
                txt_label = "Set {}".format(udsvi)
            else:
                if udsvi==0:
                    color_c = "xkcd:red"
                else:
                    color_c = "black"
                txt_label = None
            plt.plot(qas, corrprobs[udsvi, :, i], color=color_c, label=txt_label)
        axes[i].set_xlabel("attack data complexity")
        axes[i].set_ylabel(r'Pr($k_{}^* | \boldsymbol{{l}}$)'.format(i))
        axes[i].set_title("Byte {}".format(i))      
        if use_colors:
            axes[i].legend()
    plt.show()

def key_rank_approximation_scalib(subkey_probs, correct_subkeys, max_nb_bin=2**18):
    """
    subkey_probs: array of shape (nvars, 256), containing the probabilities associated to the values of 'nvars' subkey bytes.
    correct_subkeys: an array of shape  (nvars,), where the i-th element is the correct value of the i-th subkey. 
    return (rmin, r, rmax), where `r` is the approximated rank of the full key, `rmin` and `rmax` are respectively the minimal and maximal bounds.
    """
    return rank_accuracy(-np.log(subkey_probs), correct_subkeys, max_nb_bin=max_nb_bin)
        
### Multi param model 
class MultiLdaAccParams:
    def __init__(self, params_set, pois, nc):
        self.params = params_set
        self.nc = nc

        # Create the groups 
        psets = {}
        for (id, npois, ndims)  in self.params:
            if (npois, ndims) in psets:
                psets[(npois, ndims)] += [id]
            else:
                psets[(npois, ndims)] = [id]

        # Create the LDA instances 
        self.mods = []
        for (npois, ndims), ids in psets.items():
            lda_acc = LdaAcc(nc=nc, pois=[pois[e][:npois] for e in ids]) 
            self.mods += [[lda_acc, ids, ndims]]

    def fit_u(self, traces, labels):
        for mi in range(len(self.mods)):
            self.mods[mi][0].fit_u(traces, labels[:,self.mods[mi][1]])

class MultiLdaParams:
    def __init__(self, multi_ldaacc_params):
        self.ldas=[]
        self.nc = multi_ldaacc_params.nc
        self.nv = 0
        for [lda_acc, ids, ndims] in multi_ldaacc_params.mods:
            lda = Lda(lda_acc, p=ndims)
            self.nv += len(ids)
            self.ldas += [[lda, ids]]

    def predict_proba(self, traces):
        probas = np.zeros([self.nv, traces.shape[0], self.nc], dtype=np.float64)
        for [lda, ids] in self.ldas:
            probas[ids, :, :] = lda.predict_proba(traces)
        return probas

def mlda_params_fit(traces, labels, params_set, chunk_size=1000, nc=256):
    # Compute the SNR to identify the SNR
    poisu = POI_selection_SNR(traces, labels, nc)

    # Create the Multiparam LdaAcc
    mlda_acc = MultiLdaAccParams(params_set, poisu, nc)

    # Compute the sies to use for chunking
    csizes = resolve_chunk_size(traces.shape[0], chunk_size)

    # Fit
    off = 0
    for cs in csizes:
        mlda_acc.fit_u(
            np.round(traces[off:off+cs,:]).astype(np.int16),
            labels[off:off+cs, :].astype(np.uint16)
        )
        off += cs
    # Return the LDA
    return MultiLdaParams(mlda_acc)

### Shuffling exhaustive perm model 
class ModelRSI:
    def __init__(self, mperm, minterns, chunk_size):
        self.mperm = mperm
        self.minterns = minterns
        self.chunk_size = chunk_size # RESERVED

    def predict_proba(self, traces): 
        prs_perm = self.mperm.predict_proba(traces)
        prs_inters = self.minterns.predict_proba(traces)
        return RI_probas_intermediates(prs_inters, prs_perm[0, :, :])

    def predict_log2_proba_class(self, traces, labels):
        # nv x n x nc
        l2prsall = np.log2(self.predict_proba(traces))
        #
        l2prs = np.zeros([labels.shape[1], labels.shape[0]])
        # 
        for nvi in range(labels.shape[1]):
            for ni in range(labels.shape[0]):
                l2prs[nvi, ni] = l2prsall[nvi, ni, labels[ni, nvi]]
        return l2prs


def mlda_RSI_model_fit(traces, labels_RSI, labels_interns, params_set_RSI, params_set_interns, chunk_size=1000):
    # Fit the model for the RSI
    model_RSI = mlda_params_fit(traces, labels_RSI[:,np.newaxis], params_set_RSI, chunk_size=chunk_size,nc=16)
    # Fit the model for the interns 
    model_inters = mlda_params_fit(traces, labels_interns, params_set_interns, chunk_size=chunk_size, nc=256)
    return ModelRSI(model_RSI, model_inters, chunk_size)


### SASCA related
def explo_SASCA_msk_pSB(patcks, mlda_pSB_shares, qas, f_SASCA, bidx=range(4)):
    corrprobs = np.zeros([
        len(patcks),
        len(qas),
        len(bidx)
    ])
    allprobs = np.zeros([
        len(patcks),
        len(qas),
        len(bidx),
        256
    ])
    correct_kbytes = np.zeros([len(patcks), len(bidx)], dtype=np.uint8)

    for dsi, dsp in enumerate(patcks):
        # Load dataset
        dsa = utils_files.load_dataset(dsp, seed_shuffle=0, remove_first=True, load_msk=True)
        correct_kbytes[dsi] = dsa['ks'][0,bidx]
        for qavi, q_a in enumerate(qas):
            # Perform the SASCA 
            kprobs = f_SASCA(
                np.round(dsa['traces'][:q_a]).astype(np.int16),
                dsa['pts'][:q_a, bidx].astype(np.uint16),
                mlda_pSB_shares
            )
            allprobs[dsi, qavi] = kprobs.copy()
            for vi, kc in enumerate(correct_kbytes[dsi]):
                corrprobs[dsi, qavi, vi] = kprobs[vi, kc]
    return (allprobs, corrprobs, None, qas, correct_kbytes)

def explo_SASCA_msk_pSB_pAK(patcks, mlda_pSB_shares, mlda_pAK_shares, qas, f_SASCA, bidx=range(4)):
    corrprobs = np.zeros([
        len(patcks),
        len(qas),
        len(bidx)
    ])
    allprobs = np.zeros([
        len(patcks),
        len(qas),
        len(bidx),
        256
    ])
    correct_kbytes = np.zeros([len(patcks), len(bidx)], dtype=np.uint8)

    for dsi, dsp in enumerate(patcks):
        # Load dataset
        dsa = utils_files.load_dataset(dsp, seed_shuffle=0, remove_first=True, load_msk=True)
        correct_kbytes[dsi] = dsa['ks'][0,bidx]
        for qavi, q_a in enumerate(qas):
            # Perform the SASCA 
            kprobs = f_SASCA(
                np.round(dsa['traces'][:q_a]).astype(np.int16),
                dsa['pts'][:q_a, bidx].astype(np.uint16),
                mlda_pSB_shares,
                mlda_pAK_shares
            )
            allprobs[dsi, qavi] = kprobs.copy()
            for vi, kc in enumerate(correct_kbytes[dsi]):
                corrprobs[dsi, qavi, vi] = kprobs[vi, kc]
    return (allprobs, corrprobs, None, qas, correct_kbytes)



        


