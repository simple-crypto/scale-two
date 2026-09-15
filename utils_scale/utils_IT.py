import numpy as np 
import tqdm
from utils_scale import utils_eval, utils_ta, utils_infometrics, test_scale

def pi_LDA_multi_TA(train_trs, train_labels, test_trs, test_labels, npois, ndim, nclasses=256, chunk_size=1000):
    """
    train_trs: the training traces as an array of shape (train_ntraces, nsamples)
    train_labels: training labels (train_ntraces, nvars)
    test_trs: the training traces as an array of shape (test_ntraces, nsamples)
    test_labels: test labels (test_ntraces, nvars)
    npois: amount of pois used by the LDA
    ndim: amount of dimensions of the linear subspace projection. 
    test_labels: test labels (test_ntraces, nvars)
    return: (pis, lprobs) where
        - `pis` is a vector of shape (nvars,) containing the PI computed for each variable
        - `lprobas` is a matrix of shape (nvars, test_ntraces) containing the log2 proba used to compute the PI
    """
    # SNR 
    pois = utils_ta.POI_selection_SNR(train_trs, train_labels, nclasses)
    # Compute the models
    if npois==1:
        upois = pois[:, :npois]#[:,np.newaxis] CHECK
    else:
        upois = pois[:,:npois]
    models = utils_ta.multivariate_gaussian_models(train_trs, train_labels, upois, ndim, nclasses=nclasses, chunk_size=chunk_size)
    # Perform the IT computation 
    MPIC = utils_infometrics.MultiPIComputer(nclasses, test_labels.shape[1])
    csizes = utils_ta.resolve_chunk_size(test_trs.shape[0], chunk_size)
    off = 0
    for cs in csizes:
        # Compute log2 prob 
        log2pr = models.predict_log2_proba_class(
            test_trs[off:off+cs].astype(np.int16),
            test_labels[off:off+cs].astype(np.uint16)
        )
        MPIC.fit_from_log2_proba_class(log2pr)
        off += cs

    # Compute the PI
    return MPIC


def pi_LDA_multi_TA_RSI(train_trs, train_labels_rsi, train_labels_interns, test_trs, test_labels, params_rsi, params_interns, nclasses=256, chunk_size=1000):
    # Build model
    models = utils_ta.mlda_RSI_model_fit(
        train_trs,
        train_labels_rsi,
        train_labels_interns,
        params_rsi,
        params_interns,
        chunk_size=chunk_size
    )
    # Perform the IT computation 
    MPIC = utils_infometrics.MultiPIComputer(nclasses, test_labels.shape[1])
    csizes = utils_ta.resolve_chunk_size(test_trs.shape[0], chunk_size)
    off = 0
    for cs in csizes:
        # Compute log2 prob 
        log2pr = models.predict_log2_proba_class(
            test_trs[off:off+cs].astype(np.int16),
            test_labels[off:off+cs].astype(np.uint16)
        )
        MPIC.fit_from_log2_proba_class(log2pr)
        off += cs

    # Compute the PI
    return MPIC

def ti_LDA_multi_TA(train_trs, train_labels, npois, ndim, nclasses=256, chunk_size=1000):
    """
    train_trs: the training traces as an array of shape (train_ntraces, nsamples)
    train_labels: training labels (train_ntraces, nvars)
    npois: amount of pois used by the LDA
    ndim: amount of dimensions of the linear subspace projection. 
    return: (pis, lprobs) where
        - `pis` is a vector of shape (nvars,) containing the PI computed for each variable
        - `lprobas` is a matrix of shape (nvars, test_ntraces) containing the log2 proba used to compute the PI
    """
    return pi_LDA_multi_TA(train_trs, train_labels, train_trs, train_labels, npois, ndim, nclasses=nclasses, chunk_size=chunk_size)

def ti_LDA_multi_TA_RSI(train_trs, train_labels_rsi, train_labels_interns, train_labels, params_rsi, params_interns, nclasses=256, chunk_size=1000):
    return pi_LDA_multi_TA_RSI(train_trs, train_labels_rsi, train_labels_interns, train_trs, train_labels, params_rsi, params_interns, nclasses=nclasses, chunk_size=chunk_size)

def explore_params_LDA(traces, labels, ntraces_pi, explo_npois, explo_ndims, q_t=None, nclasses=256, chunk_size=1000):
    # Verify the shape 
    assert traces.shape[0] == labels.shape[0], "Mismatch between traces and labels shape"
    if q_t is not None:
        assert q_t <= traces.shape[0] - ntraces_pi

    return utils_eval.explore_params(pi_LDA_multi_TA, ti_LDA_multi_TA, traces, labels, ntraces_pi, explo_npois, explo_ndims, qp=q_t, nclasses=nclasses, chunk_size=chunk_size)

def identify_best_params(explo_res):
    (e_npois, e_ndims, exp_pis, _) = explo_res
    nvs = exp_pis.shape[0]
    params_set = []
    for vid in range(nvs):
        [npois_loc, ndim_loc] = np.unravel_index(np.nanargmax(exp_pis[vid]), exp_pis[vid].shape)
        params_set.append((vid, e_npois[npois_loc], e_ndims[ndim_loc]))
    return params_set

def compute_PI_curves(training_traces, training_labels, test_traces, test_labels, qt_s, param_sets, nclasses=256, chunk_size=1000):
    # First create the mapping of parameters set to enable efficient grouping of same params
    psets = {}
    for (vi, npois, ndim) in param_sets:
        if (npois, ndim) in psets:
            psets[(npois, ndim)] += [vi]
        else:
            psets[(npois, ndim)] = [vi]

    # Then, iterate over the different paramset and compute the PI
    nv = training_labels.shape[1]
    pis = np.zeros([nv, len(qt_s)])
    mB = np.zeros([nv, len(qt_s)])
    MB = np.zeros([nv, len(qt_s)])
    for (npois, ndims), vis in tqdm.tqdm(psets.items(), total=len(psets), desc="PI curves"):
        # Create the wrapped PI function
        w_pi_method = lambda a,b,c,d: pi_LDA_multi_TA(a,b,c,d, npois, ndims, nclasses=nclasses, chunk_size=chunk_size)
        # Compute the PIs
        res = test_scale.compute_pi_estimations(
            training_traces,
            training_labels[:,vis],
            test_traces,
            test_labels[:,vis],
            w_pi_method,
            qt_s
        )
        pis[vis, :] = res["it"]
        mB[vis, :] = res["mB"]
        MB[vis, :] = res["MB"]
    return dict(
        dtype="PI",
        qt_s=qt_s,
        it=pis,
        mB=mB,
        MB=MB
    )

def compute_PI_curves_RSI(train_trs, train_labels_rsi, train_labels_interns, test_trs, test_labels, qt_s, param_sets_rsi, param_sets_interns, nclasses=256, chunk_size=1000):
    # Create the wrapped PI function
    w_pi_method = lambda a,b,c,d,e: pi_LDA_multi_TA_RSI(a,b,c,d,e, param_sets_rsi, param_sets_interns, nclasses=nclasses, chunk_size=chunk_size)
    # Compute the PIs
    res = test_scale.compute_pi_estimations_RSI(
        train_trs,
        train_labels_rsi,
        train_labels_interns,
        test_trs,
        test_labels,
        w_pi_method,
        qt_s
    )
    return dict(
        dtype="PI",
        qt_s=qt_s,
        it=res["it"],
        mB=res["mB"],
        MB=res["MB"]
    )
    
def compute_TI_curves(training_traces, training_labels, qt_s, param_sets, nclasses=256, chunk_size=1000):
    # First create the mapping of parameters set to enable efficient grouping of same params
    psets = {}
    for (vi, npois, ndim) in param_sets:
        if (npois, ndim) in psets:
            psets[(npois, ndim)] += [vi]
        else:
            psets[(npois, ndim)] = [vi]

    # Then, iterate over the different paramset and compute the PI
    nv = training_labels.shape[1]
    tis = np.zeros([nv, len(qt_s)])
    mB = np.zeros([nv, len(qt_s)])
    MB = np.zeros([nv, len(qt_s)])
    for (npois, ndims), vis in tqdm.tqdm(psets.items(), total=len(psets), desc="TI curves"):
        # Create the wrapped PI function
        w_ti_method = lambda a,b: ti_LDA_multi_TA(a,b, npois, ndims, nclasses=nclasses, chunk_size=chunk_size)
        # Compute the PIs
        res = test_scale.compute_ti_estimations(
            training_traces,
            training_labels[:,vis],
            w_ti_method,
            qt_s
        )
        tis[vis, :] = res["it"]
        mB[vis, :] = res["mB"]
        MB[vis, :] = res["MB"]
    return dict(
        dtype="TI",
        qt_s=qt_s,
        it=tis,
        mB=mB,
        MB=MB
    )

def compute_TI_curves_RSI(train_trs, train_labels_rsi, train_labels_interns, train_labels, qt_s, param_sets_rsi, param_sets_interns, nclasses=256, chunk_size=1000):
    # Create the wrapped PI function
    w_ti_method = lambda a,b,c,d: ti_LDA_multi_TA_RSI(a,b,c,d, param_sets_rsi, param_sets_interns, nclasses=nclasses, chunk_size=chunk_size)
    # Compute the PIs
    res = test_scale.compute_ti_estimations_RSI(
        train_trs,
        train_labels_rsi,
        train_labels_interns,
        train_labels,
        w_ti_method,
        qt_s
    )
    return dict(
        dtype="TI",
        qt_s=qt_s,
        it=res["it"],
        mB=res["mB"],
        MB=res["MB"]
    )
