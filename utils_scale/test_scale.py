import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scalib.metrics import SNR
import tqdm
import time

from utils_scale.utils_aes import Sbox
from utils_scale import utils_files

import datetime

# Version 2025
MY_COLORS = [
        "xkcd:blue",
        "xkcd:green",
        "xkcd:red",
        "xkcd:orange",
        "xkcd:pink",
        ]

def print_install_success():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from scalib.metrics import SNR
    from scalib.modeling import Lda, LdaAcc
    from utils_scale.utils_aes import Sbox
    from utils_scale import utils_files, utils_aes, utils_plot, utils_IT, utils_ta, utils_flow
    import datetime
    print("Your installation seems successful :)")

def log_start(test_id, log=True):
    if log:
        print("##### START TEST: {} #####".format(test_id))

def log_end(log=True):
    if log:
        print("##### FINISHED #####")

def TIME_start(label):
    print("{} started...".format(label))
    return (label, datetime.datetime.now())

def TIME_end(tstart_res):
    (label, ts) = tstart_res
    tstop = datetime.datetime.now()
    print("{} done. [{} elapsed]".format(label, tstop-ts))

def styledstr(st, background=None, color=None):
    # reset, MUST be put at the end
    RESET  = "\033[0m",
    # color
    colors = {
            "BLACK"  : "\033[30m",
            "RED"    : "\033[31m",
            "GREEN"  : "\033[32m",
            "YELLOW" : "\033[33m",
            "BLUE"   : "\033[34m",
            "PURPLE" : "\033[35m",
            "CYAN"   : "\033[36m",
            "WHITE"  : "\033[37m",
            }
    
    # background color
    bgcolors = {
            "BLACK"  : "\033[40m",
            "RED"    : "\033[41m",
            "GREEN"  : "\033[42m",
            "YELLOW" : "\033[43m",
            "BLUE"   : "\033[44m",
            "PURPLE" : "\033[45m",
            "CYANB"   : "\033[46m",
            "WHITEB"  : "\033[47m",
    }

    return "{}{}\033[0m".format(bgcolors[background], st)


def ref_POI_selection_SNR(traces, classes, nclasses):
    snrs = ref_snr_scalib(traces, classes, nclasses)
    return np.flip(np.argsort(snrs,axis=1),axis=1)

def bytes2str(v):
    st = ""
    for e in v:
        st = "{:02x}{}".format(e,st)
    return "0x{}".format(st)

def kg2ststr(k, status):
    st = ""
    for e,s in zip(k,status):
        if s:
            st = "{:02x}{}".format(e,st)
        else:
            newc = styledstr("{:02x}".format(e), background="RED")
            st = "{}{}".format(styledstr("{:02x}".format(e), background="RED"),st)
    return "0x{}".format(st)

def display_snr_sbox_output(traces, labels, byte_index, compute_byte_snr, subplots=[]):
    # Compute Sbox output
    ulabels = labels[:,byte_index]
    # Comptue the snr
    snr = compute_byte_snr(traces, ulabels)
    # Print some stuff
    print("Max SNR ({}) found at time index {}.".format(np.max(snr), np.argmax(snr)))
    # Plot the result
    naxes = 2+len(subplots)
    f = plt.figure(figsize=(10,6))
    ax0 = f.add_subplot(naxes,1,1)
    ax1 = f.add_subplot(naxes,1,2)
    ax0.plot(traces[:1,:].T)
    ax1.plot(snr)
    ax0.set_ylabel("Power")
    ax1.set_ylabel("SNR")
    ax0.set_title(r"$y_{{{}}}=\text{{Sbox}}[p_{{{}}} \oplus k_{{{}}}]$".format(byte_index, byte_index, byte_index))
    axes = [ax0,ax1]
    for i, (sub, lab) in enumerate(subplots):
        ax = f.add_subplot(naxes, 1, 3+i)
        metr = sub(traces, ulabels)
        ax.plot(metr)
        ax.set_ylabel(lab)
        axes.append(ax)

    axes[-1].set_xlabel("Time")
    f.tight_layout() 
    plt.show()



def display_snr_sbox_output_SCALIB(traces, labels, indexes):
    # Compute snr
    n_p = labels.shape[1]     
    snr = SNR(256,traces.shape[1],n_p)
    snr.fit_u(traces.astype(np.int16), labels.astype(np.uint16))
    snr_val = snr.get_snr() 
    # Plot the res
    ax_sx_inch = 5
    ax_sy_inch = 3
    figsize=(4*ax_sx_inch, 4*ax_sy_inch)
    f = plt.figure("mulSNR",figsize=figsize)
    axes = []
    for i in indexes:
        axes.append(f.add_subplot(4,4,i+1))
        axes[i].plot(snr_val[i,:])
        axes[i].set_ylabel("SNR")
        axes[i].set_xlabel("time index")
        axes[i].set_title(r"SCALib SNR for $y_{{{}}}=\text{{Sbox}}[p_{{{}}} \oplus k_{{{}}}]$".format(i, i, i))
    f.tight_layout()


def display_snrs_sbox_output(traces, labels, byte_index, compute_byte_snr, compute_byte_snr_scalib):
    import datetime
    # Compute Sbox output
    ulabels = labels[:,byte_index]
    # Comptue the snr
    t0_0 = datetime.datetime.now()
    my_snr = compute_byte_snr(traces, ulabels)
    t0_1 = datetime.datetime.now()
    t1_0 = datetime.datetime.now()
    scalib_snr = compute_byte_snr_scalib(traces, ulabels[:,np.newaxis])
    t1_1 = datetime.datetime.now()
    # Print some stuff
    print("Max SNR custom ({}) found at time index {}. [{} elapsed]".format(np.max(my_snr), np.argmax(my_snr),t0_1-t0_0))
    print("Max SNR SCALIB ({}) found at time index {}. [{} elapsed]".format(np.max(scalib_snr), np.argmax(scalib_snr),t1_1-t1_0))
    # Plot the result
    f = plt.figure()
    ax0 = f.add_subplot(3,1,1)
    ax1 = f.add_subplot(3,1,2)
    ax2 = f.add_subplot(3,1,3)
    ax0.plot(traces[:1,:].T)
    ax1.plot(my_snr)
    ax2.plot(scalib_snr)
    ax0.set_ylabel("Power")
    ax1.set_ylabel("My SNR")
    ax0.set_title(r"SNR for $y_{{{}}}=\text{{Sbox}}[p_{{{}}} \oplus k_{{{}}}]$".format(byte_index, byte_index, byte_index))
    ax2.set_xlabel("Time")
    ax2.set_ylabel("SCALib SNR")


def display_TA_qa_results(TA_res, indexes, use_colors=False, log=True):
    log_start("display_TA_qa_results", log=log)
    # unpack the results
    (probs, correct_kbytes, complexities_attack) = TA_res
    # Plot the res
    ax_sx_inch = 5
    ax_sy_inch = 3
    figsize=(4*ax_sx_inch, 4*ax_sy_inch)
    f = plt.figure("TAqares",figsize=figsize)
    axes = []
    for i in range(len(indexes)):
        axes.append(f.add_subplot(4,4,i+1))
        # Plot the wrong guess
        for fpi in range(probs.shape[0]):
            for b in range(256):
                axes[i].plot(complexities_attack, probs[fpi,i,b,:],color="xkcd:light grey")
        # Plot the valid guess
        for fpi in range(probs.shape[0]):
                # Flag for color
                if use_colors:
                    color_correct = MY_COLORS[fpi % len(MY_COLORS)]
                    txtlabel="Set {}".format(fpi)
                else:
                    color_correct = "xkcd:black"
                    txtlabel=None
                axes[i].plot(complexities_attack, probs[fpi,i,correct_kbytes[fpi,indexes[i]]],color=color_correct, label=txtlabel)
        # Label
        axes[i].set_xlabel("attack data complexity")
        axes[i].set_ylabel(r'Pr($k_{}^* | \boldsymbol{{l}}$)'.format(i))
        axes[i].set_title("Likelihood byte index {}".format(indexes[i]))
        if use_colors:
            axes[i].legend()
    f.tight_layout()
    log_end(log=log)

# Find the amount of traces required such that, for each variable, all classes appears at least twice
# in order to allows gaussian model
def compute_minimal_traces_for_all_classes(labels, mins, maxs,MINam=3):
    if maxs-mins == 1:
        return maxs
    else:
        # Compute bounds 
        half = (maxs + mins) // 2
        # Compute amount of unique var for each variables
        status_h0 = labels.shape[1]*[0]
        status_h1 = labels.shape[1]*[0]
        for i in range(labels.shape[1]):
            _, count_h0 = np.unique(labels[:half,i], return_counts=True)
            _, count_h1 = np.unique(labels[:maxs,i], return_counts=True)
            flag_len_h0 = len(count_h0)==256
            flag_am_h0 = (count_h0>=MINam).all()
            flag_len_h1 = len(count_h1)==256
            flag_am_h1 = (count_h1>=MINam).all()
            status_h0[i] = flag_len_h0 and flag_am_h0
            status_h1[i] = flag_len_h1 and flag_am_h1

        gstatus_h0 = status_h0 == (labels.shape[1]*[1])
        gstatus_h1 = status_h1 == (labels.shape[1]*[1])
        if gstatus_h0:
            return compute_minimal_traces_for_all_classes(labels, mins, half)
        elif gstatus_h1:
            return compute_minimal_traces_for_all_classes(labels, half, maxs)    
        else:
            return -1

def string_status(str_id, value, max_value):
    return "{}/{} {}".format(value, max_value, str_id)

def write_progress_status_ranks(**args):
    # Build the progress status string
    status_str = None
    for str_id, (mV, MV) in args.items():
        if status_str == None:
            status_str = "{}".format(string_status(str_id, mV, MV))
        else:
            status_str = "{}; {}".format(status_str, string_status(str_id, mV, MV))
    print("\rProgress: {}".format(status_str),end="",flush=True)

def display_TA_qa_key_rank(TA_res, indexes, key_rank_computation, use_colors=False, log=True, prefix="Exhaustive"):
    log_start("display_TA_qa_key_rank",log=log)
    f = plt.figure("keyrank")
    ax0 = f.add_subplot(1,1,1)
    # Unpack results
    (probs, correct_kbytes, complexities_attack) = TA_res
    # Compute key_ranks
    key_ranks = np.zeros([probs.shape[0], len(complexities_attack)],dtype=int)
    for fpi in range(probs.shape[0]):
        # Compute key ranks
        for qa_i, qa in enumerate(complexities_attack):
            key_ranks[fpi,qa_i] = key_rank_computation(probs[fpi, indexes,:,qa_i].reshape([len(indexes),256]), correct_kbytes[fpi,indexes]) 
            write_progress_status_ranks(validation_set=(fpi+1,probs.shape[0]), complexities=(qa_i+1,len(complexities_attack)))
        # Plot 
        if use_colors:
            color_correct = MY_COLORS[fpi % len(MY_COLORS)]
            txtlabel = "Set {}".format(fpi)
        else:
            color_correct = "xkcd:black"
            txtlabel = None
        ax0.semilogy(complexities_attack, key_ranks[fpi], base=2, color=color_correct, label=txtlabel)
        if use_colors:
            ax0.legend()
    ax0.set_ylabel("Key Rank")
    ax0.set_xlabel("Attack data complexity")
    ax0.set_title("{} {}-byte key rank".format(prefix,len(indexes)))
    log_end(log=log)
    return key_ranks

def test_key_rank_computation(TA_res, indexes, key_rank_computation):
    log_start("test_key_rank_computation")
    display_TA_qa_results(TA_res, indexes, use_colors=True, log=False)
    qa_key_ranks_results = display_TA_qa_key_rank(TA_res, indexes, key_rank_computation, use_colors=True, log=False)
    print()
    log_end()
    return qa_key_ranks_results
    
def zero2negligeable(probs, neg=1e-20):
    zeros_cmp = np.zeros(probs.shape)
    # Find probs that are equal to 0
    loc0 = np.isclose(probs, zeros_cmp)
    probs_corr = np.copy(probs)
    probs_corr[loc0] = neg
    return probs_corr

def display_rank_approx_vs_exact_rank(TA_res, indexes, qa_key_ranks_results, key_rank_approx):
    # Unpack results
    (probs, correct_kbytes, complexities_attack) = TA_res
    # Tweak the probas to avoid 0 probabilities 
    probs = zero2negligeable(probs)
    # Plot the res
    ax_sx_inch = 4
    ax_sy_inch = 3
    figsize=(3*ax_sx_inch, 2*ax_sy_inch)
    f = plt.figure("TArankapp",figsize=figsize)
    axes = []
    # Compute key_ranks
    app_key_ranks = np.zeros([probs.shape[0], len(complexities_attack)])
    rmin_ranks = np.zeros([probs.shape[0], len(complexities_attack)])
    rmax_ranks = np.zeros([probs.shape[0], len(complexities_attack)])
    for fpi in range(probs.shape[0]):
        # Create the axes
        axes.append(f.add_subplot(2,3,fpi+1))
        for qa_i, qa in enumerate(complexities_attack):
            # Compute the approx_key_rank
            (rmin, r, rmax) = key_rank_approx(probs[fpi, indexes,:,qa_i].reshape([len(indexes),256]), correct_kbytes[fpi,indexes])
            rmin_ranks[fpi, qa_i] = rmin
            rmax_ranks[fpi, qa_i] = rmax
            app_key_ranks[fpi, qa_i] = r
        # Plot results
        axes[fpi].fill_between(complexities_attack, rmin_ranks[fpi,:], rmax_ranks[fpi,:], color=MY_COLORS[fpi % len(MY_COLORS)], alpha=.3, label="Bounds")
        axes[fpi].semilogy(complexities_attack, qa_key_ranks_results[fpi, :], color="xkcd:black", base=2, label="Exact")
        axes[fpi].semilogy(complexities_attack, app_key_ranks[fpi, :], color=MY_COLORS[fpi % len(MY_COLORS)],linestyle="dashed", base=2, label="Approx.")
        axes[fpi].set_xlabel("attack data complexity")
        axes[fpi].set_ylabel("key rank")
        axes[fpi].set_title("{}-byte key rank for attack set {}".format(len(indexes),fpi))
        axes[fpi].legend()
        f.tight_layout()

def display_rank_esti_full_key_axe(ax,TA_res, byte_indexes, key_rank_approx):
    # Unpack results
    # Here,
    # - probs of shape (n_ds_validation, nvars, 256, n_qas)
    # - correct_kbytes of shape (n_ds_validation, nvars)
    (mulTA_probs, mulTA_correct_kbytes, mulTA_complexities_attack) = TA_res
    mulTA_probs = zero2negligeable(mulTA_probs)
    # Display the ranks 

    # Allocate memory
    kr_mul = np.zeros([mulTA_probs.shape[0], len(mulTA_complexities_attack)])  
    kr_mul_mb = np.zeros(kr_mul.shape)
    kr_mul_Mb = np.zeros(kr_mul.shape)
    # Approximate rank on the full key
    pbar_max = mulTA_probs.shape[0]*len(mulTA_complexities_attack)
    with tqdm.tqdm(total=pbar_max, desc="Ranks computation") as pbar:
        for fpi in range(mulTA_probs.shape[0]):
            # Iterate over the complexity
            for qa_i, qa in enumerate(mulTA_complexities_attack):
                # Rank for multivariate attack 
                (umb, ur, uMb) = key_rank_approx(mulTA_probs[fpi, byte_indexes, :, qa_i], mulTA_correct_kbytes[fpi,byte_indexes])
                kr_mul[fpi, qa_i] = ur
                kr_mul_mb[fpi, qa_i] = umb
                kr_mul_Mb[fpi, qa_i] = uMb
                pbar.update(1)
            # Plot the axis
            ax.fill_between(mulTA_complexities_attack, kr_mul_mb[fpi], kr_mul_Mb[fpi], color=MY_COLORS[fpi % len(MY_COLORS)], alpha=.3)
    # Plot estimated rank on top
    for fpi in range(mulTA_probs.shape[0]):
        ax.semilogy(mulTA_complexities_attack, kr_mul[fpi], base=2, color=MY_COLORS[fpi % len(MY_COLORS)],label="Set {}".format(fpi))
    # Plot median
    ax.semilogy(mulTA_complexities_attack, np.median(kr_mul,axis=0), base=2, color="xkcd:black", linestyle="dashed", label="Median")
    # Add legend
    ax.legend()
    # Add label and title
    ax.set_xlabel("attack data complexity")
    ax.set_ylabel("Key rank")
    ax.set_title("{}-byte rank with TA".format(len(byte_indexes)))

def display_rank_esti_full_key(uni_TA_res, multi_TA_res, byte_indexes, key_rank_approx, title2=None):
    # Unpack results
    # Here,
    # - probs of shape (n_ds_validation, nvars, 256, n_qas)
    # - correct_kbytes of shape (n_ds_validation, nvars)
    (uniTA_probs, uniTA_correct_kbytes, uniTA_complexities_attack) = uni_TA_res
    uniTA_probs = zero2negligeable(uniTA_probs)
    (mulTA_probs, mulTA_correct_kbytes, mulTA_complexities_attack) = multi_TA_res
    mulTA_probs = zero2negligeable(mulTA_probs)
    assert (uniTA_complexities_attack == mulTA_complexities_attack).all(), "Mismatch between attack results"
    assert (uniTA_correct_kbytes == mulTA_correct_kbytes).all(), "Mismatch between correct keys"
    # Display the ranks 
    ax_sx_inch = 5
    ax_sy_inch = 3
    figsize=(2*ax_sx_inch, 1*ax_sy_inch)
    f = plt.figure(figsize=figsize)
    ax0 = f.add_subplot(1,2,1)
    ax1 = f.add_subplot(1,2,2)
    # Allocate memory
    kr_uni = np.zeros([uniTA_probs.shape[0], len(uniTA_complexities_attack)])  
    kr_uni_mb = np.zeros(kr_uni.shape) 
    kr_uni_Mb = np.zeros(kr_uni.shape) 

    kr_mul = np.zeros([mulTA_probs.shape[0], len(mulTA_complexities_attack)])  
    kr_mul_mb = np.zeros(kr_mul.shape)
    kr_mul_Mb = np.zeros(kr_mul.shape)
    # Approximate rank on the full key
    pbar_max = uniTA_probs.shape[0]*len(uniTA_complexities_attack)
    with tqdm.tqdm(total=pbar_max, desc="Test") as pbar:
        for fpi in range(uniTA_probs.shape[0]):
            # Iterate over the complexity
            for qa_i, qa in enumerate(uniTA_complexities_attack):
                # Rank for univariate attack
                (umb, ur, uMb) = key_rank_approx(uniTA_probs[fpi, byte_indexes, :, qa_i], uniTA_correct_kbytes[fpi,byte_indexes])
                kr_uni[fpi, qa_i] = ur
                kr_uni_mb[fpi, qa_i] = umb
                kr_uni_Mb[fpi, qa_i] = uMb
                # Rank for multivariate attack 
                (umb, ur, uMb) = key_rank_approx(mulTA_probs[fpi, byte_indexes, :, qa_i], mulTA_correct_kbytes[fpi,byte_indexes])
                kr_mul[fpi, qa_i] = ur
                kr_mul_mb[fpi, qa_i] = umb
                kr_mul_Mb[fpi, qa_i] = uMb
                pbar.update(1)
            # Plot the axis
            ax0.fill_between(uniTA_complexities_attack, kr_uni_mb[fpi], kr_uni_Mb[fpi], color=MY_COLORS[fpi % len(MY_COLORS)], alpha=.3)
            ax1.fill_between(mulTA_complexities_attack, kr_mul_mb[fpi], kr_mul_Mb[fpi], color=MY_COLORS[fpi % len(MY_COLORS)], alpha=.3)
    # Plot estimated rank on top
    for fpi in range(uniTA_probs.shape[0]):
        ax0.semilogy(uniTA_complexities_attack, kr_uni[fpi], base=2, color=MY_COLORS[fpi % len(MY_COLORS)],label="Set {}".format(fpi))
        ax1.semilogy(mulTA_complexities_attack, kr_mul[fpi], base=2, color=MY_COLORS[fpi % len(MY_COLORS)],label="Set {}".format(fpi))
    # Plot median
    ax0.semilogy(uniTA_complexities_attack, np.median(kr_uni,axis=0), base=2, color="xkcd:black", linestyle="dashed", label="Median")
    ax1.semilogy(mulTA_complexities_attack, np.median(kr_mul,axis=0), base=2, color="xkcd:black", linestyle="dashed", label="Median")
    # Add legend
    ax0.legend()
    ax1.legend()
    # Add label and title
    ax0.set_xlabel("attack data complexity")
    ax1.set_xlabel("attack data complexity")
    ax0.set_ylabel("Key rank")
    ax1.set_ylabel("Key rank")
    ax0.set_title("{}-byte rank with univaritate TA".format(len(byte_indexes)))
    if title2 is None:
        ax1.set_title("{}-byte rank with LDA + multivariate TA".format(len(byte_indexes)))
    else:
        ax1.set_title(title2)


def kfold_indexes(total_size, kfold, index):
    ksize = total_size // kfold
    test_index = np.arange(index*ksize, (index+1)*ksize)
    tindex0 = np.arange(0, index*ksize)
    tindex1 = np.arange((index+1)*ksize, total_size)
    training_index = np.hstack([tindex0, tindex1])
    return test_index, training_index

def IT_TwoSigmas_CI(data, Hbits=8):
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    mBs = means - (2*stds/np.sqrt(data.shape[1])) + Hbits
    MBs = means + (2*stds/np.sqrt(data.shape[1])) + Hbits
    return mBs, MBs

def compute_pi_estimations(training_traces, training_labels, test_traces, test_labels, pi_method, qt_s, nclasses=256):
    # Fetch config
    qt_max, n_p = training_labels.shape
    # Some config
    assert max(qt_s)<=qt_max, "The provided training data complexities cannot be reached given the amount of traces in the test set"
    # Allocate memory
    pis= np.zeros([n_p, len(qt_s)])
    mB = np.zeros([n_p, len(qt_s)])
    MB = np.zeros([n_p, len(qt_s)])
    # Iterate over qt_s
    for qt_i, qt in enumerate(qt_s):
        # Compute PI univariate
        mpic = pi_method(
            training_traces[:qt, :], 
            training_labels[:qt, :], 
            test_traces, 
            test_labels
        )
        pis[:, qt_i] = mpic.get_pi()
        std_pi = mpic.get_pi_std()
        # Compute 2sigmas interval
        mB[:, qt_i] = pis[:, qt_i] - (2*std_pi)
        MB[:, qt_i] = pis[:, qt_i] + (2*std_pi)

    # Return results
    return dict(
        dtype="PI",
        qt_s=qt_s,
        it=pis,
        mB=mB,
        MB=MB
    )

def compute_pi_estimations_RSI(training_traces, training_labels_rsi, training_labels_interns, test_traces, test_labels, pi_method, qt_s, nclasses=256):
    # Fetch config
    qt_max, n_p = training_labels_interns.shape
    # Some config
    assert max(qt_s)<=qt_max, "The provided training data complexities cannot be reached given the amount of traces in the test set"
    # Allocate memory
    pis= np.zeros([n_p, len(qt_s)])
    mB = np.zeros([n_p, len(qt_s)])
    MB = np.zeros([n_p, len(qt_s)])
    # Iterate over qt_s
    for qt_i, qt in tqdm.tqdm(enumerate(qt_s), desc="PI computations", total=len(qt_s)):
        # Compute PI univariate
        mpic = pi_method(
            training_traces[:qt, :], 
            training_labels_rsi[:qt], 
            training_labels_interns[:qt, :], 
            test_traces, 
            test_labels,
        )
        pis[:, qt_i] = mpic.get_pi()
        std_pi = mpic.get_pi_std()
        # Compute 2sigmas interval
        mB[:, qt_i] = pis[:, qt_i] - (2*std_pi)
        MB[:, qt_i] = pis[:, qt_i] + (2*std_pi)

    # Return results
    return dict(
        dtype="PI",
        qt_s=qt_s,
        it=pis,
        mB=mB,
        MB=MB
    )

def compute_ti_estimations(training_traces, training_labels, ti_method, qt_s, nclasses=256):
    # Fetch config
    qt_max, n_p = training_labels.shape
    # Some config
    assert max(qt_s)<=qt_max, "The provided training data complexities cannot be reached given the amount of traces in the test set"
    # Allocate memory
    tis= np.zeros([n_p, len(qt_s)])
    # Iterate over qt_s
    for qt_i, qt in enumerate(qt_s):
        # Compute PI univariate
        mpic = ti_method(
            training_traces[:qt, :], 
            training_labels[:qt, :], 
        )
        tis[:, qt_i] = mpic.get_pi()
    # Return results
    return dict(
        dtype="TI",
        qt_s=qt_s,
        it=tis,
        mB=None,
        MB=None
    )

def compute_ti_estimations_RSI(training_traces, training_labels_rsi, training_labels_interns, training_labels, ti_method, qt_s, nclasses=256):
    # Fetch config
    qt_max, n_p = training_labels_interns.shape
    # Some config
    assert max(qt_s)<=qt_max, "The provided training data complexities cannot be reached given the amount of traces in the test set"
    # Allocate memory
    tis= np.zeros([n_p, len(qt_s)])
    # Iterate over qt_s
    for qt_i, qt in tqdm.tqdm(enumerate(qt_s), desc="TI computations", total=len(qt_s)):
        # Compute PI univariate
        mpic = ti_method(
            training_traces[:qt, :], 
            training_labels_rsi[:qt], 
            training_labels_interns[:qt], 
            training_labels[:qt], 
        )
        tis[:, qt_i] = mpic.get_pi()
    # Return results
    return dict(
        dtype="TI",
        qt_s=qt_s,
        it=tis,
        mB=None,
        MB=None
    )


class ITDisplayEntry:
    def __init__(self, results, modelID):
        assert results["dtype"] in ["PI", "TI"], "Dtype not handled"
        assert "qt_s" in results, "Provided results do not contain 'qt_s' value."
        assert "it" in results, "Provided results do not contain 'it' value"
        self.res = results
        self.mID = modelID
        self.color= None
        self._set_linestyle()
        self._set_label()

    def _set_linestyle(self):
        if self.res["dtype"]=="PI":
            self.linestyle = "solid"
        elif self.res["dtype"]=="TI":
            self.linestyle = "dashed"
    
    def _set_label(self):
        self.label = "{} {}".format(self.res["dtype"], self.mID)

    def _set_color(self, color):
        self.color=color


class ITDisplayConFig:
    def __init__(self, config_list):
        self.map_model = {}
        self.am_id = 0
        self.n_p = None
        self.cfgs = []
        for c in config_list:
            if self.am_id == 0:
                self.n_p= c.res["it"].shape[0]
            else:
                assert c.res["it"].shape[0]==self.n_p, "All the provided results must be generated for the same amount on intermediate state"
            if c.mID in self.map_model:
                c._set_color(self.map_model[c.mID][0].color)
                self.map_model[c.mID].append(c)
                self.cfgs.append(c)
            else:
                c._set_color(MY_COLORS[self.am_id % len(MY_COLORS)])
                self.map_model[c.mID] = [c]
                self.cfgs.append(c)
                self.am_id += 1

    def models2plot(self):
        return self.map_model

def display_IT_results(byte_indexes, list_tuples_cfg, scale=1.0, disable_legend=False):
    # Create the config
    list_config = [ITDisplayEntry(e,k) for (k,e) in list_tuples_cfg]
    display_config = ITDisplayConFig(list_config)
    # Shape 
    n_p = display_config.n_p
    # Plot the res
    ax_sx_inch = 5
    ax_sy_inch = 3
    figsize=(4*ax_sx_inch*scale, 4*ax_sy_inch*scale)
    f = plt.figure(figsize=figsize)
    axes = []
    for i in byte_indexes:
        axes.append(f.add_subplot(4,4,i+1))
        # Iterate over the config
        for cfg in display_config.cfgs:
            if cfg.res["mB"] is not None:
                axes[i].fill_between(cfg.res["qt_s"], cfg.res["mB"][i,:], cfg.res["MB"][i,:], color=cfg.color, alpha=0.3)
            axes[i].plot(cfg.res["qt_s"], cfg.res["it"][i,:], color=cfg.color, linestyle=cfg.linestyle, label=cfg.label)
        if not(disable_legend):
            axes[i].legend()
        axes[i].set_ylabel("IT metric [bits]")
        axes[i].set_xlabel("training data complexity")
        axes[i].set_title("byte index {}".format(i))
    # Global 
    f.tight_layout()
    plt.show()

def display_ttest_result(ttest_result, example_trace, title="Ttest results"):
    f = plt.figure()
    ax0 = f.add_subplot(2,1,1)
    ax1 = f.add_subplot(2,1,2)
    ax0.plot(example_trace)
    ax1.plot(np.abs(ttest_result))
    ax1.hlines(y=4.5,xmin=0,xmax=len(example_trace),color="xkcd:red",linestyle="solid")
    ax0.set_title(title)
    ax0.set_ylabel("Power")
    ax1.set_ylabel("t-statistic")
    ax1.set_xlabel("time index")
    f.tight_layout()
    plt.show()


