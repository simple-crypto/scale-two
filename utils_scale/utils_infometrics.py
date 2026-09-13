import numpy as np
import scalib.tools 

class MultiInformationEstimator:
    def __init__(self, nc: int, prs, enth=None):
        self._nc = nc
        self._nv = prs.shape[0]
        self._ntraces = prs.shape[1]
        self._logprob_mean = np.mean(prs, axis=1)
        self._logprob_varN = self._ntraces * np.var(prs, axis=1)
        if enth is None:
            self.enth = np.log2(nc)
        else:
            self.enth = enth

    def merge(self, other):
        "return the combined stats of self and other"
        # Chan et al.'s generalization of Welford's algorithm.
        # Chan, Tony F. et al. "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances
        # as reported in
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        assert self._nv == other._nv
        assert self._nc == other._nc
        res = type(self).__new__(type(self))
        res._nc = self._nc
        res._nv = self._nv
        res.enth = self.enth
        res._ntraces = self._ntraces + other._ntraces
        delta = other._logprob_mean - self._logprob_mean
        res._logprob_mean = self._logprob_mean + (other._ntraces / res._ntraces) * delta
        res._logprob_varN = (
            self._logprob_varN
            + other._logprob_varN
            + (self._ntraces * other._ntraces / res._ntraces) * delta**2
        )
        return res

    def info(self):
        return self.enth + self._logprob_mean

    def info_std(self):
        return np.sqrt(self._logprob_varN / self._ntraces**2)

    @classmethod
    def from_legacy(cls, old):
        enth = np.log2(old._nc)
        new = cls(
            old._nc, 
            np.zeros([1,1]),
        )
        new._nc = old._nc
        new._nv = old._nv
        new._ntraces = old._ntraces
        new._logprob_mean = old._logprob_mean
        new._logprob_varN = old._logprob_varN
        new.enth = np.log2(old._nc)
        return new

class MultiPIComputer:
    def __init__(self, nc, nv, nthreads=None, enth=None):
        self.nc = nc
        self.nv = nv
        self._init = False
        self.enth = enth


    def fit(self, prs, classes):
        assert (
            len(prs) == self.nv
        ), "Wrong format for prs, must be a list of length nv, containing np.array of shape (Nt, nc)"
        assert (
            classes.shape[1] == self.nv
        ), "Wrong format for classes, must be of dimension (Nt, nv)"
        n = classes.shape[0]
        probas_class = np.array(
            [prs[vi][np.arange(n), classes[:, vi]] for vi in range(self.nv)]
        )
        self.fit_from_log2_proba_class(np.log2(probas_class))

    def fit_from_log2_proba_class(self, prs):
        """
        Fit method directly relying on the scalib.Lda.predict_log2_proba_class results
        """
        assert (
            prs.shape[0] == self.nv
        ), "Wrong format for 'prs', must be of shape (nv, n)."
        new_est = MultiInformationEstimator(self.nc, prs, enth=self.enth)
        if self._init:
            self._inner = self._inner.merge(new_est)
        else:
            self._init = True
            self._inner = new_est

    def get_pi(self):
        return self._inner.info()

    def get_pi_std(self):
        return self._inner.info_std()

    @classmethod
    def from_legacy(cls, old):
        new = cls(
            old.nc,
            old.nv
        )
        new.nc = old.nc 
        new.nv = old.nv 
        new._init = old._init 
        new.enth = np.log2(old.nc)
        new._inner = MultiInformationEstimator.from_legacy(old._inner)
        return new

if __name__ == "__main__":
    print("run utils_models")
    import datetime

    nc = 256
    nv = 100
    Nt = 20000

    # Simulate classes
    classes = np.random.randint(0,nc, [Nt, nv],dtype=np.uint8)
    # Simulate probas
    prs = np.random.randint(0,100, [nv, Nt, nc]).astype(np.float64)
    prs_sum = np.sum(prs,axis=2)
    for i in range(nv):
        for j in range(Nt):
            prs[i,j,:] /= prs_sum[i,j]

    # PI computer multithread
    mpi = MultiPIComputer(nc, nv, nthreads=5)
    t0 = datetime.datetime.now()
    mpi.fit(prs, classes)
    t1 = datetime.datetime.now()
    print("Fit: {}".format(t1-t0))
    mpi_res = mpi.get_pi()

