import numpy as np

class welfordVarEstimator:
    def __init__(self, ns, dtype=float):
        self.acc = self.init(ns, dtype)

    def init(self, ns, dtype):
        count = 0
        m2 = np.zeros(ns, dtype=dtype)
        mx = np.zeros(ns, dtype=dtype)
        return (count, m2, mx)

    def fit_u(self, nvs):
        (na, m2a, mxa) = self.acc
        nb = nvs.shape[0]
        nab = na+nb
        mxb = np.mean(nvs,axis=0)
        delta = mxb - mxa 
        mxab = mxa + delta * (nb / nab)
        # m2b 
        m2b = np.sum((nvs - mxb)**2, axis=0)
        m2ab = m2a + m2b + (delta**2) * (na*nb/(nab)) 
        self.acc = (nab, m2ab, mxab)

    def get_var(self):
        (n, m2_n, _) = self.acc
        return m2_n / (n - 1)

class onlineExpectationEstimator:
    def __init__(self, ns, dtype=float):
        self.sum = np.zeros(ns, dtype=dtype)
        self.n = 0

    def fit_u(self, nvs):
        self.sum += np.sum(nvs, axis=0)
        self.n += nvs.shape[0]

    def get_mean(self):
        return self.sum / self.n
        

class onlineCorr:
    def __init__(self):
        self.sum_x = None
        self.sum_l = None
        self.sum_xl = None
        self.init = False
        # Runner for online var 
        self.welf_l_state = None
        self.welf_x_state = None

    def fit_u(self, ls, x):
        if not(self.init):
            self.welf_l_state = welfordVarEstimator(ls.shape[1])
            self.welf_x_state = welfordVarEstimator(ls.shape[1])
            self.sum_x = onlineExpectationEstimator(ls.shape[1])
            self.sum_l = onlineExpectationEstimator(ls.shape[1])
            self.sum_xl = onlineExpectationEstimator(ls.shape[1])
            self.init = True
        # Update the var
        self.welf_l_state.fit_u(ls)
        self.welf_x_state.fit_u(x)
        # Update the running sums
        self.sum_x.fit_u(x)
        self.sum_l.fit_u(ls)
        self.sum_xl.fit_u(x*ls)

    def get_cov(self):
        return self.sum_xl.get_mean() - (self.sum_l.get_mean()*self.sum_x.get_mean() )

    def get_corr(self):
        std_l = np.sqrt(self.welf_l_state.get_var())
        std_x = np.sqrt(self.welf_x_state.get_var())
        return self.get_cov() / (std_x * std_l)
        
def gen_cases(Nt, ns):
    traces = np.random.randint(0,2**16, [Nt, ns], dtype=int)
    models = np.random.randint(1,10, traces.shape)
    return dict(
            traces=traces,
            models=models
    )

def test_var():
    nt = 8000
    ntd2 = nt//2
    ns = 3

    dic = gen_cases(nt, ns)
    traces = dic["traces"]

    # fit online
    welf = welfordVarEstimator(ns)
    welf.fit_u(traces[:ntd2])
    welf.fit_u(traces[ntd2:])
    v = welf.get_var()
    ref_v = np.var(traces,axis=0,ddof=1)
    assert np.allclose(v, ref_v)

def test_cov():
    nt = 8000
    ntd2 = nt//2
    ns = 3

    dic = gen_cases(nt, ns)
    traces = dic["traces"]
    models = dic["models"]

    # fit online
    cov = onlineCorr()
    cov.fit_u(traces[:ntd2], models[:ntd2])
    cov.fit_u(traces[ntd2:], models[ntd2:])
    v = cov.get_cov()

    ut = traces - np.mean(traces,axis=0)
    um = models - np.mean(models,axis=0)
    ref_cov = np.mean(ut*um, axis=0)
    print(ref_cov)
    print(v)
    assert np.allclose(v, ref_cov)

def test_corr():
    def pearson_corr(x,y):
        """
        x: raw traces, as an np.array of shape (nb_traces, nb_samples) and type np.float64
        y: model, as an np.array of shape (nb_traces, nb_samples) and type np.int16
        return: the pearson coefficient of each samples as a np.array of shape (1,nb_samples)
        """
        xc = x - np.mean(x, axis=0)
        yc = y - np.mean(y, axis=0)
        cov = np.mean(xc*yc, axis=0)
        x_std = np.std(x,axis=0)
        y_std = np.std(y,axis=0)
        return cov / (x_std * y_std)

    Nt = 1000
    Ntd2 = Nt//2
    ns = 3
    dic = gen_cases(Nt, ns)
    traces = dic["traces"]
    models = dic["models"]

    # Fit online
    corr_obj = onlineCorr() 
    corr_obj.fit_u(traces[:Ntd2], models[:Ntd2])
    corr_obj.fit_u(traces[Ntd2:], models[Ntd2:])
    corrv = corr_obj.get_corr()

    # Reference correlation
    ref_corr = pearson_corr(traces, models)

    print(corrv)
    print(ref_corr)
    assert np.allclose(corrv, ref_corr, rtol=1e-3)

if __name__ == "__main__":
    test_corr()





