import numpy as np
from scipy.stats import gamma, norm, binom
from scipy.integrate import quad


def compute_gamma_q(q, check_point):
    qs = []
    for t in check_point:
        qs.append(gamma.ppf(q=q,a=t))
    return np.array(qs)


def compute_ind_q(q, ind_delta, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*(1-ind_delta)+ q*np.sqrt(t*(1-ind_delta)*ind_delta))
    return np.array(qs)


def compute_general_q(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)


def compute_ind_q_new(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)


####################################################
##
## Compute test statistics for Gumbel-max watermarks
##
####################################################


def h_ars(Ys,  alpha=0.05):
    # Compute critical values
    check_points = np.arange(1, 1+Ys.shape[-1])
    h_ars_qs = compute_gamma_q(1-alpha, check_points)

    # Compute the test scores
    Ys = np.array(Ys)
    h_ars_Ys = -np.log(1-Ys)
    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)

    results = (cumsum_Ys >= h_ars_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_log(Ys,  alpha=0.05):
    # Compute critical values
    check_points = np.arange(1, 1+Ys.shape[-1])
    h_log_qs = compute_gamma_q(alpha, check_points)

    # Compute the test scores
    Ys = np.array(Ys)
    h_log_Ys = np.log(Ys)
    cumsum_Ys = np.cumsum(h_log_Ys, axis=1)

    results = (cumsum_Ys >= -h_log_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_ind(Ys, ind_delta=0.5, alpha=0.05):
    # Compute critical values
    check_points = np.arange(1, 1+Ys.shape[-1])
    h_ind_qs = binom.ppf(n=check_points, p = 1-ind_delta, q = 1-alpha)

    # Compute the test scores
    Ys = np.array(Ys)
    h_ind_Ys = (Ys >= ind_delta)
    cumsum_Ys = np.cumsum(h_ind_Ys, axis=1)
    
    results = (cumsum_Ys >= h_ind_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_opt_gum(Ys, delta0=0.2,theo=True, alpha=0.05):
    # Compute critical values
    Ys = np.array(Ys)
    check_points = np.arange(1, 1+Ys.shape[-1])
    # np.log(Ys**(1/(1-delta0)-1)+Ys**(1/delta0-1))

    def f(r, delta):
        inte_here = np.floor(1/(1-delta))
        rest = 1-(1-delta)*inte_here
        return np.log(inte_here*r**(delta/(1-delta))+ r**(1/rest-1))
    h_ars_Ys = f(Ys, delta0)
    
    if theo:
        mu = quad(lambda x: f(x, delta0), 0, 1,epsabs = 1e-10,epsrel=1e-10)
        EX2 = quad(lambda x: f(x, delta0)**2, 0, 1,epsabs = 1e-10,epsrel=1e-10)
        # print(delta0,"acc",mu[1],EX2[1])
        mu, EX2 = mu[0], EX2[0]
        Var = EX2 - mu**2

        h_help_qs = compute_general_q(1-alpha, mu, Var, check_points)
    else:
        def find_q(N=500):
            Null_Ys = np.random.uniform(size=(N, Ys.shape[1]))
            Simu_Y = f(Null_Ys, delta0)
            Simu_Y = np.cumsum(Simu_Y, axis=1)
            h_help_qs = np.quantile(Simu_Y, 1-alpha, axis=0)
            return h_help_qs
        
        q_lst = []
        for N in [500] * 10:
            q_lst.append(find_q(N))
        h_help_qs = np.mean(np.array(q_lst),axis=0)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_help_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


####################################################
##
## Compute test statistics for inverse watermarks
##
####################################################


def h_id_dif(Ds, alpha=0.05):
    # Compute critical values
    Ds = np.array(Ds)
    check_points = np.arange(1, 1+Ds.shape[-1])

    mu_dif = -1/3
    var_dif = 1/6 - 1/9
    h_id_dif_qs = compute_general_q(1-alpha, mu_dif, var_dif, check_points)

    # Compute the test scores
    cumsum_Ds = np.cumsum(Ds, axis=1)

    results = (cumsum_Ds >= h_id_dif_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_ind_dif(Ds,delta=-0.05,alpha=0.05):
    assert -1 <= delta <= 0

    # Compute critical values
    Ds = np.array(Ds)
    check_points = np.arange(1,1+Ds.shape[1])
    h_id_dif_qs = binom.ppf(n=check_points, p = 1-(1+delta)**2, q = 1-alpha)

    # Compute the test scores
    h_ind_Ds = Ds >= delta
    cumsum_Ds = np.cumsum(h_ind_Ds, axis=1)

    results = (cumsum_Ds >= h_id_dif_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_opt_dif(dif,delta=0.1,theo=False, weight=1e-6, vocab_size=32000, alpha=0.05, model_name="2p7B"):
    dif = np.array(dif)
    final_Y = dif

    def transform(Y):
        ## weight=1e-6 is used to avoid numerical blow-up
        return np.log(np.maximum(1+Y/(1-delta),weight)/np.maximum(1+Y,0)/(1-delta))
    
    # final_Y = np.log(np.maximum(1+final_Y/(1-delta),0)/np.maximum(1+final_Y,0))
    final_Y = transform(final_Y)
    cumsum_Ys = np.cumsum(final_Y, axis=1)

    # Compute critical values
    if theo:
        ## Thereotical critical values don't perform well due to the added truncation.
        def log_likelihood(x):
            return np.log(np.maximum(1+x/(1-delta), weight)/(1-delta)/(1+x))
        
        def rho(x):
            return 2*(1+x)
        
        mu =  quad(lambda x: rho(x)*log_likelihood(x), -1+1e-6, 0)[0]
        V2 = quad(lambda x: rho(x)*log_likelihood(x)**2, -1+1e-6, 0)[0]
        # print("mean", mu, "Var", V2-mu**2)

        h_opt_qs = compute_ind_q(1-alpha, mu, V2-mu**2, dif.shape[1])
    else:
        ## We use simulation to compute the critical values
        def find_q(N=1000):
            Null_Ys_U = np.random.uniform(size=(N, dif.shape[1]))
            Null_Ys_pi_s = np.random.randint(low=0, high=vocab_size, size=(N, dif.shape[1]))
            Null_etas = np.array(Null_Ys_pi_s)/(vocab_size-1)
            null_final_Y = -np.abs(Null_Ys_U-Null_etas)
            null_final_Y = transform(null_final_Y)
            null_cumsum_Ys = np.cumsum(null_final_Y, axis=1)
            h_opt_qs = np.quantile(null_cumsum_Ys, 1-alpha, axis=0)
            return h_opt_qs

        q_lst = []
        if model_name == "2p7B":
            ## If we use the 2.7B model, we should pay more efforts to control the Type I error
            for N in [500] * 10 + [200, 1000, 2000]: 
                q_lst.append(find_q(N))
            h_opt_qs = np.min(np.array(q_lst),axis=0)
        else:
            for N in [500] * 10: 
                q_lst.append(find_q(N))
            h_opt_qs = np.mean(np.array(q_lst),axis=0)

    results = (cumsum_Ys >= h_opt_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)
