import numpy as np
from scipy.sparse.linalg import gmres, spsolve
from scipy.sparse.linalg import splu
import scipy.sparse as sp
from joblib import Parallel, delayed


def ldgm_R_times(P, x, Pidn0):
    if len(Pidn0) == 0:
        return np.array([])
    else:
        y = np.zeros(P.shape[0])
        y[Pidn0] = x
        # return spsolve(P, y)[Pidn0]
        return gmres(P, y, rtol=1e-8)[0][Pidn0]


def ldgm_P_times(P, x, Pidn0):
    Pid0 = list(set(range(P.shape[0])) - set(Pidn0))
    if len(Pid0) == 0:
        return P[Pidn0][:, Pidn0].dot(x)
    else:
        # return P[Pidn0][:, Pidn0].dot(x) - P[Pidn0][:, Pid0].dot(
        #     spsolve(P[Pid0][:, Pid0], P[Pid0][:, Pidn0].dot(x))
        # )
        return P[Pidn0][:, Pidn0].dot(x) - P[Pidn0][:, Pid0].dot(
            gmres(P[Pid0][:, Pid0], P[Pid0][:, Pidn0].dot(x), rtol=1e-8)[0]
        )


def ldgm_gibbs_block_grid_parallel_subprocess(subinput):
    (
        PM,
        snplist,
        beta_hat,
        curr_beta,
        R_curr_beta,
        NN,
        h2_per_var,
        inv_odd_p,
        h2,
        para,
        i,
        k,
    ) = subinput
    if i % 137 == 0:
        print("Run ldgm_gibbs_block_grid block:", i, "Iteration:", k)
    res_beta_hat = beta_hat - (R_curr_beta - curr_beta)
    N = NN / (1 - h2)
    C1 = h2_per_var * N
    C2 = 1 / (1 + 1 / C1)
    C3 = C2 * res_beta_hat
    C4 = C2 / N
    m = len(beta_hat)
    post_p = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
    q = np.random.rand(m)
    q = (q < post_p).astype(int)
    id0 = np.where(q == 0)[0]
    idn0 = np.where(q != 0)[0]
    Pidn0 = snplist["index"][idn0]
    mean_beta = np.zeros(m)
    if len(Pidn0) != 0:
        P = PM["precision"].copy()
        beta_hat_copy = beta_hat.copy()
        beta_hat_copy = beta_hat_copy * np.sqrt(1 - h2)
        beta_hat_copy[id0] = 0
        P[Pidn0, Pidn0] += C1[idn0]
        mean = ldgm_R_times(
            P,
            C1 * (ldgm_P_times(PM["precision"], beta_hat_copy, snplist["index"])),
            snplist["index"],
        )
        # mean = ldgm_R_times(
        #     P,
        #     C1
        #     * (ldgm_P_times(P, beta_hat_copy, snplist["index"]) - C1 * beta_hat_copy),
        #     snplist["index"],
        # )
        curr_beta[idn0] = np.random.randn(len(idn0))
        x = curr_beta[idn0]
        coef = 1
        for l in range(1, para["taylor_num"] + 1):
            x = ldgm_R_times(P, x, Pidn0) * C1[idn0]
            coef *= -(0.5 - l + 1) / l
            curr_beta[idn0] += coef * x
        curr_beta[idn0] *= np.sqrt(h2_per_var)
        curr_beta[idn0] += mean[idn0]
        mean_beta[idn0] = mean[idn0]
    curr_beta[id0] = 0
    R_curr_beta = ldgm_R_times(PM["precision"], curr_beta, snplist["index"])
    return curr_beta, R_curr_beta, mean_beta


def ldgm_gibbs_block_grid_parallel(PM, snplist, sumstats, para):
    p = para["p"]
    h2 = para["h2"]
    M = 0
    beta_ldgm = []
    curr_beta = []
    avg_beta = []
    beta_hat = []
    N = []
    scale_size = []
    R_curr_beta = []
    for i in range(len(PM)):
        M += len(sumstats[i]["beta"])
    for i in range(len(PM)):
        beta_hat.append(np.array(sumstats[i]["beta"]).astype(float))
        N.append(np.array(sumstats[i]["N"]).astype(float))
        scale_size.append(
            np.sqrt(
                N[i] * np.array(sumstats[i]["beta_se"]).astype(float) ** 2
                + beta_hat[i] ** 2
            )
        )
        # sacle_size[i] = 1
        beta_hat[i] = beta_hat[i] / scale_size[i]
        m = len(beta_hat[i])
        curr_beta.append(np.zeros(m))
        R_curr_beta.append(np.zeros(m))
        avg_beta.append(np.zeros(m))
        snplist[i]["index"] = np.array(snplist[i]["index"])
    for k in range(-para["ldgm_burn_in"], para["ldgm_num_iter"]):
        print("step:", k, "p:", p, "h2:", h2)
        h2_per_var = h2 / (M * p)
        inv_odd_p = (1 - p) / p
        subinput = []
        for i in range(len(PM)):
            subinput.append(
                (
                    PM[i],
                    snplist[i],
                    beta_hat[i],
                    curr_beta[i],
                    R_curr_beta[i],
                    N[i],
                    h2_per_var,
                    inv_odd_p,
                    h2,
                    para,
                    i,
                    k,
                )
            )
        results = Parallel(n_jobs=-1)(
            delayed(ldgm_gibbs_block_grid_parallel_subprocess)(d) for d in subinput
        )
        for i in range(len(PM)):
            curr_beta[i], R_curr_beta[i], mean_beta = results[i]
            if k >= 0:
                avg_beta[i] += mean_beta
    for i in range(len(PM)):
        beta_ldgm.append(avg_beta[i] / para["ldgm_num_iter"])
        beta_ldgm[i] = beta_ldgm[i] * scale_size[i]
        snplist[i]["index"] = snplist[i]["index"].tolist()
    return beta_ldgm


def ldgm_gibbs_block_auto_parallel_subprocess(subinput):
    (
        PM,
        snplist,
        beta_hat,
        curr_beta,
        R_curr_beta,
        NN,
        h2_per_var,
        inv_odd_p,
        h2,
        para,
        i,
        k,
    ) = subinput
    if i % 137 == 0:
        print("Run ldgm_gibbs_block_auto block:", i, "Iteration:", k)
    res_beta_hat = beta_hat - (R_curr_beta - curr_beta)
    N = NN / (1 - h2)
    C1 = h2_per_var * N
    C2 = 1 / (1 + 1 / C1)
    C3 = C2 * res_beta_hat
    C4 = C2 / N
    m = len(beta_hat)
    post_p = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
    q = np.random.rand(m)
    q = (q < post_p).astype(int)
    id0 = np.where(q == 0)[0]
    idn0 = np.where(q != 0)[0]
    Pidn0 = snplist["index"][idn0]
    mean_beta = np.zeros(m)
    if len(Pidn0) != 0:
        P = PM["precision"].copy()
        beta_hat_copy = beta_hat.copy()
        beta_hat_copy = beta_hat_copy * np.sqrt(1 - h2)
        beta_hat_copy[id0] = 0
        P[Pidn0, Pidn0] += C1[idn0]
        mean = ldgm_R_times(
            P,
            C1 * (ldgm_P_times(PM["precision"], beta_hat_copy, snplist["index"])),
            snplist["index"],
        )
        # mean = ldgm_R_times(
        #     P,
        #     C1
        #     * (ldgm_P_times(P, beta_hat_copy, snplist["index"]) - C1 * beta_hat_copy),
        #     snplist["index"],
        # )
        curr_beta[idn0] = np.random.randn(len(idn0))
        x = curr_beta[idn0]
        coef = 1
        for l in range(1, para["taylor_num"] + 1):
            x = ldgm_R_times(P, x, Pidn0) * C1[idn0]
            coef *= -(0.5 - l + 1) / l
            curr_beta[idn0] += coef * x
        curr_beta[idn0] *= np.sqrt(h2_per_var)
        curr_beta[idn0] += mean[idn0]
        mean_beta[idn0] = mean[idn0]
    curr_beta[id0] = 0
    R_curr_beta = ldgm_R_times(PM["precision"], curr_beta, snplist["index"])
    return curr_beta, R_curr_beta, len(idn0), np.dot(curr_beta, R_curr_beta), mean_beta


def ldgm_gibbs_block_auto_parallel(PM, snplist, sumstats, para):
    p = para["p"]
    p = np.random.rand()
    h2 = para["h2"]
    h2 = np.random.rand()
    M = 0
    beta_ldgm = []
    curr_beta = []
    avg_beta = []
    beta_hat = []
    N = []
    scale_size = []
    R_curr_beta = []
    for i in range(len(PM)):
        M += len(sumstats[i]["beta"])
    for i in range(len(PM)):
        beta_hat.append(np.array(sumstats[i]["beta"]).astype(float))
        N.append(np.array(sumstats[i]["N"]).astype(float))
        scale_size.append(
            np.sqrt(
                N[i] * np.array(sumstats[i]["beta_se"]).astype(float) ** 2
                + beta_hat[i] ** 2
            )
        )
        # sacle_size[i] = 1
        beta_hat[i] = beta_hat[i] / scale_size[i]
        m = len(beta_hat[i])
        curr_beta.append(np.zeros(m))
        R_curr_beta.append(np.zeros(m))
        avg_beta.append(np.zeros(m))
        snplist[i]["index"] = np.array(snplist[i]["index"])
    for k in range(-para["ldgm_burn_in"], para["ldgm_num_iter"]):
        print("step:", k, "p:", p, "h2:", h2)
        h2 = max(h2, para["h2_min"])
        h2 = min(h2, para["h2_max"])
        p = max(h2, para["p_min"])
        p = min(h2, para["p_max"])
        Mc = 0
        h2_per_var = h2 / (M * p)
        inv_odd_p = (1 - p) / p
        subinput = []
        for i in range(len(PM)):
            subinput.append(
                (
                    PM[i],
                    snplist[i],
                    beta_hat[i],
                    curr_beta[i],
                    R_curr_beta[i],
                    N[i],
                    h2_per_var,
                    inv_odd_p,
                    h2,
                    para,
                    i,
                    k,
                )
            )
        h2 = 0
        results = Parallel(n_jobs=-1)(
            delayed(ldgm_gibbs_block_auto_parallel_subprocess)(d) for d in subinput
        )
        for i in range(len(PM)):
            curr_beta[i], R_curr_beta[i], Mc_add, h2_add, mean_beta = results[i]
            Mc += Mc_add
            h2 += h2_add
            if k >= 0:
                avg_beta[i] += mean_beta
        p = np.random.beta(1 + Mc, 1 + M - Mc)
    for i in range(len(PM)):
        beta_ldgm.append(avg_beta[i] / para["ldgm_num_iter"])
        beta_ldgm[i] = beta_ldgm[i] * scale_size[i]
        snplist[i]["index"] = snplist[i]["index"].tolist()
    return beta_ldgm, p, h2
