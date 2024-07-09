import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
import scipy.sparse as sp
from joblib import Parallel, delayed


def ldgm(PM, sumstats, para):
    p = para["p"]
    h2 = para["h2"]
    M = 0
    beta_ldgm = []
    for i in range(len(PM)):
        M += len(sumstats[i]["beta"])
    sigma2 = h2 / (M * p)
    inv_odd_p = (1 - p) / p
    for i in range(len(PM)):
        beta_hat = np.array(sumstats[i]["beta"]).astype(float)
        beta_tilde = PM[i]["precision"].dot(beta_hat)
        m = len(beta_hat)
        logp = np.zeros(para["ldgm_num_iter"])
        post_p = np.random.rand(m)
        beta_ldgm_block = np.zeros(m)
        beta_set = np.random.multivariate_normal(
            beta_tilde,
            PM[i]["precision"].toarray() / para["N"],
            size=para["ldgm_num_iter"],
        ).T
        logsum = 0
        post_p = np.ones(m) * 0.5
        for k in range(para["ldgm_num_iter"]):
            q = np.random.rand(m)
            q = (q < post_p).astype(int)
            N = np.array(sumstats[i]["N"]).astype(float)
            C1 = sigma2 * N
            C2 = 1 / (1 + 1 / C1)
            C3 = C2 * beta_ldgm_block
            C4 = C2 / N
            beta_set[:, k] = q * beta_set[:, k]
            # post_p = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
            # print(post_p)
            rsid = np.where(beta_set[:, k] == 0)[0]
            nrsid = np.where(beta_set[:, k] != 0)[0]
            logp[k] = -np.dot(beta_set[nrsid, k], beta_set[nrsid, k]) / sigma2 / 2
            print(logp[k])
            # logp[k] += (
            #     -(np.dot(beta_tilde[rsid], beta_tilde[rsid]) * para["N"] - len(rsid))
            #     * 0.5
            # )
            # logp[k] += np.sum(np.log(post_p[rsid] / (1 - post_p[rsid]) * inv_odd_p))
            # print(
            #     (
            #         -(
            #             np.dot(beta_tilde[rsid], beta_tilde[rsid]) * para["N"]
            #             - len(rsid)
            #         )
            #         * 0.5
            #     )
            # )
            # print(beta_tilde[rsid])
            # logp[k] += len(rsid) * para["N"]
            # logp[k] += (
            #     -np.dot(
            #         beta_tilde[rsid],
            #         spsolve(PM[i]["precision"][rsid][:, rsid], beta_tilde[rsid])
            #         * N[rsid],
            #     )
            #     / 2
            # )
            # logp[k] += -np.dot(
            #     beta_tilde[rsid],
            #     PM[i]["precision"][rsid][:, nrsid]
            #     @ (beta_set[nrsid, k] - beta_tilde[nrsid])
            #     * N[rsid],
            # )
            if k == 0:
                C5 = logp[k]
                beta_ldgm_block = beta_set[:, k]
            else:
                if logp[k] - logsum >= 20:
                    C5 = logp[k] - logsum
                else:
                    C5 = np.log(1 + np.exp(logp[k] - logsum))
                beta_ldgm_block = beta_ldgm_block * np.exp(-C5) + beta_set[
                    :, k
                ] * np.exp(logp[k] - logsum - C5)
            # print(k, beta_ldgm_block)
            logsum += C5
        # print(len(np.exp(logp - logsum)))
        beta_ldgm.append(beta_ldgm_block)
    return beta_ldgm


# def ldgm_gibbs_block(PM, sumstats, para):
#     p = para["p"]
#     h2 = para["h2"]
#     M = 0
#     beta_ldgm = []
#     for i in range(len(PM)):
#         M += len(sumstats[i]["beta"])
#     sigma2 = h2 / (M * p)
#     for i in range(len(PM)):
#         beta_hat = np.array(sumstats[i]["beta"]).astype(float)
#         beta_tilde = PM[i]["precision"].dot(beta_hat)
#         m = len(beta_hat)
#         beta_ldgm_block = np.zeros(m)
#         beta_set = np.zeros(m)
#         num_beta_0 = np.ones(m)
#         num_beta_n0 = np.ones(m)
#         z = np.random.rand(m)
#         z = (z < p).astype(int)
#         for k in range(-para["ldgm_burn_in"], para["ldgm_num_iter"]):
#             idn0 = np.where(z != 0)[0]
#             id0 = np.where(z == 0)[0]
#             num_beta_0[id0] += 1
#             num_beta_n0[idn0] += 1
#             beta_set[id0] = 0
#             cov = np.linalg.inv(
#                 PM[i]["LD"][idn0][:, idn0].toarray() * para["N"] + sigma2
#             )
#             if len(idn0) > 0:
#                 beta_set[idn0] = np.random.multivariate_normal(
#                     cov
#                     @ spsolve(PM[i]["precision"][idn0][:, idn0], beta_tilde[idn0])
#                     * para["N"],
#                     cov,
#                 )
#             post_p = (num_beta_n0 * p) / (num_beta_0 * (1 - p) + num_beta_n0 * p)
#             print(post_p)
#             z = np.random.rand(m)
#             z = (z < post_p).astype(int)
#             if k >= 0:
#                 beta_ldgm_block += beta_set
#         beta_ldgm.append(beta_ldgm_block / para["ldgm_num_iter"])
#     return beta_ldgm


def ldgm_gibbs_block(PM, sumstats, para):
    p = para["p"]
    h2 = para["h2"]
    M = 0
    beta_ldgm = []
    for i in range(len(PM)):
        M += len(sumstats[i]["beta"])
    h2_per_var = h2 / (M * p)
    inv_odd_p = (1 - p) / p
    for i in range(len(PM)):
        beta_hat = np.array(sumstats[i]["beta"]).astype(float)
        beta_tilde = PM[i]["precision"].dot(beta_hat)
        N = np.array(sumstats[i]["N"]).astype(float)
        P = PM[i]["precision"].copy()
        P.setdiag(0)
        m = len(beta_hat)
        curr_beta = np.zeros(m)
        avg_beta = np.zeros(m)
        for k in range(-para["ldgm_burn_in"], para["ldgm_num_iter"]):
            res_beta_hat = beta_hat - (PM[i]["LD"].dot(curr_beta) - curr_beta)
            # res_beta_hat = beta_tilde
            # res_beta_hat = beta_tilde - P.dot(curr_beta)
            C1 = h2_per_var * N
            C2 = 1 / (1 + 1 / C1)
            C3 = C2 * res_beta_hat
            C4 = C2 / N
            post_p = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
            q = np.random.rand(m)
            q = (q < post_p).astype(int)
            id0 = np.where(q == 0)[0]
            idn0 = np.where(q == 1)[0]
            curr_beta[id0] = 0
            curr_beta[idn0] = (np.random.randn(len(idn0)) + C3[idn0]) * np.sqrt(
                C4[idn0]
            )
            if k >= 0:
                avg_beta += C3 * post_p
        beta_ldgm.append(avg_beta / para["ldgm_num_iter"])
    return beta_ldgm


def ldgm_R_times(P, x, Pidn0):
    if len(Pidn0) == 0:
        return np.array([])
    else:
        y = np.zeros(P.shape[0])
        y[Pidn0] = x
        return spsolve(P, y)[Pidn0]


def ldgm_P_times(P, x, Pidn0):
    Pid0 = list(set(range(P.shape[0])) - set(Pidn0))
    if len(Pid0) == 0:
        return P[Pidn0][:, Pidn0].dot(x)
    else:
        return P[Pidn0][:, Pidn0].dot(x) - P[Pidn0][:, Pid0].dot(
            spsolve(P[Pid0][:, Pid0], P[Pid0][:, Pidn0].dot(x))
        )


def ldgm_gibbs_block_auto(PM, snplist, sumstats, para):
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
    p_curr_beta = []
    for i in range(len(PM)):
        M += len(sumstats[i]["beta"])
    for i in range(len(PM)):
        beta_hat.append(np.array(sumstats[i]["beta"]).astype(float))
        N.append(np.array(sumstats[i]["N"]).astype(float))
        m = len(beta_hat[i])
        curr_beta.append(np.zeros(m))
        p_curr_beta.append(np.zeros(m))
        avg_beta.append(np.zeros(m))
        snplist[i]["index"] = np.array(snplist[i]["index"])
    for k in range(-para["ldgm_burn_in"], para["ldgm_num_iter"]):
        print(k, p, h2)
        Mc = 0
        h2_per_var = h2 / (M * p)
        inv_odd_p = (1 - p) / p
        h2 = 0
        for i in range(len(PM)):
            print("Run ldgm_gibbs_block_auto block:", i, "Iteration:", k)
            res_beta_hat = beta_hat[i] - (p_curr_beta[i] - curr_beta[i])
            C1 = h2_per_var * N[i]
            C2 = 1 / (1 + 1 / C1)
            C3 = C2 * res_beta_hat
            C4 = C2 / N[i]
            m = len(beta_hat[i])
            post_p = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
            q = np.random.rand(m)
            q = (q < post_p).astype(int)
            id0 = np.where(q == 0)[0]
            idn0 = np.where(q != 0)[0]
            Pidn0 = snplist[i]["index"][idn0]
            if len(Pidn0) != 0:
                P = PM[i]["precision"].copy()
                beta_hat_copy = beta_hat[i].copy()
                beta_hat_copy[id0] = 0
                P[Pidn0, Pidn0] += C1[idn0]
                mean = ldgm_R_times(
                    P,
                    C1
                    * (
                        ldgm_P_times(P, beta_hat_copy, snplist[i]["index"])
                        - C1 * beta_hat_copy
                    ),
                    snplist[i]["index"],
                )
                curr_beta[i][idn0] = np.random.randn(len(idn0))
                x1 = ldgm_R_times(P, curr_beta[i][idn0], Pidn0) * C1
                x2 = ldgm_R_times(P, x1, Pidn0) * C1
                # x2 = np.zeros(len(idn0))
                x3 = ldgm_R_times(P, x2, id0, idn0) * C1
                # x3 = np.zeros(len(idn0))
                curr_beta[i][idn0] = curr_beta[i][idn0] - x1 / 2 - x2 / 8 - x3 / 16
                curr_beta[i][idn0] *= np.sqrt(h2_per_var)
                curr_beta[i][idn0] += mean[idn0]
            curr_beta[i][id0] = 0
            Mc += len(idn0)
            if k >= 0:
                avg_beta[i] += curr_beta[i]
            p_curr_beta[i] = ldgm_R_times(
                PM[i]["precision"], curr_beta[i], snplist[i]["index"]
            )
            h2 += np.dot(curr_beta[i], p_curr_beta[i])
        p = np.random.beta(1 + Mc, 1 + M - Mc)
    for i in range(len(PM)):
        beta_ldgm.append(avg_beta[i] / para["ldgm_num_iter"])
        snplist[i]["index"] = snplist[i]["index"].tolist()
    return beta_ldgm


def ldgm_gibbs_block_auto_parallel_subprocess(subinput):
    (
        PM,
        snplist,
        beta_hat,
        curr_beta,
        R_curr_beta,
        N,
        h2_per_var,
        inv_odd_p,
        para,
        i,
        k,
    ) = subinput
    print("Run ldgm_gibbs_block_auto block:", i, "Iteration:", k)
    res_beta_hat = beta_hat - (R_curr_beta - curr_beta)
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
        beta_hat_copy[id0] = 0
        P[Pidn0, Pidn0] += C1[idn0]
        mean = ldgm_R_times(
            P,
            C1
            * (ldgm_P_times(P, beta_hat_copy, snplist["index"]) - C1 * beta_hat_copy),
            snplist["index"],
        )
        curr_beta[idn0] = np.random.randn(len(idn0))
        x1 = ldgm_R_times(P, curr_beta[idn0], Pidn0) * C1[idn0]
        x2 = ldgm_R_times(P, x1, Pidn0) * C1[idn0]
        x3 = ldgm_R_times(P, x2, Pidn0) * C1[idn0]
        # x3 = np.zeros(len(idn0))
        curr_beta[idn0] = curr_beta[idn0] - x1 / 2 - x2 / 8 - x3 / 16
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
    R_curr_beta = []
    for i in range(len(PM)):
        M += len(sumstats[i]["beta"])
    for i in range(len(PM)):
        beta_hat.append(np.array(sumstats[i]["beta"]).astype(float))
        N.append(np.array(sumstats[i]["N"]).astype(float))
        m = len(beta_hat[i])
        curr_beta.append(np.zeros(m))
        R_curr_beta.append(np.zeros(m))
        avg_beta.append(np.zeros(m))
        snplist[i]["index"] = np.array(snplist[i]["index"])
    for k in range(-para["ldgm_burn_in"], para["ldgm_num_iter"]):
        print("step:", k, "p:", p, "h2:", h2)
        h2 = max(h2, para["h2_min"])
        h2 = min(h2, para["h2_max"])
        Mc = 0
        h2_per_var = h2 / (M * p)
        inv_odd_p = (1 - p) / p
        h2 = 0
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
                    para,
                    i,
                    k,
                )
            )
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
        snplist[i]["index"] = snplist[i]["index"].tolist()
    return beta_ldgm
