import numpy as np
from scipy.sparse.linalg import inv, spsolve
from scipy.sparse import csr_matrix, eye


def get_marginal_beta(sumstats):
    beta_marginal = []
    for i in range(len(sumstats)):
        beta_marginal.append(list(map(float, sumstats[i]["beta"])))
    return beta_marginal


def ldpred2_inf(PM, sumstats, para):
    beta_inf_set = []
    m = 0
    for i in range(len(sumstats)):
        m += len(sumstats[i]["N"])
    for i in range(len(PM)):
        if len(sumstats[i]["N"]) == 0:
            beta_inf_set.append([])
            continue
        N = np.array(sumstats[i]["N"]).astype(float)
        beta_se = np.array(sumstats[i]["beta_se"]).astype(float)
        beta = np.array(sumstats[i]["beta"]).astype(float)
        scale = np.sqrt(N * beta_se**2 + beta**2)
        beta_hat = beta / scale
        LD = PM[i]["LD"]
        LD += eye(LD.shape[0], format="csr") * (m / (para["h2"] * para["N"]))
        beta_inf = spsolve(LD, beta_hat)
        beta_inf_set.append(beta_inf * scale)
    return beta_inf_set


def ldpred2_gibbs_one_sampling(PM, beta_hat, N, M, para):
    LD = PM["LD"]
    # LD = inv(PM["precision"])
    if isinstance(LD[0], np.float64):
        LD = csr_matrix([[LD[0]]])
    m = len(beta_hat)
    h2 = para["h2"]
    p = para["p"]
    curr_beta = np.zeros(m)
    avg_beta = np.zeros(m)
    dotprods = np.zeros(m)
    h2_per_var = h2 / (M * p)
    inv_odd_p = (1 - p) / p
    for k in range(-para["burn_in"], para["num_iter"]):
        for j in range(m):
            res_beta_hat_j = beta_hat[j] - (dotprods[j] - curr_beta[j])
            # res_beta_hat_j = beta_hat[j] - dotprods[j]
            C1 = h2_per_var * N[j]
            C2 = 1 / (1 + 1 / C1)
            C3 = C2 * res_beta_hat_j
            C4 = C2 / N[j]
            post_p_j = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
            diff = -curr_beta[j]
            # if post_p_j < p:
            #     curr_beta[j] = 0
            if post_p_j > np.random.rand():
                curr_beta[j] = np.random.normal(C3, np.sqrt(C4))
                diff += curr_beta[j]
            else:
                curr_beta[j] = 0
            if k >= 0:
                avg_beta[j] += C3 * post_p_j
            if diff != 0:
                dotprods += LD[:, j].toarray().flatten() * diff
    return avg_beta / para["num_iter"]


def ldpred2_grid(PM, sumstats, para):
    beta_gibbs_set = []
    M = 0
    for i in range(len(sumstats)):
        M += len(sumstats[i]["beta"])
    for i in range(len(PM)):
        if len(sumstats[i]["beta"]) == 0:
            beta_gibbs_set.append([])
            continue
        N = np.array(sumstats[i]["N"]).astype(float)
        beta_se = np.array(sumstats[i]["beta_se"]).astype(float)
        beta = np.array(sumstats[i]["beta"]).astype(float)
        # scale = np.sqrt(N * beta_se**2 + beta**2)
        scale = 1
        beta_hat = beta / scale
        beta_gibbs = ldpred2_gibbs_one_sampling(PM[i], beta_hat, N, M, para)
        beta_gibbs_set.append(beta_gibbs * scale)
    return beta_gibbs_set


def ldpred2_gibbs_auto(PM, beta_hat, N, M, para):
    LD = PM["LD"]
    # LD = inv(PM["precision"])
    if isinstance(LD[0], np.float64):
        LD = csr_matrix([[LD[0]]])
    m = len(beta_hat)
    h2 = para["h2"]
    p = para["p"]
    curr_beta = np.zeros(m)
    avg_beta = np.zeros(m)
    dotprods = np.zeros(m)
    h2_per_var = h2 / (M * p)
    inv_odd_p = (1 - p) / p
    for k in range(-para["burn_in"], para["num_iter"]):
        for j in range(m):
            res_beta_hat_j = beta_hat[j] - (dotprods[j] - curr_beta[j])
            # res_beta_hat_j = beta_hat[j] - dotprods[j]
            C1 = h2_per_var * N[j]
            C2 = 1 / (1 + 1 / C1)
            C3 = C2 * res_beta_hat_j
            C4 = C2 / N[j]
            post_p_j = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
            diff = -curr_beta[j]
            # if post_p_j < p:
            #     curr_beta[j] = 0
            if post_p_j > np.random.rand():
                curr_beta[j] = np.random.normal(C3, np.sqrt(C4))
                diff += curr_beta[j]
            else:
                curr_beta[j] = 0
            if k >= 0:
                avg_beta[j] += C3 * post_p_j
            if diff != 0:
                dotprods += LD[:, j].toarray().flatten() * diff
    return avg_beta / para["num_iter"]


def ldpred2_auto(PM, sumstats, para):
    p = np.random.rand()
    h2 = np.random.rand()
    curr_beta = []
    avg_beta = []
    dotprods = []
    M = 0
    for i in range(len(sumstats)):
        m = len(sumstats[i]["beta"])
        M += m
        curr_beta.append(np.zeros(m))
        avg_beta.append(np.zeros(m))
        dotprods.append(np.zeros(m))

    for k in range(-para["burn_in"], para["num_iter"]):
        Mc = 0
        # print(k, h2, p)
        h2_per_var = h2 / (M * p)
        inv_odd_p = (1 - p) / p
        h2 = 0
        for i in range(len(PM)):
            if len(sumstats[i]["beta"]) == 0:
                continue
            N = np.array(sumstats[i]["N"]).astype(float)
            beta_hat = np.array(sumstats[i]["beta"]).astype(float)
            LD = PM[i]["LD"]
            if isinstance(LD[0], np.float64):
                LD = csr_matrix([[LD[0]]])
            m = len(beta_hat)
            for j in range(m):
                res_beta_hat_j = beta_hat[j] - (dotprods[i][j] - curr_beta[i][j])
                # res_beta_hat_j = beta_hat[j] - dotprods[j]
                C1 = h2_per_var * N[j]
                C2 = 1 / (1 + 1 / C1)
                C3 = C2 * res_beta_hat_j
                C4 = C2 / N[j]
                post_p_j = 1 / (
                    1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2)
                )
                diff = -curr_beta[i][j]
                # if post_p_j < p:
                #     curr_beta[j] = 0
                if post_p_j > np.random.rand():
                    curr_beta[i][j] = np.random.normal(C3, np.sqrt(C4))
                    diff += curr_beta[i][j]
                    Mc += 1
                else:
                    curr_beta[i][j] = 0
                if k >= 0:
                    avg_beta[i][j] += C3 * post_p_j
                if diff != 0:
                    dotprods[i] += LD[:, j].toarray().flatten() * diff
            h2 += np.dot(curr_beta[i], LD.dot(curr_beta[i]))
        p = np.random.beta(1 + Mc, 1 + M - Mc)
    beta_auto_set = []
    for i in range(len(sumstats)):
        if len(sumstats[i]["beta"]) == 0:
            beta_auto_set.append([])
        else:
            beta_auto_set.append(avg_beta[i] / para["num_iter"])
    return beta_auto_set
