import numpy as np


def generate_sumstats_beta(PM, snplist, para):
    sumstats = []
    beta_true = []
    m = 0
    for i in range(len(snplist)):
        m += len(snplist[i]["rsid"])

    sigma = para["h2"] / (m * para["p"])
    for i in range(len(snplist)):
        sumstats_block = {}
        mm = len(snplist[i]["rsid"])
        sumstats_block["N"] = np.ones(mm) * para["N"]
        sumstats_block["beta_se"] = np.ones(mm) * np.sqrt(sigma)
        beta_true.append([])
        if mm == 0:
            sumstats_block["beta"] = []
            sumstats.append(sumstats_block)
            continue
        for j in range(mm):
            if np.random.rand() < para["p"]:
                beta_true[i].append(np.random.normal(0, np.sqrt(sigma)))
            else:
                beta_true[i].append(0)
        beta_true[i] = np.array(beta_true[i])
        R = PM[i]["LD"].toarray()
        sumstats_block["beta"] = np.random.multivariate_normal(
            R @ beta_true[i], R / para["N"]
        )
        sumstats.append(sumstats_block)

    return sumstats, beta_true


def generate_phenotype(phestats, beta, para):
    phestats["Phenotype"] = phestats["X"] @ beta + np.random.randn(
        len(phestats["Phenotype"])
    ) * np.sqrt(1 - para["h2"])
