import numpy as np
import scipy.sparse as sp
import multiprocessing


def filter_by_PM(PM, snplist):
    for i in range(len(PM)):
        PMid = np.where(np.diff(PM[i]["precision"].indptr) != 0)[0]
        PM[i]["precision"] = PM[i]["precision"][PMid][:, PMid]
        for key in list(snplist[i].keys()):
            if isinstance(snplist[i][key], list):
                snplist[i][key] = np.array(snplist[i][key])[PMid].tolist()
        if i % 30 == 0:
            print("Fliter_by_PM block:", i)
        snplist[i]["index"] = np.arange(len(snplist[i]["rsid"])).tolist()


def normalize_PM_subprocess(subinput):
    P, i = subinput
    print("normalize_PM_subprocess block:", i)
    D = np.sqrt(sp.linalg.inv(P.tocsc()).diagonal())
    return P.multiply(np.outer(D, D)).tocsr()


def normalize_PM(PM):
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    subinput = []
    for i in range(len(PM)):
        subinput.append((PM[i]["precision"], i))
    results = pool.map(normalize_PM_subprocess, subinput)
    for i in range(len(PM)):
        PM[i]["precision"] = results[i]


def normalize_dense_matrix(A):
    diag_elements = np.diag(A)
    sqrt_diag_outer = np.sqrt(np.outer(diag_elements, diag_elements))
    return A / sqrt_diag_outer


def fliter_by_sumstats_subprocess(subinput):
    rsid_sumstats, snplist_rsid, i = subinput
    rsid1 = [
        index for index, value in enumerate(snplist_rsid) if value in rsid_sumstats
    ]
    rsid2 = [
        rsid_sumstats[value]
        for index, value in enumerate(snplist_rsid)
        if value in rsid_sumstats
    ]
    print("Fliter_by_sumstats parallel block:", i)
    return rsid1, rsid2


def fliter_by_sumstats_parallel(PM, snplist, sumstats):
    sumstats_set = []
    rsid_sumstats = {value: index for index, value in enumerate(sumstats["rsid"])}
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    subinput = []
    for i in range(len(PM)):
        subinput.append((rsid_sumstats, snplist[i]["rsid"], i))
    for key in list(sumstats.keys()):
        if isinstance(sumstats[key], list):
            sumstats[key] = np.array(sumstats[key])
    results = pool.map(fliter_by_sumstats_subprocess, subinput)
    for i in range(len(PM)):
        rsid1, rsid2 = results[i]
        for key in list(snplist[i].keys()):
            if isinstance(snplist[i][key], list):
                snplist[i][key] = np.array(snplist[i][key])[rsid1].tolist()
        sumstats_block = {}
        for key in list(sumstats.keys()):
            if isinstance(sumstats[key], np.ndarray):
                sumstats_block[key] = sumstats[key][rsid2].tolist()
        sumstats_set.append(sumstats_block)
        print("Fliter_by_sumstats results block:", i)
    return sumstats_set


def fliter_by_sumstats(PM, snplist, sumstats):
    sumstats_set = []
    rsid_sumstats = {value: index for index, value in enumerate(sumstats["rsid"])}
    for i in range(len(PM)):
        rsid = [
            index
            for index, value in enumerate(snplist[i]["rsid"])
            if value in rsid_sumstats
        ]
        # I_dense = np.eye(PM[i]["precision"].shape[0])
        # PM[i]["LD"] = csr_matrix(
        #     normalize_dense_matrix(
        #         splu(PM[i]["precision"]).solve(I_dense[:, rsid])[rsid]
        #     )
        # )
        # PM[i]["precision"] = PM[i]["precision"][rsid][:, rsid]
        # PM[i]["LD"] = PM[i]["LD"][rsid][:, rsid]
        for key in list(snplist[i].keys()):
            if isinstance(snplist[i][key], list):
                snplist[i][key] = np.array(snplist[i][key])[rsid].tolist()
        rsid = [
            rsid_sumstats[value]
            for index, value in enumerate(snplist[i]["rsid"])
            if value in rsid_sumstats
        ]
        sumstats_block = {}
        for key in list(sumstats.keys()):
            if isinstance(sumstats[key], list):
                sumstats_block[key] = np.array(sumstats[key])[rsid].tolist()
        sumstats_set.append(sumstats_block)
        print("Fliter_by_sumstats block:", i)
    return sumstats_set


def merge_vcf_fam(vcfstats, famstats):
    phestats = {}
    N = len(famstats["IndividualID"])
    M = len(vcfstats["rsid"])
    X = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            u = vcfstats[famstats["IndividualID"][i]][j]
            v = u.strip().split("/")
            X[i][j] = int(v[0]) + int(v[1])
    phestats["X"] = X
    phestats["chr"] = vcfstats["#CHROM"][:]
    phestats["rsid"] = vcfstats["rsid"][:]
    phestats["position"] = vcfstats["POS"][:]
    phestats["REF"] = vcfstats["REF"][:]
    phestats["ALT"] = vcfstats["ALT"][:]
    phestats["Phenotype"] = np.array(famstats["Phenotype"])
    return phestats


def fliter_by_vcf(beta_list, snplist, phestats):
    phestats_total = {}
    rsid_total = []
    rsid_phestats = {value: index for index, value in enumerate(phestats["rsid"])}
    for key in list(phestats.keys()):
        if isinstance(phestats[key], list):
            phestats_total[key] = []
    for i in range(len(snplist)):
        rsid = [
            index
            for index, value in enumerate(snplist[i]["rsid"])
            if value in rsid_phestats
        ]
        for key in list(snplist[i].keys()):
            if isinstance(snplist[i][key], list):
                snplist[i][key] = np.array(snplist[i][key])[rsid].tolist()
        for beta in beta_list:
            beta[i] = np.array(beta[i])[rsid].tolist()
        rsid_snplist_block = {
            value: index for index, value in enumerate(snplist[i]["rsid"])
        }
        rsid = [
            index
            for index, value in enumerate(phestats["rsid"])
            if value in rsid_snplist_block
        ]
        for key in list(phestats.keys()):
            if isinstance(phestats[key], list):
                phestats_total[key].append(np.array(phestats[key])[rsid].tolist())
        rsid_total += rsid
    phestats_total["X"] = phestats["X"][:, rsid_total]
    phestats_total["Phenotype"] = phestats["Phenotype"]
    return phestats_total


def merge_beta(beta):
    beta_total = []
    for i in range(len(beta)):
        if isinstance(beta[i], list) == 0:
            beta_total += beta[i].tolist()
        else:
            beta_total += beta[i]
    return np.array(beta_total)


def fliter_by_REF_ALT(snplist, phestats_total):
    rsid_total = []
    # base_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
    si = 0
    for i in range(len(snplist)):
        for j in range(len(phestats_total["rsid"][i])):
            if (
                snplist[i]["REF"][j] == phestats_total["ALT"][i][j]
                and snplist[i]["ALT"][j] == phestats_total["REF"][i][j]
            ):
                phestats_total["X"][:, si + j] = 2 - phestats_total["X"][:, si + j]
        si += len(phestats_total["rsid"][i])


# def PM_get_LD(PM):
#     for i in range(len(PM)):
#         PM[i]["LD"] = inv(PM[i]["precision"])
