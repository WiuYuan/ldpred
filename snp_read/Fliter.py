import numpy as np
import scipy.sparse as sp
from mpi4py import MPI


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


def normalize_PM_parallel_subprocess(subinput):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
        subinput = comm.recv(source=0)
    if isinstance(subinput, str):
        return 0
    P, i = subinput
    print("normalize_PM_subprocess block:", i)
    D = np.sqrt(sp.linalg.inv(P.tocsc()).diagonal())
    if rank != 0:
        comm.send(P.multiply(np.outer(D, D)).tocsr(), dest=0)
        return 1
    return P.multiply(np.outer(D, D)).tocsr()


def normalize_PM_parallel(PM):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    results = []
    for i in range(len(PM)):
        subinput = (PM[i]["precision"], i)
        if i % size == size - 1:
            result0 = normalize_PM_parallel_subprocess(subinput)
            for i in range(1, size):
                results.append(comm.recv(source=i))
            results.append(result0)
        else:
            dest = i % size + 1
            comm.send(subinput, dest=dest)
    if len(PM) % size != 0:
        for i in range(1, len(PM) % size + 1):
            results.append(comm.recv(source=i))
    for i in range(len(PM)):
        PM[i]["precision"] = results[i]
    for i in range(1, size):
        comm.send("done", dest=i)


def normalize_dense_matrix(A):
    diag_elements = np.diag(A)
    sqrt_diag_outer = np.sqrt(np.outer(diag_elements, diag_elements))
    return A / sqrt_diag_outer


def fliter_by_sumstats_parallel_subprocess(rsid_sumstats, subinput):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
        subinput = comm.recv(source=0)
    if isinstance(subinput, str):
        return 0
    snplist_rsid, i = subinput
    rsid1 = [
        index for index, value in enumerate(snplist_rsid) if value in rsid_sumstats
    ]
    rsid2 = [
        rsid_sumstats[value]
        for index, value in enumerate(snplist_rsid)
        if value in rsid_sumstats
    ]
    print("Fliter_by_sumstats parallel block:", i)
    if rank != 0:
        comm.send((rsid1, rsid2), dest=0)
        return 1
    return rsid1, rsid2


def fliter_by_sumstats_parallel(snplist, sumstats):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    sumstats_set = []
    rsid_sumstats = {value: index for index, value in enumerate(sumstats["rsid"])}
    rsid_sumstats = comm.bcast(rsid_sumstats, root=0)
    subinput = []
    results = []
    for i in range(len(snplist)):
        subinput = (snplist[i]["rsid"], i)
        if i % size == size - 1:
            result0 = fliter_by_sumstats_parallel_subprocess(rsid_sumstats, subinput)
            for i in range(1, size):
                results.append(comm.recv(source=i))
            results.append(result0)
        else:
            dest = i % size + 1
            comm.send(subinput, dest=dest)
    if len(snplist) % size != 0:
        for i in range(1, len(snplist) % size + 1):
            results.append(comm.recv(source=i))
    for key in list(sumstats.keys()):
        if isinstance(sumstats[key], list):
            sumstats[key] = np.array(sumstats[key])
    for i in range(len(snplist)):
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
    for i in range(1, size):
        comm.send("done", dest=i)
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
