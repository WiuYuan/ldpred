import LDpred as ld
import time
from mpi4py import MPI

if __name__ == "__main__":
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pwd = "/Users/yuanwen/Desktop/Docker_Environment/intern/1"
    precision_folder_path = pwd + "/data/EUR"
    snplist_folder_path = pwd + "/data/snplists_GRCh38positions"
    sumstats_path = pwd + "/data/sumstats/body_BMIz.sumstats"
    output_path = pwd + "/data/sumstats/output.sumstats"
    if rank == 0:
        PM = ld.Read.PM_read(precision_folder_path)
        snplist = ld.Read.snplist_read(snplist_folder_path)
        ld.Fliter.filter_by_PM(PM, snplist)
        sumstats = ld.Read.sumstats_read(sumstats_path)
        sumstats_set = ld.Fliter.fliter_by_sumstats_parallel(snplist, sumstats)
        ld.Fliter.normalize_PM_parallel(PM)
        para = ld.generate.get_para()
        para["N"] = float(sumstats["N"][0])
        beta_ldgm = ld.mymodel.ldgm_gibbs_block_auto_parallel(
            PM, snplist, sumstats_set, para
        )
        end_time = time.time()
        ld.Write.sumstats_beta_write(
            sumstats_set, beta_ldgm, output_path, end_time - start_time
        )
    else:
        rsid_sumstats = None
        rsid_sumstats = comm.bcast(rsid_sumstats, root=0)
        jud = 1
        while jud:
            jud = ld.Fliter.fliter_by_sumstats_parallel_subprocess(rsid_sumstats, 0)
        jud = 1
        while jud:
            jud = ld.Fliter.normalize_PM_parallel_subprocess(0)
        jud = 1
        while jud:
            jud = ld.mymodel.ldgm_gibbs_block_auto_parallel_subprocess(0)
