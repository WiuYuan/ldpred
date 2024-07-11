import LDpred as ld
import time

if __name__ == "__main__":
    start_time = time.time()
    pwd = "/Users/yuanwen/Desktop/Docker_Environment/intern/1"
    precision_folder_path = pwd + "/data/EUR"
    snplist_folder_path = pwd + "/data/snplists_GRCh38positions"
    sumstats_path = pwd + "/data/sumstats/BMI_no_dup.txt"
    output_path = pwd + "/data/sumstats/output.sumstats"
    sumstats = ld.Read.sumstats_read(sumstats_path)
    ld.Fliter.fliter_by_unique_sumstats(sumstats)
    PM = ld.Read.PM_read(precision_folder_path)
    snplist = ld.Read.snplist_read(snplist_folder_path)
    ld.Fliter.filter_by_PM(PM, snplist)
    ld.Fliter.normalize_PM(PM)
    sumstats_set = ld.Fliter.fliter_by_sumstats_parallel(PM, snplist, sumstats)
    ld.Check.check_same_rsid(snplist, sumstats_set)
    para = ld.generate.get_para()
    beta_ldgm, p, h2 = ld.mymodel.ldgm_gibbs_block_auto_parallel(
        PM, snplist, sumstats_set, para
    )
    end_time = time.time()
    ld.Write.sumstats_beta_write(
        sumstats_set, beta_ldgm, output_path, end_time - start_time, p, h2
    )
