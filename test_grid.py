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
    pset = [
        1.0e-04,
        5.6e-04,
        3.2e-03,
        1.0e-02,
        5.6e-02,
        1.8e-01,
        5.6e-01,
        1.0e00,
    ]
    h2set = [0.0890, 0.2076, 0.4152]
    for i in range(len(pset)):
        for j in range(len(h2set)):
            output_path = (
                pwd
                + "/data/sumstats/grid/output_p"
                + str(i)
                + "_h2_"
                + str(j)
                + ".sumstats"
            )
            para["p"] = pset[i]
            para["h2"] = h2set[j]
            beta_ldgm = ld.mymodel.ldgm_gibbs_block_grid_parallel(
                PM, snplist, sumstats, para
            )
            end_time = time.time()
            ld.Write.sumstats_beta_write(
                sumstats,
                beta_ldgm,
                output_path,
                end_time - start_time,
                para["p"],
                para["h2"],
            )
