import LDpred as ld

if __name__ == "__main__":
    pwd = "/Users/yuanwen/Desktop/Docker_Environment/intern/1"
    precision_folder_path = pwd + "/data/EUR"
    snplist_folder_path = pwd + "/data/snplists_GRCh38positions"
    sumstats_path = pwd + "/data/sumstats/body_BMIz.sumstats"
    output_path = pwd + "/data/sumstats/output.sumstats"
    sumstats = ld.Read.sumstats_read(sumstats_path)
    PM = ld.Read.PM_read(precision_folder_path)
    snplist = ld.Read.snplist_read(snplist_folder_path)
    ld.Fliter.filter_by_PM(PM, snplist)
    sumstats_set = ld.Fliter.fliter_by_sumstats_parallel(PM, snplist, sumstats)
    para = ld.generate.get_para()
    para["N"] = float(sumstats["N"][0])
    beta_ldgm = ld.mymodel.ldgm_gibbs_block_auto_parallel(
        PM, snplist, sumstats_set, para
    )
    ld.Write.sumstats_beta_write(sumstats_set, beta_ldgm, output_path)
