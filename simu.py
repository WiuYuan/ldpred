import LDpred as ld
import time

if __name__ == "__main__":
    start_time = time.time()
    pwd = "/Users/yuanwen/Desktop/Docker_Environment/intern/1"
    output_path = pwd + "/data/sumstats/output.sumstats"
    para = ld.generate.get_para()
    PM, snplist = ld.generate.generate_PM_snplist(para)
    sumstats, beta_true = ld.generate.generate_sumstats_beta(PM, snplist, para)
    beta_true_total = ld.Fliter.merge_beta(beta_true)
    beta_ldgm, p, h2 = ld.mymodel.ldgm_gibbs_block_auto_parallel(
        PM, snplist, sumstats, para
    )
    beta_ldgm_total = ld.Fliter.merge_beta(beta_ldgm)
    beta_auto = ld.LDpred2.ldpred2_auto(PM, snplist, sumstats, para)
    beta_auto_total = ld.Fliter.merge_beta(beta_auto)
    beta_auto2 = ld.LDpred2.ldpred2_auto(PM, snplist, sumstats, para)
    beta_auto2_total = ld.Fliter.merge_beta(beta_auto2)
    beta_inf = ld.LDpred2.ldpred2_inf(PM, snplist, sumstats, para)
    beta_inf_total = ld.Fliter.merge_beta(beta_inf)
    print(
        "inf r2:",
        ld.predict.check_r_squared_beta(beta_inf_total, beta_true_total),
        "ldgm_auto r2:",
        ld.predict.check_r_squared_beta(beta_ldgm_total, beta_true_total),
        "auto r2:",
        ld.predict.check_r_squared_beta(beta_auto_total, beta_true_total),
        "auto r2:",
        ld.predict.check_r_squared_beta(beta_auto2_total, beta_true_total),
    )
    end_time = time.time()
    ld.Write.sumstats_beta_write(
        sumstats, beta_ldgm, output_path, end_time - start_time, p, h2
    )
