def sumstats_beta_write(sumstats, beta, output_path, run_time, p, h2):
    # output_key = ["rsid", "CHR", "POS", "REF", "ALT", "beta", "beta_se", "P", "N"]
    output_key = ["rsid", "REF"]
    with open(output_path, "w") as f:
        f.write(f"## Running time: {run_time:.4f} seconds\n")
        f.write(f"## p = {p:.4f} h2 = {h2:.4f}\n")
        f.write("\t".join(output_key))
        f.write("\tbeta_joint\n")
        for i in range(len(sumstats)):
            for j in range(len(beta[i])):
                for key in output_key:
                    if key in sumstats[i]:
                        if isinstance(sumstats[i][key][j], str):
                            f.write(sumstats[i][key][j] + "\t")
                        elif key == "N":
                            f.write(f"{int(sumstats[i][key][j])}\t")
                        else:
                            f.write(f"{sumstats[i][key][j]:.6f}\t")
                    else:
                        f.write("NA\t")
                f.write(f"{beta[i][j]:.6f}")
                f.write("\n")
