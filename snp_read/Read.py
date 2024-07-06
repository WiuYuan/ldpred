# read precision matrix

import os
from scipy.sparse import csr_matrix


def snplist_read(snplist_folder_path):
    snplist = []
    for snplist_file in sorted(
        os.listdir(snplist_folder_path),
        key=lambda x: (
            int(
                x.split("_")[1][3:]
            ),  # Extract numeric part after 'chr' and convert to int
            int(x.split("_")[2]),  # Start position, convert to int
            int(x.split("_")[3].split(".")[0]),  # End position, convert to int
        ),
    ):
        snplist_block = {}
        with open(snplist_folder_path + "/" + snplist_file, "r") as file:
            tot = 0
            title_list = []
            index_dict = {}
            for line in file:
                line_list = line.strip().split(",")
                if tot > 0 and line_list[0] in index_dict:
                    continue
                if tot == 0:
                    title_list = line_list
                    for i in range(len(title_list)):
                        snplist_block[title_list[i]] = []
                else:
                    for i in range(len(title_list)):
                        snplist_block[title_list[i]].append(line_list[i])
                    index_dict[snplist_block["index"][tot - 1]] = tot - 1
                tot = tot + 1
        snplist_block["filename"] = snplist_file
        snplist_block["rsid"] = snplist_block.pop("site_ids")
        snplist_block["REF"] = snplist_block.pop("anc_alleles")
        snplist_block["ALT"] = snplist_block.pop("deriv_alleles")
        snplist.append(snplist_block)
        print("Read snplist file:", snplist_file)
    return snplist


def PM_read(precision_folder_path):
    PM = []
    for PM_file in sorted(
        os.listdir(precision_folder_path),
        key=lambda x: (
            int(
                x.split("_")[1][3:]
            ),  # Extract numeric part after 'chr' and convert to int
            int(x.split("_")[2]),  # Start position, convert to int
            int(x.split("_")[3].split(".")[0]),  # End position, convert to int
        ),
    ):
        PM_block = {}
        rows = []
        cols = []
        data = []
        with open(precision_folder_path + "/" + PM_file, "r") as file:
            for line in file:
                row_idx, col_idx, value = map(float, line.strip().split(","))
                rows.append(int(row_idx))
                cols.append(int(col_idx))
                data.append(value)
        PM_block["precision"] = csr_matrix((data, (rows, cols)))
        PM_block["filename"] = PM_file
        PM.append(PM_block)
        print("Read Precision matrix file:", PM_file)
    return PM


# def sumstats_read(sumstats_path):
#     sumstats = {}
#     with open(sumstats_path, "r") as file:
#         tot = 0
#         for line in file:
#             line_list = line.strip().split(",")
#             if tot == 0:
#                 title_list = line_list
#                 for i in range(len(title_list)):
#                     sumstats[title_list[i]] = []
#             else:
#                 for i in range(len(title_list)):
#                     sumstats[title_list[i]].append(line_list[i])
#             tot = tot + 1
#     sumstats["REF"] = sumstats.pop("a0")
#     sumstats["ALT"] = sumstats.pop("a1")
#     return sumstats


def sumstats_read(sumstats_path):
    sumstats = {}
    sumstats["ALT"] = []
    with open(sumstats_path, "r") as file:
        tot = 0
        for line in file:
            line_list = line.strip().split("\t")
            if tot == 0:
                title_list = line_list
                for i in range(len(title_list)):
                    sumstats[title_list[i]] = []
            else:
                for i in range(len(title_list)):
                    sumstats[title_list[i]].append(line_list[i])
                if sumstats["A1"][tot - 1] == sumstats["REF"][tot - 1]:
                    sumstats["ALT"].append(sumstats["A2"][tot - 1])
                else:
                    sumstats["ALT"].append(sumstats["A1"][tot - 1])
            tot = tot + 1
            if tot % 100000 == 0:
                print("Read sumstats line:", tot)
    sumstats["rsid"] = sumstats.pop("SNP")
    sumstats["beta"] = sumstats.pop("Beta")
    sumstats["beta_se"] = sumstats.pop("se")
    return sumstats


def vcf_read(vcf_path):
    vcfstats = {}
    with open(vcf_path, "r") as file:
        tot = 0
        for line in file:
            if line[0:2] == "##":
                continue
            line_list = line.strip().split("\t")
            if tot == 0:
                title_list = line_list
                for i in range(len(title_list)):
                    vcfstats[title_list[i]] = []
            else:
                for i in range(len(title_list)):
                    vcfstats[title_list[i]].append(line_list[i])
            tot = tot + 1
    vcfstats["rsid"] = vcfstats.pop("ID")
    return vcfstats


def fam_read(fam_path):
    famstats = {}
    title_list = [
        "FamilyID",
        "IndividualID",
        "PaternalID",
        "MaternalID",
        "Sex",
        "Phenotype",
    ]
    for i in range(len(title_list)):
        famstats[title_list[i]] = []
    with open(fam_path, "r") as file:
        for line in file:
            line_list = line.strip().split("\t")
            for i in range(len(title_list)):
                famstats[title_list[i]].append(line_list[i])
    return famstats
