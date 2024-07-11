def check_same_rsid(snplist, sumstats):
    for i in range(len(snplist)):
        if snplist[i]["rsid"] != sumstats[i]["rsid"]:
            raise Exception("The rsid of the snplist are not same as sumstats!")
