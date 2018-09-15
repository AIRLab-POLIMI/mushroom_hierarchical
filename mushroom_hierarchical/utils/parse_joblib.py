def parse_joblib(res):
    res_list = list()

    n_results = len(res[0])

    for i in range(n_results):
        res_i = [r[i] for r in res]
        res_list.append(res_i)

    return tuple(res_list)
