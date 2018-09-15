def check_no_of_eps(dataset):
    no_of_eps = 0

    for dataset_step in dataset:
        if dataset_step[-1]:
            no_of_eps +=1
    return no_of_eps