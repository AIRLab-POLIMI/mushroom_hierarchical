
def pick_eps(dataset, start, end):

    dataset_ep = list()
    dataset_ep_list = list()
    for dataset_step in dataset:
        if not dataset_step[-1]:
            dataset_ep.append(dataset_step)
        else:
            dataset_ep.append(dataset_step)
            dataset_ep_list.append(dataset_ep)
            dataset_ep = list()
    return dataset_ep_list[start:end]


