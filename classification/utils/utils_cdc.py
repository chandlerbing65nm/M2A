import numpy as np


def create_cdc_sequence(num_domains: int,
                        distribution: str = "dirichlet",
                        delta: float = 1.0,
                        num_total_batches: int = 1):
    """Create a continual domain curriculum (CDC) sequence over domains.

    This is a generalized version of the CDC helper used in the CIFAR CTTA
    code. It returns a list of domain indices in ``[0, num_domains-1]`` that
    specifies, for each batch step, from which domain/corruption the batch
    should be drawn.

    Args:
        num_domains: Number of distinct domains/corruptions.
        distribution: Either 'dirichlet' or 'multinomial' (default: 'dirichlet').
        delta: Concentration parameter for the Dirichlet distribution.
        num_total_batches: Approximate number of batches per domain.

    Returns:
        A list of integers of length roughly ``num_domains * num_total_batches``
        indicating the domain index for each batch step.
    """

    assert num_domains > 0, "num_domains must be positive"
    num_total_batches = max(1, int(num_total_batches))

    domains = list(range(num_domains))
    domain_order = []

    if distribution == "multinomial":
        remaining_batches = {d: num_total_batches for d in domains}
        while remaining_batches:
            selected = np.random.choice(list(remaining_batches.keys()), 1)[0]
            num_sel = np.random.choice(
                list(range(1, remaining_batches[selected] + 1))
            )
            remaining_batches[selected] -= num_sel
            if remaining_batches[selected] == 0:
                del remaining_batches[selected]
            domain_order.extend([selected] * num_sel)

    else:  # 'dirichlet' or any other string defaults to dirichlet behavior
        slot_num = 3
        # For each domain, sample a Dirichlet over slots
        label_distribution = np.random.dirichlet([delta] * slot_num, num_domains)
        slot_indices = [[] for _ in range(slot_num)]
        class_indices = [[d] * num_total_batches for d in domains]

        for c_ids, partition in zip(class_indices, label_distribution):
            # Split this domain's occurrences across slots according to partition
            splits = np.split(
                np.array(c_ids),
                (np.cumsum(partition)[:-1] * len(c_ids)).astype(int),
            )
            for s, ids in enumerate(splits):
                if len(ids) > 0:
                    slot_indices[s].append(ids)

        # For each slot, randomly permute the domain segments and concatenate
        for s_ids in slot_indices:
            if not s_ids:
                continue
            permutation = np.random.permutation(len(s_ids))
            for idx in permutation:
                ids = s_ids[idx]
                # ids is an array filled with a single domain index
                domain_order.extend(ids.tolist())

    return domain_order
