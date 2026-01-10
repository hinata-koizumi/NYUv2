import numpy as np

def align_file_ids(target_ids, source_ids, source_data):
    """
    Align source_data to match target_ids order.
    Assumes all target_ids exist in source_ids.
    """
    sorter = np.argsort(source_ids)
    indices = sorter[np.searchsorted(source_ids, target_ids, sorter=sorter)]
    return source_data[indices]
