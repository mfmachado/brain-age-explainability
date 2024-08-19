import os

import numpy as np
import pandas as pd

from src.base.data_types import ImageType, BrainTissue
from src.base.database_columns import DATASETS
from src.settings import RESULTS_DIR


def get_dataset_name(dataset, site):
    dataset_name = dataset.name
    if site is not None:
        dataset_name = f"{dataset.name}--{'-'.join(site)}"
    return dataset_name


image_type = ImageType.PREPROCESSED_IMAGE

variable = "age"
all_rows = []

brain_tissues_all = [BrainTissue.DF, BrainTissue.ALL_TISSUE_SANLM_LOCAL, BrainTissue.GM, BrainTissue.WM,
                     BrainTissue.CSF]
datasets = [(DATASETS.IXI, ["HH"]), (DATASETS.DIAMARKER, None), (DATASETS.COBRE, None),
            (DATASETS.CIBIT_CONTROL_AD, None)]
for brain_tissue_i in brain_tissues_all:

    tissues_dataset_jaccard = []

    for i, (dataset_i, site_i) in enumerate(datasets):

        dataset_name_i = get_dataset_name(dataset_i, site_i)
        baseline_path_file = os.path.join(RESULTS_DIR,
                                          f"ancova_{variable}_{image_type.name}_mean_{dataset_name_i}_"
                                          f"{brain_tissue_i.name}.csv")
        df = pd.read_csv(baseline_path_file)
        for j, (dataset_j, site_j) in enumerate(datasets[i + 1:]):
            dataset_name_j = get_dataset_name(dataset_j, site_j)

            sig_rois = df["roi"]

            path_dt = os.path.join(RESULTS_DIR, f"ancova_{variable}_{image_type.name}_mean_{dataset_name_j}_"
                                                f"{brain_tissue_i.name}.csv")
            df_dt = pd.read_csv(path_dt)

            rois_intersection = set(sig_rois).intersection(set(df_dt["roi"]))
            dice = (2 * len(rois_intersection) / (df_dt["roi"].shape[0] + df["roi"].shape[0]))
            all_positive_rois = list(set(df["roi"].to_list() + df_dt["roi"].to_list()))
            jaccard = len(rois_intersection) / len(all_positive_rois)
            tissues_dataset_jaccard.append(jaccard)
    all_rows.append([brain_tissue_i.name, np.mean(tissues_dataset_jaccard), np.max(tissues_dataset_jaccard),
                     np.min(tissues_dataset_jaccard)])

df_jaccard_ = pd.DataFrame(all_rows, columns=["tissue", "mean jaccard", "max jaccard", "min jaccard"])

df_jaccard_.to_csv(os.path.join(RESULTS_DIR, f"overlap_{variable}_with_{image_type.name}_jaccard_tissues.csv"))
