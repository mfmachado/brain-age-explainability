import os

import numpy as np
import pandas as pd

from src.base.data_types import ImageType, BrainTissue
from src.base.database_columns import DATASETS
from src.settings import RESULTS_DIR

for image_type in [ImageType.PREPROCESSED_IMAGE, ImageType.EXPLAINABILITY_MAPS_SINGLE]:

    all_rows = []
    for brain_tissue in [BrainTissue.ALL_TISSUE_SANLM_LOCAL, BrainTissue.GM, BrainTissue.WM, BrainTissue.CSF,
                         BrainTissue.DF]:

        baseline_path_file = os.path.join(RESULTS_DIR, f"ancova_age_{image_type.name}_mean_IXI--HH_{brain_tissue.name}.csv")

        df = pd.read_csv(baseline_path_file)
        sig_rois = df["roi"]
        for dataset in [DATASETS.DIAMARKER, DATASETS.COBRE, DATASETS.CIBIT_CONTROL_AD]:
            path_dt = os.path.join(RESULTS_DIR,
                                   f"ancova_age_{image_type.name}_mean_{dataset.name}_{brain_tissue.name}.csv")
            df_dt = pd.read_csv(path_dt)

            rois_intersection = set(sig_rois).intersection(set(df_dt["roi"]))
            dice = (2 * len(rois_intersection) / (df_dt["roi"].shape[0] + df["roi"].shape[0]))
            all_positive_rois = list(set(df["roi"].to_list() + df_dt["roi"].to_list()))
            jaccard = len(rois_intersection) / len(all_positive_rois)
            all_rows.append([brain_tissue.name, dataset.name, dice, jaccard])

    df_overlap = pd.DataFrame(all_rows, columns=["tissue", "dataset", "dice", "jaccard"])

    df_ = pd.pivot_table(data=df_overlap, index=["tissue"],
                         values=["jaccard"],
                         columns=["dataset"]).round(2)

    df_ = pd.concat([df_, (df_.sum() / df_.count()).to_frame().T])

    df_ = df_.reindex(index=[BrainTissue.ALL_TISSUE_SANLM_LOCAL.name, BrainTissue.GM.name,
                             BrainTissue.WM.name, BrainTissue.CSF.name, BrainTissue.DF.name])

    df_.columns = df_.columns.droplevel()
    df_.to_csv(os.path.join(RESULTS_DIR, f"overlap_with_IXIHH_{image_type.name}.csv"))

    df_ = df_[["COBRE", "DIAMARKER", "CIBIT_CONTROL_AD"]]
    latex_txt = df_.to_latex().replace("0000 ", " ").replace("ALL_TISSUE_SANLM_LOCAL",
                                                             "Minimally processed").replace("CIBIT_CONTROL_AD", "CIBIT AD")

    table_dir = "/home/fmachado/Dropbox/thesis/Chapter7/Tables"
    path_latex = os.path.join(table_dir, f"overlap_with_IXIHH_{image_type.name}.txt")
    with open(path_latex, 'w') as f:
        f.write(latex_txt)
