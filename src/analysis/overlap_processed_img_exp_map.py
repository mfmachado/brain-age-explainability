import os

import pandas as pd

from src.base.data_types import ImageType, BrainTissue
from src.base.database_columns import DATASETS
from src.settings import RESULTS_DIR

image_type_baseline = ImageType.PREPROCESSED_IMAGE
image_type_analyse = ImageType.EXPLAINABILITY_MAPS_SINGLE

for variable in ["age", "diagnosticGroup"]:
    all_rows = []
    for brain_tissue in [BrainTissue.ALL_TISSUE_SANLM_LOCAL, BrainTissue.GM, BrainTissue.WM, BrainTissue.CSF,
                         BrainTissue.DF]:

        for dataset, site in [(DATASETS.IXI, ["HH"]), (DATASETS.DIAMARKER, None), (DATASETS.COBRE, None),
                              (DATASETS.CIBIT_CONTROL_AD, None)]:
            dataset_name = dataset.name
            if site is not None:
                dataset_name = f"{dataset.name}--{'-'.join(site)}"

            baseline_path_file = os.path.join(RESULTS_DIR,
                                              f"ancova_{variable}_{image_type_baseline.name}_mean_{dataset_name}_"
                                              f"{brain_tissue.name}.csv")

            df = pd.read_csv(baseline_path_file)
            sig_rois = df["roi"]

            path_dt = os.path.join(RESULTS_DIR,
                                   f"ancova_{variable}_{image_type_analyse.name}_mean_{dataset_name}_{brain_tissue.name}.csv")
            df_dt = pd.read_csv(path_dt)

            rois_intersection = set(sig_rois).intersection(set(df_dt["roi"]))
            den_dice = (df_dt["roi"].shape[0] + df["roi"].shape[0])
            all_positive_rois = list(set(df["roi"].to_list() + df_dt["roi"].to_list()))
            den_jaccard = len(all_positive_rois)

            if den_dice == 0:
                dice = 0
            else:
                dice = (2 * len(rois_intersection) / (df_dt["roi"].shape[0] + df["roi"].shape[0]))
            if den_jaccard == 0:
                jaccard = 0
            else:
                jaccard = len(rois_intersection) / len(all_positive_rois)

            all_rows.append([brain_tissue.name, dataset.name, dice, jaccard])

    df_overlap = pd.DataFrame(all_rows, columns=["tissue", "dataset", "dice", "jaccard"])

    df_ = pd.pivot_table(data=df_overlap, index=["tissue"],
                         values=["jaccard"],
                         columns=["dataset"]).round(2)

    df_ = pd.concat([df_, (df_.sum() / df_.count()).to_frame().T])

    df_.columns = df_.columns.droplevel()

    df_ = df_.reindex(index=[BrainTissue.ALL_TISSUE_SANLM_LOCAL.name, BrainTissue.GM.name,
                             BrainTissue.WM.name, BrainTissue.CSF.name, BrainTissue.DF.name])

    df_.to_csv(os.path.join(RESULTS_DIR, f"overlap_{variable}_with_"
                                         f"{image_type_baseline.name}_vs_{image_type_analyse.name}.csv"))

    df_ = df_[["COBRE", "DIAMARKER", "CIBIT_CONTROL_AD"]]
    latex_txt = df_.to_latex().replace("0000 ", " ").replace("ALL_TISSUE_SANLM_LOCAL",
                                                             "Minimally processed").replace("CIBIT_CONTROL_AD", "CIBIT AD")

    table_dir = "/home/fmachado/Dropbox/thesis/Chapter7/Tables"
    path_latex = os.path.join(table_dir, f"overlap_{variable}_with_"
                                         f"{image_type_baseline.name}_vs_{image_type_analyse.name}.txt")
    with open(path_latex, 'w') as f:
        f.write(latex_txt)
