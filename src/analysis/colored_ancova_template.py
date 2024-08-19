import os

import numpy as np
import pandas as pd

from src.base.data_types import BrainTissue, ImageType
from src.base.database_columns import DATASETS, DemographicsColumns
from src.settings import RESULTS_DIR

roi_metric = "mean"

for source_ancova in [DemographicsColumns.AGE_AT_SCAN, DemographicsColumns.DIAGNOSTIC_GROUP]:

    for var_evaluate in [ImageType.EXPLAINABILITY_MAPS_SINGLE, ImageType.PREPROCESSED_IMAGE]:
        var_name = f"{var_evaluate.name}_{roi_metric}"

        path_save_img = "/home/fmachado/Dropbox/thesis/Chapter7/Figures/pathology"

        for dataset, site in [(DATASETS.IXI, ["HH"]), (DATASETS.DIAMARKER, None), (DATASETS.COBRE, None),
                              (DATASETS.CIBIT_CONTROL_AD, None)]:

            dataset_name = dataset.name
            if site is not None:
                dataset_name = f"{dataset.name}--{'-'.join(site)}"

            for tissue in [BrainTissue.GM, BrainTissue.WM, BrainTissue.ALL_TISSUE_SANLM_LOCAL,
                           BrainTissue.CSF, BrainTissue.DF]:
                filename = f"ancova_{source_ancova}_{var_name}_{dataset_name}_{tissue.name}"
                path_rois = os.path.join(RESULTS_DIR, filename + ".csv")

                df = pd.read_csv(path_rois)
                col_roi_name = 'roi'
                col_value = 'p-value'

                template = "/home/fmachado/Desktop/templates_volumes/neuromorphometrics.nii"
                filename_ids_names = "/home/fmachado/Desktop/templates_volumes/neuromorphometrics.csv"

                df_map_roi_names = pd.read_csv(filename_ids_names, sep=";")
                col_roi_name_map = "ROIname"
                col_roi_id_map = "ROIid"

                import nibabel as nib

                img = nib.load(template)
                data = img.get_fdata()

                colored_img = np.zeros_like(data) + (data > 1).astype(int)
                for _, row in df.iterrows():
                    roi_name = row[col_roi_name]
                    roi_id = df_map_roi_names.loc[df_map_roi_names[col_roi_name_map] == roi_name][col_roi_id_map].iloc[0]
                    if roi_id == 0:
                        continue
                    mask = (data == roi_id) * 2
                    colored_img += mask

                from nilearn.plotting import plot_anat
                import matplotlib.pyplot as plt
                img_mask = nib.Nifti1Image(colored_img, affine=img.affine)
                plot_anat(img_mask, cut_coords=(9, -31, 4), cmap='inferno', black_bg="auto", vmin=0, vmax=5,
                          output_file=os.path.join(path_save_img, f"{filename}.png"))
                plt.show()
