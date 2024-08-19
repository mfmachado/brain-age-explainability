import os

import numpy as np
import pandas as pd

from src.base.data_types import BrainTissue, ImageType
from src.base.database_columns import DATASETS, DemographicsColumns
from src.settings import RESULTS_DIR

roi_metric = "mean"

var_evaluate1 = f"{ImageType.EXPLAINABILITY_MAPS_SINGLE.name}_mean"
var_evaluate2 = "age"
col_value = "0"
mask_img = False
path_save_img = "/home/fmachado/Dropbox/thesis/Chapter9/Figures/pathology"
for dataset, site in [(DATASETS.IXI, ["HH"])]:

    dataset_name = dataset.name
    if site is not None:
        dataset_name = f"{dataset.name}--{'-'.join(site)}"

    for tissue in [BrainTissue.ALL_TISSUE_SANLM_LOCAL]:
        filename = f"{var_evaluate1}_pearson_{var_evaluate2}_{dataset_name}_{tissue.name}"
        path_rois = os.path.join(RESULTS_DIR, filename + ".csv")

        col_roi_name = 'level_1'

        df = pd.read_csv(path_rois)
        df = df.groupby(col_roi_name)[[col_value]].mean().reset_index()

        template = "/home/fmachado/Desktop/templates_volumes/neuromorphometrics.nii"
        filename_ids_names = "/home/fmachado/Desktop/templates_volumes/neuromorphometrics.csv"

        df_map_roi_names = pd.read_csv(filename_ids_names, sep=";")
        col_roi_name_map = "ROIname"
        col_roi_id_map = "ROIid"

        import nibabel as nib

        img = nib.load(template)
        data = img.get_fdata()

        colored_img = np.zeros_like(data)
        for _, row in df.iterrows():
            roi_name = row[col_roi_name]
            roi_id = df_map_roi_names.loc[df_map_roi_names[col_roi_name_map] == roi_name][col_roi_id_map].iloc[0]
            if roi_id == 0:
                continue
            roi_colored = row[col_value]
            if mask_img:
                roi_colored = roi_colored > 0
            mask = (data == roi_id)*roi_colored
            colored_img += mask

        from nilearn.plotting import plot_anat
        import matplotlib.pyplot as plt
        img_mask = nib.Nifti1Image(colored_img, affine=img.affine)
        filename_save = f"{filename}.png"
        if mask_img:
            filename_save = "mask_" + filename_save
        plot_anat(img_mask, cut_coords=(9, -31, 4), cmap='inferno', black_bg="auto",
                  vmin=0, vmax=1, colorbar=True)
        #output_file=os.path.join(path_save_img, filename_save)
        plt.show()

