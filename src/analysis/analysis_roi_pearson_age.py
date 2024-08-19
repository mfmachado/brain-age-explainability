import os.path

import pandas as pd
from scipy.stats import pearsonr

from src import settings
from src.base.data_types import BrainTissue, ImageType
from src.base.database_columns import PreprocessingPipeline, DATASETS, ExplainabilityMapsFusionCollection, \
    PreprocessedDataColumns, ROIValuesCollection, ExplainabilityMapsSingleCollection
from src.preprocessing.load_roi_features import ROIFeaturesLoader
from src.preprocessing.loader_explainability_maps import ExplainedMapsLoader
from src.preprocessing.loader_processed_data import PreprocessedImagesLoader
from src.preprocessing.loader_subjects import SubjectsLoader
from src.settings import RESULTS_DIR


def r2(X, y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X.to_numpy(), y.to_numpy())
    return reg.score(X.to_numpy(), y.to_numpy())


if __name__ == '__main__':
    exp_map_type = ImageType.EXPLAINABILITY_MAPS_SINGLE
    column_path = "col_abs_path"
    column_path_exp = "col_exp_abs_path"

    roi_metric = "mean"

    var_eval1 = (ImageType.PREPROCESSED_IMAGE.name, roi_metric) #(ImageType.EXPLAINABILITY_MAPS_SINGLE.name, roi_metric)
    var_eval2 = "age" #(ImageType.PREPROCESSED_IMAGE.name, roi_metric)

    for dataset, site in [(DATASETS.IXI, ["HH"])]:

        dataset_name = dataset.name
        if site is not None:
            dataset_name = f"{dataset.name}--{'-'.join(site)}"

        print("#########################################")
        print(dataset)
        print("#########################################")

        df_sub = SubjectsLoader().get_subjects(dataset, True, None,
                                               1, True, sites=site)
        noise_rows = []

        model_ids = {"ALL_TISSUE_SANLM_LOCAL": ("6543152bcdb3137a8d9cd131", 0.28),
                     "GM": ("6542f110f4cfa554fb83541d", 0.02),
                     "CSF": ("6542fd79abc67df1db788c8f", 0.06),
                     "WM": ("654309e88a995a5e67e23092", 0.02),
                     "DF": ("6542e4dc487d0e69df5e3410", 0.04)}

        tissues_ = [(BrainTissue.WM, "WM"), (BrainTissue.ALL_TISSUE_SANLM_LOCAL, "ALL_TISSUE_SANLM_LOCAL"),
                    (BrainTissue.GM, "GM"), (BrainTissue.CSF, "CSF"), (BrainTissue.DF, "DF")]

        for tissue, tissue_name in [(BrainTissue.ALL_TISSUE_SANLM_LOCAL, "ALL_TISSUE_SANLM_LOCAL")]:
            print("#########################################")
            print(dataset, tissue)
            print("#########################################")
            df_processed, _ = PreprocessedImagesLoader().get_images_per_subject(df_sub,
                                                                                PreprocessingPipeline[
                                                                                    settings.PREPROCESS_PIPELINE],
                                                                                settings.PATH_PROCESSED_DATA,
                                                                                tissue.name, column_path)
            noise = model_ids[tissue.name][1]

            filter_exp_db = {"algorithm": "SMOOTHGRAD", "algorithm_parameters.stdev_spread": noise,
                             "model_id": model_ids[tissue.name][0]}

            df_maps, _ = ExplainedMapsLoader(exp_map_type).get_expmaps_per_processed_image(df_processed,
                                                                                           column_path_exp,
                                                                                           verify_exists=False,
                                                                                           filter_db=filter_exp_db)

            print(noise, df_maps.shape)
            if df_maps.shape[0] == 0:
                continue
            df_rois_exp_maps = ROIFeaturesLoader(exp_map_type).get_roi_values_per_image(df_maps,
                                                                                        column_path_exp,
                                                                                        normalize=False)

            df_rois_processed = ROIFeaturesLoader(ImageType.PREPROCESSED_IMAGE).get_roi_values_per_image(df_processed,
                                                                                                         column_path,
                                                                                                         normalize=False)

            if df_rois_exp_maps.shape[0] == 0:
                continue

            col_processed = PreprocessedDataColumns.PATH_PREPROCESSED
            if exp_map_type == ImageType.EXPLAINABILITY_MAPS_FUSION:
                col_exp_map = ExplainabilityMapsFusionCollection.COLUMN_PATH_EXPLAINABILITY_MAP
            elif exp_map_type == ImageType.EXPLAINABILITY_MAPS_SINGLE:
                col_exp_map = ExplainabilityMapsSingleCollection.COLUMN_PATH_EXPLAINABILITY_MAP
            else:
                raise NotImplementedError

            df_rois_processed = df_rois_processed.reset_index().set_index([ROIValuesCollection.COLUMN_IMAGE_PATH,
                                                                           (ImageType.PREPROCESSED_IMAGE.name,
                                                                            'roi_name')])
            df_rois_exp_maps = df_rois_exp_maps.reset_index().set_index([col_processed,
                                                                         (exp_map_type.name, 'roi_name')])

            df_results = pd.concat([df_rois_exp_maps, df_rois_processed], axis=1)
            df_results = df_results.loc[:, ~df_results.columns.duplicated()].copy()

            if isinstance(var_eval1, tuple):
                name_var_test1 = f"{var_eval1[0]}_{var_eval1[1]}"
            else:
                name_var_test1 = var_eval1
            if isinstance(var_eval2, tuple):
                name_var_test2 = f"{var_eval2[0]}_{var_eval2[1]}"
            else:
                name_var_test2 = var_eval2

            df_results.reset_index(inplace=True)
            df_results.to_csv(os.path.join(RESULTS_DIR, f"{name_var_test1}_{name_var_test2}_"
                                                        f"{dataset.name}_{tissue.name}.csv"))
            rows_significant = []

            res = df_results.groupby("level_1").apply(lambda x: pearsonr(x[var_eval1],
                                                                         x[var_eval2]).statistic)

            res.to_csv(os.path.join(RESULTS_DIR, f"{name_var_test1}_pearson_{name_var_test2}_{dataset_name}_{tissue_name}.csv"))

