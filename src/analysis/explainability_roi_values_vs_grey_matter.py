import os.path

import pandas as pd

from src import settings
from src.base.data_types import BrainTissue, ImageType, DiagnosticGroupCondition
from src.base.database_columns import PreprocessingPipeline, DATASETS, PreprocessedDataColumns, ROIValuesCollection, \
    DemographicsColumns, ExplainabilityMapsSingleCollection
from src.preprocessing.load_roi_features import ROIFeaturesLoader
from src.preprocessing.loader_explainability_maps import ExplainedMapsLoader
from src.preprocessing.loader_processed_data import PreprocessedImagesLoader
from src.preprocessing.loader_subjects import SubjectsLoader

tissue = BrainTissue.GM

dataset = DATASETS.IXI
exp_map_type = ImageType.EXPLAINABILITY_MAPS_SINGLE
column_path = "col_abs_path"
column_path_exp = "col_abs_path"
filter_exp_db = {"algorithm": "SMOOTHGRAD", "algorithm_parameters.stdev_spread": 0.15}

df_sub = SubjectsLoader().get_subjects(dataset, True, [DiagnosticGroupCondition.CONTROL.name],
                                       1, True, sites=["IOP"])

df_processed, _ = PreprocessedImagesLoader().get_images_per_subject(df_sub,
                                                                    PreprocessingPipeline[
                                                                        settings.PREPROCESS_PIPELINE],
                                                                    settings.PATH_PROCESSED_DATA,
                                                                    tissue.name, column_path)

df_maps, _ = ExplainedMapsLoader(exp_map_type).get_expmaps_per_processed_image(df_processed,
                                                                               column_path_exp, verify_exists=False,
                                                                               filter_db=filter_exp_db)

df_rois_exp_maps = ROIFeaturesLoader(exp_map_type).get_roi_values_per_image(df_maps,
                                                                            column_path_exp,
                                                                            normalize=True)

df_rois_processed = ROIFeaturesLoader(ImageType.PREPROCESSED_IMAGE).get_roi_values_per_image(df_processed,
                                                                                             column_path,
                                                                                             normalize=False)

col_processed = PreprocessedDataColumns.PATH_PREPROCESSED
col_exp_map = ExplainabilityMapsSingleCollection.COLUMN_PATH_EXPLAINABILITY_MAP

df_rois_processed = df_rois_processed.reset_index().set_index([ROIValuesCollection.COLUMN_IMAGE_PATH,
                                                               (ImageType.PREPROCESSED_IMAGE.name, 'roi_name')])
df_rois_exp_maps = df_rois_exp_maps.reset_index().set_index([col_processed,
                                                             (exp_map_type.name, 'roi_name')])

df_results = pd.concat([df_rois_exp_maps, df_rois_processed], axis=1)
df_results = df_results.loc[:, ~df_results.apply(lambda x: x.duplicated(), axis=1).all()].copy()

from scipy.stats import pearsonr

df_all_group = df_results.reset_index().set_index("level_0").groupby("level_1")
df_processed_group = df_rois_processed.reset_index().set_index(ROIValuesCollection.COLUMN_IMAGE_PATH).groupby(
    (ImageType.PREPROCESSED_IMAGE.name,
     "roi_name"))
df_exp_maps_group = df_rois_exp_maps.reset_index().set_index(ROIValuesCollection.COLUMN_IMAGE_PATH).groupby(
    (exp_map_type.name, "roi_name"))

## Analysis between a metric of grey matter and explainability maps
for metric in ["max", "mean", "std"]:
    df_roi_relation = df_all_group.apply(lambda x: pd.Series(pearsonr(x[(ImageType.PREPROCESSED_IMAGE.name, metric)],
                                                                      x[(exp_map_type.name,
                                                                         metric)])))

    path_save_csv = "/home/fmachado/Documents/explainability_maps/analysis_gm_total_vs_explainability"
    df_roi_relation.to_csv(os.path.join(path_save_csv, f"{dataset.name}_{metric}.csv"))

    # Relation between each and the
    df_proc_age_relation = df_processed_group.apply(lambda x: pd.Series(pearsonr(x[(ImageType.PREPROCESSED_IMAGE.name,
                                                                                    metric)],
                                                                                 x[DemographicsColumns.AGE_AT_SCAN])))

    df_exp_age_relation = df_exp_maps_group.apply(
        lambda x: pd.Series(pearsonr(x[(exp_map_type.name, metric)],
                                     x[DemographicsColumns.AGE_AT_SCAN])))

    path_save_csv = "/home/fmachado/Documents/explainability_maps/analysis_age_relation"
    df_proc_age_relation.to_csv(os.path.join(path_save_csv, f"{dataset.name}_gm_{metric}.csv"))
    df_exp_age_relation.to_csv(os.path.join(path_save_csv, f" {dataset.name}_gm_explainability-map_{metric}.csv"))
