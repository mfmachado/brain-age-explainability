import pandas as pd
from scipy.stats import pearsonr

from src import settings
from src.base.data_types import BrainTissue, ImageType
from src.base.database_columns import PreprocessingPipeline, DATASETS, ExplainabilityMapsFusionCollection, \
    PreprocessedDataColumns, ROIValuesCollection, DemographicsColumns, ExplainabilityMapsSingleCollection
from src.preprocessing.load_roi_features import ROIFeaturesLoader
from src.preprocessing.loader_explainability_maps import ExplainedMapsLoader
from src.preprocessing.loader_processed_data import PreprocessedImagesLoader
from src.preprocessing.loader_subjects import SubjectsLoader

if __name__ == '__main__':

    dataset = DATASETS.IXI
    exp_map_type = ImageType.EXPLAINABILITY_MAPS_SINGLE
    column_path = "col_abs_path"
    column_path_exp = "col_exp_abs_path"

    roi_metric = "mean"
    df_sub = SubjectsLoader().get_subjects(dataset, True, None,
                                           1, True, sites=["HH"])
    noise_rows = []

    model_ids = {"ALL_TISSUE_SANLM_LOCAL": "6543152bcdb3137a8d9cd131",
                 "GM": "6542f110f4cfa554fb83541d", "CSF": "6542fd79abc67df1db788c8f",
                 "WM": "654309e88a995a5e67e23092",
                 "DF": "6542e4dc487d0e69df5e3410"}

    for tissue, tissue_name in [(BrainTissue.WM, "WM"),
                                (BrainTissue.ALL_TISSUE_SANLM_LOCAL, "ALL_TISSUE_SANLM_LOCAL"),
                                (BrainTissue.GM, "GM"),
                                (BrainTissue.CSF, "CSF"),
                                (BrainTissue.DF, "DF")
                                ]:
        df_processed, _ = PreprocessedImagesLoader().get_images_per_subject(df_sub,
                                                                            PreprocessingPipeline[
                                                                                settings.PREPROCESS_PIPELINE],
                                                                            settings.PATH_PROCESSED_DATA,
                                                                            tissue.name, column_path,
                                                                            verify_exists=False)

        for noise_int in range(0, 51, 1):
            noise = noise_int / 100
            filter_exp_db = {"algorithm": "SMOOTHGRAD", "algorithm_parameters.stdev_spread": noise,
                             "model_id": model_ids[tissue.name]}

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

            df_all_group = df_results.reset_index().set_index("level_0").groupby(["level_1",
                                                                                  DemographicsColumns.DIAGNOSTIC_GROUP])

            df_exp_maps_group = df_rois_exp_maps.reset_index().set_index(ROIValuesCollection.COLUMN_IMAGE_PATH).groupby(
                [DemographicsColumns.DIAGNOSTIC_GROUP, (exp_map_type.name, "roi_name")])

            # Relation between each and the
            df_exp_age = df_exp_maps_group.apply(
                lambda x: pd.Series(pearsonr(x[(exp_map_type.name, roi_metric)],
                                             x[DemographicsColumns.AGE_AT_SCAN]).statistic)).reset_index()

            for diagnostic_group in df_exp_age[DemographicsColumns.DIAGNOSTIC_GROUP].unique():
                df_diag = df_exp_age.loc[df_exp_age[DemographicsColumns.DIAGNOSTIC_GROUP] == diagnostic_group]
                total = df_diag[0].abs().mean()
                noise_rows.append([noise, diagnostic_group, tissue_name, total])

    df_noise = pd.DataFrame(noise_rows, columns=["noise", DemographicsColumns.DIAGNOSTIC_GROUP, "tissue", "total"])
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_noise.to_csv(f"{dataset.name}_age_relation.csv")
    f, ax = plt.subplots(1, 1, figsize=(15, 5))
    sns.set(font_scale=2)
    sns.lineplot(data=df_noise, x="noise", y="total", hue="tissue", style=DemographicsColumns.DIAGNOSTIC_GROUP, ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.ylabel("Mean of $R^2$ between \n saliency maps and age")
    plt.tight_layout()
    plt.show()
