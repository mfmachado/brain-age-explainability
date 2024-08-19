import os.path

import numpy as np
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

    for var_evaluate in [ImageType.EXPLAINABILITY_MAPS_SINGLE, ImageType.PREPROCESSED_IMAGE]:

        for source_ancova in [DemographicsColumns.DIAGNOSTIC_GROUP, DemographicsColumns.AGE_AT_SCAN]:
            rows_correlation = []
            for dataset, site in [(DATASETS.IXI, ["HH"]), (DATASETS.IXI, ["IOP"]), (DATASETS.COBRE, None),
                                  (DATASETS.DIAMARKER, None), (DATASETS.CIBIT_CONTROL_AD, None)]:
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

                for tissue, tissue_name in [(BrainTissue.ALL_TISSUE_SANLM_LOCAL, "ALL_TISSUE_SANLM_LOCAL"),
                                            (BrainTissue.GM, "GM"),
                                            (BrainTissue.WM, "WM"),
                                            (BrainTissue.CSF, "CSF"),
                                            (BrainTissue.DF, "DF")]:
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

                    df_rois_processed = ROIFeaturesLoader(ImageType.PREPROCESSED_IMAGE).get_roi_values_per_image(
                        df_processed,
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

                    from pingouin import ancova

                    dataset_name = dataset.name
                    if site is not None:
                        dataset_name = f"{dataset.name}--{'-'.join(site)}"

                    name_var_test = f"{var_evaluate.name}_{roi_metric}"

                    df_results[name_var_test] = df_results[(var_evaluate.name, roi_metric)]
                    df_results.reset_index(inplace=True)
                    df_results.to_csv(os.path.join(RESULTS_DIR, f"{name_var_test}_{dataset_name}_{tissue.name}.csv"))

                    rows_p_values = []
                    for roi in df_results["level_1"].unique():
                        if roi == "Background":
                            continue
                        df_roi = df_results.loc[df_results["level_1"] == roi]
                        if len(df_roi[DemographicsColumns.DIAGNOSTIC_GROUP].unique()) > 1:
                            result = ancova(data=df_roi, dv=name_var_test, covar=DemographicsColumns.AGE_AT_SCAN,
                                            between=DemographicsColumns.DIAGNOSTIC_GROUP)
                            p_value = result.loc[result["Source"] == source_ancova]["p-unc"].iloc[0]
                        elif source_ancova == DemographicsColumns.AGE_AT_SCAN:
                            result = pearsonr(df_roi[name_var_test], df_roi[DemographicsColumns.AGE_AT_SCAN])
                            p_value = result.pvalue
                        else:
                            continue

                        rows_p_values.append([roi, p_value])
                    df_p_values = pd.DataFrame(rows_p_values, columns=["roi", "p-corr"])
                    from statsmodels.stats.multitest import fdrcorrection

                    df_p_values["p-value"] = fdrcorrection(df_p_values["p-corr"])[1]
                    df_p_values_significant = df_p_values.loc[df_p_values["p-value"] < 0.05]

                    percent_significant = (df_p_values_significant.shape[0]) * 100 / (
                            len(df_results["level_1"].unique()) - 1)
                    print(percent_significant)
                    rows_correlation.append([dataset_name, tissue_name, var_evaluate, source_ancova, percent_significant,
                                             df_p_values_significant.shape[0]])

                    filename = os.path.join(RESULTS_DIR, f"ancova_{source_ancova}_{name_var_test}_"
                                                         f"{dataset_name}_{tissue.name}.csv")
                    print(filename)
                    df_p_values_significant.to_csv(filename)

            df_percent = pd.DataFrame(rows_correlation, columns=["dataset", "tissue", "variable",
                                                                 "source", "percentage",
                                                                 "number of significant ROIS"])
            filename = os.path.join(RESULTS_DIR, f"percent_significant_{var_evaluate.name}_{source_ancova}.csv")
            df_percent.to_csv(filename)

            df_percent_ = pd.pivot_table(data=df_percent, index=["tissue"],
                                         values=["percentage"], columns=["dataset"]).round(2)

            df_percent_.columns = df_percent_.columns.droplevel()

            df_percent_ = df_percent_.reindex(index=[BrainTissue.ALL_TISSUE_SANLM_LOCAL.name, BrainTissue.GM.name,
                                                     BrainTissue.WM.name, BrainTissue.CSF.name, BrainTissue.DF.name])
            cols = [["COBRE", "DIAMARKER", "CIBIT_CONTROL_AD"]]
            if source_ancova == DemographicsColumns.AGE_AT_SCAN:
                cols = [["IXI--HH", "COBRE", "DIAMARKER", "CIBIT_CONTROL_AD"]]

            df_percent_ = df_percent_[["IXI--HH", "COBRE", "DIAMARKER", "CIBIT_CONTROL_AD"]]

            table_dir = "/home/fmachado/Dropbox/thesis/Chapter7/Tables"
            path_latex = os.path.join(table_dir, f"percent_significant_{var_evaluate.name}_{source_ancova}.txt")
            with open(path_latex, 'w') as f:
                latex_txt = df_percent_.to_latex()
                latex_txt = latex_txt.replace("ALL_TISSUE_SANLM_LOCAL",
                                              "Minimally processed").replace("CIBIT_CONTROL_AD",
                                                                             "CIBIT AD").replace("0000 ", " ")
                f.write(latex_txt)
