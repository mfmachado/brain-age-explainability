import os.path
from pathlib import Path
from typing import List, Union

import pandas as pd
import pingouin
from sklearn.metrics import mean_absolute_error

from src import settings
from src.base.data_types import BrainTissue, DiagnosticGroupCondition
from src.base.database_columns import DATASETS, DemographicsColumns, PreprocessingPipeline
from src.models.utils import load_data_from_df
from src.preprocessing.loader_processed_data import PreprocessedImagesLoader
from src.preprocessing.loader_subjects import SubjectsLoader
from src.settings import RESULTS_DIR


def get_data(dataset: Union[DATASETS, List[DATASETS]], tissue: BrainTissue, sites: Union[List, None],
             path_col: str, col_label: str, diagnostics: Union[list, None] = None):
    df_sub = SubjectsLoader().get_subjects(dataset, True, diagnostics, 1, True, sites=sites)

    df, _ = PreprocessedImagesLoader().get_images_per_subject(df_sub,
                                                              PreprocessingPipeline[
                                                                  settings.PREPROCESS_PIPELINE],
                                                              settings.PATH_PROCESSED_DATA,
                                                              tissue.name, path_col)

    data = load_data_from_df(df, path_col, col_label)
    return df, data


def latex_float(float_str, number_round):
    number_decimal = len(str(1 / float(float_str)).split(".")[0]) if "e" not in str(1 / float(float_str)) else 1000
    if number_decimal > 4:
        float_str = f"{float(float_str):.2e}"
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(round(float(base), number_round), int(exponent))
    else:
        return round(float(float_str), number_round + number_decimal - 1)


if __name__ == '__main__':
    import matplotlib

    tissue_legend = {}
    tissue_legend_box = {}
    for el in BrainTissue:
        tissue_legend[el] = el.name if el != BrainTissue.ALL_TISSUE_SANLM_LOCAL else "Minimal processed image"
        tissue_legend_box[el] = el.name if el != BrainTissue.ALL_TISSUE_SANLM_LOCAL else "MP"

    matplotlib.rcParams.update({'font.size': 16})

    palette = {DiagnosticGroupCondition.CONTROL.name: "#2A3049",
               DiagnosticGroupCondition.ALZHEIMER.name: "#318573",
               DiagnosticGroupCondition.SCHIZOPHRENIA.name: "#DCA425",
               DiagnosticGroupCondition.DIABETES.name: "#FA5012",
               DiagnosticGroupCondition.BIPOLAR_DISORDER.name: "#BB0000"}

    rows_mae = []

    datasets = [DATASETS.CIBIT_CONTROL_AD, DATASETS.DIAMARKER, DATASETS.COBRE]
    diagnostics_dataset = [DiagnosticGroupCondition.ALZHEIMER]
    model_id = "ntrain954"

    disease_dataset = {DATASETS.CIBIT_CONTROL_AD: DiagnosticGroupCondition.ALZHEIMER.name,
                       DATASETS.DIAMARKER: DiagnosticGroupCondition.DIABETES.name,
                       DATASETS.COBRE: DiagnosticGroupCondition.SCHIZOPHRENIA.name}

    for dataset in datasets:
        print()
        print(dataset.name)
        all_bag = []
        column_path = "path_abs"
        column_label = DemographicsColumns.AGE_AT_SCAN

        result_ancova_list = []
        result_ancova_corrected_list = []

        rows_bag = []
        for tissue in [BrainTissue.ALL_TISSUE_SANLM_LOCAL, BrainTissue.GM, BrainTissue.WM, BrainTissue.CSF,
                       BrainTissue.DF]:
            print(dataset.name, tissue.name)

            col_fusion = f"{tissue.name}_fusion"
            filename = f"correctedIXIHH_{model_id}_{tissue.name}_{dataset.name}.csv"

            df_test = pd.read_csv(os.path.join(RESULTS_DIR, "predictions", filename))

            df_control2 = df_test.loc[
                df_test[DemographicsColumns.DIAGNOSTIC_GROUP] == DiagnosticGroupCondition.CONTROL.name].copy()
            # df_train_bias = df_control2.iloc[:50]

            col_prediction_corrected = f"{tissue.name} predictions corrected"

            df_test.groupby(["repositoryName", "diagnosticGroup"]).apply(lambda x: mean_absolute_error(x[column_label],
                                                                                                       x[
                                                                                                           col_prediction_corrected]))

            for pathology in df_test[DemographicsColumns.DIAGNOSTIC_GROUP].unique():
                row_mae = {"dataset": dataset.name, "tissue": tissue.name, "model_id": model_id}
                df_filtered = df_test.loc[df_test[DemographicsColumns.DIAGNOSTIC_GROUP] == pathology]

                mae = mean_absolute_error(df_filtered[DemographicsColumns.AGE_AT_SCAN],
                                          df_filtered[col_fusion])
                mae_corrected = mean_absolute_error(df_filtered[DemographicsColumns.AGE_AT_SCAN],
                                                    df_filtered[col_prediction_corrected])
                row_mae["disease"] = pathology
                row_mae["mae"] = mae
                row_mae["mae_corrected"] = mae_corrected
                rows_mae.append(row_mae)

            path_save_img_and_tables = os.path.join(RESULTS_DIR, "single_model")

            import seaborn as sns
            import matplotlib.pyplot as plt

            bag = df_test.copy()
            bag["BrainAGE [years]"] = df_test[col_prediction_corrected] - df_test["age"]
            bag["dataset"] = dataset.name
            bag["tissue"] = tissue_legend_box[tissue]
            all_bag.append(bag)

            path_img = os.path.join(path_save_img_and_tables, "Figures",
                                    tissue.name, model_id)
            Path(path_img).mkdir(exist_ok=True, parents=True)

            bag_col = "BAG"
            bag_col_corrected = "BAG_corrected"

            df_test[bag_col] = df_test[col_fusion] - df_test[DemographicsColumns.AGE_AT_SCAN]
            df_test[bag_col_corrected] = df_test[col_prediction_corrected] - df_test[DemographicsColumns.AGE_AT_SCAN]

            result_ancova = pingouin.ancova(data=df_test, dv=bag_col, covar=DemographicsColumns.AGE_AT_SCAN,
                                            between='diagnosticGroup')
            df_test["gender_binary"] = df_test[DemographicsColumns.GENDER] == "FEMALE"
            result_ancova_corrected = pingouin.ancova(data=df_test, dv=bag_col_corrected,
                                                      covar=[DemographicsColumns.AGE_AT_SCAN,
                                                             "gender_binary"],
                                                      between='diagnosticGroup')

            path_table = os.path.join(path_save_img_and_tables, "Tables", model_id)
            Path(path_table).mkdir(exist_ok=True, parents=True)

            result_ancova["tissue"] = tissue.name
            result_ancova_corrected["tissue"] = tissue.name

            result_ancova_list.append(result_ancova)
            result_ancova_corrected_list.append(result_ancova_corrected)

        for list_result, filename_csv in [(result_ancova_list, f"stats_{dataset.name}"),
                                          (result_ancova_corrected_list, f"stats_{dataset.name}_corrected")]:
            df_ancova = pd.concat(list_result)
            df_ancova = df_ancova.loc[df_ancova["Source"] != "Residual"]
            df_ancova = df_ancova[["tissue", "Source", "SS", "F", "p-unc", "np2"]]

            for col in ["SS", "F", "p-unc", "np2"]:
                df_ancova[col] = df_ancova[col].apply(lambda x: latex_float(x, 2))

            df_ancova.to_csv(os.path.join(path_table, f"{filename_csv}.csv"))

            path_tex_file = os.path.join(path_table, f"{filename_csv}.txt")
            if "cor" in filename_csv:
                print(df_ancova[["tissue", "Source", "p-unc"]])

            with open(path_tex_file, 'w') as f:
                print()
                f.write(df_ancova.to_latex().replace("ALL_TISSUE_SANLM_LOCAL", "Minimal processed"))
        df_bag = pd.concat(all_bag).reset_index(drop=True)
        df_bag["Modality"] = df_bag["tissue"]
        g = sns.boxplot(data=df_bag, x="Modality", y="BrainAGE [years]", hue='diagnosticGroup',
                          hue_order=[DiagnosticGroupCondition.CONTROL.name,
                                     disease_dataset[dataset]],
                          palette=[palette[DiagnosticGroupCondition.CONTROL.name],
                                   palette[disease_dataset[dataset]]])

        g.legend(loc='upper left', ncol=2, title=None)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{dataset}.png")             
        plt.show()

