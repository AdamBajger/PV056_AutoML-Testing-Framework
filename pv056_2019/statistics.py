import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from pv056_2019.schemas import StatisticsSchema


def compile_reg(s):
    try:
        return re.compile(s)
    except Exception as e:
        print("Regex error:", e, file=sys.stderr)
        exit(1)


def calculate_accuracy(prediction_file):
    dataframe = pd.read_csv(prediction_file)
    all_results = dataframe.shape[0]
    return np.sum(dataframe["error"] != "+") / all_results


def main():
    parser = argparse.ArgumentParser(
        description="Script for counting basic statistic (Accuracy, )"
    )
    parser.add_argument("--config-file", "-c", required=True, help="JSON configuration")
    parser.add_argument("--datasets-csv-in", "-di", required=True,
                        help="CSV file with paths to predictions, test files and configurations histories")
    parser.add_argument("--datasets-csv-baseline", "-db", required=False,
                        help="CSV file with paths to predictions, test files and configurations histories")

    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        statistic_conf: StatisticsSchema = StatisticsSchema(**json.load(json_file))

    # reg = re.compile(r"removed-0*")
    # Previously these files were listed from a directory
    # files = sorted([x for x in os.listdir(conf.results_dir) if x.endswith(".csv")])

    # load datasets, sort them by size, if some do not exists, exclude them
    with open(args["datasets_csv_in"], "r") as csv_in:
        # csv_reader = csv.reader(csv_in, delimiter=",")
        list_of_rows = []
        for row in csv.reader(csv_in, delimiter=","):
            #print("ROW:", row)
            #print("FILE:", row[0])
            if os.path.exists(row[0]):
                #print("EXISTS")
                list_of_rows.append(row)
            else:
                print("FILE:", row[0], "DOES NOT EXISTS")

        #csv_rows = sorted([row for row in csv.reader(csv_in, delimiter=",") if os.path.exists(row[0])], key=lambda x: os.path.getsize(x[0]))
        csv_rows = sorted(list_of_rows,
                          key=lambda x: os.path.getsize(x[0]))

    # if a baseline predictions csv was supplied
    baseline_supplied = False
    # load datasets, sort them by size, if some do not exists, exclude them
    if args["datasets_csv_baseline"] is not None:
        baseline_supplied = True
        with open(args["datasets_csv_in"], "r") as csv_baseline:
            # csv_reader = csv.reader(csv_in, delimiter=",")
            csv_b_rows = sorted([row for row in csv.reader(csv_baseline, delimiter=",") if os.path.exists(row[0])],
                                key=lambda x: os.path.getsize(x[0]))

    # pattern = compile_reg(conf.pattern)

    # TODO: migh be used again in future versions
    config_file_paths = [
        # Previously these jsons were also listed from a directory
        # x for x in os.listdir(conf.results_dir) if x.endswith(".json")
    ]

    # for config_file_path in config_file_paths:
    #     with open(os.path.join(conf.results_dir, config_file_path)) as config_file:
    #         basename = os.path.basename(config_file_path)
    #         conf_hash = basename.split("_")[1].replace(".json", "")
    #         config_dict[conf_hash] = json.load(config_file)

    # We actually take into account future upgrades, which might include non-identical number of preprocessing steps applied
    # ==========================
    data = []
    config_dict = {}
    greatest_steps_count = 0
    for csv_row in csv_rows:
        # if not pattern.match(fl):
        #    continue
        prediction_file = csv_row[0]
        conf_path = csv_row[2]  # this is created in the CLF step. Third element of a CSV row is path to config json.

        accuracy = calculate_accuracy(prediction_file)  #"{:.4f}".format(calculate_accuracy(prediction_file))
        file_split = os.path.basename(prediction_file).split("_")

        # file_split[-1] = file_split[-1].replace(".csv", "")

        # We dont care about filenames anymore
        # if "removed-" in file_split[-1]:
        #     file_split[-1] = reg.sub("", file_split[-1])
        # else:
        #     file_split.append(0)

        datest, split, *_ = file_split

        # we still do care about json filenames (practical reasons) Here we extract a hash from that filename
        conf_hash = conf_path.split(".")[0]
        if conf_hash not in config_dict:
            with open(conf_path) as config_file:
                config_dict[conf_hash] = json.load(config_file)

        _conf = config_dict[conf_hash]
        clf_fullname_dotsplit = _conf["model_config"].get("class_name").split(".")
        classifier = clf_fullname_dotsplit[-1]
        clf_family = clf_fullname_dotsplit[-2]
        classifier_args = str(_conf["model_config"].get("args"))
        steps_count = _conf["steps_count"]
        # previous step concluded all the preprocessing configurations into a list of json dicts and saved the into JSON
        # we are here loading the json dicts and turning them into strings
        preprocessing_steps_config_strings = [str(conf_dict) for conf_dict in _conf["preprocessing_configs"]]
        if steps_count > greatest_steps_count:
            greatest_steps_count = steps_count

        # od_name = config_dict[conf_hash]["od_configs"].get("name", "")

        # based on the maximum
        data_entry = [datest, split, classifier, clf_family, classifier_args, *preprocessing_steps_config_strings, accuracy]
        # for entry in data_entry:
        #     if not isinstance(entry, str):
        #         raise TypeError(entry, "should be string, is", type(entry))
        data.append(data_entry)

        # ====================================
    headers_steps = ["step_{}".format(x) for x in range(greatest_steps_count)]
    _str_accuracy = "accuracy"
    _str_base = "_base"
    headers_proccessed = [
        "dataset",
        "fold",
        "clf",
        "clf_family",
        "clf_params",
        *headers_steps,
        "accuracy",
    ]
    headers_baseline = [
        "dataset",
        "fold",
        "clf",
        "clf_family",
        "clf_params",
        _str_accuracy,
    ]

    #print("DATA from classifier prediction")
    # print(headers_proccessed)
    # print(*data, sep="\n")
    dataframe = pd.DataFrame(data, columns=headers_proccessed)

    if baseline_supplied:
        baseline_data = []
        for csv_row in csv_b_rows:
            prediction_file = csv_row[0]
            conf_path = csv_row[2]

            accuracy = calculate_accuracy(prediction_file)
            file_split = os.path.basename(prediction_file).split("_")

            datest, split, *_ = file_split

            conf_hash = conf_path.split(".")[0]
            if conf_hash not in config_dict:
                with open(conf_path) as config_file:
                    config_dict[conf_hash] = json.load(config_file)

            _conf = config_dict[conf_hash]
            dotsplit = _conf["model_config"].get("class_name").split(".")
            classifier = dotsplit[-1]
            classifier_family = dotsplit[-2]
            classifier_args = str(_conf["model_config"].get("args"))

            baseline_data.append([datest, split, classifier, classifier_family, classifier_args, accuracy])

        dataframe_baseline = pd.DataFrame(baseline_data, columns=headers_baseline)
        # print("DATAFRAME")
        # print(dataframe.dtypes)
        # print("DATAFRAME_BASELINE")
        # print(dataframe_baseline.dtypes)

        dataframe = dataframe.merge(dataframe_baseline, on=headers_baseline[:-1], how="right", suffixes=("", _str_base))

    # od_times = pd.read_csv(conf.od_times_path)
    # clf_times = pd.read_csv(conf.clf_times_path)

    # times = od_times.merge(clf_times, on=["dataset", "fold", "od_hex"], how="outer")
    # times['fold'] = times['fold'].astype(str)
    # times.loc[(times.removed == 0), 'od_time'] = 0.0
    # times['total_time'] = times['od_time'] + times['clf_time']

    # dataframe['removed'] = dataframe['removed'].astype(float)

    # dataframe['clf_params'] = [re.sub("'|\"|,", "", str(config_dict[ax]["model_config"].get("args")))
    #                           for ax in dataframe['clf_hex']]
    # dataframe['od_params'] = [str(config_dict[ax]["ad_config"].get("parameters")).replace(",", ";").replace("'", "")
    #                          for ax in dataframe['clf_hex']]
    #print("DATA BEFORE AGGREGATION")
    #print(dataframe)
    if statistic_conf.aggregate:
        dataframe = dataframe.groupby(
            by=[
                "dataset",
                "clf",
                "clf_family",
                "clf_params",
                *headers_steps
            ]
        ).mean().reset_index()
        #print(group_fold_df)
        #print(dataframe.columns != "fold")
        #dataframe = group_fold_df.loc[:, dataframe.columns != "fold"]

    dataframe = dataframe.round(5)
    if baseline_supplied:
        dataframe["gain"] = dataframe[_str_accuracy] - dataframe[_str_accuracy+_str_base]
    #print("FINAL DATA:")
    #print(dataframe.to_csv())
    with open(statistic_conf.output_table, "w") as ot:
        dataframe.to_csv(ot, index=False)
        #print(dataframe.to_csv(), file=ot)
    backup_ts = "backups/" + statistic_conf.output_table.split("/")[-1].replace(".csv", datetime.now()
                                                                      .strftime("_backup_%d-%m-%Y_%H-%M.csv"))
    with open(backup_ts, "w+") as ot:
        print(dataframe.to_csv(index=False), file=ot)


if __name__ == "__main__":
    main()
