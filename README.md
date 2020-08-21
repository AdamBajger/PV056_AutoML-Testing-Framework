# PV056-AutoML-testing-framework
* *PV056 Machine learning and knowledge discovery*

## How to use PV056-AutoML-testing-framework
* First, follow the [Installation guide](#installation-guide) section
* Then follow the [Usage](#usage) section
 
## Installation guide 
### Prerequisites
- Python version >=3.6 (including python3-dev, python3-wheel)
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/) (comes with the framework)
- Java 8 or later (for Weka execution)
- UNIX-like environment

If you are a Windows user, you can either set up a 
virtual machine or run the script on the Aisa server (the whole script can be run through the terminal, 
even [remotely](https://www.fi.muni.cz/tech/lets-get-started-at-fi.html.cs)).
However, keep in mind that these options will impact the running time of the script.

### Installation to python virtual env (recommended)
It's highly recommended to install this testing framework to python virtual environment.
- Simple python virtual environment guide: https://realpython.com/python-virtual-environments-a-primer/

Simply run commands below in the root folder of this repository.
```
$ python3 -m venv venv
$ source venv/bin/activate # venv/bin/activate.fish # for fish shell
(venv)$ pip install -r requirements.txt
(venv)$ pip install -e .
```

### Installation without python virtual env (not recommended)
```
$ pip install .
```


### Downloading datasets
The framework supports datasets in `.arff` format. The individual datasets can be downloaded from [OpenML](https://www.openml.org).

By default, you should place the downloaded datasets to the `data/datasets/` folder.


## Usage
If you have chosen to install this tester in the virtual environment, you must activate it to proceed.
* *Pipeline 1:* split data -> run classifiers -> (optional) statistics
* *Pipeline 2 (outlier detection):* split data -> apply outlier detectors -> remove outliers -> run classifiers -> (optional) statistics


### Split data
Because we want to cross-validate every classifier, we need to split data into the five train-test tuples.  Before this was a part of weka classifiers, jet now, because we want to work with training datasets we need to do it manually.

For this purpose, we have the `pv056-split-data`. As specified in its configuration file (see `config_split_example.json`) it will split datasets into five train-test tuples and generates CSV (`datasets.csv`) file which can be used for classification without any changes to the training splits.

```
(venv)$ pv056-split-data --help
usage: pv056-split-data [-h] --config-file CONFIG_FILE --datasets-file
                        DATASETS_FILE

Script splits datasets for cross-validation

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
  --datasets-file DATASETS_FILE, -d DATASETS_FILE
                        Filename of output datasets config
```
#### Example usage
```
(venv)$ pv056-split-data -c configs/split/default.json -d datasets.csv
```

#### Example config file
* *data_path*
    * Directory with datasets in arff format
* *train_split_dir*
    * Directory where generated **train** datasets should be saved
* *test_split_dir*
    * Directory where generated **test** datasets should be saved

```json
{
    "data_path": "data/datasets/",
    "train_split_dir": "data/train_split/",
    "test_split_dir": "data/test_split/"
}
```

### Apply outlier detection methods
To apply outlier detection methods to all training splits, we have the `pv056-apply-od-methods`. This script takes all the training splits from the `train_split_dir` (it will only take files which basename ends with `_train.arff`) and adds a column with outlier detection value. It will generate new training file for every outlier detection method!
```
(venv)$ pv056-apply-od-methods --help
usage: pv056-apply-od-methods [-h] --config-file CONFIG_FILE

Apply outlier detection methods to training data

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
```
#### Example usage
```
(venv)$ pv056-apply-od-methods -c configs/od/default.json
```

#### Example config file
* *train_split_dir*
    * Directory with splitted **train** datasets
* *train_od_dir*
    * Directory where generated **train** datasets with outlier detection values should be saved
* *n_jobs*
    * number of parallel workers
* *times_output*
    * path to file where the run times should be stored
* *od_methods*
    * List with Outlier detection methods
    * Outlier detection method schema:
        * *name* - OD name
        * *parameters* - Dictionary "parameter_name": "value"

```json
{
    "train_split_dir": "data/train_split/",
    "train_od_dir": "data/train_od/",
    "n_jobs": 2,
    "times_output": "outputs/od_times.csv",
    "od_methods": [
        {
            "name": "IsolationForest",
            "parameters": {
                "contamination": "auto",
                "behaviour": "new"
            }
        },
        {
            "name": "LOF",
            "parameters": {
                "contamination": "auto"
            }
        }
    ]
}
```

#### Outlier detector names and parameters:
| Name | Full name | Parameters |
|:----:|:----------:|:----------:|
| **LOF** | Local Outlier Factor | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) |
| **NearestNeighbors** | Nearest Neighbors | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) |
| **IsolationForest** | Isolation Forest | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
| **OneClassSVM** | One-class SVM | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) |
| **EllipticEnvelope** | Elliptic Envelope | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html) |
| **ClassLikelihood** | Class Likelihood | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html)
| **ClassLikelihoodDifference** | Class Likelihood Difference | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html)
| **KDN** | K-Disagreeing Neighbors | n_neighbors |
| **DS** | Disjunct size | -- |
| **DCP** | Disjunct class percentage | min_impurity_split [docs](https://blog.nelsonliu.me/2016/08/05/gsoc-week-10-scikit-learn-pr-6954-adding-pre-pruning-to-decisiontrees/) |
| **TD** | Tree Depth with and without prunning | -- |
| **TDWithPrunning** | Tree Depth with prunning | min_impurity_split |
| **Random** | Random OD score | seed |
| **CODB** | CODB | See below |


#### CODB
* path to CODB jar file jar_path, must be defined
* k nearest neighbors (default = 7) -k "\<int\>"
* Alpha coeffecient (default = 100) -a "\<double\>"
* Beta coeffecient (default = 0.1) -b "\<double\>"
* distance-type (default = motaz.util.EuclidianDataObject) -D "\<String\>"
* Replace Missing Vaules (default = false) -r 
* Remove Missing Vaules (default = false) -m

Example:
```
{
    "train_split_dir": "data/train_split/",
    "train_od_dir": "data/train_od/",
    "n_jobs": 2,
    "times_output": "outputs/od_times.csv",
    "od_methods": [
        {
            "name": "CODB",
            "parameters": {
                "jar_path" : "data/java/WEKA-CODB.jar",
                "-k" : "10",
                "-r" : "",
                "-m": ""
            }
        }
    ]
}
```

* New methods for outlier detection coming soon!


### Remove outliers
As the name suggests, this script will remove the biggest outliers from each training split with outlier detection value. Based on his configuration it will generate CSV file (`datasets.csv`) which can be then used for running the weka classifiers.
```
(venv)$ pv056-remove-outliers  --help
usage: pv056-remove-outliers [-h] --config-file CONFIG_FILE --datasets-file
                             DATASETS_FILE

Removes the percentage of the largest outliers.

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
  --datasets-file DATASETS_FILE, -d DATASETS_FILE
                        Filename of output datasets config
```
#### Example usage
```
(venv)$ pv056-remove-outliers  -c configs/rm/default.json -d datasets.csv
```

#### Example config file
* *test_split_dir*
    * Directory with splitted **test** datasets
* *train_od_dir*
    * generated **train** datasets with outlier detection values
* *train_removed_dir*
    * Directory where train data with **removed** outliers should be saved
* *keep_original*
    * Setting this to true will produce baseline datasets as well,
    ie. datasets without any instances removed (default: `true`)
* *percentage*
    * How many percents of the largest outliers should be removed (0.0-100.0)
    * int or List[int]
```json
{
    "test_split_dir": "data/test_split/",
    "train_od_dir": "data/train_od/",
    "train_removed_dir": "data/train_removed/",
    "keep_original": true,
    "percentage": [
        0.5,
        5,
        10
    ]
}
```

### Select features
As the name suggests, this script will remove the worst features from each training split based on feature selection settings.
```
(venv)$ pv056-evalueate-features  --help
usage: pv056-evaluate-features [-h] -c CONFIG_FS [-do DATASETS_CSV_OUT] -di
                               DATASETS_CSV_IN

PV056-AutoML-testing-framework

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FS, --config-fs CONFIG_FS
                        path to feature selection config file
  -do DATASETS_CSV_OUT, --datasets-csv-out DATASETS_CSV_OUT
                        path to a csv file which contains datasets used for FS
                        and their respective result files
  -di DATASETS_CSV_IN, --datasets-csv-in DATASETS_CSV_IN
                        Path to csv file that contains previous data files
                        mappings, locations and for example OD configurations

```
#### Example usage
```
(venv)$ pv056-remove-outliers  -c configs/fs/default.json -di datasets-rm.csv -do datasets-fs.csv
```

#### Example config file
* *test_split_dir*
    * Directory with splitted **test** datasets
* *train_od_dir*
    * generated **train** datasets with outlier detection values
* *train_removed_dir*
    * Directory where train data with **removed** outliers should be saved
* *keep_original*
    * Setting this to true will produce baseline datasets as well,
    ie. datasets without any instances removed (default: `true`)
* *percentage*
    * How many percents of the largest outliers should be removed (0.0-100.0)
    * int or List[int]
```json
{
  "output_folder_path": "data/fs_outputs/",
  "weka_jar_path": "data/java/weka.jar",
  "blacklist_file_path": "data/fs_blacklist.csv",
  "selection_methods": [
    {
      "source_library": "SCIKIT",
      "leave_attributes_binarized": true,
      "fs_method": {
        "name": "SelectFpr",
        "parameters": {
          "alpha": 0.2
        }
      },
      "score_func": {
        "name": "chi2",
        "parameters": {}
      }
    },
    {
      "source_library": "SCIKIT",
      "leave_attributes_binarized": false,
      "fs_method": {
        "name": "SelectFpr",
        "parameters": {
          "alpha": 0.2
        }
      },
      "score_func": {
        "name": "chi2",
        "parameters": {}
      }
    },
    {
      "source_library": "WEKA",
      "eval_class": {
        "name": "weka.attributeSelection.CfsSubsetEval",
        "parameters": {
          "M": true,
          "L": true,
          "Z": true,
          "P": 3,
          "E": 4
        }
      },
      "search_class": {
        "name": "weka.attributeSelection.BestFirst",
        "parameters": {
          "N": 5
        }
      }
    },
    {
      "source_library": "WEKA",
      "eval_class": {
        "name": "weka.attributeSelection.InfoGainAttributeEval",
        "parameters": {}
      },
      "search_class": {
        "name": "weka.attributeSelection.Ranker",
        "parameters": {
          "T": 5E-3,
          "N": -1
        }
      }
    }
  ]
}

```



### Run weka classifiers
To run a weka classifier using this framework, first setup virtual environment, install required modules and download weka tool.
1) Activate your virtual Python environment with this project.
2) Generate `datasets.csv` file using `pv056-split-data` or `pv056-remove-outliers` (See [Split data](#split-data) and [Remove outliers](#remove-outliers) )
3) Create a `config_clf_example.json` file, with weka classifiers and their configuration (See [Config file for weka classifiers](#example-of-config-file-for-weka-classifiers))
5) Run `pv056-run-clf` script, see command below

```
(venv)$ pv056-run-clf --help
usage: pv056-run-clf [-h] -c CONFIG_CLF -d DATASETS_CSV

PV056-AutoML-testing-framework

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_CLF, --config-clf CONFIG_CLF
                        path to classifiers config file
  -d DATASETS_CSV, --datasets-csv DATASETS_CSV
                        Path to csv with data files
```

#### Example usage
```
(venv)$ pv056-run-clf -c configs/clf/default.json -d datasets.csv
```

#### Example of config file for weka classifiers
* *output_folder*
    * path to output folder, where outputs from your classifiers will be saved
* *weka_jar_path*
    * path to a weka.jar file
* *n_jobs*
    * number of parallel workers
* *times_output*
    * path to file where the run times should be stored
* *timeout*
    * max run time before a classifier is terminated (in seconds)
* *classifiers*
    * list of classifiers which you want to run
    * you can run an arbitrary number of classifiers, even same classifier with different configuration
    * list of [weka classifiers](http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html)
    * *class_name*
        * name of weka classifier class
    * *args*
        * optional value
        * list of arguments for specific classifier
        * you can find all arguments for specific classifier using weka command: ```$ java -cp weka.jar weka.classifiers.trees.J48 --help``` or in [weka documentation](http://weka.sourceforge.net/doc.dev/weka/classifiers/trees/J48.html)
        * you can find more information about Weka-CLI in the section below [How to work with Weka 3](#how-to-work-with-Weka-3), but I don't think you need that for using this tool.
    * *filters*
        * optional value
        * you can use any filter from Weka
        * you have to specify name of filter and arguments for it

```json
{
    "output_folder": "clf_outputs/",
    "weka_jar_path": "data/java/weka.jar",
    "n_jobs": 2,
    "times_output": "outputs/clf_times.csv",
    "timeout": 1800,
    "classifiers": [
        {
            "class_name": "weka.classifiers.trees.J48",
            "args": [
                "-C",
                0.25,
                "-M",
                2
            ],
            "filters": [
                {
                    "name": "weka.filters.unsupervised.attribute.RemoveByName",
                    "args": [
                        "-E",
                        "^size$"
                    ]
                }
            ]
        },
        {
            "class_name": "weka.classifiers.trees.J48",
            "args": [
                "-C",
                0.35
            ]
        },
        {
            "class_name": "weka.classifiers.bayes.BayesNet"
        }
    ]
}
```

### Count accuracy
To count accuracy simply run `pv056-statistics` script. Script will put it all together
and generate output in csv format (see example below).
```
(venv)$ pv056-statistics --help
usage: pv056-statistics [-h] --config-file CONFIG_FILE

Script for counting basic statistic (Accuracy, )

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
```
#### Example usage
```
pv056-statistics -c configs/stats/default.json
```

#### Example of config file for statistics
* *results_dir*
    * directory with predictions generated by ```pv056-run-clf```
* *od_times_path*
    * path to file with OD run times generated by ```pv056-apply-od-methods```
* *clf_times_path*
    * path to file with classification run times generated by ```pv056-run-clf```
* *aggregate*
    * whether to aggregate values across folds (using *mean*)
* *pattern*
    * regex pattern which of the files in ```results_dir``` should be taken into account
    (e.g. when you are only interested in the results on some datasets)

#### Example output
```
(venv)$ pv056-statistics -c configs/stats/default.json
dataset,clf,clf_family,clf_params,od_name,od_params,removed,accuracy,od_time,clf_time,total_time
zoo,J48,trees,[],IsolationForest,{behaviour: new; contamination: auto; random_state: 123},0.5,0.92048,0.12022,0.91727,1.0375
zoo,J48,trees,[],IsolationForest,{behaviour: new; contamination: auto; random_state: 123},1.0,0.92048,0.12022,0.89028,1.0105
zoo,J48,trees,[],IsolationForest,{behaviour: new; contamination: auto; random_state: 123},10.0,0.92048,0.12022,0.85894,0.97916
zoo,J48,trees,[],LOF,{contamination: auto},0.5,0.92048,0.05723,0.84975,0.90698
zoo,J48,trees,[],LOF,{contamination: auto},1.0,0.92048,0.05723,0.86782,0.92505
zoo,J48,trees,[],LOF,{contamination: auto},10.0,0.90048,0.05723,0.83007,0.88731
zoo,RandomForest,trees,[-I 1000],IsolationForest,{behaviour: new; contamination: auto; random_state: 123},0.5,0.94048,0.12022,1.12301,1.24323
zoo,RandomForest,trees,[-I 1000],IsolationForest,{behaviour: new; contamination: auto; random_state: 123},1.0,0.96048,0.12022,1.20243,1.32265
zoo,RandomForest,trees,[-I 1000],IsolationForest,{behaviour: new; contamination: auto; random_state: 123},10.0,0.95048,0.12022,1.25996,1.38018
zoo,RandomForest,trees,[-I 1000],LOF,{contamination: auto},0.5,0.94048,0.05723,1.24674,1.30397
zoo,RandomForest,trees,[-I 1000],LOF,{contamination: auto},1.0,0.94048,0.05723,1.23285,1.29009
zoo,RandomForest,trees,[-I 1000],LOF,{contamination: auto},10.0,0.95048,0.05723,1.22809,1.28532
```

## All-in-one script
To make the script easier to use, we have created a runnable shell script which runs the full
workflow pipeline without the need to execute each step separately.

The script runs each step with its respective default configuration file found in the ```configs/```
folder and its subdirectories. Assuming the virtual environment is set up (see [Installation guide](#installation-guide)), all you need to do is set the config files according to your
preference and then run the script from the root directory of the project.

The outputs for each step can be found in the ```logs/``` directory.

***Note:** there is no need to activate the virtual environment first as the the
script takes care of it, as well as deactivating it after the script finishes*

#### Example usage
```
$ scripts/script.sh
05/12/2019-17:21:33 - Splitting data...
05/12/2019-17:21:35 - SPLIT done.

05/12/2019-17:21:35 - Applying OD methods...
05/12/2019-17:21:36 - OD done.

05/12/2019-17:21:36 - Removing outliers...
05/12/2019-17:21:38 - RM done.

05/12/2019-17:21:38 - Running classification...
05/12/2019-17:21:57 - CLF done.

05/12/2019-17:21:57 - Counting accuracy...
05/12/2019-17:21:58 - ACC done.

Script finished.

```

### Checking progress
Depending on your setup, the script may take a long time to finish. 
Most steps will now output ongoing progress after each finished file. If it is not the case,
you might try to peek into the `od_times.csv` and `clf_times.csv` files possibly generated during execution for the progress
on outlier detection and classification, respectively.
These files list the datasets already processed with the time it took for each of them.

## Reporting problems
All encountered problems regarding running the script should be posted to the subject [forum](https://is.muni.cz/auth/discussion/predmetove/fi/jaro2020/PV056/). This is mainly
because if someone struggles with something, chances are other students have encountered the same problem.

## How to work with Weka 3
* Download Weka from https://www.cs.waikato.ac.nz/ml/weka/downloading.html
* Weka classifiers http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html
* Documentation to Weka in `The WEKA Workbench` & `Weka manual` documents at https://www.cs.waikato.ac.nz/ml/weka/documentation.html
```shell
# Example of running J48 classifier with diabetes dataset
java -cp weka.jar weka.classifiers.trees.J48 -t data/diabetes.arff

# General & Specific configuration options
java -cp weka.jar weka.classifiers.trees.J48 --help
```

## Developers guide

As developers, we have chosen to use [Black](https://github.com/ambv/black/) auto-formater and [Flake8](https://gitlab.com/pycqa/flake8) style checker. Both of these tools are pre-prepared for pre-commit. It's also recommended to use [mypy](https://github.com/python/mypy).


Since there is the typing module in the standard Python library, it would be a shame not to use it.  A wise old man once said: More typing, fewer bugs. [Typing module](https://docs.python.org/3/library/typing.html)


To prepare dev env run commands below.
```
$ python3 -m venv venv
$ source venv/bin/activate # venv/bin/activate.fish # for fish shell
(venv)$ pip install -r requirements.txt
(venv)$ pip install -r requirements-dev.txt
(venv)$ pre-commit install
(venv)$ pip install -e .
```

For generating `requirements.txt` we are using pip-compile from [pip-tools](https://github.com/jazzband/pip-tools).
For keeping your packages updated, use `pip-sync requirements.txt requirements-dev.txt`.
