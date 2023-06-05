import sys
import os
import json
import pandas
import argparse

DATASET_TO_FILTER = "['GS', 'EMO', 'GTZAN', 'MTT']" # sometimes the log may only con
# DATASET_TO_FILTER = ['GS', 'EMO', 'GTZAN', ]
# DATASET_TO_FILTER = ['MTT', ]

SLIENT_MODE = True
# SLIENT_MODE = False

TOTAL_LAYER=14
# TOTAL_LAYER=26

def get_all_results(log_file, dataset_names="['GS', 'EMO', 'GTZAN', 'MTT']", total_layer=14):
    state = 'get_checkpoint'
    dataset_result_patterns = {
        'MTT': ["valid_aucroc", "test_aucroc", "test_ap"],
        'GS':  ["valid_best_ensemble_score", "test_ensemble_score_val-select","test_ensemble_gmean_score"],
        'EMO': ["valid_r2", "test_r2", 'test_arousal_r2', 'test_valence_r2'],
        'GTZAN': ["valid_acc", "test_acc"],
        'NSynthI': ["valid_acc", "test_acc"],
        'NSynthP': ["valid_acc", "test_acc"],
    }

    current_checkpoint = ''
    current_dataset = ''
    current_layer = ''

    all_results = {}

    with open(log_file) as f:
        for line in f:
            # print(line)
            # break
            if state == 'get_checkpoint':
                # detect current checkpoint name
                if 'Probing' in line and 'with model' in line:
                    # eg: Probing GS dataset with model: HF_HuBERT_base_MPD_train_1Kh_MuBERT_MPD_10Kh_HPO-v3_DC-v6_ckpt_23_360k
                    # current_checkpoint = os.path.basename(line.strip())
                    line = line.strip().split()
                    current_checkpoint = line[-1]
                    current_dataset = line[1]
                    # print(line)
                    if current_dataset in dataset_names:
                        if not SLIENT_MODE:
                            print(f'start to get results for checkpoint {current_checkpoint}, dataset {current_dataset}')

                        if current_checkpoint not in all_results:
                            all_results[current_checkpoint] = {}
                        if current_dataset not in all_results[current_checkpoint]:
                            all_results[current_checkpoint][current_dataset] = {}
                        state = 'get_dataset_layer'
                    else:
                        print(f'skip getting results for checkpoint {current_checkpoint}, dataset {current_dataset}, since its not in the list {dataset_names}')

            elif state == 'get_dataset_layer':
                # detect current layer and evaluation dataset
                if 'Probing' in line and 'dataset with Layer' in line:
                    #  eg. # Probing GS dataset with Layer: all
                    line = line.strip().split()
                    # current_dataset = line[1]
                    assert current_dataset == line[1]
                    current_layer = line[-1]
                    
                    # if current_dataset not in all_results[current_checkpoint]:
                    #     all_results[current_checkpoint][current_dataset] = {}
                    all_results[current_checkpoint][current_dataset][current_layer] = {}

                    if not SLIENT_MODE:
                        print(f'dataset {current_dataset}, layer {current_layer}')
                    state = 'get_test_result'

            elif state == 'get_test_result':
                # get the result of the current run
                n_metrics = len(dataset_result_patterns[current_dataset])
                n_retrieved_metrics = len(all_results[current_checkpoint][current_dataset][current_layer].keys())
                # print(f'need keys: {all_results[current_checkpoint][current_dataset][current_layer].keys()}')
                # print(f'{n_metrics}, {n_retrieved_metrics}')
                if  n_retrieved_metrics < n_metrics:
                    # if not enough results, continue to retrive
                    for pattern in dataset_result_patterns[current_dataset]:
                        if pattern in line and not ('wandb' in line) and not ('|' in line) and len(line.split())==2:
                            line = line.strip().split()
                            if not SLIENT_MODE:
                                print(f'retrieved: {line}')
                            metric_name = line[0]
                            result = float(line[1])
                            # result = line[1]
                            all_results[current_checkpoint][current_dataset][current_layer][metric_name] = result
                            
                            # update retrieved number
                            n_retrieved_metrics = len(all_results[current_checkpoint][current_dataset][current_layer].keys()) 
                            if n_retrieved_metrics == n_metrics:
                                if len(all_results[current_checkpoint][current_dataset]) == total_layer:
                                    state = 'get_checkpoint' # get results for the next checkpoint
                                    if not SLIENT_MODE:
                                        print(f'finish retrieval for {current_checkpoint}')
                                else:
                                    state = 'get_dataset_layer' # get results for the next layer but the same checkpoint
                                break
                
            else:
                raise NotImplementedError('unimplemented states')
    return all_results
# def get_best_metric(result_dict):
#     for dataset in result_dict:
#         for layer in result_dict[dataset]:


if  __name__ == '__main__':
    '''
    usage:
    this program will take the probing logs as input, and output a json file to store all the results at `exp_results/` folder.
    example script:
    1. directly use: python filter_result_from_probing_log.py [probing_log_file]
    2. specify output file name:     python filter_result_from_probing_log.py [probing_log_file] [output_file_name]
    '''

    # log_file = sys.argv[1]
    # if len(sys.argv) == 3:
    #     output_file_name = sys.argv[2] 
    # else:
    #     output_file_name = None

    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--log_file", type=str, required=True, default=None)
    parser.add_argument("-s", "--result_folder", type=str, default="exp_results")
    parser.add_argument("-o", "--output_file_name", type=str, default=None)
    parser.add_argument("-d", "--filter_datasets", type=str, default=DATASET_TO_FILTER)
    parser.add_argument("-tl", "--total_layer", type=int, default=TOTAL_LAYER)
    # parser.add_argument("--not_rewrite", action='store_false')

    args = parser.parse_args()
    # total_shard = args.total_shard

    args.filter_datasets = eval(args.filter_datasets)
    # print(f'filtering results of the following datasets:', args.filter_datasets)
    # print(type(args.filter_datasets), args.filter_datasets)
    # exit()
    if not SLIENT_MODE:
        print(f'total used layer: {args.total_layer}')
        
    all_results = get_all_results(args.log_file, args.filter_datasets, total_layer=args.total_layer)

    # output_file_path = f'exp_results/{args.log_file.replace(".log","")}.json' if args.output_file_name is None else f'exp_results/{args.output_file_name}.json'
    
    if args.output_file_name is None:
        args.output_file_name = args.log_file.replace(".log","") 
    output_file_path = os.path.join(args.result_folder, args.output_file_name+'.json')

    with open(output_file_path, 'w') as fp:
        json.dump(all_results, fp)

    # for f in `ls eval.HPO-*.log`; do echo $f; python filter_result_from_probing_log.py $f; done
