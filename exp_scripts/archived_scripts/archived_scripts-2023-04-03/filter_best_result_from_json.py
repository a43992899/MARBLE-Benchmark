
import sys
import os
import json
import pandas
import argparse
import numpy as np

DATASET_TO_FILTER = "['MTT', 'GTZAN', 'GS', 'EMO', 'NSynthI', 'NSynthP']"

EVAL_MODE='layer_as_competitor'
# EVAL_MODE='valid'

def get_best_metric(result_dict):
    '''retrieve results of a checkpoint'''
    # dataset_result_patterns = {
    #     'MTT': ["test_aucroc", "test_ap"],
    #     'GS':  ["test_ensemble_score_val-select"],
    #     'EMO': ["test_r2", 'test_arousal_r2', 'test_valence_r2'],
    #     'GTZAN': ["test_acc"],
    # }
    best_result = {}
    for dataset in result_dict:
        # best_result[dataset] = []
        results = []
        for layer in result_dict[dataset]:
            results.append(result_dict[dataset][layer])

        if EVAL_MODE == 'layer_as_competitor':
            if dataset == 'MTT':
                sorted_results = sorted(results, key=lambda x:x['test_aucroc'], reverse=True)
            elif dataset == 'GTZAN':
                sorted_results = sorted(results, key=lambda x:x['test_acc'], reverse=True)
            elif dataset == 'GS':
                sorted_results = sorted(results, key=lambda x:x['test_ensemble_gmean_score'], reverse=True)
            elif dataset == 'EMO':
                sorted_results = sorted(results, key=lambda x:x['test_r2'], reverse=True)
            elif dataset == 'NSynthI':
                sorted_results = sorted(results, key=lambda x:x['test_acc'], reverse=True)
            elif dataset == 'NSynthP':
                sorted_results = sorted(results, key=lambda x:x['test_acc'], reverse=True)
            else:
                raise NotImplementedError
        else:
            if dataset == 'MTT':
                sorted_results = sorted(results, key=lambda x:x['valid_aucroc'], reverse=True)
            elif dataset == 'GTZAN':
                sorted_results = sorted(results, key=lambda x:x['valid_acc'], reverse=True)
            elif dataset == 'GS':
                sorted_results = sorted(results, key=lambda x:x['valid_best_ensemble_score'], reverse=True)
            elif dataset == 'EMO':
                sorted_results = sorted(results, key=lambda x:x['valid_r2'], reverse=True)
            elif dataset == 'NSynthI':
                sorted_results = sorted(results, key=lambda x:x['valid_acc'], reverse=True)
            elif dataset == 'NSynthP':
                sorted_results = sorted(results, key=lambda x:x['valid_acc'], reverse=True)
            else:
                raise NotImplementedError
        best_result[dataset] = sorted_results[0]
        # for metric in result_dict[dataset]:

            # pass
    
    return best_result

if  __name__ == '__main__':

    # result_file = sys.argv[1]
    # rewrite_best_result = True

    parser = argparse.ArgumentParser()
    # parser.add_argument("-l","--log_file", type=str, required=True, default=None)
    # parser.add_argument("-s", "--result_folder", type=str, default="exp_results")
    # parser.add_argument("-o", "--output_file_name", type=str, default=None)
    parser.add_argument("-d", "--filter_datasets", type=str, default=DATASET_TO_FILTER)

    parser.add_argument("-r", "--result_file", type=str, default=None)
    
    # parser.add_argument("--not_rewrite", action='store_false')

    args = parser.parse_args()
    args.filter_datasets = eval(args.filter_datasets)
    # print(f'filtering results of the following datasets:', args.filter_datasets)

    # result_file = os.path.join(args.result_folder, args.output_file_name+'.json')

    result_file = args.result_file

    # assert os.path.exists(result_file)

    with open(result_file)  as f:
        all_results = json.load(f)

    # best_csv = 'exp_result/checkpoint_best_results.csv'
    # if os.path.exists(best_csv):
    #     df = pandas.read_csv(best_csv)
    # else:
    #     df = pandas.DataFrame(columns=['ckpt', 'MTT_aucroc', 'MTT_ap', 'GTAN_acc', 'GS_score', 'EMO_r2_arousal', 'EMO_r2_valence'])

    sorted_checkpoints = sorted(list(all_results.keys()), reverse=True)
    print(f'detecting checkpoints: {sorted_checkpoints}')
    # print(all_results)
    for checkpoint_name in sorted_checkpoints:
        # print(checkpoint_name)
        # if not (df['ckpt'] == checkpoint_name).any():
        best_results = get_best_metric(all_results[checkpoint_name])
        # print(f'best for ckpt {checkpoint_name}')
        # print(best_results)
        
        # row_to_write = [
        #     best_results['MTT']['test_aucroc'], best_results['MTT']['test_ap'],
        #     best_results['GTZAN']['test_acc'], 
        #     best_results['GS']['test_ensemble_score_val-select'], 
        #     best_results['EMO']['test_arousal_r2'], best_results['EMO']['test_valence_r2'],
        #     ]
        print(f'get best results for', best_results.keys())
        print(best_results)
        row_to_write = []
        # print(args.filter_datasets)
        if 'MTT' in args.filter_datasets:
            row_to_write.append(best_results['MTT']['test_aucroc'])
            row_to_write.append(best_results['MTT']['test_ap'])
        if 'GTZAN' in args.filter_datasets:
            row_to_write.append(best_results['GTZAN']['test_acc'])
        if 'GS'in args.filter_datasets:
            # row_to_write.append(best_results['GS']['test_ensemble_score_val-select'])
            row_to_write.append(best_results['GS']['test_ensemble_gmean_score'])
        if 'EMO' in args.filter_datasets:
            row_to_write.append(best_results['EMO']['test_arousal_r2'])
            row_to_write.append(best_results['EMO']['test_valence_r2'])
        if 'NSynthI' in args.filter_datasets:
            row_to_write.append(best_results['NSynthI']['test_acc'])
        if 'NSynthP' in args.filter_datasets:
            row_to_write.append(best_results['NSynthP']['test_acc'])


        # row_to_write.append(np.mean(row_to_write))

        print(','.join([str(s) for s in row_to_write])+',')
