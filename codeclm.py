import os
from transformers import LlamaTokenizer, LlamaForCausalLM, EncodecModel, AutoProcessor
import torch
import json
# import torchaudio
from scipy.io.wavfile import write
import einops
from glob import glob
import argparse
from pydub import AudioSegment

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def offset_tok_ids(x, device_num, global_offset=0, offset_by_codebook=True, codebook_size=2048, num_codebooks=4):
    """
    x: (K, T)
    """
    device_cpu = torch.device("cpu")
    device_gpu = torch.device(device_num if torch.cuda.is_available() else 'cpu')
    x = x.to(device_cpu).numpy()
    assert x.max() < codebook_size, f"max(x)={x.max()}, codebook_size={codebook_size}"
    assert x.min() >= 0, f"min(x)={x.min()}"
    assert x.shape[0] == num_codebooks, f"x.shape[0]={x.shape[0]}, num_codebooks={num_codebooks}"

    _x = x.copy()
    for k in range(num_codebooks):
        if offset_by_codebook:
            _x[k] += global_offset + k * codebook_size
        else:
            _x[k] += global_offset
    _x = torch.tensor(_x).to(device_gpu)
    return _x

def unoffset_tok_ids(x, device_num, global_offset=0, offset_by_codebook=True, codebook_size=2048, num_codebooks=4):
    """
    x: (K, T)
    """
    device_cpu = torch.device("cpu")
    device_gpu = torch.device(device_num if torch.cuda.is_available() else 'cpu')
    x = x.to(device_cpu).numpy()
    assert x.max() < codebook_size * num_codebooks, f"max(x)={x.max()}, codebook_size={codebook_size}"
    assert x.min() >= global_offset, f"min(x)={x.min()}, global_offset={global_offset}"
    assert x.shape[0] == num_codebooks, f"x.shape[0]={x.shape[0]}, num_codebooks={num_codebooks}"

    _x = x.copy()
    for k in range(num_codebooks):
        if offset_by_codebook:
            _x[k] -= global_offset + k * codebook_size
        else:
            _x[k] -= global_offset
    _x = torch.tensor(_x).to(device_gpu)
    return _x

def flatten(x):
    x = x.squeeze()
    assert len(x.shape) == 2
    assert x.shape[0] == 4
    return einops.rearrange(x, 'K T -> (T K)')

def unflatten(x):
    x = x.squeeze()
    if x.shape[0] % 4 != 0:
        n = x.shape[0] // 4
        x = x[:4*n]
    assert len(x.shape) == 1
    assert x.shape[0] % 4 == 0
    return einops.rearrange(x, '(T K) -> K T', K=4)

def ckpt_generate(jsonl_filepath, 
                  sample_list, 
                  prompt_len, 
                  top_k, top_p, temperature, 
                  exp, epo, rp, reconstructed_rootpath,
                  device):
    with open(jsonl_filepath, 'r') as f:
        lines = f.readlines()
        # # continuous id
        # for i in range(sample_id_start, sample_id_end):
        # human chosen
        for i in sample_list:
            data = json.loads(lines[i])
            input_codec = torch.tensor(data['text'][:prompt_len]).to(device)
            input_codec = torch.reshape(input_codec, (1, prompt_len))
            outputs = model.generate(inputs = input_codec, 
                                max_length = 8000,
                                top_k = top_k,
                                top_p = top_p,
                                do_sample=True,
                                temperature = temperature,
                                repetition_penalty = rp)
            # print(tokenizer.decode(outputs[0]))
            # print(outputs[0])
            if not os.path.exists(f"{reconstructed_rootpath}/codes_pt/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(rp)}_ep{epo}_pl{prompt_len}"):
                os.makedirs(f"{reconstructed_rootpath}/codes_pt/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(rp)}_ep{epo}_pl{prompt_len}")
            torch.save(outputs[0], 
                       f"{reconstructed_rootpath}/codes_pt/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(rp)}_ep{epo}_pl{prompt_len}/output_id{str(i)}.pt")
            # torch.save(input_codec, 
            #            f"codes_pt/exp1.5_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(rp)}_ep{ep}_pl{prompt_len}/input_id{str(i)}.pt")

def reconstruct_audios(sample_list, 
                       prompt_len, 
                       top_k, top_p, temperature,
                       exp, epo,
                       reconstructed_rootpath, device_num,
                       num_codebooks, codebook_size):
    # # reconstruct the audio: continuous id
    # for i in range(sample_id_start, sample_id_end):
    # reconstruct the audio:
    device = torch.device(device_num if torch.cuda.is_available() else 'cpu')
    for i in sample_list:
        # input_codec = torch.load(f"codes_pt/exp1.5_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{ep}_pl{prompt_len}/input_id{str(i)}.pt")
        try:
            output_codec = torch.load(f"{reconstructed_rootpath}/codes_pt/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{epo}_pl{prompt_len}/output_id{str(i)}.pt")
            output_unflattened = unflatten(output_codec.to(device))
            output_unoffseted = unoffset_tok_ids(output_unflattened, device_num, global_offset=0, offset_by_codebook=True, codebook_size=codebook_size, num_codebooks=num_codebooks)

            # remove the negative codes
            error_index = output_unoffseted.shape[1] # define the variable
            for codebook in output_unoffseted:
                for timestep in range(codebook.shape[0]):
                    if codebook[timestep] < 0 and timestep < error_index:
                        error_index = timestep
                        break # finish checking this codebook
            output_unoffseted = output_unoffseted[:,:error_index]

            output_unoffseted = torch.reshape(output_unoffseted, (1, 1, 4, -1))
            audio_values_output = model_enc.decode(output_unoffseted.to(device), [None])[0]

            # tensor to numpy
            # audio_values_output = audio_values_output.to(torch.device('cpu')).detach().numpy()
            sr = processor.sampling_rate
            if not os.path.exists(f"{reconstructed_rootpath}/reconstructed_audios/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{epo}_pl{prompt_len}"):
                os.makedirs(f"{reconstructed_rootpath}/reconstructed_audios/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{epo}_pl{prompt_len}")
            torchaudio.save(f"{reconstructed_rootpath}/reconstructed_audios/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{epo}_pl{prompt_len}/output_id{str(i)}.mp3", 
                            audio_values_output, sample_rate=sr)

            # Use AudioSegment to write audio 
            # audio = AudioSegment(
            #                     audio_values_output.tobytes(),
            #                     frame_rate=sr,
            #                     sample_width=audio_values_output.dtype.itemsize,
            #                     channels=1
            #                 )
            # audio.export(f"{reconstructed_rootpath}/reconstructed_audios/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{epo}_pl{prompt_len}/output_id{str(i)}.mp3", format='mp3')

            # use scipy.wavfile write audio
            # write(f"{reconstructed_rootpath}/reconstructed_audios/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{epo}_pl{prompt_len}/output_id{str(i)}.wav", sr, audio_values_output)
        except Exception as e:
            if not os.path.exists(f"{reconstructed_rootpath}/reconstructed_audios/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{epo}_pl{prompt_len}"):
                os.makedirs(f"{reconstructed_rootpath}/reconstructed_audios/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{epo}_pl{prompt_len}")
            with open(f"{reconstructed_rootpath}/reconstructed_audios/{exp}_topk{str(top_k)}_topp{str(top_p)}_tem{str(temperature)}_rp{str(repetition_penalty)}_ep{epo}_pl{prompt_len}/error_logs.txt", "a") as f:
                f.write(f"output_id{str(i)}.wav" + "\t" + str(e) + "\n")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstructed_rootpath", type=str, default="/project/buildlam/eval/") # "/workspace/dataset/music/audio-2m_encodec_32khz"
    parser.add_argument("--testset_jsonl_path", type=str, default="/scratch/buildlam/codeclm/Megatron-LMM/dataset/music/youtube-130khr/jsonl/youtube-130khr_encodec_32khz.jsonl") # "/workspace/dataset/music/audio-2m_encodec_32khz.jsonl"
    parser.add_argument("--ckpt_path", type=str, default="/project/buildlam/checkpoints/clm_2b_nl_tp1_pp1_mb1_gb1024_gas2/exp1.4/hf_ckpt/")
    parser.add_argument("--exp_num", type=str, default="exp1.4")
    parser.add_argument("--encodec_ckpt_type", type=str, default="32khz")
    parser.add_argument("--sample_list_txt", type=str, default="sample_list.txt")
    parser.add_argument("--epo_index", type=int, default=-1) # the last epoch
    parser.add_argument("--prompt_len", type=int, default=4000)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--gpu_device", type=str, default='cuda:0')
    parser.add_argument("--num_codebooks", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=2048)
    args = parser.parse_args()

    reconstructed_rootpath = args.reconstructed_rootpath
    testset_jsonl_path = args.testset_jsonl_path
    ckpt_path = args.ckpt_path
    encodec_ckpt_type = args.encodec_ckpt_type
    sample_list_txt = args.sample_list_txt
    epo_list = os.listdir(ckpt_path)
    epo_index = args.epo_index
    prompt_len = args.prompt_len
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    repetition_penalty = args.repetition_penalty
    exp_num = args.exp_num
    gpu_device = args.gpu_device
    num_codebooks = args.num_codebooks
    codebook_size = args.codebook_size

    sample_list = []
    with open(sample_list_txt, "r") as f:
        for line in f.readlines():
            sample_list.append(int(line.strip("\n")))

    reconstructed_rootpath = "/project/buildlam/eval/"
    # change parameters here
    epo = epo_list[epo_index] # define the epoch

    # sample_id_start = 100
    # sample_id_end = 110

    # model load
    model_enc = EncodecModel.from_pretrained(f"facebook/encodec_{encodec_ckpt_type}")
    processor = AutoProcessor.from_pretrained(f"facebook/encodec_{encodec_ckpt_type}")
    tokenizer = LlamaTokenizer.from_pretrained("/scratch/buildlam/codeclm/Megatron-LMM/codeclm/hf/hkg_amber_hf")
    model = LlamaForCausalLM.from_pretrained(ckpt_path + epo) # exp1.4
    device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model_enc.to(device)

    # generate the codes
    ckpt_generate(jsonl_filepath = testset_jsonl_path, 
                sample_list = sample_list,
                prompt_len = prompt_len,
                top_k = top_k,
                top_p = top_p,
                temperature = temperature,
                exp = exp_num,
                epo = epo,
                rp = repetition_penalty,
                reconstructed_rootpath = reconstructed_rootpath,
                device = device)

    reconstruct_audios(sample_list = sample_list, 
                       prompt_len = prompt_len, 
                       top_k = top_k, 
                       top_p = top_p, 
                       temperature = temperature,
                       exp = exp_num, 
                       epo = epo,
                       reconstructed_rootpath = reconstructed_rootpath, 
                       device_num = gpu_device,
                       num_codebooks = num_codebooks,
                       codebook_size = codebook_size)
