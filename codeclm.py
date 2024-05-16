import os
import argparse
import torchaudio
import einops
from transformers import EncodecModel
from transformers.models.llama import LlamaForCausalLM

def flatten (x) :
    x = x.squeeze ()
    assert len (x.shape) == 2
    assert x.shape[0] == 4
    return einops.rearrange (x, 'K T -> (T K)')

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser ()
    parser.add_argument ("c", type = str, default = "./ckpt/hf_ckpt_exp1.3/20.32B")
    parser.add_argument ("--audio", type = str, default = "./data/EMO/clips_45seconds/1.mp3")
    args = parser.parse_args ()
    llamacep = args.c

    device = 'cuda'
    encodecmodel = EncodecModel.from_pretrained (f"facebook/encodec_32khz")
    encodecmodel.to (device)
    audio, sr = torchaudio.load (args.audio)
    to_encodec = audio[None].expand (1, -1, -1).cuda ()
    audio_codes = encodecmodel (to_encodec).audio_codes
    print (f"audio_codes.shape = {audio_codes.shape}")
    flattened_encodec_tokens = flatten (audio_codes)
    to_codeclm = flattened_encodec_tokens[None].expand (1, -1)
    print (f"to_codeclm.shape = {to_codeclm.shape}")
    llamamodel = LlamaForCausalLM.from_pretrained (llamacep).cuda ()
    hidden_states = llamamodel (to_codeclm, return_dict = True, output_hidden_states = True)['hidden_states']
    print (f"hidden_states.shape = {hiddenstate[0].shape}")
