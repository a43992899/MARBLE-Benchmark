import benchmark as bench
from benchmark.utils.config_utils import load_config, override_config, print_config, merge_args_to_config

def main(args):
    from benchmark.models.musichubert_hf.extract_bert_features import main as extract_hubert_features_main #mert
    from benchmark.models.music2vec.extract_music2vec_features import main as extract_data2vec_features_main #music2vec
    from benchmark.models.data2vec.extract_data2vec_features import main as extract_data2vec_audio_features_main #data2vec-audio
    from benchmark.models.handcrafted.extract_handcrafted_features import main as extract_handcrafted_features_main
    from benchmark.models.jukebox.extract_jukemir_features import main as extract_jukemir_features_main
    from benchmark.models.musicnn.extract_musicnn_features import main as extract_musicnn_features_main
    from benchmark.models.clmr.extract_clmr_features import main as extract_clmr_features_main
    from benchmark.models.mule.extract_mule_features import main as extract_mule_features_main
    from benchmark.models.hubert.extract_hubert_features import main as extract_speech_hubert_features_main #hubert

    config = load_config(args.config, namespace=True)

    if args.override is not None and args.override.lower() != "none":
        override_config(args.override, config)
    
    config = merge_args_to_config(args, config)
    
    print_config(config)

    representation_name = config.dataset.pre_extract.feature_extractor.pretrain.name
    extract_main = bench.NAME_TO_EXTRACT_FEATURES_MAIN[representation_name]
    extract_main = eval(extract_main)
    extract_main(config)
