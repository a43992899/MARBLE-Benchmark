TASK_TYPE_MAPPER = {
    "MTT": "multilabel",
    "MAESTRO": "multilabel",
    "MTGGenre": "multilabel",
    "MTGInstrument": "multilabel",
    "MTGMood": "multilabel",
    "MTGTop50": "multilabel",
    "GTZAN": "multiclass",
    "GS": "multiclass",
    "NSynthI": "multiclass",
    "NSynthP": "multiclass",
    "VocalSetS": "multiclass",
    "VocalSetT": "multiclass",
    "EMO": "regression",
    "GTZANBT": "multiclass",
    "MUSDB18": "regression",
}

SUPPORTED_TASKS = list(TASK_TYPE_MAPPER.keys())
FEWSHOT_TASKS = [k for k, v in TASK_TYPE_MAPPER.items() if (v == "multiclass" and k != "GTZANBT")]
