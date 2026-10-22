"""zerowhite.chat - a conversational model you train yourself, entirely in numpy.

    config.py      layered, schema-driven configuration (nothing is hard-coded; derived values are measured)
    hardware.py    CPU / RAM / thread probing used to derive sizes and budgets
    paths.py       the workspace layout (data/ inbox, models/, state/, logs/ ...)
    corpus/        reading, converting, filtering and storing datasets (the data/ folder pipeline)
    tokenizer.py   byte-level BPE trained from scratch on your data + the chat template
    model.py       a decoder-only transformer with hand-written backward pass
    train.py       the real trainer (resume, validation, early stop, checkpoints)
    online.py      online learning from live feedback with replay and a regression gate
    generate.py    sampling with a KV cache
    export.py      model conversion (npz / int8 / safetensors)
    service/       background daemon, HTTP API and web UI
    ../voice/      speech in / speech out

There is no external AI service and no pre-trained weights: everything the chat model knows comes from the files
you put in the data folder.
"""
