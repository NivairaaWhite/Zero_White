# Real-time multimodal streaming (6.5)

`zerowhite/stream/`. Until 6.4 every input was a finished file: a text, a recording, a frame batch. This package lets text, audio and video arrive **as they happen**, be tokenised the moment enough has arrived, and be fed to one language model whose context window slides.

```
zerowhite stream demo            # a text + audio + video stream through one pipeline, with the counters
zerowhite stream check           # the chunk-invariance and window-exactness checks, on random chunkings
```

## Parts

| piece | what it guarantees |
|---|---|
| `Ring` (`ring.py`) | fixed capacity; overflow is `drop_oldest`, `drop_newest` or `error`, and **counted**: `pushed == consumed + dropped + len` always. One producer thread and one consumer thread need no other locking. |
| `TextStream`, `AudioStream`, `AudioDecoder`, `PixelCodec`, `VideoStream` (`tokens.py`) | **chunk invariance**: feeding a stream in *any* chunking gives exactly the tokens tokenising it whole gives (tested on random chunkings, including a UTF-8 character split across chunks). The tail that could still change (an unfinished word, half a character) is held back until the next chunk or `finish()`. |
| `TokenSpace`, `interleave` | one integer id space for text, audio and pixel tokens plus modality markers, so a single transformer can read a merged, time-ordered stream. |
| `WindowLM` (`window.py`) | a model that reads a stream longer than its context. `mode="reprefill"` is **exact**: the cache equals a fresh pass over the current window to 1e-9 (the oldest `stride` tokens go, the first `sink` stay). `mode="rotate"` is cheap and **approximate** (see below). |
| `StreamPipeline` (`pipeline.py`) | rings, then streaming tokenisers, then one callback (typically `WindowLM.feed`); a deterministic `pump()` or a background thread. `stats()` reports rows dropped and push-to-handoff latency per source. |
| `FrameStackEnv`, `MaskObsEnv` (`envs.py`) | observation wrappers for the reinforcement-learning families: `stack3:cartpole`, `pomdp:cartpole`, `stack2:pomdp:cartpole`. Stacking restores the velocity a partial observation hides; any family working on observation vectors gets short-term memory with no change of its own. |

## What is *not* claimed

* **No trained audio or vision understanding.** The ids are a *layout*. A chat model has to be trained on this vocabulary before it does anything with the audio and pixel ids. The streaming layer guarantees the plumbing (exact tokens, bounded memory, counted loss), nothing about meaning.
* **No hard real-time deadline.** Latency is measured and reported; nothing enforces it. An audio frame costs one hop (10 ms) of algorithmic latency; the rest is however fast numpy is on the machine.
* **`rotate` mode is not equal to a fresh pass.** Rotating the cache is exact for the rotary embedding at the first layer; deeper layers' keys and values were computed from hidden states that had attended to the dropped tokens, so they are stale by that context. The tests measure the difference. Use `reprefill` when exactness matters, `rotate` when the per-token cost does.
* **The pixel codec is a patch codec** (residual k-means over p x p patches), not a perceptual model. Reconstruction quality is whatever the codebooks fitted on your frames give.
* Dropped data is *reported*, not recovered: a source that outruns the model loses rows, and `stats()` says how many.
