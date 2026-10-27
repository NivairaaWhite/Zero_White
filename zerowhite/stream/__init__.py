"""stream - real-time multimodal streaming (6.5): ring buffers, tokenisers whose chunked output equals their batch output, a sliding-window language model,
a pipeline that wires live sources to it, and observation wrappers (frame stacking, partial observability) for the reinforcement-learning families.

    from zerowhite.stream import Ring, TextStream, AudioStream, AudioDecoder, PixelCodec, VideoStream, TokenSpace, interleave, WindowLM, StreamPipeline
    from zerowhite.stream import FrameStackEnv, MaskObsEnv

docs/STREAM.md has the guarantees (and the list of what is NOT claimed: no trained audio/vision understanding, no hard real-time deadlines).
"""
from .envs import FrameStackEnv, MaskObsEnv, make_wrapped                                          # noqa: F401
from .pipeline import StreamPipeline                                                                # noqa: F401
from .ring import Ring, RingOverflow                                                                # noqa: F401
from .tokens import AudioDecoder, AudioStream, PixelCodec, TextStream, TokenSpace, VideoStream, interleave   # noqa: F401
from .window import WindowLM                                                                        # noqa: F401
