"""zerowhite.chat.corpus - the data-folder pipeline: read any dataset, normalise, filter, store incrementally."""
from .adapters import ADAPTERS, AdapterError, Ctx, detect_adapter  # noqa: F401
from .filters import RecordFilter, describe_filters  # noqa: F401
from .readers import AUDIO_EXTENSIONS, ReaderError, iter_rows, supported_extensions  # noqa: F401
from .records import canonical_text, make_chat, make_pref, make_text, record_hash  # noqa: F401
from .store import FileEntry, IngestReport, Ingestor, ShardInfo  # noqa: F401
