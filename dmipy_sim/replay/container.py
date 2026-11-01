"""The ``.rpk`` container on the producer's side (RPK.md 12): :func:`write_rpk` writes a pack's tensors and metadata,
:func:`save_pack` writes a pack. The file is dmipy-replay's safetensors container
(:func:`dmipy_replay.replay.container.write_safetensors`), the metadata under the ``"rpk"`` header key."""
import json

from dmipy_replay.replay.container import write_safetensors

__all__ = ["write_rpk", "save_pack"]


def write_rpk(path, arrays, metadata):
    """Write a replay pack: ``arrays`` (name -> ndarray, or a list of ndarrays stored as their concatenation along the
    first axis, as a merge stores its shards' tensors) as safetensors tensors, ``metadata`` (a dict) serialised to JSON
    under the ``"rpk"`` header key: byte for byte what ``safetensors.numpy.save_file`` writes for the contiguous arrays,
    streamed into a sibling file that replaces ``path`` once complete, so a pack mapped from ``path`` keeps reading the
    file it was mapped from (:func:`~dmipy_replay.replay.container.write_safetensors`)."""
    write_safetensors(path, arrays, {"rpk": json.dumps(dict(metadata))})


def save_pack(pack, path):
    """Write ``pack`` (a :class:`~dmipy_replay.replay.replay.ReplayPack`) to a ``.rpk`` file at ``path`` and make the file
    the pack's source; returns ``path``."""
    write_rpk(path, {k: v for k, v in pack.arrays.items() if v is not None}, pack.meta)
    pack.source = str(path)
    return path
