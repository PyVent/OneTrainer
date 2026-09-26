from contextlib import contextmanager
from threading import Lock

from mgds.pipelineModules.EncodeVAE import EncodeVAE


class EncodeAnimaVAE(EncodeVAE):
    """Serialize VAE inference because both Anima VAEs have mutable temporal caches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vae_lock = Lock()

    @contextmanager
    def _all_contexts(self, contexts):
        # Image loading/augmentation remains parallel, only the encode is serialized.
        with self._vae_lock, super()._all_contexts(contexts):
            yield
