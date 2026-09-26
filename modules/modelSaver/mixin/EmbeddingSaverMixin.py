from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modules.util.enum.ModelFormat import ModelFormat
from modules.util.path_util import safe_filename

import torch
from torch import Tensor

from safetensors.torch import save_file

if TYPE_CHECKING:
    from modules.model.BaseModel import BaseModel


class EmbeddingSaverMixin:
    # Model component attribute and its on-disk tensor key, primary encoder first.
    embedding_fields: tuple[tuple[str, str], ...]

    def _to_state_dict(
            self,
            embedding: Any | None,
            embedding_state_dict: dict[str, Tensor] | None,
            dtype: torch.dtype | None,
    ) -> dict[str, Tensor]:
        state_dict = dict(embedding_state_dict) if embedding_state_dict is not None else {}
        if embedding is not None:
            for attribute, key in self.embedding_fields:
                component = getattr(embedding, attribute)
                if component.vector is not None:
                    state_dict[key] = component.vector
                if component.output_vector is not None:
                    state_dict[f"{key}_out"] = component.output_vector

        return {
            key: value.detach().to(device="cpu", dtype=dtype).contiguous()
            for key, value in state_dict.items()
        }

    def _primary_embedding(self, embedding: Any) -> Any:
        return getattr(embedding, self.embedding_fields[0][0])

    def save_single(
            self,
            model: BaseModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embedding = model.embedding
        if embedding is not None:
            embedding_uuid = self._primary_embedding(embedding).uuid
        elif model.embedding_state_dicts:
            embedding_uuid = next(iter(model.embedding_state_dicts))
        else:
            raise ValueError("No embedding available to save")

        self._save_embedding(
            embedding,
            model.embedding_state_dicts.get(embedding_uuid),
            embedding_uuid,
            output_model_format,
            output_model_destination,
            dtype,
        )

    def save_multiple(
            self,
            model: BaseModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embeddings = {self._primary_embedding(x).uuid: x for x in model.additional_embeddings}
        embedding_uuids = model.embedding_state_dicts.keys() | embeddings.keys()
        if model.embedding is not None:
            embedding_uuids.discard(self._primary_embedding(model.embedding).uuid)

        for embedding_uuid in sorted(embedding_uuids):
            embedding = embeddings.get(embedding_uuid)
            embedding_state = model.embedding_state_dicts.get(embedding_uuid)
            if embedding is None and embedding_state is None:
                continue

            destination = output_model_destination
            if output_model_format == ModelFormat.SAFETENSORS:
                name = self._primary_embedding(embedding).placeholder if embedding is not None else embedding_uuid
                name = safe_filename(name, allow_spaces=False, max_length=None)
                destination = os.path.join(f"{destination}_embeddings", f"{name}.safetensors")

            self._save_embedding(
                embedding, embedding_state, embedding_uuid, output_model_format, destination, dtype,
            )

    def _save_embedding(
            self,
            embedding: Any | None,
            embedding_state: dict[str, Tensor] | None,
            embedding_uuid: str,
            output_model_format: ModelFormat,
            destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.SAFETENSORS:
                self._save_safetensors(embedding, embedding_state, destination, dtype)
            case ModelFormat.INTERNAL:
                self._save_internal(embedding, embedding_state, embedding_uuid, destination)
            case _:
                raise NotImplementedError(f"Unsupported embedding output format: {output_model_format}")

    def _save_safetensors(
            self,
            embedding: Any | None,
            embedding_state_dict: dict[str, Tensor] | None,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        state_dict = self._to_state_dict(
            embedding,
            embedding_state_dict,
            dtype,
        )

        save_file(state_dict, destination)

    def _save_internal(
            self,
            embedding: Any | None,
            embedding_state: dict[str, Tensor] | None,
            embedding_uuid: str,
            destination: str,
    ):
        safetensors_embedding_name = os.path.join(
            destination,
            "embeddings",
            f"{embedding_uuid}.safetensors",
        )
        self._save_safetensors(
            embedding,
            embedding_state,
            safetensors_embedding_name,
            None,
        )
