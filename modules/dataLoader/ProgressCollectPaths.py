import os

from mgds.pipelineModules.CollectPaths import CollectPaths

from tqdm import tqdm


class ProgressCollectPaths(CollectPaths):
    """Collect sample paths while reporting the actual directory entries checked."""

    def _iter_entries(self, path: str, include_subdirectories: bool):
        subdirectories = []
        with os.scandir(path) as entries:
            for entry in entries:
                is_file = entry.is_file()
                if include_subdirectories and not is_file and entry.is_dir() and not entry.name.startswith("."):
                    subdirectories.append(entry.path)
                yield entry.path, is_file

        for subdirectory in subdirectories:
            yield from self._iter_entries(subdirectory, include_subdirectories)

    def start(self, variation: int):
        checked_files = 0
        # Recursive directory contents are unknown until scanned, so a percentage would be misleading.
        with tqdm(desc="checking sample paths", unit="path") as progress:
            for in_index in range(self._get_previous_length(self.concept_in_name)):
                concept = self._get_previous_item(variation, self.concept_in_name, in_index)
                if not concept[self.enabled_in_name]:
                    continue

                include_subdirectories = self._get_previous_item(
                    variation, self.include_subdirectories_in_name, in_index
                )
                file_names = []
                for path, is_file in self._iter_entries(concept[self.path_in_name], include_subdirectories):
                    progress.update()
                    if is_file:
                        checked_files += 1
                        stem, extension = os.path.splitext(path)
                        if (
                            extension.lower() in self.extensions
                            and (
                                not self.include_postfix
                                or any(stem.endswith(postfix) for postfix in self.include_postfix)
                            )
                            and (
                                not self.exclude_postfix
                                or not any(stem.endswith(postfix) for postfix in self.exclude_postfix)
                            )
                        ):
                            file_names.append(path)
                    if progress.n % 256 == 0:
                        progress.set_postfix(
                            files=checked_files, samples=len(self.paths) + len(file_names), refresh=False
                        )

                file_names.sort()
                self.paths.extend(file_names)
                self.concepts.extend([concept] * len(file_names))
                progress.set_postfix(files=checked_files, samples=len(self.paths), refresh=False)
