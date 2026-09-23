"""Run pip without loading backports.zstd into the pip process.

On Windows, pip's bundled urllib3 imports backports.zstd when it starts.
That keeps the extension DLL open and prevents pip from uninstalling or
upgrading backports.zstd in the same process (WinError 5).
"""

import runpy
import sys


class _SkipZstdForPip:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "backports.zstd":
            raise ModuleNotFoundError("backports.zstd is disabled in the pip process")
        return None


sys.meta_path.insert(0, _SkipZstdForPip())
runpy.run_module("pip", run_name="__main__")
