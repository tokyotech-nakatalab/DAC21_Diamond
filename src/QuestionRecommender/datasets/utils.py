import base64
import copy
import glob
import os
import pickle
import shelve
import shutil
import tempfile
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor as TPE
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from tqdm import tqdm


def dumps(x: Any) -> bytes:
    return zlib.compress(pickle.dumps(x))


def loads(x: bytes) -> Any:
    return pickle.loads(zlib.decompress(x))


class CacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args, cache_len: int = 1024, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len
        self._lock = Lock()

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)
            super().move_to_end(key)

            while len(self) > self.cache_len:
                oldkey = next(iter(self))
                super().__delitem__(oldkey)

    def __getitem__(self, key):
        with self._lock:
            val = super().__getitem__(key)
            super().move_to_end(key)

        return val


class ShelveDict:
    """
    Dictのように使えるShelve

     - Dictから作成するには，`ShelveDict.from_dict(d)`
     - `ShalveDict()`や`ShelveDict.from_dict(d)`で作成したShelveDictをshelveとして保存するには，`shelve_dict.save_shelve(path)`
     - 保存されたshelveから作成するには， `ShelveDict.load_shelve(path)`
     - グループディスクから読んだshelveをノードの一時フォルダに移動したい場合は`shelve_dict.localize()`
     - dictに変換したい場合は， `shelve_dict.to_dict()`
     - shelveから先読みしておいてほしい場合は，`shelve_dict.prefetch(key)`
    """

    suffix: str = "data"

    def __init__(self, max_workers=32, cache_len=8192):
        self.exe = TPE(max_workers=max_workers)
        self._cache = CacheDict(cache_len=cache_len)

    def clear_cache(self):
        self._cache.clear()

    def _create_shelve(self):
        self._shelve = shelve.open(self.path)

    @property
    def tmpdir(self):
        if not hasattr(self, "_tmpdir"):
            self._tmpdir = tempfile.TemporaryDirectory()
        return self._tmpdir.name

    @property
    def path(self):
        if hasattr(self, "_path"):
            return self._path
        else:
            return os.path.join(self.tmpdir, self.suffix)

    @staticmethod
    def key_dumps(x):
        return base64.b64encode(pickle.dumps(x)).decode("ascii")

    @staticmethod
    def key_loads(x):
        return pickle.loads(base64.b64decode(x.encode("ascii")))

    @classmethod
    def from_dict(cls, d: Dict[Any, Any]):
        assert isinstance(d, dict)
        self = cls()
        self._create_shelve()

        for k, v in tqdm(d.items(), total=len(d), desc="Converting dict to shelve"):
            self._shelve[self.key_dumps(k)] = v

        self._shelve.sync()
        return self

    @classmethod
    def load_shelve(cls, path):
        self = cls()
        self._path = path
        self._shelve = shelve.open(path)
        return self

    def _update_cache(self, key, value):
        self._cache[key] = value

    def __getitem__(self, key):
        if key in self._cache:
            try:
                return self._cache[key]
            except KeyError:
                pass
        value = self._shelve[self.key_dumps(key)]
        return value

    def _prefetch(self, key):
        self._update_cache(key, self[key])

    def prefetch(self, key):
        """
        別スレッドでshelveから取得してキャッシュする
        欲しいkeyを早めにprefetchして，別の処理をしてから取得すると効率的
        """
        self.exe.submit(self._prefetch, key)

    def __setitem__(self, key, value):
        if not hasattr(self, "_shelve"):
            self._create_shelve()
        self._shelve[self.key_dumps(key)] = value

    def __del__(self):
        if hasattr(self, "_shelve"):
            self._shelve.close()
        if hasattr(self, "_tmpdir"):
            self._tmpdir.cleanup()
        self.exe.shutdown()

    def save_shelve(self, path):
        self._shelve.sync()
        path = Path(path)
        save_dir = path.parent
        name = path.name
        os.makedirs(save_dir, exist_ok=True)
        assert os.path.isdir(save_dir)
        for sname in glob.glob(os.path.join(self.tmpdir, self.suffix) + "*"):
            dname = Path(save_dir) / Path(sname).name.replace(self.suffix, name)
            shutil.copy(sname, dname)

    def localize(self):
        if not hasattr(self, "_tmpdir"):
            path = copy.copy(self._path)
            path = Path(path)
            save_dir = path.parent
            name = path.name
            del self._path
            self._shelve.sync()
            for sname in glob.glob(f"{path}*"):
                dname = Path(self.tmpdir) / Path(sname).name.replace(name, self.suffix)
                shutil.copy(sname, dname)
            self._shelve.close()
            self._shelve = shelve.open(self.path)

    def to_dict(self):
        d = dict()
        self._shelve.sync()
        for k, v in self._shelve.items():
            d[self.key_loads(k)] = v
        return d
