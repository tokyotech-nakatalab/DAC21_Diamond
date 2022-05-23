import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from QuestionRecommender.datasets.utils import ShelveDict, dumps, loads


@pytest.mark.parametrize("obj", [1, "a", {"1": 2}, [3, (4, 5)]])
def test_loads_and_dumps(obj):
    x = dumps(obj)
    assert isinstance(x, bytes)
    y = loads(x)
    assert obj == y
    x = ShelveDict.key_dumps(obj)
    assert isinstance(x, str)
    y = ShelveDict.key_loads(x)
    assert obj == y


def test_ShelveDict(tmp_path):
    arr = np.random.randn(1000, 10)
    df = pd.DataFrame(arr)
    d = {i: df.copy() for i in range(100)}
    od = ShelveDict.from_dict(d)
    for k in d.keys():
        np.testing.assert_allclose(d[k].values, od[k].values)

    path = (tmp_path / "od2").as_posix()
    shutil.copytree(od.tmpdir, path)
    od2 = ShelveDict.load_shelve(path + "/data")
    for k in d.keys():
        np.testing.assert_allclose(od2[k].values, od[k].values)

    od3 = ShelveDict()
    for k in d.keys():
        od3[k] = d[k]
    for k in d.keys():
        np.testing.assert_allclose(d[k].values, od3[k].values)

    shutil.rmtree(Path(od2.path).parent)
    path = path + "/dadada"
    od3.save_shelve(path)
    od4 = ShelveDict.load_shelve(path)
    for k in d.keys():
        np.testing.assert_allclose(od4[k].values, od3[k].values)
    od4.localize()
    od5 = od3.to_dict()
    od4.clear_cache()
    for k in d.keys():
        od4.prefetch(k)
        np.testing.assert_allclose(od4[k].values, od5[k].values)
