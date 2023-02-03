import gc
import logging
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from psutil import Process
from scipy.sparse import coo_matrix, csr_matrix

import emcfile as ef


@pytest.fixture()
def data():
    np.random.seed(123)
    num_data, num_pix = 1000, 4096
    dense = (10 * np.random.rand(num_data, num_pix) ** 5).astype(int)
    den_data = ef.patterns(dense)
    np.testing.assert_almost_equal(den_data.get_mean_count(), np.sum(dense) / num_data)
    np.testing.assert_equal(den_data.todense(), dense)
    return den_data


def test_operation(data):
    data.get_mean_count()
    data.nbytes
    data.sparsity()


def test_readcoomatrix():
    np.random.seed(123)
    dense = (5 * np.random.rand(400, 128**2) ** 5).astype("i4")
    p2 = ef.patterns(dense)  # reading dense form
    co = coo_matrix(dense)
    p = ef.patterns(co)  # reading coo_matrix form
    assert p == p2


def test_getitem(data):
    for i in np.random.choice(data.num_data, 5):
        assert np.sum(data[i] == 1) == data._ones[i]

    mask = np.random.rand(data.num_data) < 0.5
    idx = np.where(mask)[0]
    t0 = time.time()
    subdata1 = data[mask]
    subdata2 = data[idx]
    logging.info(f"Select dataset: {(time.time()-t0)/2}s")
    assert subdata1 == subdata2
    for _ in np.random.choice(subdata1.num_data, 5):
        assert np.all(subdata1[_] == data[idx[_]])


def test_vstack(data):
    patterns = [ef.patterns(data.num_pix)] + [
        deepcopy(data[i * 10 : (i + 1) * 10]) for i in range(5)
    ]
    ans = ef.vstack(patterns)
    assert data[:50] == ans
    ans = ef.vstack(patterns, destroy=True)
    assert data[:50] == ans

    patterns = [ef.patterns(np.full((10000, 1000), 2)) for _ in range(2)]
    process = Process()
    m0 = process.memory_info().rss
    ans = ef.vstack(patterns, destroy=True)
    gc.collect()
    m1 = process.memory_info().rss
    assert (m1 - m0) < ans.nbytes * 0.9
    assert len(patterns) == 0


def test_sum():
    num_data, num_pix = 30, 4
    dense = (10 * np.random.rand(num_data, num_pix) ** 5).astype(int)
    data = ef.patterns(dense)
    np.testing.assert_equal(data.sum(), dense.sum())
    np.testing.assert_equal(data.sum(axis=1), dense.sum(axis=1))
    np.testing.assert_equal(
        data.sum(axis=1, keepdims=True), dense.sum(axis=1, keepdims=True)
    )
    np.testing.assert_equal(data.sum(axis=0), dense.sum(axis=0))
    np.testing.assert_equal(
        data.sum(axis=0, keepdims=True), dense.sum(axis=0, keepdims=True)
    )


def test_empty():
    num_pix = 32
    empty_data = ef.patterns(num_pix)
    assert empty_data.num_pix == num_pix
    assert empty_data.num_data == 0


def test_io(data):
    with tempfile.NamedTemporaryFile(suffix=".emc") as f:
        with pytest.raises(Exception):
            data.write(Path(f.name))
        t0 = time.time()
        data.write(Path(f.name), overwrite=True)
        logging.info(f"Writing {data.num_data} patterns to emc file: {time.time()-t0}")

        t0 = time.time()
        d_read = ef.patterns(f.name)
        logging.info(
            f"Reading {data.num_data} patterns from emc file: {time.time()-t0}"
        )
        assert d_read == data

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        t0 = time.time()
        data.write(f.name, overwrite=True, h5version="1")
        logging.info(
            f"Writing {data.num_data} patterns to h5 file(v1): {time.time()-t0}"
        )

        start = data.num_data // 3
        end = start * 2
        t0 = time.time()
        d_read = ef.patterns(f.name, start=start, end=end)
        logging.info(
            f"Reading {d_read.num_data} patterns from h5 file(v1): {time.time()-t0}"
        )
        d_read.offset = (0, d_read.num_data)
        assert d_read == data[start:end]

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        t0 = time.time()
        data.write(ef.h5path(f.name, "data"), overwrite=True, h5version="2")
        logging.info(
            f"Writing {data.num_data} patterns to h5 file(v2): {time.time()-t0}"
        )
        t0 = time.time()
        d_read = ef.patterns(ef.h5path(f.name, "data"), start=start, end=end)
        logging.info(
            f"Reading {d_read.num_data} patterns from h5 file(v2): {time.time()-t0}"
        )
        d_read.offset = (0, d_read.num_data)
        assert d_read == data[start:end]


def gen_write_photons():
    data = ef.patterns("data/photons.emc")[:512]
    for i in 2 ** np.arange(0, 10, 2):
        yield ".emc", [data] * i
        yield ".h5", [data] * i


@pytest.mark.parametrize("suffix, data_list", gen_write_photons())
def test_write_photons(suffix, data_list):
    with tempfile.NamedTemporaryFile(suffix=suffix) as f0, tempfile.NamedTemporaryFile(
        suffix=suffix
    ) as f1:
        t = time.time()
        all_data = ef.vstack(data_list)
        all_data.write(f1.name, overwrite=True)
        t1 = time.time() - t
        logging.info(f"speed[single]: {all_data.nbytes * 1e-9 /t1:.2f} GB/s")

        t = time.time()
        ef.write_photons(data_list, f0.name, overwrite=True)
        t0 = time.time() - t
        logging.info(
            f"speed[multiple; #patterns={len(data_list)}]: {all_data.nbytes * 1e-9 /t0:.2f} GB/s"
        )

        logging.info(f"speed ratio [{suffix}]: {t1 / t0:.3f}")
        assert ef.patterns(f0.name) == all_data


def test_pattern_mul():
    m = ef.patterns("data/photons.emc", end=10)
    mtx = np.random.rand(m.num_pix, 10)
    np.testing.assert_almost_equal(m @ mtx, m.todense() @ mtx)
    mtx = mtx > 0.4
    np.testing.assert_almost_equal(m @ mtx, m.todense() @ mtx)
    mtx = coo_matrix(mtx)
    np.testing.assert_equal((m @ mtx).todense(), m.todense() @ mtx)
    mtx = csr_matrix(mtx)
    np.testing.assert_equal((m @ mtx).todense(), m.todense() @ mtx)


def test_PatternsSOneEMC():
    p0 = ef.patterns("data/photons.emc")
    p1 = ef.PatternsSOneEMC("data/photons.emc")
    np.testing.assert_equal(p0[10], p1[10])
    assert p0[::2] == p1[::2]


def test_PatternsSOneEMC_getitem():
    np.random.seed(12)
    p = Path("data/photons.emc")
    d1 = ef.patterns(p)
    d2 = ef.PatternsSOneEMC(p)
    assert d2.sparsity() == d1.sparsity()
    np.testing.assert_equal(d2[3], d1[3])
    idx = np.random.rand(d1.num_data) > 0.5
    assert d2[idx] == d1[idx]
    idx = np.where(idx)[0]
    assert d2[idx] == d1[idx]
    assert d2[np.array([], dtype=np.int32)].num_data == 0
