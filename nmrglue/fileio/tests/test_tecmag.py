"""Unit tests for nmrglue/fileio/tecmag.py module.

These tests use hand-crafted byte sequences to exercise the parser
without requiring external .tnt test data. The synthetic files mimic
the minimum structure needed: the TNT1.NNN magic header, TLV section
records for TMAG/PSEQ/DATA/TMG2, and valid Pascal-string table pairs
inside PSEQ.
"""
import os
import struct
import tempfile

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import nmrglue as ng
from nmrglue.fileio.tecmag import (
    TNTTMAG, TNTTMG2, guess_tables, _read_pascal_pair, _convert_si
)


# =============================================================================
# Helpers — synthesize minimal valid .tnt byte sequences
# =============================================================================

def _pascal_string(s):
    """Build a length-prefixed Pascal string."""
    if isinstance(s, str):
        s = s.encode('ascii')
    return struct.pack('<i', len(s)) + s


def _tlv_section(tag, payload, flag=0):
    """Build a TLV (tag-length-value) section record.

    Layout: 4-byte tag + 4-byte bool flag + 4-byte LE length + payload.
    """
    tag_bytes = tag.encode('ascii') if isinstance(tag, str) else tag
    assert len(tag_bytes) == 4, "Tag must be exactly 4 bytes"
    return tag_bytes + struct.pack('<II', flag, len(payload)) + payload


def _build_minimal_tnt(table_pairs=(), n_points=8, n_records=4):
    """Synthesize a minimal valid .tnt file with TMAG/PSEQ/DATA/TMG2.

    Parameters
    ----------
    table_pairs : iterable of (name, content) tuples
        Pascal-string pairs to embed in PSEQ.
    n_points, n_records : int
        Shape of the DATA section (1D complex64 buffer).
    """
    # --- TNTMAGIC: 8-byte magic header ---
    magic = b'TNT1.008'

    # --- TMAG: acquisition parameters (zeroed out, just satisfy size) ---
    tmag_payload = bytes(TNTTMAG.itemsize)
    # Patch actual_npts so it matches the DATA section we'll provide
    tmag_arr = np.frombuffer(tmag_payload, dtype=TNTTMAG).copy()
    tmag_arr[0]['actual_npts'][0] = n_points
    tmag_arr[0]['actual_npts'][1] = n_records
    tmag_arr[0]['actual_npts'][2] = 1
    tmag_arr[0]['actual_npts'][3] = 1
    tmag_arr[0]['sw'][0] = 20000.0
    tmag_arr[0]['ob_freq'][0] = 21.31
    tmag_payload = tmag_arr.tobytes()

    # --- DATA: complex64 buffer of zeros ---
    n_complex = n_points * n_records
    data_payload = np.zeros(n_complex, dtype='<c8').tobytes()

    # --- PSEQ: table pairs separated by some padding ---
    pseq_payload = b'\x00' * 32
    for name, content in table_pairs:
        pseq_payload += _pascal_string(name)
        pseq_payload += _pascal_string(content)
        pseq_payload += b'\x00' * 8

    # --- TMG2: processing parameters (zeroed) ---
    tmg2_payload = bytes(TNTTMG2.itemsize)

    sections = (
        _tlv_section('TMAG', tmag_payload) +
        _tlv_section('DATA', data_payload) +
        _tlv_section('PSEQ', pseq_payload) +
        _tlv_section('TMG2', tmg2_payload)
    )

    return magic + sections


def _write_tnt(table_pairs=(), n_points=8, n_records=4):
    """Write a synthetic .tnt to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix='.tnt')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(_build_minimal_tnt(table_pairs, n_points, n_records))
    except Exception:
        os.unlink(path)
        raise
    return path


# =============================================================================
# Tests for read()
# =============================================================================

def test_read_returns_dic_and_data():
    """read() returns a 2-tuple of (dict, ndarray) with expected shape."""
    path = _write_tnt(n_points=16, n_records=4)
    try:
        dic, data = ng.tecmag.read(path)
        assert isinstance(dic, dict)
        assert data.shape == (16, 4, 1, 1)
        assert data.dtype == np.dtype('<c8')
        assert dic['actual_npts'][0] == 16
        assert dic['actual_npts'][1] == 4
        assert_allclose(dic['sw'][0], 20000.0)
        assert_allclose(dic['ob_freq'][0], 21.31)
    finally:
        os.unlink(path)


def test_read_rejects_non_tnt_magic():
    """read() raises ValueError on a file without TNT magic header."""
    fd, path = tempfile.mkstemp(suffix='.tnt')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(b'NOT_TNT_' + b'\x00' * 128)
        with pytest.raises(ValueError, match="Invalid magic number"):
            ng.tecmag.read(path)
    finally:
        os.unlink(path)


def test_guess_udic_basic():
    """guess_udic() returns a per-axis dict matching the file's metadata."""
    path = _write_tnt(n_points=8, n_records=4)
    try:
        dic, data = ng.tecmag.read(path)
        udic = ng.tecmag.guess_udic(dic, data)
        # Axis 0 should reflect the values we wrote into TMAG
        assert udic[0]['size'] == 8
        assert udic[0]['sw'] == 20000.0
        assert_allclose(udic[0]['obs'], 21.31 * 1e6)
    finally:
        os.unlink(path)


# =============================================================================
# Tests for guess_tables()
# =============================================================================

def test_guess_tables_basic_numeric():
    """A simple multi-value numeric table is found and parsed."""
    path = _write_tnt([('rfamp', '1.0 4.222 7.444 10.667 13.889 17.111')])
    try:
        tables = guess_tables(path)
        assert 'rfamp' in tables
        assert isinstance(tables['rfamp'], np.ndarray)
        assert len(tables['rfamp']) == 6
        assert_allclose(
            tables['rfamp'],
            [1.0, 4.222, 7.444, 10.667, 13.889, 17.111],
        )
    finally:
        os.unlink(path)


def test_guess_tables_si_prefix_conversion():
    """SI-prefix tokens like '100u' are converted to base units."""
    path = _write_tnt([('tau', '5u 10u 20u 50u 100u')])
    try:
        tables = guess_tables(path)
        assert_allclose(tables['tau'], [5e-6, 10e-6, 20e-6, 50e-6, 100e-6])
    finally:
        os.unlink(path)


def test_guess_tables_skips_scalars():
    """Single-token (scalar) entries are excluded from the result."""
    path = _write_tnt([
        ('rfamp',    '1.0 5.0 10.0'),
        ('tp',       '50u'),
        ('deadtime', '20u'),
    ])
    try:
        tables = guess_tables(path)
        assert 'rfamp' in tables
        assert 'tp' not in tables
        assert 'deadtime' not in tables
    finally:
        os.unlink(path)


def test_guess_tables_name_filter():
    """The `names` argument restricts the result to a subset."""
    path = _write_tnt([
        ('rfamp', '1.0 2.0 3.0'),
        ('d1',    '1m 2m 3m'),
        ('d2',    '1s 2s 3s'),
    ])
    try:
        tables = guess_tables(path, names=['rfamp', 'd1'])
        assert set(tables.keys()) == {'rfamp', 'd1'}
    finally:
        os.unlink(path)


def test_guess_tables_convert_si_disabled():
    """When convert_si=False, values are returned as token lists."""
    path = _write_tnt([('d1', '1m 2m 3m')])
    try:
        tables = guess_tables(path, convert_si=False)
        assert tables['d1'] == ['1m', '2m', '3m']
    finally:
        os.unlink(path)


def test_guess_tables_no_pseq_section():
    """A file without PSEQ returns an empty dict (no exception)."""
    fd, path = tempfile.mkstemp(suffix='.tnt')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(b'TNT1.008' + b'\x00' * 256)
        assert guess_tables(path) == {}
    finally:
        os.unlink(path)


def test_guess_tables_rejects_invalid_length_prefix():
    """A name preceded by a wrong length prefix is rejected.

    Guards against false positives: short substrings like 'tp' appearing
    inside other binary data must not be parsed as table names.
    """
    fd, path = tempfile.mkstemp(suffix='.tnt')
    try:
        # Length prefix claims 99 but only 5 chars follow → invalid pair
        bogus = struct.pack('<i', 99) + b'rfamp' + b'extra junk'
        pseq = _tlv_section('PSEQ', b'\x00' * 32 + bogus + b'\x00' * 32)
        with os.fdopen(fd, 'wb') as f:
            f.write(b'TNT1.008' + b'\x00' * 64 + pseq + b'\x00' * 64)
        assert 'rfamp' not in guess_tables(path)
    finally:
        os.unlink(path)


def test_guess_tables_skips_non_ascii_names():
    """Names containing control bytes are not treated as tables."""
    fd, path = tempfile.mkstemp(suffix='.tnt')
    try:
        # 5-byte "name" containing control char 0x01
        bad = _pascal_string(b'\x01\x02bad') + _pascal_string('1.0 2.0 3.0')
        pseq = _tlv_section('PSEQ', b'\x00' * 32 + bad + b'\x00' * 32)
        with os.fdopen(fd, 'wb') as f:
            f.write(b'TNT1.008' + b'\x00' * 64 + pseq + b'\x00' * 64)
        assert len(guess_tables(path)) == 0
    finally:
        os.unlink(path)


def test_guess_tables_mixed_separators():
    """Content separated by mixed CRLF and spaces is parsed correctly."""
    path = _write_tnt([('rfamp', '1.0\r\n2.0\r\n3.0 4.0 5.0')])
    try:
        assert_allclose(
            guess_tables(path)['rfamp'],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
    finally:
        os.unlink(path)


def test_guess_tables_alongside_read():
    """read() and guess_tables() can both be applied to the same file."""
    path = _write_tnt(
        [('rfamp', '1.0 5.0 10.0 15.0')],
        n_points=8,
        n_records=4,
    )
    try:
        # Both APIs work on the same file
        dic, data = ng.tecmag.read(path)
        tables = guess_tables(path)
        assert data.shape == (8, 4, 1, 1)
        assert_allclose(tables['rfamp'], [1.0, 5.0, 10.0, 15.0])
    finally:
        os.unlink(path)


# =============================================================================
# Tests for internal helpers
# =============================================================================

def test_convert_si_plain_floats():
    """_convert_si() returns plain floats unchanged."""
    result = _convert_si(['1.0', '2.5', '-3.14', '1e-6'])
    assert_allclose(result, [1.0, 2.5, -3.14, 1e-6])


def test_convert_si_with_prefixes():
    """_convert_si() converts SI-prefix-suffixed tokens correctly."""
    result = _convert_si(['100u', '5m', '21.31M', '2k'])
    assert_allclose(result, [100e-6, 5e-3, 21.31e6, 2e3])


def test_convert_si_invalid_token_raises():
    """_convert_si() raises ValueError on garbage input."""
    with pytest.raises(ValueError):
        _convert_si(['not_a_number'])


def test_read_pascal_pair_validates_strictly():
    """_read_pascal_pair() rejects mismatched length prefixes."""
    # Length prefix says 5, name is 5 chars, content length 3, content 3 chars
    good = struct.pack('<i', 5) + b'rfamp' + struct.pack('<i', 3) + b'1 2'
    result = _read_pascal_pair(good, 0)
    assert result is not None
    name, content, next_pos = result
    assert name == 'rfamp'
    assert content == '1 2'

    # Length prefix says 99 — invalid (longer than _NAME_MAX_LEN)
    bad = struct.pack('<i', 99) + b'rfamp' + struct.pack('<i', 3) + b'1 2'
    assert _read_pascal_pair(bad, 0) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
