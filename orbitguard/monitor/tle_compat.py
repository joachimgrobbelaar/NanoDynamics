"""TLE compatibility: Alpha-5 catalog fields and line checksums.

Prefer GP/OMM JSON for new data. This module only reads the legacy
fixed-width fields when a TLE is supplied. It does not turn mean elements
into an osculating Cartesian state.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Strict Space-Track alphabet: digits, then A-Z with I and O removed.
# A=10 ... H=17, J=18 ... N=22, P=23 ... Z=33. Maximum catalog number is
# Z9999 = 339999. python-sgp4's decoder also accepts I, O, and lowercase;
# this decoder does not.
_ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def encode_alpha5(catalog_number: int) -> str:
    if not isinstance(catalog_number, int) or isinstance(catalog_number, bool):
        raise TypeError("Catalog number must be an int")
    if catalog_number < 0 or catalog_number > 339999:
        raise ValueError("Alpha-5 encodes catalog numbers from 0 through 339999")
    if catalog_number < 100000:
        return f"{catalog_number:05d}"
    prefix, rest = divmod(catalog_number, 10000)
    return _ALPHABET[prefix] + f"{rest:04d}"


def decode_alpha5(field: str) -> int:
    if not isinstance(field, str):
        raise TypeError("Alpha-5 field must be a string")
    text = field.strip()
    if len(text) != 5:
        raise ValueError("Alpha-5 catalog field must be exactly 5 characters")
    if text != text.upper() or any(ch.islower() for ch in text):
        raise ValueError("Alpha-5 letters must be uppercase")
    if any(ch in "IO" for ch in text):
        raise ValueError("Alpha-5 omits I and O")
    if text[0] not in _ALPHABET or not text[1:].isdigit():
        raise ValueError(f"Not a strict Alpha-5 catalog field: {text!r}")
    prefix = _ALPHABET.index(text[0])
    return prefix * 10000 + int(text[1:])


def checksum(line: str) -> int:
    """TLE checksum: digits plus one for each leading minus, modulo 10.

    The checksum character itself (column 69) is not included.
    """
    if len(line) < 68:
        raise ValueError("TLE line is shorter than 68 characters")
    total = 0
    for char in line[:68]:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1
    return total % 10


def checksum_matches(line: str) -> bool:
    if len(line) < 69 or not line[68].isdigit():
        return False
    try:
        expected = checksum(line)
    except ValueError:
        return False
    return int(line[68]) == expected


def catalog_number_from_line1(line1: str) -> int:
    if len(line1) < 7 or line1[0] != "1":
        raise ValueError("Line 1 must start with '1'")
    return decode_alpha5(line1[2:7])


def epoch_from_line1(line1: str) -> datetime:
    """UTC datetime of the TLE epoch. Two-digit years use the 1957 pivot.

    Years 00-56 map to 2000-2056. Year 57 and above map to 1957-1999.
    This is the historical TLE convention and is not defined through 2057.
    """
    if len(line1) < 32:
        raise ValueError("Line 1 has no epoch field")
    year_2 = int(line1[18:20])
    day = float(line1[20:32])
    year = 2000 + year_2 if year_2 < 57 else 1900 + year_2
    if not 1.0 <= day < 367.0:
        raise ValueError("TLE epoch day is outside 1..366")
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    return start + timedelta(days=day - 1.0)
