"""Tests for ise/data/process.py — data pipeline classes.

These tests cover only constructor-level behaviour that does not require
NetCDF files on disk.  File-I/O-heavy methods (process(), merge_dataset())
are intentionally excluded — they depend on the ISMIP6 directory tree.
"""

from ise.data.process import ProjectionProcessor


class TestProjectionProcessorConstructor:
    def test_ais_ice_sheet_stored(self):
        pp = ProjectionProcessor(
            ice_sheet="AIS",
            forcings_directory="/tmp",
            projections_directory="/tmp",
        )
        assert pp.ice_sheet == "AIS"

    def test_gis_normalisation(self):
        pp = ProjectionProcessor(
            ice_sheet="GrIS",
            forcings_directory="/tmp",
            projections_directory="/tmp",
        )
        assert pp.ice_sheet == "GIS"

    def test_ais_resolution_8km(self):
        pp = ProjectionProcessor("AIS", "/tmp", "/tmp")
        assert pp.resolution == 8

    def test_gis_resolution_5km(self):
        pp = ProjectionProcessor("GrIS", "/tmp", "/tmp")
        assert pp.resolution == 5

    def test_directories_stored(self):
        pp = ProjectionProcessor("AIS", "/forcing/path", "/proj/path")
        assert pp.forcings_directory == "/forcing/path"
        assert pp.projections_directory == "/proj/path"

    def test_optional_paths_default_none(self):
        pp = ProjectionProcessor("AIS", "/tmp", "/tmp")
        assert pp.scalefac_path is None
        assert pp.densities_path is None
