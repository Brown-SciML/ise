"""Climatology-based anomaly conversion for ISEFlow inputs.

The ISEFlow models are trained on forcing *anomalies* (departures from a
historical baseline), not raw absolute values.  This module provides
``AnomalyConverter`` — a lightweight class that looks up the pre-extracted
ISMIP6 climatological baselines (stored in ``data_files/``) and subtracts
them from user-supplied raw time-series arrays to produce the anomaly arrays
expected by ``ISEFlowAISInputs`` and ``ISEFlowGrISInputs``.

Supported ice sheets
--------------------
AIS:
    Atmospheric variables ``pr``, ``evspsbl``, ``smb``, ``ts``  →  anomalies.
    The baseline is the 1995-2014 spatial mean over each AIS sector from the
    ISMIP6 atmospheric forcing files (``AIS_atmos_climatologies.csv``).

GrIS:
    Atmospheric variables ``smb``, ``st``  →  anomalies.
    The baseline is the 1960-1989 MAR long-term mean over each GrIS drainage
    sector from the ISMIP6 Reference files (``GrIS_atmos_climatologies.csv``).

Variables that are **not** anomalies (passed through unchanged):
    AIS:  ``ocean_thermal_forcing``, ``ocean_salinity``, ``ocean_temperature``
    GrIS: ``ocean_thermal_forcing``, ``basin_runoff``

Usage
-----
With a bundled ISMIP6 climatology::

    converter = AnomalyConverter("AIS")
    anomalies = converter.compute_ais(
        aogcm="noresm1-m_rcp85",
        sector=10,
        pr=pr_array,
        evspsbl=evspsbl_array,
        smb=smb_array,
        ts=ts_array,
    )
    # anomalies = {"pr_anomaly": ..., "evspsbl_anomaly": ...,
    #              "smb_anomaly": ..., "ts_anomaly": ...}

With a user-supplied climatology (e.g. a new CMIP model not in ISMIP6)::

    converter = AnomalyConverter("AIS")
    anomalies = converter.compute_ais(
        sector=10,
        pr=pr_array,
        evspsbl=evspsbl_array,
        smb=smb_array,
        ts=ts_array,
        custom_climatology={
            "pr": 1.3e-5,
            "evspsbl": 4e-6,
            "smb": 9e-6,
            "ts": 253.7,
        },
    )
"""

import os
import warnings

import numpy as np
import pandas as pd

# Paths to bundled climatology CSVs
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data_files")
_AIS_ATMOS_CLIM_PATH = os.path.join(_DATA_DIR, "AIS_atmos_climatologies.csv")
_GRIS_ATMOS_CLIM_PATH = os.path.join(_DATA_DIR, "GrIS_atmos_climatologies.csv")

# ---------------------------------------------------------------------------
# Canonical AOGCM name normalisation
#
# Users may supply AOGCM names in various capitalisation/separator styles.
# We normalise everything to lowercase with hyphens separating model parts
# and an underscore before the scenario string — matching the canonical form
# used in ismip6_experiments_updated.csv and the climatology CSVs.
# ---------------------------------------------------------------------------
_AIS_ALIASES: dict[str, str] = {
    # common alternate spellings/capitalisations
    "noresm1-m_rcp8.5": "noresm1-m_rcp85",
    "noresm1-m_rcp2.6": "noresm1-m_rcp26",
    "miroc-esm-chem_rcp8.5": "miroc-esm-chem_rcp85",
    "miroc-esm-chem_rcp2.6": "miroc-esm-chem_rcp26",
    "ccsm4_rcp8.5": "ccsm4_rcp85",
    "ccsm4_rcp2.6": "ccsm4_rcp26",
    "hadgem2-es_rcp8.5": "hadgem2-es_rcp85",
    "csiro-mk3-6-0_rcp8.5": "csiro-mk3.6_rcp85",
    "csiro-mk3.6_rcp8.5": "csiro-mk3.6_rcp85",
    "ipsl-cm5a-mr_rcp8.5": "ipsl-cm5-mr_rcp85",
    "ipsl-cm5a-mr_rcp2.6": "ipsl-cm5-mr_rcp26",
    "ipsl-cm5-mr_rcp8.5": "ipsl-cm5-mr_rcp85",
    "ipsl-cm5-mr_rcp2.6": "ipsl-cm5-mr_rcp26",
    "cnrm-cm6-1_ssp585": "cnrm-cm6_ssp585",
    "cnrm-cm6-1_ssp126": "cnrm-cm6_ssp126",
    "cnrm-esm2-1_ssp585": "cnrm-esm2_ssp585",
    "ukesm1-0-ll_ssp585": "ukesm1-0-ll_ssp585",
    "cesm2_ssp585": "cesm2_ssp585",
}

_GRIS_ALIASES: dict[str, str] = {
    "access1-3_rcp85": "access1.3_rcp85",
    "access1.3_rcp8.5": "access1.3_rcp85",
    "csiro-mk3-6-0_rcp85": "csiro-mk3.6_rcp85",
    "csiro-mk3.6_rcp8.5": "csiro-mk3.6_rcp85",
    "hadgem2-es_rcp8.5": "hadgem2-es_rcp85",
    "ipsl-cm5-mr_rcp8.5": "ipsl-cm5-mr_rcp85",
    "ipsl-cm5a-mr_rcp85": "ipsl-cm5-mr_rcp85",
    "miroc5_rcp8.5": "miroc5_rcp85",
    "miroc5_rcp2.6": "miroc5_rcp26",
    "noresm1_rcp85": "noresm1-m_rcp85",
    "noresm1-m_rcp8.5": "noresm1-m_rcp85",
    "cnrm-cm6-1_ssp585": "cnrm-cm6_ssp585",
    "cnrm-cm6-1_ssp126": "cnrm-cm6_ssp126",
    "cnrm-esm2-1_ssp585": "cnrm-esm2_ssp585",
    "ukesm1-cm6_ssp585": "ukesm1-0-ll_ssp585",
    "ukesm1-0-ll_ssp585": "ukesm1-0-ll_ssp585",
    "cesm2_ssp585": "cesm2_ssp585",
}


def _normalise_aogcm(name: str, aliases: dict) -> str:
    """Lower-case the name and apply alias lookup if needed."""
    lowered = name.strip().lower()
    return aliases.get(lowered, lowered)


# ---------------------------------------------------------------------------
# AnomalyConverter
# ---------------------------------------------------------------------------


class AnomalyConverter:
    """Convert raw absolute forcing arrays to anomalies using ISMIP6 climatologies.

    Parameters
    ----------
    ice_sheet : str
        ``'AIS'`` or ``'GrIS'``.

    Attributes
    ----------
    ice_sheet : str
    climatology : pd.DataFrame
        The loaded climatology table for the selected ice sheet.
    """

    def __init__(self, ice_sheet: str) -> None:
        ice_sheet = ice_sheet.upper()
        if ice_sheet not in ("AIS", "GRIS"):
            raise ValueError("ice_sheet must be 'AIS' or 'GrIS'")
        # Normalise GrIS variant spellings
        if ice_sheet == "GRIS":
            ice_sheet = "GrIS"
        self.ice_sheet = ice_sheet
        self._clim: pd.DataFrame | None = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def climatology(self) -> pd.DataFrame:
        """Return the climatology DataFrame, loading it on first access."""
        if self._clim is None:
            self._clim = self._load_climatology()
        return self._clim

    def list_aogcms(self) -> list[str]:
        """Return the list of AOGCM names available in the bundled climatology."""
        return sorted(self.climatology["aogcm"].unique().tolist())

    def get_climatology(self, aogcm: str, sector: int) -> dict:
        """Return the climatological mean values for a given AOGCM and sector.

        Parameters
        ----------
        aogcm : str
            Canonical AOGCM name (see ``list_aogcms()``).  Common alternate
            spellings are normalised automatically.
        sector : int
            Sector / drainage basin number.

        Returns
        -------
        dict
            Variable name → scalar mean value for the baseline period.

        Raises
        ------
        KeyError
            If ``aogcm`` is not found in the bundled climatology.
        """
        canonical = _normalise_aogcm(
            aogcm, _AIS_ALIASES if self.ice_sheet == "AIS" else _GRIS_ALIASES
        )
        row = self.climatology[
            (self.climatology["aogcm"] == canonical) & (self.climatology["sector"] == int(sector))
        ]
        if row.empty:
            available = self.list_aogcms()
            raise KeyError(
                f"No climatology found for aogcm='{aogcm}' (normalised: '{canonical}'), "
                f"sector={sector}.  Available AOGCMs: {available}"
            )
        return row.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # AIS
    # ------------------------------------------------------------------

    def compute_ais(
        self,
        sector: int,
        pr: np.ndarray,
        evspsbl: np.ndarray,
        smb: np.ndarray,
        ts: np.ndarray,
        aogcm: str | None = None,
        custom_climatology: dict | None = None,
        mrro: np.ndarray | None = None,
    ) -> dict:
        """Compute AIS atmospheric anomalies from raw annual time-series arrays.

        Exactly one of ``aogcm`` (use bundled ISMIP6 climatology) or
        ``custom_climatology`` (user-supplied baseline scalars) must be provided.

        Parameters
        ----------
        sector : int
            AIS drainage sector number (1-18).
        pr : np.ndarray
            Raw precipitation time series (86 values, kg m⁻² s⁻¹).
        evspsbl : np.ndarray
            Raw evaporation/sublimation time series (86 values, kg m⁻² s⁻¹).
        smb : np.ndarray
            Raw surface mass balance time series (86 values, kg m⁻² s⁻¹).
        ts : np.ndarray
            Raw surface temperature time series (86 values, K).
        aogcm : str, optional
            AOGCM name to look up in the bundled climatology.  Common alternate
            spellings are normalised automatically (e.g. ``'NorESM1-M_rcp8.5'``
            → ``'noresm1-m_rcp85'``).
        custom_climatology : dict, optional
            User-supplied baseline means.  Must contain keys ``'pr'``,
            ``'evspsbl'``, ``'smb'``, ``'ts'`` (and optionally ``'mrro'`` if
            ``mrro`` is provided).  Use this for CMIP models not in ISMIP6.
        mrro : np.ndarray, optional
            Raw runoff time series (86 values).  Required only for v1.0.0.
            Ignored if ``custom_climatology`` is given and does not contain
            ``'mrro'``.

        Returns
        -------
        dict
            ``{'pr_anomaly', 'evspsbl_anomaly', 'smb_anomaly', 'ts_anomaly'}``
            as 86-element numpy arrays.  ``'mrro_anomaly'`` is included when
            ``mrro`` is provided and a baseline is available.

        Raises
        ------
        ValueError
            If neither or both of ``aogcm`` / ``custom_climatology`` are given,
            or if array lengths are not 86.
        """
        self._validate_ais_args(aogcm, custom_climatology)
        arrays = {"pr": pr, "evspsbl": evspsbl, "smb": smb, "ts": ts}
        if mrro is not None:
            arrays["mrro"] = mrro
        self._check_lengths(arrays)

        clim = self._resolve_clim_ais(aogcm, sector, custom_climatology)

        result = {
            "pr_anomaly": np.asarray(pr) - clim["pr"],
            "evspsbl_anomaly": np.asarray(evspsbl) - clim["evspsbl"],
            "smb_anomaly": np.asarray(smb) - clim["smb"],
            "ts_anomaly": np.asarray(ts) - clim["ts"],
        }

        if mrro is not None:
            if "mrro" in clim:
                result["mrro_anomaly"] = np.asarray(mrro) - clim["mrro"]
            else:
                warnings.warn(
                    "mrro was provided but no mrro baseline is available for this "
                    "AOGCM; mrro_anomaly will not be included in the output."
                )

        return result

    # ------------------------------------------------------------------
    # GrIS
    # ------------------------------------------------------------------

    def compute_gris(
        self,
        sector: int,
        smb: np.ndarray,
        st: np.ndarray,
        aogcm: str | None = None,
        custom_climatology: dict | None = None,
    ) -> dict:
        """Compute GrIS atmospheric anomalies from raw annual time-series arrays.

        The GrIS baseline period is 1960-1989 (MAR long-term mean), matching
        the ISMIP6 Reference files.

        Parameters
        ----------
        sector : int
            GrIS drainage basin number (1-6).
        smb : np.ndarray
            Raw surface mass balance time series (86 values, mm w.e. yr⁻¹ or
            equivalent units matching the MAR reference).
        st : np.ndarray
            Raw surface temperature time series (86 values, K or °C, consistent
            with the MAR reference).
        aogcm : str, optional
            AOGCM name to look up in the bundled climatology.
        custom_climatology : dict, optional
            Must contain keys ``'smb'`` and ``'st'``.

        Returns
        -------
        dict
            ``{'aSMB': ..., 'aST': ...}`` as 86-element numpy arrays.
            Variable names match ``ISEFlowGrISInputs`` field names.

        Raises
        ------
        ValueError
            If neither or both of ``aogcm`` / ``custom_climatology`` are given,
            or if array lengths are not 86.
        """
        self._validate_gris_args(aogcm, custom_climatology)
        self._check_lengths({"smb": smb, "st": st})

        clim = self._resolve_clim_gris(aogcm, sector, custom_climatology)

        return {
            "aSMB": np.asarray(smb) - clim["smb"],
            "aST": np.asarray(st) - clim["st"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_climatology(self) -> pd.DataFrame:
        if self.ice_sheet == "AIS":
            path = _AIS_ATMOS_CLIM_PATH
        else:
            path = _GRIS_ATMOS_CLIM_PATH

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Bundled climatology file not found: {path}\n"
                "Run extract_climatologies.py from the repo root to generate it."
            )
        return pd.read_csv(path)

    def _resolve_clim_ais(
        self,
        aogcm: str | None,
        sector: int,
        custom: dict | None,
    ) -> dict:
        """Return a dict with keys 'pr', 'evspsbl', 'smb', 'ts' (and 'mrro' if present)."""
        if custom is not None:
            required = {"pr", "evspsbl", "smb", "ts"}
            missing = required - set(custom.keys())
            if missing:
                raise ValueError(
                    f"custom_climatology is missing required keys: {missing}. "
                    f"Must include: {required}"
                )
            return {k: float(v) for k, v in custom.items()}

        row = self.get_climatology(aogcm, sector)
        clim = {
            "pr": row["pr_clim"],
            "evspsbl": row["evspsbl_clim"],
            "smb": row["smb_clim"],
            "ts": row["ts_clim"],
        }
        if "mrro_clim" in row and not pd.isna(row["mrro_clim"]):
            clim["mrro"] = row["mrro_clim"]
        return clim

    def _resolve_clim_gris(
        self,
        aogcm: str | None,
        sector: int,
        custom: dict | None,
    ) -> dict:
        """Return a dict with keys 'smb' and 'st'."""
        if custom is not None:
            required = {"smb", "st"}
            missing = required - set(custom.keys())
            if missing:
                raise ValueError(
                    f"custom_climatology is missing required keys: {missing}. "
                    f"Must include: {required}"
                )
            return {k: float(v) for k, v in custom.items()}

        row = self.get_climatology(aogcm, sector)
        return {"smb": row["smb_clim"], "st": row["st_clim"]}

    @staticmethod
    def _validate_ais_args(aogcm, custom_climatology):
        if aogcm is None and custom_climatology is None:
            raise ValueError(
                "Provide either 'aogcm' (to use the bundled ISMIP6 climatology) "
                "or 'custom_climatology' (a dict of baseline means for a new CMIP model)."
            )
        if aogcm is not None and custom_climatology is not None:
            raise ValueError("Provide only one of 'aogcm' or 'custom_climatology', not both.")

    @staticmethod
    def _validate_gris_args(aogcm, custom_climatology):
        if aogcm is None and custom_climatology is None:
            raise ValueError(
                "Provide either 'aogcm' (to use the bundled ISMIP6 climatology) "
                "or 'custom_climatology' (a dict of baseline means for a new CMIP model)."
            )
        if aogcm is not None and custom_climatology is not None:
            raise ValueError("Provide only one of 'aogcm' or 'custom_climatology', not both.")

    @staticmethod
    def _check_lengths(arrays: dict):
        for name, arr in arrays.items():
            arr = np.asarray(arr)
            if arr.ndim != 1 or len(arr) != 86:
                raise ValueError(
                    f"Array '{name}' must be a 1-D array of length 86 (one value "
                    f"per year 2015-2100); got shape {arr.shape}."
                )
