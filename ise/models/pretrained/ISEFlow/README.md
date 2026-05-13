# ISEFlow Pretrained Weights

The canonical source for ISEFlow weights is the HuggingFace Hub repository:

**https://huggingface.co/pvankatwyk/ISEFlow**

The weights checked in under this directory are a convenience copy used as an
offline fallback (e.g. for air-gapped HPC environments) and are loaded
automatically by `ise.models.pretrained.get_model_dir()` only when the
HuggingFace download is unavailable.

If the files here ever diverge from what is on the Hub, treat the HuggingFace
version as authoritative.

These weights are not shipped with the `ise-py` PyPI wheel — pip users always
download them from HuggingFace on first use.
