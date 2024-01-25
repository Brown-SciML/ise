r"""

# Grids

This module contains all necessary functions for end-to-end creation of emulators for both the Antarctic 
ice sheet and the Greenland ice sheet.The data processing is done on close to native grids on the 
kilometer scale rather than sector averages as seen in [A variational LSTM emulator of sea level 
contribution from the Antarctic ice sheet](https://doi.org/10.22541/essoar.168874913.31606296/v1).
This module also contains all processing functions all training functions and necessary tools for 
analyzing the emulator performance.
"""

from ise.grids.utils import get_all_filepaths