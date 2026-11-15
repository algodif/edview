# EDView
#### (**E**)nergy-(**D**)ispersive X-ray diffraction data (**View**)er

GUI software for viewing data of the Energy-Dispersive X-ray Diffraction
Tomography (EDDT) technique, processed by [Algodif](https://github.com/algodif/algodif).
Built on Algodif, PySide6 and pyqtgraph.

Tabs
====

- **Data browser**: folder tree, file list and preview of images, text/JSON
  files and HDF datasets. Send the selected file or folder to any other tab
  with *Open in* (or double-click).
- **Data correction**:
  - *Correction*: correct raw scans (a scan folder with a `tifs` sub-folder,
    a folder of tif files, files, an HDF file, or a range of `scan_xxxxx`
    folders) with q or energy calibration coefficients, with optional
    background removal. Output per scan: tif (`tifs_with_bgr`,
    `tifs_without_bgr`), HDF, or xy/csv (raw or corrected), plus
    `metadata.json` (with the output step), the coefficients and the map of
    the raw pixels.
  - *Conversion*: tif folder to 3D HDF, 3D HDF to tif folder, tif folder of
    a 2D scan to 4D HDF, tif folder or HDF to xy/csv files.
- **EDDT viewer**: image of a 1D scan, q-slice across the scan, and the
  diffraction pattern of the selected row.
- **Multiple-lines viewer**: as above, with the patterns of all images at
  the selected row.
- **Multiple-scans viewer**: several 1D scans; patterns of all scans at the
  same image and row.
- **2D-scan viewer**: image at a scan position (tif files named
  `..._<row>_<col>.tif`, loaded on demand, or a 4D HDF dataset), its
  pattern, and the q-slice along one scan axis.

The viewers read the x-step (q-step or energy step) and the motor positions
from the JSON metadata next to the data (e.g. written by the Data correction
tab or `process_data.py` of Algodif), or from the HDF file. Clicking on the
image selects a row and an x-value; clicking on the q-slice selects a row
and a scan position.

Installation
============

```commandline
pip install .
```
Launch with
```commandline
edview
```
or, without installing, from the repository root:
```commandline
python -m edview.main
```
