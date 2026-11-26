# Plugin workflow and outputs

The plugin works with a selected ATLAS section set and supports both raw
acquisition folders and a watcher output copy containing a master
`stitched_sections.zarr` store.

## Raw acquisition folder

Select the folder containing the `S_...` section folders. The plugin discovers
valid sections directly inside it. Each section needs a completed `.ve-mif`
file and the TIFF tiles referenced by that metadata. Stitch selected sections
or all listed sections; the plugin stores stitched images and section metadata
in `stitched_sections.zarr` beside the section folders. Section stitching is
performed concurrently according to the thread count in **Options...**.

## Watcher output copy

If a selected folder has no raw `S_` folders but contains
`stitched_sections.zarr`, the plugin opens it in watcher mode. Completed
sections can be displayed and aligned; click **Refresh** to discover sections
completed since the last refresh. Stitching controls are unavailable in this
mode. The section count reflects the store at the most recent refresh.

If watcher alignment results are present, the plugin offers to import the
alignment table. **Render and Preview** uses the saved shifts. Otherwise, use
**Initial Z Alignment** to calculate shifts from the completed sections and
make a preview in the selected folder.

## Review and correct alignment

Set the axial pixel size and unit before alignment. Initial alignment uses
numeric ATLAS section order by default, or the order in `correct_order.txt`.
The file can select a subset, but every listed section must be complete in the
store. Inspect the downsampled preview by scrolling through its Z planes.

For manual correction, click **Enter Correction** and choose a mode:

- **Trace: one point per plane**: place one point on the same recognizable
  feature in each consecutive plane. Gaps start separate traces. Click
  **Apply Correction** to update the alignment and rebuild the preview. Add
  new points on the updated preview for each subsequent correction.
- **Explicit point pairs**: place corresponding landmarks in the Reference
  Points and Adjustable Points layers, in matching order, then apply the
  correction.

Inspect the corrected preview, repeating correction if needed. The original
automatic result is kept as
`alignment_results/z_alignment_results_before_manual_correction.pkl`; the
latest corrected result is stored in `z_alignment_results.pkl` and used by
final alignment.

## Final alignment and exports

Click **Final Z Alignment** after reviewing the preview to write a
full-resolution aligned Zarr and an OME-Zarr pyramid. The OME-Zarr can be
opened with **View**. **Export to CZI** writes a CZI file, and **Upload to
Webknossos...** creates and uploads a dataset. WebKnossos fine alignment can
also be downloaded as a tiled CZI using **Download from webknossos...**.

The aligned outputs and their exports use per-section intensity stretching
based on the lower and upper percentiles in **Options...**. This changes saved
pixel intensities. Use the original `stitched_sections.zarr` when the
unmodified stitched intensities are required. Final alignment and exports can
need substantial disk space and processing time. See
[backend outputs and alignment](../backend/outputs.md) for the corresponding
storage and alignment concepts.
