# 04_Comparison: reconstruction vs CAD ground truth

`compare_to_cad.py` measures how far a reconstructed cloud from `03_Reconstruction` is from the
object's CAD model. It uses the saved board-frame clouds as they are: mm, z = height above the board.

```
src/venv/bin/python src/04_Comparison/compare_to_cad.py <cloud.ply> [<cloud2.ply> ...] \
    [--reference data/GroundTruth/roughness_blocks_STL/PYRAMID_1.stl]
```

Use the `*_clean.ply` clouds (cropped, outliers removed), not the meshes.

## What it does
1. **Reference.** It loads the STL and drops the faces pointing down, which sit on the board and can't be seen.
2. **Object points.** It keeps points more than 3 mm above the board, in the connected cluster that reaches highest.
3. **Two fits of the CAD model onto the scan.** Both use robust point-to-plane ICP against the exact CAD triangles, started from 24 rotations about z:
   - **on_board** (x, y, yaw): the object sits flat on the board, and the board frame already fixes its height and tilt. Height and tilt errors in the reconstruction therefore show up in the numbers. **Use this one for the absolute accuracy.**
   - **shape** (6 DOF): the best rigid fit, so it measures the accuracy of the shape alone.
4. **Scoring,** over the CAD footprint plus 2 mm:
   - the signed distance of each point to the surface (positive = outside), with its bias, RMS, median and 95th percentile;
   - per face, how much of it the scan covers, and the angle between a plane fitted through its points and the designed face.

   If a fit covers less than 15% of the surface, it is flagged **UNRELIABLE**.

## Outputs (`results/comparison/<cloud>/`)
| File | Contents |
|---|---|
| `metrics.json` | Every number for both fits, the transforms, and the settings used |
| `<cloud>_comparison.png` | Top-down error map, histogram, and profiles through the apex |
| `<cloud>_distances.ply` | Object points coloured by error, with a `signed_distance` scalar field |
| `<cloud>_reference_aligned.ply` | The CAD mesh placed by the on_board fit, in the board frame |

`results/comparison/summary.csv` gets one row per cloud and fit on every run. It is the table to compare scans with.

## Looking at it in CloudCompare
CloudCompare 2.13.2 is installed as a user Flatpak. To open a comparison:
```
D=results/comparison/mjpg_pyr_lights_2_cloud_clean
flatpak run org.cloudcompare.CloudCompare $D/*_distances.ply $D/*_reference_aligned.ply
```
- **Colour by distance.** Select the cloud, and in the *Properties* panel set the active scalar field to `signed_distance`. Set *Color Scale* to a blue-white-red one and make the range symmetric (for example ±5 mm). Menu names may differ slightly between CloudCompare versions.
- **Cross-sections.** The *Cross Section* tool cuts slices through the apex.
- **Check in CloudCompare's own code.** Run *Cloud/Mesh Dist* (under *Distances*) with the aligned reference as the mesh. It should reproduce `signed_distance`, since both are already in the same frame.
- **Measuring.** *Point picking* measures a point, a distance or an angle by hand.

## Results so far (24 Sep 2026, PYRAMID_1, on_board fit)
| Cloud | Coverage | Bias | RMS | Median abs. error | Face slope errors |
|---|---|---|---|---|---|
| SGBM, lights (`mjpg_pyr_lights_2`) | 42% | +0.88 mm | 2.75 mm | 1.91 mm | 0.1 to 2.1° |
| SGBM, no lights (`mjpg_pyr2`) | 36% | +2.28 mm | 4.67 mm | 3.35 mm | 2.4 to 14.6° |
| COLMAP quick run (`mjpg_pyr2`) | 8% | UNRELIABLE | | | |
