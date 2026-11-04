# AgroScope Data Card

The data behind AgroScope's model and maps, as of 2026-11-04.

## Sources and licences

| Data | Source | Dates | Licence and conditions |
| :--- | :--- | :--- | :--- |
| Satellite imagery | Sentinel-2 Level-2A, through [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) | 2023-10-06 to 2024-03-29 | [Copernicus Sentinel data terms](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice): free and open, with credit |
| Weather | [Open-Meteo](https://open-meteo.com/) historical archive | 2023-01-01 to 2024-05-31 | CC BY 4.0, with credit. The free API is for non-commercial use, under 10,000 calls a day |
| Canal boundary | `data/raw/study_area/command_area.geojson`, in this repository, marked by hand by the project author | Static | **Approximate, and the exact method isn't recorded** (see known problems) |

**Map display only:** Carto Voyager or OpenStreetMap for street maps, and EOxCloudless 2025 for satellite. Their credits show on the map, and they aren't model inputs.

## Coverage

| Item | Value |
| :--- | :--- |
| Area | Nira Left Bank Canal, Pune district, Maharashtra |
| Grid | 22,580 cells of 250 m, or 6.25 ha each |
| Sentinel-2 tiles | 43QDA and 43QDV, with at most 15% cloud |
| Scene list | 61 scene records across 27 dates |
| Weather points | 24, at 0.1 degree spacing, each with 517 days. Each cell uses its nearest point |
| Training table | 566,319 rows and 38 columns, with no missing values or duplicates |
| Vegetated rows | 539,852, using NDVI of at least 0.15. The rest are left out of training |

The training and later-date periods split on 2024-02-01, with 308,726 and 231,126 vegetated rows.

## Inputs

There are 25 model inputs:

- **Satellite (12):** mean and spread of NDVI, NDWI and EVI, and mean reflectance for bands B02, B03, B04, B08, B11 and B12. Cloud-affected pixels are removed, and ESA's +1000 reflectance offset is corrected.
- **Weather (13):** temperature, humidity, wind, solar radiation, and rolling rainfall, temperature and evapotranspiration totals.

## Labels

**Labels are calculated, not observed.** No field measurements exist in this project.

```text
demand  = et0_8day_sum x 0.95
rain    = effective 8-day rainfall
buffer  = clip((ndwi_mean - 0.02) / 0.16, 0, 1) x demand x 0.75
deficit = max(0, demand - rain - buffer)

LOW: deficit < 12 mm    MODERATE: 12-28 mm    HIGH: > 28 mm
```

| Class | All rows | Vegetated rows before 2024-02-01 | Vegetated rows from 2024-02-01 |
| :--- | :--- | :--- | :--- |
| LOW | 99,067 | 94,424 | 3,922 |
| MODERATE | 155,787 | 124,241 | 30,632 |
| HIGH | 311,465 | 90,061 | 196,572 |

The balance shifts sharply toward HIGH later in the season, as the crop matures and rain stops. Training corrects for class imbalance with balanced weights.

## Known problems

| Problem | Effect | Reference |
| :--- | :--- | :--- |
| Labels use three of the model's own inputs | High scores don't show real-world skill | B-07 |
| One season only | Monsoon and other years are untested | B-08, B-12 |
| The vegetated area looks too large | 120,625 ha is reported, more than the 66,250 ha culturable and 84,500 ha gross in the boundary file. The hand-marked boundary may be too broad, and NDVI of 0.15 can include non-crop land | B-35 |
| The boundary is hand-marked | The project author marked this 28-point outline by hand from the canal's dimensions, and the exact method isn't recorded. The file's own label, "Validated Official Geometry", overstates this, because no authority has validated it. Treat every area figure as approximate | B-35, B-47 |
| One crop coefficient for everything | Crop type and soil don't affect labels | B-19 |
| Thresholds not reviewed | They were set before the reflectance fix, and no expert has reviewed them | B-34 |
| Uneven dates | Some dates cover part of the area, down to 6,537 vegetated cells on 2024-03-29. The published date is the latest covering at least 90% of cells | F-11 |
| Reprocessed scenes | Some dates have more than one copy of a scene, and only the latest should be used | F-19 |

## Personal data

The data has no names, farm boundaries, ownership records or demographic details. Results are for 250 m cells, each covering several farms.

## Storage and versioning

| Item | Where | Versioned by |
| :--- | :--- | :--- |
| Raw scenes, about 9.8 GB | Local, and Google Drive through DVC | `data/raw/satellite.dvc` |
| Weather, features, maps and models | Local, and Google Drive through DVC | `dvc.lock` |
| Runs and registered models | MLflow, in a Docker volume | MLflow |
| Code, settings and validation results | GitHub | Git |

Google Drive's free tier limits storage, so new raw scenes aren't downloaded (D-30).

## Quality checks

`src/validation/validate_data.py` runs in the pipeline and stops it on any failure. It checks required columns, value ranges, missing values, date coverage, duplicates, label values, and that all three classes appear in the later dates. Latest results are in `reports/validation_raw.json` and `reports/validation_features.json`, and all checks pass.
