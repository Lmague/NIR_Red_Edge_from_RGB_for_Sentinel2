"""Sentinel-2 forest patch export from Google Earth Engine to GCS TFRecords.

Generates 128x128 patches over geographically separated train/test regions
(test regions are unseen during training to evaluate cross-domain generalization).
"""
import ee

BUCKET = 'YOUR_BUCKET'
FOLDER = 'dataset_training_mst'
NUM_POINTS = 100000
PATCH_SIZE = 128
BATCH_SIZE = 500

# Set to True to validate the pipeline on a few points before launching the full export.
TEST_MODE = False

POINTS_PER_REGION_TRAIN = 10000
POINTS_PER_REGION_TEST = 3000
SCALE = 1500  # meters between sample points


def get_spectral_pair(feature):
    """Return a Feature with the 7 Sentinel-2 bands for the patch around `feature`."""
    geom = feature.geometry()

    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geom)
        .filterDate('2021-01-01', '2024-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    )

    img = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()

    # ee.Algorithms.If is required because map() cannot return None.
    return ee.Algorithms.If(
        collection.size().gt(0),
        extract_patches(img, geom),
        ee.Feature(None, {'valid': 0}),
    )


def extract_patches(img, geom):
    """Extract a patch (~129x129, cropped to 128 by the loader) of B2-B8 around `geom`."""
    bands = img.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'])
    bands = bands.unmask(0)
    bands = bands.reproject(crs='EPSG:3857', scale=10)

    region = geom.buffer(640, 1).bounds(1, 'EPSG:3857')
    arrays = bands.sampleRectangle(region=region, defaultValue=0)

    def get_flat_array(band_name):
        # GEE TFRecord export supports ee.Array but not raw Python lists.
        return ee.Array(arrays.get(band_name)).reshape([-1])

    return ee.Feature(None, {
        'valid': 1,
        'B2': get_flat_array('B2'),
        'B3': get_flat_array('B3'),
        'B4': get_flat_array('B4'),
        'B5': get_flat_array('B5'),
        'B6': get_flat_array('B6'),
        'B7': get_flat_array('B7'),
        'B8': get_flat_array('B8'),
    })


def run_test_mode(all_points_train, all_points_test,
                  region_names_train, region_names_test, points_train):
    """Sanity-check the pipeline on one point and export 10 tiles."""
    print("\n[1/6] Train points per region")
    total_train = 0
    for i, name in enumerate(region_names_train):
        count = all_points_train[i].size().getInfo()
        total_train += count
        print(f"  {name:<25} {count:>5}")
    print(f"  TOTAL TRAIN: {total_train}")

    print("\n[2/6] Test points per region")
    total_test = 0
    for i, name in enumerate(region_names_test):
        count = all_points_test[i].size().getInfo()
        total_test += count
        print(f"  {name:<25} {count:>5}")
    print(f"  TOTAL TEST: {total_test}")

    if points_train.size().getInfo() == 0:
        raise RuntimeError("No training points generated.")

    test_point = points_train.first()
    geom = test_point.geometry()
    print(f"\n[3/6] Test point coordinates: {geom.coordinates().getInfo()}")

    print("\n[4/6] Looking for Sentinel-2 imagery")
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geom)
        .filterDate('2021-01-01', '2024-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    )
    col_size = collection.size().getInfo()
    print(f"  {col_size} images found")

    if col_size == 0:
        # Fall back to a wider date range and looser cloud filter.
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geom)
            .filterDate('2019-01-01', '2024-12-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        )
        col_size = collection.size().getInfo()
        print(f"  with relaxed filters: {col_size} images")

    print("\n[5/6] Patch extraction test")
    img = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
    bands = img.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']).unmask(0)
    bands = bands.reproject(crs='EPSG:3857', scale=10)
    region = geom.buffer(640, 1).bounds(1, 'EPSG:3857')
    arrays = bands.sampleRectangle(region=region, defaultValue=0)
    b2_array = ee.Array(arrays.get('B2')).getInfo()

    if not b2_array:
        raise RuntimeError("No data extracted.")

    rows = len(b2_array)
    cols = len(b2_array[0]) if rows > 0 else 0
    print(f"\n[6/6] Result: {rows} x {cols} = {rows * cols} pixels (target >= {PATCH_SIZE})")

    if rows < PATCH_SIZE or cols < PATCH_SIZE:
        raise RuntimeError(f"Patch too small ({rows}x{cols}). Increase the buffer.")

    print("\nLaunching test export of 10 tiles")
    test_batch = ee.FeatureCollection(points_train.toList(10))
    test_features = test_batch.map(get_spectral_pair).filter(ee.Filter.eq('valid', 1))
    task = ee.batch.Export.table.toCloudStorage(
        collection=test_features,
        description='TEST_10_tiles',
        bucket=BUCKET,
        fileNamePrefix=f'{FOLDER}/test_export',
        fileFormat='TFRecord',
    )
    task.start()
    print("Task started. Validate the output with check_tfrecord.py before disabling TEST_MODE.")


def export_dataset(points_collection, folder_suffix, num_points_max):
    """Export `points_collection` to GCS TFRecords, batched by BATCH_SIZE."""
    points_list = points_collection.toList(num_points_max)
    actual_count = points_collection.size().getInfo()

    print(f"\nExport {folder_suffix.upper()}: {actual_count} points")

    tasks_launched = 0
    for i in range(0, actual_count, BATCH_SIZE):
        batch = ee.FeatureCollection(points_list.slice(i, min(i + BATCH_SIZE, actual_count)))
        features_export = batch.map(get_spectral_pair).filter(ee.Filter.eq('valid', 1))

        task = ee.batch.Export.table.toCloudStorage(
            collection=features_export,
            description=f'{folder_suffix}_{i}_{i + BATCH_SIZE}',
            bucket=BUCKET,
            fileNamePrefix=f'{FOLDER}/{folder_suffix}/data_{i}',
            fileFormat='TFRecord',
        )
        task.start()
        tasks_launched += 1
        print(f"  task {tasks_launched} sent ({i} to {min(i + BATCH_SIZE, actual_count)})")

    return tasks_launched


def main():
    ee.Authenticate()
    ee.Initialize(project='YOUR_GCP_PROJECT_ID')

    landcover = ee.Image("ESA/WorldCover/v100/2020").select('Map')
    tree_mask = landcover.eq(10).Or(landcover.eq(20))
    forest_only = landcover.updateMask(tree_mask)

    regions_train = [
        ("Amazonia", ee.Geometry.Rectangle([-80, -15, -45, 5])),
        ("Congo", ee.Geometry.Rectangle([8, -10, 32, 8])),
        ("SE Asia", ee.Geometry.Rectangle([95, -10, 150, 25])),
        ("Central America", ee.Geometry.Rectangle([-100, 10, -75, 25])),
        ("India", ee.Geometry.Rectangle([70, 8, 95, 30])),
        ("Western Europe", ee.Geometry.Rectangle([-10, 40, 25, 60])),
        ("North America", ee.Geometry.Rectangle([-130, 35, -60, 55])),
        ("China/Japan", ee.Geometry.Rectangle([105, 22, 145, 45])),
        ("South America temperate", ee.Geometry.Rectangle([-75, -45, -50, -20])),
        ("Siberia", ee.Geometry.Rectangle([60, 50, 145, 70])),
        ("Canada", ee.Geometry.Rectangle([-140, 50, -60, 68])),
        ("Eastern Australia", ee.Geometry.Rectangle([140, -40, 155, -15])),
    ]

    # Held-out regions for cross-domain evaluation.
    regions_test = [
        ("New Zealand", ee.Geometry.Rectangle([166, -47, 178, -34])),
        ("Scandinavia", ee.Geometry.Rectangle([5, 55, 30, 70])),
        ("Madagascar", ee.Geometry.Rectangle([43, -26, 50, -12])),
        ("Tasmania", ee.Geometry.Rectangle([144, -44, 148, -40])),
    ]

    all_points_train = []
    region_names_train = []
    for i, (name, region) in enumerate(regions_train):
        region_points = forest_only.sample(
            region=region,
            scale=SCALE,
            numPixels=POINTS_PER_REGION_TRAIN,
            seed=42 + i,
            geometries=True,
        )
        all_points_train.append(region_points)
        region_names_train.append(name)

    all_points_test = []
    region_names_test = []
    for i, (name, region) in enumerate(regions_test):
        region_points = forest_only.sample(
            region=region,
            scale=SCALE,
            numPixels=POINTS_PER_REGION_TEST,
            seed=100 + i,
            geometries=True,
        )
        all_points_test.append(region_points)
        region_names_test.append(name)

    points_train = ee.FeatureCollection(all_points_train).flatten()
    points_test = ee.FeatureCollection(all_points_test).flatten()

    print(f"TRAIN: {len(regions_train)} regions x {POINTS_PER_REGION_TRAIN} pts")
    print(f"TEST:  {len(regions_test)} regions x {POINTS_PER_REGION_TEST} pts")
    print(f"Scale: {SCALE}m")

    if TEST_MODE:
        run_test_mode(all_points_train, all_points_test,
                      region_names_train, region_names_test, points_train)
    else:
        train_tasks = export_dataset(
            points_train,
            "train",
            len(regions_train) * POINTS_PER_REGION_TRAIN,
        )
        test_tasks = export_dataset(
            points_test,
            "test",
            len(regions_test) * POINTS_PER_REGION_TEST,
        )
        print(f"\nTrain: {train_tasks} tasks -> gs://{BUCKET}/{FOLDER}/train/")
        print(f"Test:  {test_tasks} tasks -> gs://{BUCKET}/{FOLDER}/test/")
        print("Monitor: https://code.earthengine.google.com/tasks")


if __name__ == '__main__':
    main()
