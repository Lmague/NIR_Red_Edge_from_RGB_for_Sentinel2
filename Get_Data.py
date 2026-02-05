import ee

# 1. Initialisation (Cliquez sur le lien, validez, copiez le code)
ee.Authenticate()
ee.Initialize(project='VOTRE_ID_PROJET_CLOUD')  # <-- IMPORTANT : Mettez votre ID ici

# ==============================================================================
# CONFIGURATION DU DATASET
# ==============================================================================
BUCKET = 'votre-bucket-gcs'       # <-- Votre Bucket de stockage
FOLDER = 'dataset_training_mst'   # Dossier où les fichiers vont atterrir
NUM_POINTS = 100000               # Objectif : 100k tuiles (10 régions x 10k)
PATCH_SIZE = 128                  # Taille standard pour MST++
BATCH_SIZE = 500                  # Réduit pour éviter memory limit

# Mode test : mettre True pour tester avec 5 points avant le lancement massif
TEST_MODE = False  # 🚀 PRODUCTION - Export validé !

# ==============================================================================
# ÉTAPE 1 : CIBLAGE "FORÊT PURE" - SPLIT GÉOGRAPHIQUE TRAIN/TEST
# ==============================================================================
print("Localisation des zones forestières mondiales...")
print("⚠️  Split GÉOGRAPHIQUE : certaines régions réservées pour le TEST")

landcover = ee.Image("ESA/WorldCover/v100/2020").select('Map')
tree_mask = landcover.eq(10).Or(landcover.eq(20))  # Arbres + Broussailles
forest_only = landcover.updateMask(tree_mask)

# ==============================================================================
# RÉGIONS D'ENTRAÎNEMENT (12 régions diversifiées)
# ==============================================================================
regions_train = [
    # === FORÊTS TROPICALES ===
    ("Amazonie", ee.Geometry.Rectangle([-80, -15, -45, 5])),
    ("Congo", ee.Geometry.Rectangle([8, -10, 32, 8])),
    ("Asie SE", ee.Geometry.Rectangle([95, -10, 150, 25])),
    ("Am. Centrale", ee.Geometry.Rectangle([-100, 10, -75, 25])),
    ("Inde", ee.Geometry.Rectangle([70, 8, 95, 30])),
    
    # === FORÊTS TEMPÉRÉES ===
    ("Europe Ouest", ee.Geometry.Rectangle([-10, 40, 25, 60])),  # Sans Scandinavie (test)
    ("Am. Nord", ee.Geometry.Rectangle([-130, 35, -60, 55])),
    ("Chine/Japon", ee.Geometry.Rectangle([105, 22, 145, 45])),
    ("Am. Sud temp.", ee.Geometry.Rectangle([-75, -45, -50, -20])),
    
    # === FORÊTS BORÉALES ===
    ("Sibérie", ee.Geometry.Rectangle([60, 50, 145, 70])),
    ("Canada", ee.Geometry.Rectangle([-140, 50, -60, 68])),
    
    # === SUBTROPICAL ===
    ("Australie Est", ee.Geometry.Rectangle([140, -40, 155, -15])),
]

# ==============================================================================
# RÉGIONS DE TEST (jamais vues pendant l'entraînement !)
# ==============================================================================
regions_test = [
    # Régions géographiquement isolées pour un vrai test de généralisation
    ("Nouvelle-Zélande", ee.Geometry.Rectangle([166, -47, 178, -34])),
    ("Scandinavie", ee.Geometry.Rectangle([5, 55, 30, 70])),
    ("Madagascar", ee.Geometry.Rectangle([43, -26, 50, -12])),
    ("Tasmanie", ee.Geometry.Rectangle([144, -44, 148, -40])),
]

# Configuration
POINTS_PER_REGION_TRAIN = 10000  # Plus de points avec scale=1500
POINTS_PER_REGION_TEST = 3000
SCALE = 1500  # 1.5km entre points (léger chevauchement possible mais ok)

print(f"\n📊 Configuration:")
print(f"   TRAIN: {len(regions_train)} régions × {POINTS_PER_REGION_TRAIN} pts = {len(regions_train) * POINTS_PER_REGION_TRAIN} max")
print(f"   TEST:  {len(regions_test)} régions × {POINTS_PER_REGION_TEST} pts = {len(regions_test) * POINTS_PER_REGION_TEST} max")
print(f"   Scale: {SCALE}m (tuiles ~1.3km → léger chevauchement possible)")

# Générer les points TRAIN
all_points_train = []
region_names_train = []
for i, (name, region) in enumerate(regions_train):
    region_points = forest_only.sample(
        region=region,
        scale=SCALE,
        numPixels=POINTS_PER_REGION_TRAIN,
        seed=42 + i,
        geometries=True
    )
    all_points_train.append(region_points)
    region_names_train.append(name)

# Générer les points TEST
all_points_test = []
region_names_test = []
for i, (name, region) in enumerate(regions_test):
    region_points = forest_only.sample(
        region=region,
        scale=SCALE,
        numPixels=POINTS_PER_REGION_TEST,
        seed=100 + i,  # Seed différent
        geometries=True
    )
    all_points_test.append(region_points)
    region_names_test.append(name)

# Collections finales
points_train = ee.FeatureCollection(all_points_train).flatten()
points_test = ee.FeatureCollection(all_points_test).flatten()
points = points_train  # Pour compatibilité avec le reste du code

# ==============================================================================
# ÉTAPE 2 : EXTRACTION DES BANDES
# ==============================================================================

def get_spectral_pair(feature):
    """Extrait une paire (RGB, NIR/RedEdge) pour un point donné."""
    geom = feature.geometry()
    
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geom) \
        .filterDate('2021-01-01', '2024-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    
    img = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
    
    # Retourne une Feature avec 'valid'=0 si pas d'image (sera filtré après)
    # On ne peut PAS retourner None dans un map()
    return ee.Algorithms.If(
        collection.size().gt(0),
        extract_patches(img, geom),
        ee.Feature(None, {'valid': 0})  # Marqueur pour filtrage
    )


def extract_patches(img, geom):
    """Extrait un patch 128x128 autour du point."""
    
    # 1. Sélection des 7 bandes spectrales
    bands = img.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'])
    
    # 2. Remplacer les pixels masqués par 0
    bands = bands.unmask(0)
    
    # 3. Reprojection en Mercator pour garantir 128x128 exact
    bands = bands.reproject(crs='EPSG:3857', scale=10)
    
    # 4. Créer le carré (vise 129-130 pixels, sera croppé à 128 côté PyTorch)
    region = geom.buffer(640, 1).bounds(1, 'EPSG:3857')
    
    # 5. Extraction des arrays 2D via sampleRectangle
    arrays = bands.sampleRectangle(region=region, defaultValue=0)
    
    # 6. Garder comme ee.Array (GEE TFRecord supporte les arrays, pas les listes)
    # On aplatit en 1D mais on garde le type Array
    def get_flat_array(band_name):
        arr = ee.Array(arrays.get(band_name))
        # Flatten: reshape 2D array to 1D
        return arr.reshape([-1])
    
    return ee.Feature(None, {
        'valid': 1,
        'B2': get_flat_array('B2'),
        'B3': get_flat_array('B3'),
        'B4': get_flat_array('B4'),
        'B5': get_flat_array('B5'),
        'B6': get_flat_array('B6'),
        'B7': get_flat_array('B7'),
        'B8': get_flat_array('B8')
    })


# ==============================================================================
# ÉTAPE 3 : TEST AVANT EXPORT MASSIF
# ==============================================================================

if TEST_MODE:
    print("\n" + "="*60)
    print("🧪 MODE TEST ACTIVÉ - Debug détaillé")
    print("="*60)
    
    try:
        # ÉTAPE A : Vérifier les points TRAIN par région
        print("\n[1/6] Points d'ENTRAÎNEMENT par région...")
        
        total_train = 0
        print("\n      Région                  | Points")
        print("      " + "-"*45)
        for i, name in enumerate(region_names_train):
            count = all_points_train[i].size().getInfo()
            total_train += count
            bar = "█" * (count // 500)
            print(f"      {name:<22} | {count:>5} {bar}")
        
        print("      " + "-"*45)
        print(f"      {'TOTAL TRAIN':<22} | {total_train:>5}")
        
        # ÉTAPE B : Vérifier les points TEST par région
        print("\n[2/6] Points de TEST par région (régions réservées)...")
        
        total_test = 0
        print("\n      Région                  | Points")
        print("      " + "-"*45)
        for i, name in enumerate(region_names_test):
            count = all_points_test[i].size().getInfo()
            total_test += count
            bar = "█" * (count // 300)
            print(f"      {name:<22} | {count:>5} {bar}")
        
        print("      " + "-"*45)
        print(f"      {'TOTAL TEST':<22} | {total_test:>5}")
        
        # Résumé
        print(f"\n      📊 RÉSUMÉ:")
        print(f"         Train: {total_train} tuiles ({total_train/(total_train+total_test)*100:.1f}%)")
        print(f"         Test:  {total_test} tuiles ({total_test/(total_train+total_test)*100:.1f}%)")
        print(f"         Total: {total_train + total_test} tuiles")
        
        num_points = points_train.size().getInfo()
        
        if num_points == 0:
            raise Exception("Aucun point généré!")
        
        test_point = points_train.first()
        
        # ÉTAPE C : Prendre UN SEUL point pour le test
        print("\n[3/6] Extraction d'un point de test...")
        geom = test_point.geometry()
        coords = geom.coordinates().getInfo()
        print(f"      ✓ Coordonnées: {coords}")
        
        # ÉTAPE D : Vérifier qu'il y a des images Sentinel-2 à cet endroit
        print("\n[4/6] Recherche d'images Sentinel-2...")
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geom) \
            .filterDate('2021-01-01', '2024-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        
        col_size = collection.size().getInfo()
        print(f"      ✓ {col_size} images trouvées")
        
        if col_size == 0:
            print("\n      ❌ Aucune image! On élargit les filtres...")
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(geom) \
                .filterDate('2019-01-01', '2024-12-31') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            col_size = collection.size().getInfo()
            print(f"      → Avec filtres élargis: {col_size} images")
        
        # ÉTAPE E : Tester l'extraction
        print("\n[5/6] Test d'extraction du patch...")
        img = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        
        # Sélection et préparation
        bands = img.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'])
        bands = bands.unmask(0)
        bands = bands.reproject(crs='EPSG:3857', scale=10)
        
        # Zone d'extraction (640m = ~130 pixels, sera croppé à 128)
        region = geom.buffer(640, 1).bounds(1, 'EPSG:3857')
        
        # Extraction
        print("      → Appel sampleRectangle...")
        arrays = bands.sampleRectangle(region=region, defaultValue=0)
        
        # Récupérer les données
        print("      → Récupération des données...")
        b2_array = ee.Array(arrays.get('B2')).getInfo()
        
        if b2_array:
            rows = len(b2_array)
            cols = len(b2_array[0]) if rows > 0 else 0
            total_pixels = rows * cols
            
            print(f"\n[6/6] RÉSULTAT:")
            print(f"      ✓ Dimensions: {rows} x {cols} = {total_pixels} pixels")
            print(f"      ✓ Attendu: >= {PATCH_SIZE} x {PATCH_SIZE} (sera croppé à 128x128)")
            
            if rows >= PATCH_SIZE and cols >= PATCH_SIZE:
                print(f"\n🎉 SUCCÈS! L'extraction fonctionne parfaitement!")
                print(f"   → Dimensions {rows}x{cols} seront croppées à 128x128 côté PyTorch")
                
                # TEST EXPORT : Exporter 10 tuiles pour vérifier le format TFRecord
                print(f"\n[BONUS] Export test de 10 tuiles...")
                test_batch = ee.FeatureCollection(points_train.toList(10))
                test_features = test_batch.map(get_spectral_pair)
                test_features = test_features.filter(ee.Filter.eq('valid', 1))
                
                task = ee.batch.Export.table.toCloudStorage(
                    collection=test_features,
                    description='TEST_10_tuiles',
                    bucket=BUCKET,
                    fileNamePrefix=f'{FOLDER}/test_export',
                    fileFormat='TFRecord'
                )
                task.start()
                print(f"   ✓ Tâche 'TEST_10_tuiles' lancée")
                print(f"   → Vérifiez dans GEE Tasks, puis téléchargez et validez avec Check_TFRecord.py")
                print(f"   → Si les bandes B2-B8 sont présentes, passez TEST_MODE = False")
            else:
                print(f"\n❌ ERREUR: Dimensions trop petites ({rows}x{cols})")
                print(f"   → Augmentez le buffer")
        else:
            print("\n      ❌ Aucune donnée extraite!")
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Fin du test de debug")
    print("="*60 + "\n")

else:
    # ==============================================================================
    # ÉTAPE 4 : EXPORT MASSIF (TRAIN + TEST séparés)
    # ==============================================================================
    
    def export_dataset(points_collection, folder_suffix, num_points_max):
        """Exporte une collection de points vers GCS."""
        points_list = points_collection.toList(num_points_max)
        actual_count = points_collection.size().getInfo()
        
        print(f"\n📦 Export {folder_suffix.upper()}: {actual_count} points")
        
        tasks_launched = 0
        for i in range(0, actual_count, BATCH_SIZE):
            batch = ee.FeatureCollection(points_list.slice(i, min(i + BATCH_SIZE, actual_count)))
            
            # Transformer les points en patches
            features_export = batch.map(get_spectral_pair)
            
            # Filtrer les invalides (points sans image Sentinel-2)
            features_export = features_export.filter(ee.Filter.eq('valid', 1))
            
            desc = f'{folder_suffix}_{i}_{i+BATCH_SIZE}'
            
            task = ee.batch.Export.table.toCloudStorage(
                collection=features_export,
                description=desc,
                bucket=BUCKET,
                fileNamePrefix=f'{FOLDER}/{folder_suffix}/data_{i}',
                fileFormat='TFRecord'
            )
            
            task.start()
            tasks_launched += 1
            print(f"   ✓ Tâche {tasks_launched} envoyée ({i} à {min(i+BATCH_SIZE, actual_count)})")
        
        return tasks_launched
    
    print("="*60)
    print("🚀 LANCEMENT DES EXPORTS")
    print("="*60)
    
    # Export TRAIN
    train_tasks = export_dataset(
        points_train, 
        "train", 
        len(regions_train) * POINTS_PER_REGION_TRAIN
    )
    
    # Export TEST
    test_tasks = export_dataset(
        points_test, 
        "test", 
        len(regions_test) * POINTS_PER_REGION_TEST
    )
    
    print(f"\n" + "="*60)
    print(f"🎉 EXPORTS LANCÉS!")
    print(f"   Train: {train_tasks} tâches → gs://{BUCKET}/{FOLDER}/train/")
    print(f"   Test:  {test_tasks} tâches → gs://{BUCKET}/{FOLDER}/test/")
    print(f"\n   Vérifiez l'onglet Tasks: https://code.earthengine.google.com/tasks")
    print("="*60)