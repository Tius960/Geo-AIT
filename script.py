import os
import shutil
import tempfile
from PIL import Image
import tifffile as tiff
import math
import cv2
import yaml
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from shapely.ops import unary_union
from rtree import index
from tqdm import tqdm

# ==================== FUNGSI TILING ====================

def tile_image_with_overlap(input_path, output_folder, tile_width, tile_height):
    """
    Memecah satu file GeoTIFF besar (misal orthophoto drone) menjadi tile-tile
    kecil berukuran (tile_width x tile_height), dengan overlap antar tile.

    Kenapa perlu overlap?
    - Tanpa overlap, objek (pohon) yang posisinya persis di garis batas tile
      bisa terpotong dua dan gagal / tidak akurat terdeteksi model YOLO.
    - Dengan overlap, tiap objek punya peluang lebih besar muncul utuh
      minimal di salah satu tile (deteksi duplikat di area overlap nanti
      dibersihkan oleh fungsi filter_overlap()).

    Setiap tile disimpan sebagai file GeoTIFF baru yang tetap punya
    georeferensi (CRS + transform) sendiri, dihitung dari posisi crop-nya
    terhadap raster asli — supaya nanti koordinat hasil deteksi di tiap
    tile bisa dikonversi balik ke koordinat geografis yang benar.

    Parameters:
        input_path (str): path file GeoTIFF sumber (misal sawit.tif)
        output_folder (str): folder tujuan penyimpanan tile-tile hasil crop
        tile_width, tile_height (int): ukuran tile dalam piksel (misal 640x640,
            harus sama dengan ukuran input model YOLO)
    """
    # Baca seluruh array piksel citra sumber ke memori
    image = tiff.imread(input_path)
    img = Image.fromarray(image)
    
    img_width, img_height = img.size
    
    # Hitung jumlah tiles yang dibutuhkan.
    # Dibulatkan ke atas (ceil) supaya sisa citra yang tidak pas kelipatan
    # tile_width/tile_height tetap tercakup semua (tile terakhir nanti
    # akan "digeser" ke belakang agar tidak keluar batas gambar).
    num_tiles_x = math.ceil(img_width / tile_width)
    num_tiles_y = math.ceil(img_height / tile_height)
    
    # Hitung overlap (dalam piksel) antar tile yang berdekatan.
    # Overlap dihitung otomatis dari selisih antara total lebar semua tile
    # (jika berjajar tanpa overlap) dengan lebar citra asli, dibagi rata ke
    # jumlah "celah" antar tile (num_tiles - 1). Ini memastikan tile-tile
    # tersebar rata menutupi seluruh citra tanpa ada bagian yang terlewat.
    if num_tiles_x > 1:
        overlap_x = math.ceil((num_tiles_x * tile_width - img_width) / (num_tiles_x - 1))
    else:
        overlap_x = 0
    
    if num_tiles_y > 1:
        overlap_y = math.ceil((num_tiles_y * tile_height - img_height) / (num_tiles_y - 1))
    else:
        overlap_y = 0
    
    total_tiles = num_tiles_x * num_tiles_y
    
    count = 0
    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y

    with tqdm(total=total_tiles, desc=f"Tiling {os.path.basename(input_path)}", unit="tile") as pbar:
        # Loop ganda: iterasi tile secara grid, kolom (i) lalu baris (j)
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Hitung posisi awal (pojok kiri-atas) tile ke-(i,j) di
                # citra asli, berdasarkan step (lebar tile dikurangi overlap)
                left = i * step_x
                upper = j * step_y
                
                # Pastikan tile terakhir tidak melebihi batas gambar —
                # tile paling kanan/bawah "ditarik mundur" agar pas rata
                # dengan tepi citra, bukan mengambil area di luar gambar
                if i == num_tiles_x - 1:
                    left = img_width - tile_width
                if j == num_tiles_y - 1:
                    upper = img_height - tile_height
                
                right = left + tile_width
                lower = upper + tile_height
                
                # Crop tile dari citra PIL sesuai koordinat piksel di atas
                tile = img.crop((left, upper, right, lower))
                
                # Jaga-jaga: kalau ukuran hasil crop tidak pas (misal citra
                # asli lebih kecil dari satu tile), tempelkan ke kanvas hitam
                # berukuran tile_width x tile_height supaya semua tile punya
                # ukuran seragam (dibutuhkan model YOLO yang input-nya fix)
                if tile.size != (tile_width, tile_height):
                    padded_tile = Image.new(img.mode, (tile_width, tile_height), 0)
                    padded_tile.paste(tile, (0, 0))
                    tile = padded_tile
                
                tile_filename = os.path.join(output_folder, f"tile_{count}.tif")
                
                # Simpan sebagai GeoTIFF dengan georeferencing
                with rasterio.open(input_path) as src:
                    # Hitung transform (georeferensi) khusus untuk tile ini,
                    # dari koordinat geografis citra asli + posisi piksel
                    # crop (left, upper, right, lower) yang sudah dihitung
                    # di atas. Ini yang membuat tiap tile "tahu" posisinya
                    # di dunia nyata meskipun sudah dipisah jadi file sendiri.
                    tile_transform = rasterio.transform.from_bounds(
                        src.bounds.left + left * src.transform[0],
                        src.bounds.top - lower * abs(src.transform[4]),
                        src.bounds.left + right * src.transform[0],
                        src.bounds.top - upper * abs(src.transform[4]),
                        tile_width,
                        tile_height
                    )
                    
                    # Simpan tile sebagai GeoTIFF baru, mewarisi CRS dan
                    # jumlah band (count) dari citra sumber, tapi dengan
                    # transform (georeferensi) khusus tile ini
                    with rasterio.open(
                        tile_filename,
                        'w',
                        driver='GTiff',
                        height=tile_height,
                        width=tile_width,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=tile_transform
                    ) as dst:
                        tile_array = np.array(tile)
                        # Tulis per-band: kalau citra grayscale (2D) tulis
                        # band tunggal, kalau RGB/multi-band (3D) tulis tiap
                        # band satu per satu sesuai urutan channel-nya
                        if len(tile_array.shape) == 2:
                            dst.write(tile_array, 1)
                        else:
                            for band in range(src.count):
                                dst.write(tile_array[:, :, band], band + 1)
                
                count += 1
                pbar.update(1)

def process_tiling(input_folder, output_folder, tile_width, tile_height):
    """
    Fungsi orkestrator tiling: mencari semua file .tif/.tiff di dalam
    input_folder, lalu memanggil tile_image_with_overlap() untuk
    masing-masing file tersebut.

    Setiap file sumber mendapat sub-folder outputnya sendiri (dinamai sesuai
    nama file tanpa ekstensi), supaya tile dari file yang berbeda tidak
    tercampur/bentrok nama saat proses deteksi berikutnya membaca semuanya.

    Parameters:
        input_folder (str): folder berisi satu atau lebih file TIF sumber
        output_folder (str): folder induk tempat sub-folder tile disimpan
        tile_width, tile_height (int): ukuran tile yang diteruskan ke
            tile_image_with_overlap()
    """
    # Buat folder output induk jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Cari semua file TIF/TIFF di folder input (tidak rekursif ke sub-folder)
    tif_files = [f for f in os.listdir(input_folder) 
                 if f.endswith(".tif") or f.endswith(".tiff")]
    
    if not tif_files:
        print("❌ Tidak ada file TIF ditemukan di folder input!")
        return
    
    print(f"\n📁 Ditemukan {len(tif_files)} file TIF untuk diproses")
    print("="*60)
    
    # Proses tiap file TIF satu per satu
    for filename in tif_files:
        input_path = os.path.join(input_folder, filename)
        
        # Sub-folder khusus untuk tile dari file ini, misal:
        # output_folder/sawit/tile_0.tif, tile_1.tif, dst.
        file_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
        if not os.path.exists(file_output_folder):
            os.makedirs(file_output_folder)
        
        tile_image_with_overlap(input_path, file_output_folder, tile_width, tile_height)

# ==================== FUNGSI DETEKSI YOLO ====================

def load_labels(yaml_path):
    """
    Membaca file data.yaml (format standar YOLO) dan mengambil daftar nama
    kelas (labels) dari key 'names'. Daftar ini dipakai untuk menerjemahkan
    class_id numerik hasil deteksi model (misal 0) menjadi nama kelas yang
    bisa dibaca manusia (misal 'sawit').

    Parameters:
        yaml_path (str): path ke file data.yaml
    Returns:
        list/dict nama-nama kelas sesuai isi data.yaml
    """
    with open(yaml_path, mode='r') as f:
        data_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    return data_yaml['names']

def load_yolo_model(model_path):
    """
    Memuat model YOLO dari file .onnx menggunakan modul DNN bawaan OpenCV
    (jadi tidak perlu library PyTorch/ultralytics terpisah).
    Backend & target di-set eksplisit ke OpenCV/CPU — cocok untuk mesin
    tanpa GPU khusus / tanpa CUDA.

    Parameters:
        model_path (str): path ke file model .onnx (misal palmCounting-model.onnx)
    Returns:
        objek net OpenCV DNN yang siap dipakai untuk inferensi
    """
    yolo = cv2.dnn.readNetFromONNX(model_path)
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return yolo

def get_gsd_from_raster(raster_path):
    """
    Mengambil GSD (Ground Sample Distance) — yaitu ukuran nyata di lapangan
    yang diwakili oleh satu piksel citra (misal 0.06 meter/piksel) — dari
    metadata georeferensi (transform) file raster.

    Nilai ini penting untuk mengonversi posisi deteksi dari satuan piksel
    (hasil model YOLO) menjadi satuan jarak/koordinat geografis riil.

    Parameters:
        raster_path (str): path file GeoTIFF (tile atau citra asli)
    Returns:
        tuple (gsd_x, gsd_y): resolusi piksel arah X dan Y, dalam satuan CRS
            raster (biasanya meter, karena CRS di sini UTM)
    """
    with rasterio.open(raster_path) as src:
        transform = src.transform
        gsd_x = abs(transform[0])
        gsd_y = abs(transform[4])
    return gsd_x, gsd_y

def perform_yolo_detection(image, yolo, labels, input_size=640, conf_threshold=0.1, nms_threshold=0.25):
    """
    Menjalankan satu kali inferensi YOLO pada satu gambar tile, lalu
    mem-parsing output mentah model menjadi daftar bounding box final
    (setelah filtering confidence & Non-Maximum Suppression).

    Parameters:
        image (np.ndarray): array gambar (H x W x 3, BGR dari cv2.imread)
        yolo: objek model OpenCV DNN hasil load_yolo_model()
        labels: daftar nama kelas hasil load_labels()
        input_size (int): ukuran sisi input model (default 640, harus sama
            dengan ukuran tile & ukuran input model .onnx)
        conf_threshold (float): ambang batas confidence minimum objectness
            untuk dianggap sebagai kandidat deteksi
        nms_threshold (float): ambang batas overlap (IoU) untuk NMS —
            menghapus box duplikat yang saling tumpang tindih pada objek
            yang sama

    Returns:
        list of dict: tiap dict berisi class_name, confidence, dan posisi
            box (x, y, width, height) dalam koordinat piksel tile
    """
    # Buat kanvas persegi (square) berukuran max(row, col) x max(row, col),
    # lalu tempelkan gambar asli di pojok kiri-atas. Ini diperlukan karena
    # model YOLO expects input persegi — jika tile sudah 640x640 langkah ini
    # sebenarnya tidak mengubah apa pun, tapi tetap aman untuk input non-persegi.
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    
    # Konversi gambar menjadi "blob" (tensor) yang bisa dibaca model:
    # - normalisasi piksel ke rentang 0-1 (dikali 1/255)
    # - resize ke input_size x input_size
    # - swapRB=True karena OpenCV baca gambar BGR, tapi model dilatih RGB
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (input_size, input_size), swapRB=True, crop=False)
    yolo.setInput(blob)
    preds = yolo.forward()
    # Output model berbentuk [1, 25200, 6] -> ambil batch pertama saja
    # sehingga jadi [25200, 6]: 25200 = jumlah kandidat box (anchor points),
    # 6 kolom = [center_x, center_y, width, height, objectness_conf, class_score]
    detections = preds[0]
    
    boxes = []
    confidences = []
    classes = []
    image_w, image_h = input_image.shape[:2]
    # Faktor skala untuk mengonversi koordinat box dari skala input_size
    # (640x640) kembali ke skala gambar kanvas asli (input_image)
    x_factor = image_w / input_size
    y_factor = image_h / input_size

    # Loop tiap kandidat deteksi (baris) dari output mentah model
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # skor objectness (seberapa yakin ada objek di sini)
        if confidence > conf_threshold:
            # Karena bisa multi-kelas, ambil skor kelas tertinggi & id-nya
            # (untuk model ini praktis hanya ada 1 kelas)
            class_score = row[5:].max()
            class_id = row[5:].argmax()

            if class_score > 0.25:
                # Format YOLO: cx, cy = titik tengah box; w, h = lebar/tinggi box
                # (semua masih dalam skala 0-input_size)
                cx, cy, w, h = row[0:4]
                # Konversi dari (center, width, height) ke (top-left, width, height)
                # sekaligus discale ke ukuran kanvas asli via x_factor/y_factor
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = [left, top, width, height]
                confidences.append(float(confidence))
                boxes.append(box)
                classes.append(class_id)

    # Non-maximum suppression (NMS): karena satu objek nyata biasanya
    # menghasilkan banyak box kandidat yang saling tumpang tindih, NMS
    # memilih satu box terbaik (confidence tertinggi) per objek dan
    # membuang box lain yang overlap-nya melebihi nms_threshold
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    
    if len(indices) > 0:
        # Normalisasi bentuk array index (versi OpenCV berbeda bisa
        # mengembalikan array 1D atau 2D)
        if isinstance(indices, np.ndarray):
            if indices.ndim == 2:
                indices = indices.flatten()
        
        # Susun hasil akhir jadi list of dict yang mudah dipakai fungsi lain
        for i in indices:
            idx = int(i) if isinstance(i, (np.integer, np.ndarray)) else i
            
            box = boxes[idx]
            class_name = labels[classes[idx]]
            confidence = confidences[idx]
            results.append({
                'class_name': class_name,
                'confidence': confidence,
                'x': box[0],
                'y': box[1],
                'width': box[2],
                'height': box[3]
            })

    return results

def filter_overlap(gdf, distance):
    """
    Menggabungkan titik-titik deteksi yang berdekatan (dalam radius
    `distance`) menjadi satu titik, memakai centroid-nya.

    Ini WAJIB dilakukan karena proses tiling menggunakan overlap — artinya
    area yang sama bisa muncul di lebih dari satu tile, sehingga pohon yang
    sama berpotensi terdeteksi lebih dari sekali (di tile berbeda) dan
    dihitung ganda kalau tidak difilter. Fungsi ini pakai spatial index
    (R-tree) supaya pencarian titik-titik yang berdekatan efisien meskipun
    jumlah deteksi banyak (tidak perlu bandingkan tiap titik ke semua titik
    lain / O(n²)).

    Parameters:
        gdf (GeoDataFrame): seluruh hasil deteksi (titik Point) dari semua
            tile, sebelum difilter
        distance: jarak toleransi (dalam satuan CRS, misal meter) — titik
            yang jaraknya lebih dekat dari ini dianggap deteksi objek yang
            sama dan akan digabung

    Returns:
        GeoDataFrame: hasil deteksi yang sudah bersih dari duplikat overlap
    """
    if len(gdf) == 0:
        return gdf
    
    # Bangun spatial index (R-tree) dari semua titik deteksi, supaya
    # pencarian "titik mana saja yang dekat titik ini" jadi cepat
    spatial_index = index.Index()
    points = []

    for idx, row in gdf.iterrows():
        point = row.geometry
        points.append((idx, point))
        spatial_index.insert(idx, point.bounds)

    unique_points = []
    seen = set()  # menandai index titik yang sudah "dipakai"/digabung

    # Untuk tiap titik yang belum diproses, cari semua titik lain yang
    # berada dalam radius `distance` (pakai buffer sebagai area pencarian),
    # gabungkan semuanya jadi satu titik lewat centroid, lalu tandai semua
    # titik yang tergabung itu sebagai "seen" supaya tidak diproses ulang
    for idx, point in points:
        if idx in seen:
            continue
        overlap_indices = list(spatial_index.intersection(point.buffer(distance).bounds))
        if overlap_indices:
            overlapping_points = [points[i][1] for i in overlap_indices]
            centroid = unary_union(overlapping_points).centroid
            unique_points.append((idx, centroid))
            seen.update(overlap_indices)

    # Susun ulang jadi GeoDataFrame baru: ambil class_name & confidence dari
    # titik "wakil" (idx pertama tiap kelompok), tapi geometry-nya diganti
    # dengan titik centroid hasil penggabungan
    filtered_data = []
    for idx, centroid in unique_points:
        filtered_data.append({
            'class_name': gdf.iloc[idx]['class_name'],
            'confidence': gdf.iloc[idx]['confidence'],
            'geometry': centroid
        })

    filtered_gdf = gpd.GeoDataFrame(filtered_data, crs=gdf.crs)
    return filtered_gdf

def process_detection(folder_path, model_path, yaml_path, output_shp_path, 
                      min_distance=1.0, gsd_x=None, gsd_y=None, 
                      conf_threshold=0.1, nms_threshold=0.25):  # ← Tambahkan parameter
    """
    Fungsi orkestrator tahap deteksi: menjalankan YOLO ke semua tile GeoTIFF
    di dalam folder_path (termasuk sub-folder), mengumpulkan semua hasil
    deteksi, mengonversinya ke koordinat geografis, memfilter duplikat
    akibat overlap tile, lalu menyimpan hasil akhirnya sebagai shapefile.

    Parameters:
        folder_path (str): folder berisi tile-tile GeoTIFF hasil tiling
            (dicari secara rekursif lewat os.walk)
        model_path (str): path model .onnx
        yaml_path (str): path data.yaml (nama kelas)
        output_shp_path (str): path file .shp output
        min_distance (float): jarak toleransi untuk filter_overlap()
            (satuan sama dengan CRS raster, biasanya meter)
        gsd_x, gsd_y (float, opsional): Ground Sample Distance manual.
            Jika None, akan dideteksi otomatis dari raster tile pertama
        conf_threshold, nms_threshold (float): diteruskan ke
            perform_yolo_detection()
    """
    labels = load_labels(yaml_path)
    yolo = load_yolo_model(model_path)
    all_results = []
    
    first_crs = None
    auto_gsd_x = None
    auto_gsd_y = None

    # Kumpulkan semua file TIF (rekursif — termasuk semua sub-folder tile
    # per file sumber yang dibuat oleh process_tiling)
    all_tif_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                all_tif_files.append(os.path.join(root, filename))
    
    if not all_tif_files:
        print("❌ Tidak ada file TIF ditemukan untuk deteksi!")
        return

    # Proses dengan progress bar — loop tiap tile satu per satu
    with tqdm(total=len(all_tif_files), desc="🔍 Deteksi YOLO", unit="tile") as pbar:
        for img_path in all_tif_files:
            filename = os.path.basename(img_path)
            # Baca tile sebagai array gambar biasa (BGR) untuk input model
            img = cv2.imread(img_path)
            if img is None:
                pbar.set_postfix_str(f"❌ {filename}")
                pbar.update(1)
                continue
            
            # Baca metadata georeferensi tile ini secara terpisah (rasterio),
            # karena cv2.imread() tidak membawa info CRS/bounds
            with rasterio.open(img_path) as src:
                bounds = src.bounds
                # Simpan CRS dari tile pertama untuk dipakai di shapefile akhir
                if first_crs is None:
                    first_crs = src.crs
                
                # Jika GSD tidak diberikan manual, deteksi otomatis dari
                # tile pertama saja (asumsi semua tile punya resolusi sama)
                if auto_gsd_x is None and (gsd_x is None or gsd_y is None):
                    auto_gsd_x, auto_gsd_y = get_gsd_from_raster(img_path)
                    
            current_gsd_x = gsd_x if gsd_x is not None else auto_gsd_x
            current_gsd_y = gsd_y if gsd_y is not None else auto_gsd_y
                    
            # Jalankan deteksi YOLO pada tile ini
            detections = perform_yolo_detection(img, yolo, labels, 
                                       conf_threshold=conf_threshold,  # ← Gunakan parameter
                                       nms_threshold=nms_threshold)

            if detections:
                for det in detections:
                    # Konversi posisi box dari piksel-lokal-tile menjadi
                    # koordinat geografis riil:
                    # - ambil titik tengah box (x + width/2, y + height/2)
                    # - kalikan dengan GSD (meter per piksel) untuk dapat
                    #   jarak dalam meter dari pojok kiri-atas tile
                    # - tambahkan/kurangkan ke bounds tile (posisi tile di
                    #   dunia nyata) untuk dapat koordinat absolut
                    koorX = (det['x'] + (det['width'] / 2)) * current_gsd_x + bounds.left
                    koorY = bounds.top - (det['y'] + (det['height'] / 2)) * current_gsd_y
                    
                    all_results.append({
                        'class_name': det['class_name'],
                        'confidence': det['confidence'],
                        'geometry': Point(koorX, koorY)
                    })
                
                pbar.set_postfix_str(f"✅ {len(detections)} obj")
            else:
                pbar.set_postfix_str("⚪ 0 obj")
            
            pbar.update(1)

    if not all_results:
        print("\n❌ Tidak ada deteksi ditemukan!")
        return

    print(f"\n📊 Total deteksi sebelum filtering: {len(all_results)}")
    
    # Gabungkan semua titik deteksi (dari semua tile) jadi satu GeoDataFrame
    gdf = gpd.GeoDataFrame(all_results, crs=first_crs if first_crs else 'EPSG:4326')
    
    # Buang duplikat deteksi akibat overlap antar tile (lihat filter_overlap())
    print("🔄 Filtering overlap...")
    filtered_gdf = filter_overlap(gdf, min_distance)
    print(f"✅ Total deteksi setelah filtering: {len(filtered_gdf)}")

    # Simpan hasil akhir sebagai shapefile (.shp), siap dibuka di GIS
    # (QGIS/ArcGIS) untuk visualisasi/analisis lanjutan
    os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)
    filtered_gdf.to_file(output_shp_path)
    print(f"\n💾 Shapefile disimpan: {output_shp_path}")
    print(f"🗺️  CRS: {filtered_gdf.crs}")

# ==================== FUNGSI UTAMA ====================

def process_tif_to_shapefile(input_folder, model_path, yaml_path, output_shp_path, 
                              tile_width=640, tile_height=640, min_distance=1.0, 
                              gsd_x=None, gsd_y=None,
                              conf_threshold=0.25, nms_threshold=0.4):
    """
    Proses lengkap dari tiling hingga deteksi YOLO dan menghasilkan shapefile.

    Ini adalah fungsi "pintu masuk" yang menyatukan seluruh pipeline:
        1. Tiling  -> process_tiling()   (pecah TIF asli jadi tile-tile kecil)
        2. Deteksi -> process_detection() (jalankan YOLO ke tiap tile,
           gabungkan & filter hasil, simpan sebagai shapefile)
    Tile-tile sementara disimpan di folder temporary yang otomatis dihapus
    di akhir proses (baik proses berhasil maupun gagal, lewat try/finally),
    supaya tidak menumpuk file sampah di disk.
    
    Parameters:
    - input_folder: Folder berisi file TIF asli
    - model_path: Path ke model YOLO ONNX
    - yaml_path: Path ke file data.yaml
    - output_shp_path: Path output shapefile
    - tile_width: Lebar tile (default: 640)
    - tile_height: Tinggi tile (default: 640)
    - min_distance: Jarak minimum untuk filtering overlap (dalam unit CRS)
    - gsd_x: Ground Sample Distance X (opsional, auto-detect jika None)
    - gsd_y: Ground Sample Distance Y (opsional, auto-detect jika None)
    """
    
    # Buat folder temporary unik untuk menyimpan tile-tile hasil tiling
    # (tidak ditulis ke output_folder permanen karena hanya dipakai
    # sementara sebagai input tahap deteksi)
    temp_dir = tempfile.mkdtemp(prefix="tiles_temp_")
    print(f"\n{'='*60}")
    print(f"📂 Folder temporary: {temp_dir}")
    print(f"{'='*60}\n")
    
    try:
        # STEP 1: Tiling — pecah semua TIF di input_folder jadi tile 640x640
        print("="*60)
        print("🔷 STEP 1: PROSES TILING")
        print("="*60)
        process_tiling(input_folder, temp_dir, tile_width, tile_height)
        
        # STEP 2: Deteksi YOLO — deteksi tiap tile, gabung & filter hasil,
        # simpan sebagai shapefile
        print("\n" + "="*60)
        print("🔷 STEP 2: PROSES DETEKSI YOLO")
        print("="*60)
        process_detection(temp_dir, model_path, yaml_path, output_shp_path, 
                     min_distance, gsd_x, gsd_y,
                     conf_threshold, nms_threshold) 
        
        print("\n" + "="*60)
        print("✅ PROSES SELESAI!")
        print("="*60)
        
    finally:
        # Selalu bersihkan folder temporary di akhir, meskipun terjadi error
        # di tengah proses (try/finally) — supaya tile sementara tidak
        # menumpuk memenuhi disk tiap kali script dijalankan
        print(f"\n🗑️  Menghapus folder temporary...")
        shutil.rmtree(temp_dir)
        print("✅ Folder temporary berhasil dihapus!")

# ==================== CONTOH PENGGUNAAN ====================

if __name__ == "__main__":
    # Konfigurasi path — sesuaikan dengan lokasi file di komputer Anda.
    # input_folder harus berupa FOLDER (bisa berisi lebih dari satu TIF),
    # bukan path ke satu file TIF langsung.
    input_folder = r"input_tif\25_PT MUSIM MAS"              # Folder berisi file TIF asli
    model_path = r"model_dir\model_best\weights\best.onnx"    # Path ke model YOLO
    yaml_path = r"model_dir\data.yaml"      # Path ke data.yaml
    output_shp_path = "./output/25_PT MUSIM MAS.shp"  # Output shapefile
    
    # Parameter tiling — ukuran tile harus sama dengan ukuran input model (640x640)
    tile_width = 640
    tile_height = 640
    
    # Parameter deteksi
    min_distance = 3  # Jarak minimum untuk filtering (dalam unit CRS, misal: meter)
    
    # Jalankan seluruh pipeline: tiling -> deteksi YOLO -> simpan shapefile
    process_tif_to_shapefile(
        input_folder = r"input_folder",
        model_path = r"model_path\palmCounting-model.onnx",
        yaml_path = r"yaml_path\data.yaml",
        output_shp_path = "./output/sawit.shp",
        tile_width=640,
        tile_height=640,
        min_distance=3.0,
        conf_threshold=0.3,   # ← Atur confidence threshold (0.0 - 1.0)
        nms_threshold=0.3     # ← Atur NMS threshold (0.0 - 1.0)
    )