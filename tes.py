# import torch
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Ambil semua file dalam direktori
# paths = glob.glob("/home/ubuntu/model-cuaca/dataset/202402/*/*")
paths = glob.glob(r'D:\satelit\03\*') 

# Periksa apakah ada file yang ditemukan
batch_size = len(paths)
if batch_size == 0:
    raise ValueError("Tidak ada file yang ditemukan di path yang diberikan.")

# Inisialisasi batch array
batch = np.zeros((batch_size, 1, 850, 2350))


# Loop melalui semua file
for i, files in enumerate(paths):
    with rasterio.open(files) as src:
        # Periksa dimensi
        if src.height != 851 or src.width != 2351:
            # Resize data
            data = src.read(
                out_shape=(src.count, 851, 2351),
                resampling=Resampling.bilinear
            ).astype(np.float64)
            data = np.nan_to_num(data)
            data = data[:, :-1, :-1]
            
        else:
            data = src.read().astype(np.float64)
            data = np.nan_to_num(data)
            data = data[:, :-1, :-1]
    

        # Simpan dalam batch
        batch[i] = data

        # Hapus variabel untuk menghemat memori
        del data
        print(f'{files} telah di simpan')

print(f'jumlah data {batch_size} untuk dataset terakhir pada {paths[-1]}')
        
np.save("satelit_dataset.npy", batch)
