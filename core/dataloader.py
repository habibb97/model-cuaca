#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 16:45:00 2025

@author: habib
"""

import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling

class TiffDataLoader:
    def __init__(self, data_dir):
        """
        Inisialisasi DataLoader dengan path direktori dataset.
        """
        self.paths = glob.glob(f"{data_dir}/*/*")
        self.batch_size = len(self.paths)

        if self.batch_size == 0:
            raise ValueError("Tidak ada file yang ditemukan di path yang diberikan.")

        print(f"Jumlah data: {self.batch_size}, dataset terakhir pada: {self.paths[-1]}")

    def load_tiff(self):
        """
        Memuat dan memproses file TIFF menjadi array NumPy dengan ukuran tetap (1, 851, 2351).
        """
        batch = np.zeros((self.batch_size, 1, 851, 2351), dtype=np.float32)

        for i, file in enumerate(self.paths):
            with rasterio.open(file) as src:
                if src.height != 851 or src.width != 2351:
                    data = src.read(
                        out_shape=(1, 851, 2351),
                        resampling=Resampling.bilinear
                    )
                else:
                    data = src.read(1)
                    data = np.expand_dims(data, axis=0)  # Pastikan shape (1, 851, 2351)

                batch[i] = data
                del data  # Hapus variabel untuk optimasi memori

                print(f"{file} telah disimpan")

        return batch

    def save_numpy(self, batch, filename="satelit_dataset.npy"):
        """
        Menyimpan dataset ke dalam file .npy.
        """
        np.save(filename, batch)
        print(f"Dataset telah disimpan dalam {filename}")
