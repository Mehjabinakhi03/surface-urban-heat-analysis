import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

folder = "data/"

red_path = "data/dhakab4.tif"
nir_path = "data/dhakab5.tif"

print("Red band:", red_path)
print("NIR band:", nir_path)

with rasterio.open(red_path) as red_src:
    red = red_src.read(1).astype(float)
    profile = red_src.profile

with rasterio.open(nir_path) as nir_src:
    nir = nir_src.read(1).astype(float)

np.seterr(divide="ignore", invalid="ignore")

ndvi = (nir - red) / (nir + red)

plt.figure(figsize=(8, 6))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar(label="NDVI")
plt.title("NDVI Map")
plt.show()

profile.update(dtype=rasterio.float32, count=1)

with rasterio.open("data/ndvi_dhaka.tif", "w", **profile) as dst:
    dst.write(ndvi.astype(rasterio.float32), 1)
