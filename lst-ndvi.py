import rasterio
import numpy as np
import matplotlib.pyplot as plt


nir_path = "data/b5_merged.tif"
swir_path = "data/b6_merged.tif"
thermal_path = "data/b10_merged.tif"


ML = 0.0003342
AL = 0.1
K1 = 774.8853
K2 = 1321.0789


with rasterio.open(nir_path) as src:
    nir = src.read(1).astype(float)
    profile = src.profile

with rasterio.open(swir_path) as src:
    swir = src.read(1).astype(float)

with rasterio.open(thermal_path) as src:
    thermal = src.read(1).astype(float)

np.seterr(divide='ignore', invalid='ignore')

# NDBI
ndbi = (swir - nir) / (swir + nir)

# LST
radiance = ML * thermal + AL
bt = K2 / np.log((K1 / radiance) + 1)
lst = bt - 273.15


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(ndbi, cmap="RdYlGn")
plt.title("NDBI")
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(lst, cmap="hot")
plt.title("Land Surface Temperature (°C)")
plt.colorbar()
plt.show()

# Save outputs
profile.update(dtype=rasterio.float32, count=1)

with rasterio.open("ndbi.tif", "w", **profile) as dst:
    dst.write(ndbi.astype(rasterio.float32), 1)

with rasterio.open("lst.tif", "w", **profile) as dst:
    dst.write(lst.astype(rasterio.float32), 1)
