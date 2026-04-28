import rasterio
import numpy as np
import matplotlib.pyplot as plt

red_path = "data/dhakab4.tif"
nir_path = "data/dhakab5.tif"
swir_path = "data/dhakab6.tif"
thermal_path = "data/dhakab10.tif"
ML = 0.0003342
AL = 0.1
K1 = 774.8853
K2 = 1321.0789

mult = 3.3420e-04
add = 0.10000
k1_const = 774.8853
k2_const = 1321.0789


def calculate_ndvi(red_path, nir_path):
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


def calculate_ndbi(nir_path, swir_path, output="data/ndbi_dhaka.tif"):

    with rasterio.open(nir_path) as src:
        nir = src.read(1).astype(float)
        profile = src.profile

    with rasterio.open(swir_path) as src:
        swir = src.read(1).astype(float)

    np.seterr(divide="ignore", invalid="ignore")

    ndbi = (swir - nir) / (swir + nir)

    plt.imshow(ndbi, cmap="RdYlGn")
    plt.colorbar(label="NDBI")
    plt.title("NDBI Map")
    plt.show()

    # save
    profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(ndbi.astype(rasterio.float32), 1)


def calculate_lst(band10_path, M_L, A_L, K1, K2):
    with rasterio.open(band10_path) as src:
        dn = src.read(1).astype("float32")

        dn = np.where(dn == 0, np.nan, dn)

        profile = src.profile

    radiance = (M_L * dn) + A_L

    radiance = np.where(radiance <= 0, np.nan, radiance)

    print("Calculating Brightness Temperature...")
    bt_celsius = (K2 / np.log((K1 / radiance) + 1)) - 273.15

    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    im = plt.imshow(bt_celsius, cmap="hot")
    plt.colorbar(im, label="Temperature (°C)")
    plt.title("Landsat Brightness Temperature")
    plt.axis("off")
    plt.show()

    profile.update(dtype=rasterio.float32, count=1, compress="lzw")
    with rasterio.open(
        "data/lst_dhaka.tif",
        "w",
        **profile,
    ) as dst:
        dst.write(bt_celsius.astype(rasterio.float32), 1)

    return bt_celsius


if __name__ == "__main__":
    calculate_ndvi(red_path, nir_path)
    calculate_ndbi(swir_path, nir_path)
    calculate_lst(thermal_path, mult, add, k1_const, k2_const)
