import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm



def analyze_hotspots():
    dhaka_hotspots = pd.read_csv("data/csv/Final hotspot z score.csv")

    # Correlate Z-score with NDBI
    z_ndbi_corr = dhaka_hotspots["Z-score"].corr(dhaka_hotspots["NDBISAMPLE"])

    print(f"Correlation between Z-score and NDBI: {z_ndbi_corr:.4f}")

    if z_ndbi_corr > 0.70:
        print(
            "Conclusion: The correlation is high (> 0.70). The 'Hotspots' are not random, they are strongly clustered exactly where the buildings are."
        )
    elif z_ndbi_corr > 0.50:
        print(
            "Conclusion: There is a moderate positive correlation. Hotspots show a noticeable link to building density."
        )
    else:
        print(
            "Conclusion: The correlation is weak. Hotspots may be influenced by other factors beyond just building density."
        )

    # Correlation Heatmap ---
    cols_to_correlate = ["Z-score", "Lst_1", "NDVISAMPLE", "NDBISAMPLE"]
    corr_matrix = dhaka_hotspots[cols_to_correlate].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
    )

    plt.title("Correlation Heatmap: Hotspots, Temperature, Greenery, & Buildings")
    plt.tight_layout()
    plt.savefig("data/figures/HotspotCorrelationHeatmap.png", dpi=300)
    plt.show()


def compare_zones():
    dhaka_indices = pd.read_csv("data/csv/Dhakafinal indices.csv")
    print(dhaka_indices[["Lst_1", "NDBISAMPLE", "NDVISAMPLE"]].describe())
    # # remove water bodies with NDBI < 0
    dhaka_indices = dhaka_indices[dhaka_indices["NDBISAMPLE"] >= 0]
    ndbi_threshold = dhaka_indices["NDBISAMPLE"].quantile(0.90)
    ndvi_threshold = dhaka_indices["NDVISAMPLE"].quantile(0.90)
    print(f"High-Density Threshold (NDBI): {ndbi_threshold:.2f}")
    print(f"Green Zone Threshold (NDVI): {ndvi_threshold:.2f}")

    #  High-Density Wards (Concrete Core)
    high_density_df = dhaka_indices[dhaka_indices["NDBISAMPLE"] >= ndbi_threshold]
    # Green Wards (Parks/Forests)
    green_wards_df = dhaka_indices[dhaka_indices["NDVISAMPLE"] >= ndvi_threshold]

    avg_temp_concrete = high_density_df["Lst_1"].mean()
    avg_temp_green = green_wards_df["Lst_1"].mean()
    avg_temp_city = dhaka_indices["Lst_1"].mean()

    summary = pd.DataFrame(
        {
            "Zone Type": [
                "High-Density (Top 10% NDBI)",
                "Green Wards (Top 10% NDVI)",
                "City Average",
            ],
            "Average LST (C)": [avg_temp_concrete, avg_temp_green, avg_temp_city],
        }
    )

    print(summary)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    colors = [
        "#d63031",
        "#27ae60",
        "#7f8c8d",
    ]
    plt.bar(summary["Zone Type"], summary["Average LST (C)"], color=colors)

    plt.title("Urban Heat Island Analysis: Concrete vs. Green Space", fontsize=14)
    plt.ylabel("Average Temperature (LST)", fontsize=12)
    plt.ylim(summary["Average LST (C)"].min() - 2, summary["Average LST (C)"].max() + 2)

    for i, val in enumerate(summary["Average LST (C)"]):
        plt.text(i, val + 0.1, f"{val:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("data/figures/ZoneComparison.png", dpi=300)
    plt.show()


def zonal_statistics():
    dhaka_indices = pd.read_csv("data/csv/Dhakafinal indices.csv")
    df_land = dhaka_indices[dhaka_indices["NDBISAMPLE"] >= 0].copy()
    df_land["Density_Zone"] = pd.qcut(
        df_land["NDBISAMPLE"],
        q=4,
        labels=["Low Density", "Medium Density", "High Density", "Extreme Density"],
    )

    zonal_stats = df_land.groupby("Density_Zone")["Lst_1"].mean().reset_index()
    zonal_stats.columns = ["Urban Density Zone", "Average LST"]
    print("Zonal Statistics Result")
    print(zonal_stats)

    gap = zonal_stats.iloc[3]["Average LST"] - zonal_stats.iloc[0]["Average LST"]
    print(f"\nTemperature Gap (Extreme vs Low): {gap:.2f} degrees")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    colors = [
        "#2ecc71",
        "#f1c40f",
        "#e67e22",
        "#c0392b",
    ]
    plt.bar(zonal_stats["Urban Density Zone"], zonal_stats["Average LST"], color=colors)
    plt.title("Zonal Statistics: Average LST by Urban Density Zone", fontsize=14)
    plt.xlabel("Urban Density Zone", fontsize=12)
    plt.ylabel("Average LST (C)", fontsize=12)
    plt.ylim(zonal_stats["Average LST"].min() - 2, zonal_stats["Average LST"].max() + 2)
    for i, val in enumerate(zonal_stats["Average LST"]):
        plt.text(i, val + 0.1, f"{val:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("data/figures/ZonalStatistics.png", dpi=300)
    plt.show()


def functional_greenery_analysis():
    df = pd.read_csv("data/csv/Dhakafinal indices.csv")
    total_points = len(df)

    green_pixels = df[df["NDVISAMPLE"] > 0.3]
    green_count = len(green_pixels)
    green_percentage = (green_count / total_points) * 100
    deficit_percentage = 100 - green_percentage

    print(f"Total Urban Points Analyzed: {total_points:,}")
    print(f"Points with Functional Greenery (NDVI > 0.3): {green_count:,}")
    print(f"Green Cover: {green_percentage:.2f}%")
    print(f"Greenery Deficit: {deficit_percentage:.2f}%")

    water_pixels = df[df["NDVISAMPLE"] < 0]
    water_count = len(water_pixels)
    water_percentage = (water_count / total_points) * 100
    water_deficit = 100 - water_percentage

    print(f"\nPoints with Water Bodies (NDVI < 0): {water_count:,}")
    print(f"Water Cover: {water_percentage:.2f}%")
    print(f"Water Deficit: {water_deficit:.2f}%")

    grey_pixels = df[(df["NDVISAMPLE"] >= 0) & (df["NDVISAMPLE"] <= 0.29)]
    grey_count = len(grey_pixels)
    grey_percentage = (grey_count / total_points) * 100
    grey_deficit = 100 - grey_percentage

    print(f"\nPoints with Grey Areas (NDVI 0-0.29): {grey_count:,}")
    print(f"Grey Area Cover: {grey_percentage:.2f}%")
    print(f"Grey Area Deficit: {grey_deficit:.2f}%")

    summary_table = pd.DataFrame(
        {
            "Category": [
                "Functional Greenery (NDVI > 0.3)",
                "Water Bodies (NDVI < 0)",
                "Grey Areas (NDVI 0-0.29)",
            ],
            "Count": [green_count, water_count, grey_count],
            "Percentage": [green_percentage, water_percentage, grey_percentage],
            "Deficit": [deficit_percentage, water_deficit, grey_deficit],
        }
    )
    print(summary_table)


def hotspot_correlation_analysis():
    df = pd.read_csv("data/csv/Final hotspot z score.csv")

    trusted_cols = df[["Z-score", "Lst_1", "NDVISAMPLE"]]
    corr_matrix = trusted_cols.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )

    plt.title("Correlation Heatmap: Heat Clusters, Temperature & Greenery", fontsize=14)
    plt.tight_layout()
    plt.savefig("data/figures/HotspotCorrelationHeatmap.png", dpi=300)
    plt.show()


def urban_heat_island_analysis():
    df = pd.read_csv("data/csv/Dhakafinal indices.csv")

    control_zone = df[(df["NDVISAMPLE"] > 0.4)]
    baseline_temp = control_zone["Lst_1"].mean()

    print(f"Baseline Temperature (Rural Control Zone): {baseline_temp:.2f}C")
    impact_zone = df[df["NDVISAMPLE"] < 0.2]
    urban_temp = impact_zone["Lst_1"].mean()

    uhi_intensity = urban_temp - baseline_temp

    hot_pixels = df[df["Lst_1"] > baseline_temp]
    num_hot_pixels = len(hot_pixels)
    area_km2 = (num_hot_pixels * (30**2)) / 1000000

    print(f"Rural Baseline Temp: {baseline_temp:.2f}C")
    print(f"Urban Impact Temp: {urban_temp:.2f}C")

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        control_zone["Lst_1"], label="Rural Control Zone", shade=True, color="green"
    )
    sns.kdeplot(
        impact_zone["Lst_1"], label="Urban Impact Zone", shade=True, color="red"
    )
    plt.axvline(
        baseline_temp, color="black", linestyle="--", label="Rural Baseline Temp"
    )
    plt.title("Temperature Distribution: Control vs Impact Zones", fontsize=14)
    plt.xlabel("Land Surface Temperature (C)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/figures/UHI_Temperature_Distribution.png", dpi=300)
    plt.show()


def rural_area_analysis():
    df = pd.read_csv("data/csv/Dhakafinal indices.csv")

    rural_threshold = df["NDBISAMPLE"].quantile(0.25)
    rural_points = df[df["NDBISAMPLE"] <= rural_threshold]

    avg_rural_ndvi = rural_points["NDVISAMPLE"].mean()
    avg_rural_lst = rural_points["Lst_1"].mean()

    print(f"Average Rural NDVI: {avg_rural_ndvi:.2f}")
    print(f"Average Rural Temperature: {avg_rural_lst:.2f}C")

    if avg_rural_ndvi < 0.2:
        print("PROVEN: This rural area is deforested bare soil.")
    elif avg_rural_ndvi > 0.5:
        print("PROVEN: This rural area is a healthy forest.")


def main():
    analyze_hotspots()
    compare_zones()
    zonal_statistics()
    functional_greenery_analysis()
    hotspot_correlation_analysis()
    urban_heat_island_analysis()
    rural_area_analysis()
