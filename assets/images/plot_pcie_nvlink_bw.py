import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Data: Year, Version, Bandwidth (GB/s)
pcie_data = [
    (2003, "PCIe 1.0", 4.0),
    (2007, "PCIe 2.0", 8.0),
    (2010, "PCIe 3.0", 15.8),
    (2017, "PCIe 4.0", 31.5),
    (2019, "PCIe 5.0", 63.0),
    (2021, "PCIe 6.0", 121.0),
]

nvlink_data = [
    (2014, "NVLink 1.0", 160),
    (2017, "NVLink 2.0", 300),
    (2020, "NVLink 3.0", 600),
    (2024, "NVLink 4.0", 900),
    (2025, "NVLink 5.0", 1800),  # Projected
]

# Create DataFrames
df_pcie = pd.DataFrame(pcie_data, columns=["Year", "Version", "Bandwidth_GBs"])
df_nvlink = pd.DataFrame(nvlink_data, columns=["Year", "Version", "Bandwidth_GBs"])

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 7))

# Plot PCIe
sns.lineplot(data=df_pcie, x="Year", y="Bandwidth_GBs", marker="o", label="PCIe", color="royalblue")
for i, row in df_pcie.iterrows():
    plt.text(row["Year"], row["Bandwidth_GBs"] + 5, row["Version"], horizontalalignment="center", color="royalblue")

# Plot NVLink
sns.lineplot(data=df_nvlink, x="Year", y="Bandwidth_GBs", marker="o", label="NVLink", color="tomato")
for i, row in df_nvlink.iterrows():
    plt.text(row["Year"], row["Bandwidth_GBs"] + 20, row["Version"], horizontalalignment="center", color="tomato")

plt.title("Growth of PCIe and NVLink Bandwidth Over Time", fontsize=16, weight="bold")
plt.ylabel("Bandwidth (GB/s)")
plt.xlabel("Year")

plt.ticklabel_format(style='plain', axis='y')  # Disable scientific notation on y-axis

plt.legend(title="Interconnect", fontsize=12)
plt.tight_layout()

# Save plot and print absolute path
output_path = os.path.abspath("pcie_nvlink_bandwidth_growth.svg")
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
plt.close()

print(output_path)
