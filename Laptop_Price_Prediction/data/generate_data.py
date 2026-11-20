"""
============================================================
 Laptop Price Prediction - Synthetic Dataset Generator
============================================================
 This script generates a realistic laptop dataset with
 ~1,200 rows. Prices are computed from specifications so
 that trained models can discover real-looking patterns.
============================================================
"""

import os
import random
import pandas as pd
import numpy as np

# ─── Seed for reproducibility ────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Configuration: possible values for each column ──────

COMPANIES = {
    # company_name: base_price_multiplier
    "Dell": 1.0,
    "HP": 0.95,
    "Lenovo": 0.93,
    "Asus": 0.90,
    "Acer": 0.85,
    "Apple": 1.60,
    "MSI": 1.25,
    "Toshiba": 0.88,
    "Samsung": 1.05,
    "Microsoft": 1.30,
}

TYPE_NAMES = {
    # type: (base_price_addition, weight_range)
    "Notebook": (0, (1.8, 2.5)),
    "Gaming": (15000, (2.5, 3.5)),
    "Ultrabook": (8000, (1.0, 1.6)),
    "2 in 1 Convertible": (5000, (1.2, 1.9)),
    "Workstation": (20000, (2.3, 3.2)),
}

INCHES_OPTIONS = [11.6, 13.3, 14.0, 15.6, 17.3]

SCREEN_RESOLUTIONS = {
    # resolution_string: (x_res, y_res, price_addition)
    "1366x768": (1366, 768, 0),
    "Full HD 1920x1080": (1920, 1080, 3000),
    "IPS Panel 1920x1080": (1920, 1080, 4000),
    "IPS Panel Full HD 1920x1080": (1920, 1080, 4500),
    "Full HD 2560x1440": (2560, 1440, 8000),
    "IPS Panel Quad HD+ 2560x1440": (2560, 1440, 10000),
    "4K Ultra HD 3840x2160": (3840, 2160, 18000),
    "IPS Panel 4K Ultra HD 3840x2160": (3840, 2160, 20000),
    "IPS Panel Retina Display 2560x1600": (2560, 1600, 12000),
}

TOUCHSCREEN_PROBABILITY = 0.20  # 20% of laptops have touchscreen

CPUS = {
    # cpu_string: price_addition
    "Intel Core i3 6006U 2.0GHz": 0,
    "Intel Core i3 7100U 2.4GHz": 2000,
    "Intel Core i3 8130U 2.2GHz": 3000,
    "Intel Core i5 7200U 2.5GHz": 8000,
    "Intel Core i5 8250U 1.6GHz": 10000,
    "Intel Core i5 8300H 2.3GHz": 12000,
    "Intel Core i7 7500U 2.7GHz": 16000,
    "Intel Core i7 7700HQ 2.8GHz": 20000,
    "Intel Core i7 8550U 1.8GHz": 22000,
    "Intel Core i7 8750H 2.2GHz": 25000,
    "AMD A6-Series 9220 2.5GHz": -2000,
    "AMD A9-Series 9420 3.0GHz": 0,
    "AMD Ryzen 5 2500U 2.0GHz": 10000,
    "AMD Ryzen 7 2700U 2.2GHz": 18000,
}

RAM_OPTIONS = {
    # ram_string: price_addition
    "2GB": -8000,
    "4GB": 0,
    "8GB": 5000,
    "16GB": 15000,
    "32GB": 30000,
    "64GB": 55000,
}

MEMORY_OPTIONS = {
    # storage_string: price_addition
    "128GB SSD": 2000,
    "256GB SSD": 5000,
    "512GB SSD": 10000,
    "1TB SSD": 20000,
    "500GB HDD": 0,
    "1TB HDD": 1000,
    "2TB HDD": 3000,
    "128GB SSD + 500GB HDD": 4000,
    "128GB SSD + 1TB HDD": 5000,
    "256GB SSD + 500GB HDD": 7000,
    "256GB SSD + 1TB HDD": 8000,
    "512GB SSD + 1TB HDD": 14000,
    "256GB SSD + 2TB HDD": 10000,
}

GPUS = {
    # gpu_string: price_addition
    "Intel HD Graphics 520": 0,
    "Intel HD Graphics 620": 0,
    "Intel UHD Graphics 620": 500,
    "Intel Iris Plus Graphics 640": 2000,
    "AMD Radeon R5": 1000,
    "AMD Radeon R5 M430": 2000,
    "AMD Radeon RX 560": 8000,
    "AMD Radeon RX 580": 12000,
    "Nvidia GeForce 940MX": 5000,
    "Nvidia GeForce MX150": 6000,
    "Nvidia GeForce GTX 1050": 10000,
    "Nvidia GeForce GTX 1050 Ti": 13000,
    "Nvidia GeForce GTX 1060": 18000,
    "Nvidia GeForce GTX 1070": 25000,
    "Nvidia GeForce GTX 1080": 35000,
}

OPSYS_OPTIONS = {
    # os_name: price_addition
    "Windows 10": 5000,
    "Windows 10 S": 3000,
    "Windows 7": 2000,
    "macOS": 0,         # already accounted in Apple premium
    "Mac OS X": 0,
    "Linux": 0,
    "Chrome OS": 0,
    "No OS": -3000,
    "Android": -5000,
}

# ─── Data generation function ────────────────────────────

def generate_laptop_dataset(n_samples: int = 1200) -> pd.DataFrame:
    """
    Generate a synthetic but realistic laptop dataset.

    Each laptop's price is computed from its specifications
    plus random noise, so ML models can learn actual patterns.

    Parameters
    ----------
    n_samples : int
        Number of laptop records to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching the Kaggle Laptop
        Price dataset format.
    """
    records = []
    base_price = 25000  # base price in INR for the cheapest config

    for _ in range(n_samples):
        # --- Choose specifications randomly (with some logic) ---

        # Company
        company = random.choice(list(COMPANIES.keys()))
        company_mult = COMPANIES[company]

        # Type
        type_name = random.choice(list(TYPE_NAMES.keys()))
        type_addition, weight_range = TYPE_NAMES[type_name]

        # Apple only gets macOS, others rarely get macOS
        if company == "Apple":
            opsys = random.choice(["macOS", "Mac OS X"])
        else:
            opsys_choices = [k for k in OPSYS_OPTIONS if k not in ("macOS", "Mac OS X")]
            opsys = random.choice(opsys_choices)
        opsys_addition = OPSYS_OPTIONS[opsys]

        # Screen size
        inches = random.choice(INCHES_OPTIONS)

        # Screen resolution
        res_key = random.choice(list(SCREEN_RESOLUTIONS.keys()))
        res_addition = SCREEN_RESOLUTIONS[res_key][2]

        # Touchscreen
        has_touchscreen = random.random() < TOUCHSCREEN_PROBABILITY
        if has_touchscreen:
            screen_res_str = f"Touchscreen {res_key}"
            touch_addition = 4000
        else:
            screen_res_str = res_key
            touch_addition = 0

        # CPU
        cpu = random.choice(list(CPUS.keys()))
        cpu_addition = CPUS[cpu]

        # RAM
        ram = random.choice(list(RAM_OPTIONS.keys()))
        ram_addition = RAM_OPTIONS[ram]

        # Storage
        memory = random.choice(list(MEMORY_OPTIONS.keys()))
        memory_addition = MEMORY_OPTIONS[memory]

        # GPU – gaming laptops more likely to have dedicated GPUs
        if type_name == "Gaming":
            gpu_choices = [g for g in GPUS if "Nvidia" in g or "AMD Radeon RX" in g]
        else:
            gpu_choices = list(GPUS.keys())
        gpu = random.choice(gpu_choices)
        gpu_addition = GPUS[gpu]

        # Weight (kg)
        weight = round(random.uniform(*weight_range), 2)

        # --- Calculate price ---
        raw_price = base_price + type_addition + cpu_addition + ram_addition
        raw_price += memory_addition + gpu_addition + res_addition
        raw_price += opsys_addition + touch_addition
        raw_price = raw_price * company_mult

        # Add realistic noise (±10%)
        noise_factor = np.random.normal(1.0, 0.10)
        price = int(max(9999, raw_price * noise_factor))

        # Round price to nearest 100
        price = int(round(price / 100) * 100)

        records.append({
            "Company": company,
            "TypeName": type_name,
            "Inches": inches,
            "ScreenResolution": screen_res_str,
            "Cpu": cpu,
            "Ram": ram,
            "Memory": memory,
            "Gpu": gpu,
            "OpSys": opsys,
            "Weight": f"{weight}kg",
            "Price": price,
        })

    return pd.DataFrame(records)


# ─── Main execution ─────────────────────────────────────

if __name__ == "__main__":
    print("🔧 Generating synthetic laptop dataset...")

    df = generate_laptop_dataset(n_samples=1200)

    # Quick summary
    print(f"   ✅ Generated {len(df)} records")
    print(f"   📊 Columns: {list(df.columns)}")
    print(f"   💰 Price range: ₹{df['Price'].min():,} – ₹{df['Price'].max():,}")
    print(f"   💰 Mean price : ₹{df['Price'].mean():,.0f}")
    print()

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "laptop_data.csv")
    df.to_csv(output_path, index=False)
    print(f"   💾 Saved to: {output_path}")

    # Print first few rows
    print()
    print(df.head(10).to_string(index=False))
