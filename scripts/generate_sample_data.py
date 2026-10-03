"""
scripts/generate_sample_data.py
Generates realistic, intentionally messy retail CSVs to demo the pipeline:
- sales_transactions.csv: nulls, duplicate rows, inconsistent date formats,
  mixed-case/whitespace text, currency symbols in numeric fields, mixed
  data types in the same column, inconsistent store naming.
- stores.csv: a second file to exercise join-key discovery (store_id <-> StoreID).
- products.csv: a third file with category info for the Gold layer.

Run: python scripts/generate_sample_data.py
"""
import os
import random
from datetime import datetime, timedelta

import pandas as pd

random.seed(42)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Grocery", "Toys", "Beauty"]
PRODUCTS = {
    "Electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Phone Charger", "Smart Watch", "USB Cable"],
    "Apparel": ["Cotton T-Shirt", "Denim Jeans", "Running Shoes", "Winter Jacket", "Wool Socks"],
    "Home & Kitchen": ["Non-stick Pan", "Coffee Maker", "Throw Pillow", "LED Lamp", "Cutting Board"],
    "Grocery": ["Organic Coffee", "Olive Oil", "Pasta Pack", "Granola Bars", "Green Tea"],
    "Toys": ["Building Blocks", "Puzzle Set", "RC Car", "Board Game", "Stuffed Bear"],
    "Beauty": ["Face Moisturizer", "Shampoo", "Lip Balm", "Sunscreen SPF50", "Hand Cream"],
}
STORES = [
    ("S001", "Downtown Flagship", "New York"),
    ("S002", "Westfield Mall", "Los Angeles"),
    ("S003", "Riverside Plaza", "Chicago"),
    ("S004", "Harbor Point", "Houston"),
    ("S005", "Sunset Center", "Phoenix"),
]

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d"]


def messy_date(d: datetime) -> str:
    fmt = random.choice(DATE_FORMATS)
    return d.strftime(fmt)


def messy_store_name(store_id, name):
    variants = [store_id, name, name.upper(), name.lower(), f" {name} ", store_id.lower()]
    return random.choice(variants)


def messy_price(price: float) -> str:
    variants = [
        f"{price:.2f}", f"${price:.2f}", f"{price:.2f} USD", f"{price:.0f}",
    ]
    return random.choice(variants)


def gen_sales_transactions(n=500):
    rows = []
    start = datetime(2024, 1, 1)
    for i in range(n):
        cat = random.choice(CATEGORIES)
        product = random.choice(PRODUCTS[cat])
        store_id, store_name, city = random.choice(STORES)
        qty = random.randint(1, 8)
        unit_price = round(random.uniform(5, 250), 2)
        d = start + timedelta(days=random.randint(0, 300))
        row = {
            "TransactionID": f"TXN{10000+i}",
            "TransactionDate": messy_date(d),
            "StoreID": messy_store_name(store_id, store_name),
            "ProductName": random.choice([product, product.upper(), f"  {product}  ", product.lower()]),
            "Category": cat,
            "Quantity": qty if random.random() > 0.03 else None,  # occasional nulls
            "UnitPrice": messy_price(unit_price),
            "CustomerEmail": random.choice([
                f"customer{i}@example.com", f"CUSTOMER{i}@EXAMPLE.COM", None, f"  customer{i}@example.com  "
            ]),
            "PaymentMethod": random.choice(["Credit Card", "credit card", "CASH", "cash", "Debit Card", None]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # inject exact duplicate rows
    dupes = df.sample(frac=0.04, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)
    # inject a few completely blank rows
    for _ in range(3):
        df.loc[len(df)] = [None] * len(df.columns)
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle
    df.to_csv(os.path.join(OUT_DIR, "sales_transactions.csv"), index=False)
    print(f"wrote sales_transactions.csv ({len(df)} rows)")


def gen_stores():
    rows = []
    for store_id, name, city in STORES:
        rows.append({
            "store_id": store_id,
            "store_name": name,
            "city": city,
            "region": random.choice(["East", "West", "Central"]),
            "opened_year": random.randint(2005, 2022),
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "stores.csv"), index=False)
    print("wrote stores.csv")


def gen_products():
    rows = []
    pid = 1
    for cat, items in PRODUCTS.items():
        for item in items:
            rows.append({
                "product_id": f"P{pid:03d}",
                "product_name": item,
                "category": cat,
                "supplier": random.choice(["Acme Co", "Globex Inc", "Initech", "Umbrella Supplies"]),
            })
            pid += 1
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "products.csv"), index=False)
    print("wrote products.csv")


if __name__ == "__main__":
    gen_sales_transactions()
    gen_stores()
    gen_products()
    print("Done. Files in:", os.path.abspath(OUT_DIR))
