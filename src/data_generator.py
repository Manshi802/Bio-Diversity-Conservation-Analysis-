"""
data_generator.py
-----------------
Generates a realistic synthetic biodiversity dataset (~1500 rows)
representing species observations, environmental parameters, and conservation status.
"""

import os
import numpy as np
import pandas as pd

def generate_species_data(output_path="data/species_data.csv", n_samples=1500, random_seed=42):
    """
    Generates synthetic biodiversity records with logically consistent features.
    
    Parameters:
        output_path (str): Destination path for the CSV file.
        n_samples (int): Total number of records to create.
        random_seed (int): Seed for reproducible data generation.
    
    Returns:
        pd.DataFrame: Generated dataset.
    """
    np.random.seed(random_seed)

    # 1. Species catalog by category
    species_catalog = {
        "Mammal": [
            "Bengal Tiger", "Asian Elephant", "Snow Leopard", "Red Panda", 
            "Indian Rhinoceros", "Lion-tailed Macaque", "Sloth Bear", "Nilgiri Tahr", 
            "Ganges River Dolphin", "Clouded Leopard", "Giant Panda", "Mountain Gorilla",
            "Black Rhino", "Cheetah", "Bornean Orangutan"
        ],
        "Bird": [
            "Great Indian Bustard", "Forest Owlet", "White-bellied Heron", "Spoon-billed Sandpiper",
            "California Condor", "Kakapo", "Philippine Eagle", "Spix's Macaw",
            "Bengal Florican", "Sarus Crane", "Great Hornbill", "Harpy Eagle"
        ],
        "Reptile": [
            "King Cobra", "Gharial", "Komodo Dragon", "Hawksbill Sea Turtle",
            "Leatherback Turtle", "Galapagos Giant Tortoise", "Tuatara", "Orinoco Crocodile"
        ],
        "Amphibian": [
            "Golden Poison Frog", "Axolotl", "Purple Frog", "Chinese Giant Salamander",
            "Panamanian Golden Frog", "Hellbender", "Red-eyed Tree Frog", "Fire Salamander"
        ],
        "Plant": [
            "Rafflesia Arnoldii", "Titan Arum", "Baobab Tree", "Ghost Orchid",
            "Welwitschia Mirabilis", "Dragon Blood Tree", "Venus Flytrap", "Red Sandalwood"
        ]
    }

    # 2. Key biodiversity hotspot regions with baseline coordinates and average temperatures
    region_info = {
        "Western Ghats": {"lat": 10.5, "lon": 76.5, "base_temp": 24.0},
        "Eastern Himalayas": {"lat": 27.5, "lon": 88.5, "base_temp": 14.0},
        "Sundarbans": {"lat": 21.9, "lon": 88.9, "base_temp": 27.0},
        "Amazon Rainforest": {"lat": -3.4, "lon": -62.2, "base_temp": 26.5},
        "Congo Basin": {"lat": -0.7, "lon": 21.5, "base_temp": 25.5},
        "Borneo Rainforest": {"lat": 1.0, "lon": 114.0, "base_temp": 27.0},
        "Madagascar": {"lat": -18.7, "lon": 46.8, "base_temp": 23.0},
        "Serengeti": {"lat": -2.3, "lon": 34.8, "base_temp": 22.5},
        "Pantanal Wetlands": {"lat": -17.5, "lon": -57.0, "base_temp": 26.0},
        "Arctic Tundra": {"lat": 68.0, "lon": -133.0, "base_temp": -6.0}
    }

    regions = list(region_info.keys())
    categories = list(species_catalog.keys())

    records = []

    for _ in range(n_samples):
        # Pick category and species
        category = np.random.choice(categories, p=[0.25, 0.25, 0.18, 0.16, 0.16])
        species_name = np.random.choice(species_catalog[category])
        
        # Pick region and add slight coordinate jitter for geospatial realism
        region = np.random.choice(regions)
        reg_meta = region_info[region]
        latitude = round(reg_meta["lat"] + np.random.uniform(-1.5, 1.5), 4)
        longitude = round(reg_meta["lon"] + np.random.uniform(-1.5, 1.5), 4)

        # Observation year
        year = int(np.random.randint(2015, 2025))

        # Protected Area status (Yes / No)
        protected_area = np.random.choice(["Yes", "No"], p=[0.42, 0.58])

        # Habitat loss percent (5% to 92%)
        # Protected areas generally experience less habitat loss
        if protected_area == "Yes":
            habitat_loss_percent = round(float(np.random.beta(2, 5) * 60 + 5), 1)
        else:
            habitat_loss_percent = round(float(np.random.beta(4, 3) * 75 + 15), 1)
        habitat_loss_percent = min(max(habitat_loss_percent, 5.0), 95.0)

        # Human Activity Index (1.0 to 10.0 scale)
        if protected_area == "Yes":
            human_activity_index = round(float(np.random.uniform(1.2, 5.8)), 1)
        else:
            human_activity_index = round(float(np.random.uniform(4.0, 9.8)), 1)

        # Population count (skewed distribution: 50 to 55,000)
        # Higher habitat loss typically correlates with smaller populations
        pop_base = 50000 * np.exp(-0.035 * habitat_loss_percent)
        pop_variation = np.random.uniform(0.3, 2.2)
        population_count = int(max(40, min(55000, pop_base * pop_variation)))

        # Average temperature with local climate variance
        temp_jitter = np.random.normal(0, 1.8)
        avg_temperature = round(float(reg_meta["base_temp"] + temp_jitter), 1)

        # Logical threat calculation for conservation status assignment
        # Normalizing indicators:
        loss_norm = habitat_loss_percent / 100.0                       # 0.05 - 0.95
        pop_norm = 1.0 - (np.log10(population_count) / 4.75)           # higher when pop is small
        human_norm = human_activity_index / 10.0                      # 0.1 - 1.0
        protection_offset = -0.10 if protected_area == "Yes" else 0.08
        noise = np.random.normal(0, 0.04)

        composite_risk = (0.42 * loss_norm) + (0.38 * pop_norm) + (0.20 * human_norm) + protection_offset + noise

        # Map composite score to standard IUCN Red List categories
        if composite_risk >= 0.68 or population_count < 300:
            conservation_status = "Critically Endangered"
        elif composite_risk >= 0.52 or population_count < 1200:
            conservation_status = "Endangered"
        elif composite_risk >= 0.38:
            conservation_status = "Vulnerable"
        else:
            conservation_status = "Least Concern"

        records.append({
            "species_name": species_name,
            "category": category,
            "region": region,
            "latitude": latitude,
            "longitude": longitude,
            "year": year,
            "population_count": population_count,
            "habitat_loss_percent": habitat_loss_percent,
            "avg_temperature": avg_temperature,
            "human_activity_index": human_activity_index,
            "protected_area": protected_area,
            "conservation_status": conservation_status
        })

    df = pd.DataFrame(records)

    # Ensure output directory exists and save CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at '{output_path}' with {len(df)} rows.")
    return df

if __name__ == "__main__":
    generate_species_data()
