"""
insights.py
-----------
Generates rule-based conservation recommendations and analyzes at-risk geographic regions.
"""

import pandas as pd
import numpy as np

def generate_conservation_recommendations(habitat_loss_percent, population_count, 
                                          human_activity_index, protected_area, 
                                          avg_temperature=None, predicted_status=None):
    """
    Produces actionable, rule-based conservation recommendations based on environmental
    metrics and predicted species conservation status.
    
    Returns:
        list of dict: List containing alert level, category, and recommendation text.
    """
    recommendations = []

    # 1. Overall Status Summary
    if predicted_status in ["Critically Endangered", "Endangered"]:
        recommendations.append({
            "level": "CRITICAL",
            "icon": "🚨",
            "domain": "Priority Status",
            "action": f"Species is evaluated as **{predicted_status}**. Immediate multi-stakeholder emergency intervention required under international biodiversity frameworks (e.g. CITES Appendix I)."
        })
    elif predicted_status == "Vulnerable":
        recommendations.append({
            "level": "WARNING",
            "icon": "⚠️",
            "domain": "Priority Status",
            "action": f"Species is classified as **{predicted_status}**. Proactive protective guidelines must be enforced to halt progression toward endangered status."
        })
    else:
        recommendations.append({
            "level": "INFO",
            "icon": "✅",
            "domain": "Priority Status",
            "action": "Species is currently classified as **Least Concern**. Continue routine environmental auditing to maintain ecological equilibrium."
        })

    # 2. Habitat Loss Rules
    if habitat_loss_percent >= 60.0:
        recommendations.append({
            "level": "CRITICAL",
            "icon": "🌲",
            "domain": "Habitat Restoration",
            "action": f"**Severe Habitat Loss ({habitat_loss_percent}%)**: Establish immediate logging moratoriums, launch active native tree reforestation, and reconnect isolated canopy corridors."
        })
    elif habitat_loss_percent >= 35.0:
        recommendations.append({
            "level": "WARNING",
            "icon": "🌿",
            "domain": "Habitat Management",
            "action": f"**Moderate Habitat Fragmentation ({habitat_loss_percent}%)**: Designate 1-2 km ecological buffer perimeters around key core nesting and foraging ranges."
        })
    else:
        recommendations.append({
            "level": "INFO",
            "icon": "🌱",
            "domain": "Habitat Management",
            "action": f"**Stable Vegetation Cover ({habitat_loss_percent}% loss)**: Sustain existing canopy density and monitor for localized encroaching development."
        })

    # 3. Population Count Rules
    if population_count <= 500:
        recommendations.append({
            "level": "CRITICAL",
            "icon": "🐾",
            "domain": "Species Recovery",
            "action": f"**Critically Depleted Population ({population_count:,} individuals)**: Implement ex-situ conservation, bio-banking/genetic banking, and 24/7 dedicated anti-poaching ranger squads."
        })
    elif population_count <= 2500:
        recommendations.append({
            "level": "WARNING",
            "icon": "🔍",
            "domain": "Population Monitoring",
            "action": f"**Vulnerable Demographics ({population_count:,} individuals)**: Deploy solar-powered camera traps, acoustic bio-sensors, and seasonal mark-recapture surveys."
        })
    else:
        recommendations.append({
            "level": "INFO",
            "icon": "📊",
            "domain": "Population Monitoring",
            "action": f"**Viable Population Size ({population_count:,} individuals)**: Conduct biennial census assessments to ensure sustainable reproduction rates."
        })

    # 4. Human Activity Index Rules
    if human_activity_index >= 7.0:
        recommendations.append({
            "level": "CRITICAL",
            "icon": "🚧",
            "domain": "Human-Wildlife Conflict",
            "action": f"**Severe Anthropogenic Pressure ({human_activity_index}/10)**: Construct wildlife overpasses/underpasses on transit arteries, regulate agrochemical runoff, and establish community compensation schemes."
        })
    elif human_activity_index >= 4.0:
        recommendations.append({
            "level": "WARNING",
            "icon": "👥",
            "domain": "Community Engagement",
            "action": f"**Moderate Human Interference ({human_activity_index}/10)**: Support indigenous forest councils and promote eco-tourism benefit-sharing initiatives."
        })

    # 5. Protected Area Status Rules
    is_protected = str(protected_area).strip().lower() in ["yes", "1", "true"]
    if not is_protected:
        recommendations.append({
            "level": "CRITICAL",
            "icon": "🛡️",
            "domain": "Legal Protection",
            "action": "**Unprotected Habitat**: Advocate for formal gazettement under national wildlife protection statutes (e.g. Wildlife Sanctuary or Biosphere Reserve)."
        })
    else:
        recommendations.append({
            "level": "INFO",
            "icon": "🏰",
            "domain": "Protected Sanctuary",
            "action": "**Protected Status Confirmed**: Maintain enforcement of sanctuary boundaries and upgrade surveillance drones along boundary fringes."
        })

    # 6. Temperature Stress
    if avg_temperature is not None and avg_temperature >= 32.0:
        recommendations.append({
            "level": "WARNING",
            "icon": "🌡️",
            "domain": "Thermal Adaptation",
            "action": f"**Elevated Thermal Stress ({avg_temperature}°C)**: Establish perennial shade covers and artificial perennial wetlands to alleviate thermal vulnerability."
        })

    return recommendations

def get_top_at_risk_regions(df, top_n=10):
    """
    Computes conservation risk metrics for each region and returns the top N most vulnerable regions.
    """
    if "region" not in df.columns:
        return pd.DataFrame()

    # Define high threat categories
    threat_statuses = ["Endangered", "Critically Endangered"]

    grouped = df.groupby("region").agg(
        total_species=("species_name", "count"),
        at_risk_species=("conservation_status", lambda x: (x.isin(threat_statuses)).sum()),
        critically_endangered_count=("conservation_status", lambda x: (x == "Critically Endangered").sum()),
        avg_habitat_loss=("habitat_loss_percent", "mean"),
        avg_human_activity=("human_activity_index", "mean"),
        protected_ratio=("protected_area", lambda x: (x == "Yes").mean() * 100)
    ).reset_index()

    # Calculate at risk percentage
    grouped["at_risk_rate_pct"] = (grouped["at_risk_species"] / grouped["total_species"]) * 100

    # Composite Risk Index (0 to 100)
    # Higher rate of endangered species + higher habitat loss + higher human disturbance - protected area buffer
    grouped["composite_risk_score"] = (
        (grouped["at_risk_rate_pct"] * 0.45) +
        (grouped["avg_habitat_loss"] * 0.35) +
        (grouped["avg_human_activity"] * 10.0 * 0.20)
    )

    # Sort descending by risk score
    ranked = grouped.sort_values(by="composite_risk_score", ascending=False).head(top_n).reset_index(drop=True)

    # Format numbers for clean presentation
    ranked["avg_habitat_loss"] = ranked["avg_habitat_loss"].round(1)
    ranked["avg_human_activity"] = ranked["avg_human_activity"].round(1)
    ranked["at_risk_rate_pct"] = ranked["at_risk_rate_pct"].round(1)
    ranked["protected_ratio"] = ranked["protected_ratio"].round(1)
    ranked["composite_risk_score"] = ranked["composite_risk_score"].round(1)

    ranked.rename(columns={
        "region": "Region",
        "total_species": "Total Species",
        "at_risk_species": "At-Risk Species",
        "critically_endangered_count": "Critically Endangered",
        "at_risk_rate_pct": "At-Risk Species Rate (%)",
        "avg_habitat_loss": "Avg Habitat Loss (%)",
        "avg_human_activity": "Avg Human Disturbance (1-10)",
        "protected_ratio": "Protected Areas (%)",
        "composite_risk_score": "Composite Risk Score"
    }, inplace=True)

    return ranked
