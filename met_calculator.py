#!/usr/bin/env python3
"""
Burn-Ex -- MET Calculator
--------------------------
MET values sourced from the 2011 ACSM Compendium of Physical Activities
(Ainsworth et al., Med. Sci. Sports Exerc., 43(8), 1575-1581, 2011).

Formula:
    kcal/min = MET × 3.5 × weight_kg / 200

This is the standard indirect-calorimetry proxy used in clinical fitness
research. It is NOT an external API — it is a published constant table
baked directly into this file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MET Reference Table — Compendium of Physical Activities 2011
# Code : Activity description : MET
# ---------------------------------------------------------------------------
MET_TABLE: dict[str, float] = {
    # Sedentary / very light
    "idle":            1.3,   # code 01009 - sitting quietly
    "standing":        1.8,   # code 09030 - standing, light activity

    # Ambulation
    "walking":         3.5,   # code 17151 - walking, 3.0 mph, level, moderate pace
    "walking_slow":    2.5,   # code 17160 - walking, 2.0 mph, slow
    "walking_fast":    4.3,   # code 17160 - walking, 3.5 mph, brisk pace

    # Rhythmic / calisthenics
    "jumping_jacks":   8.0,   # code 02050 - calisthenics, vigorous effort
    "squats":          5.0,   # code 02030 - calisthenics, moderate effort (weight-bearing)

    # Running
    "jogging":         7.0,   # code 12050 - running, 5 mph (12 min/mile)
    "running":         9.8,   # code 12070 - running, 6 mph (10 min/mile)

    # Unlabeled — use resting metabolic rate
    "unlabeled":       1.0,
}

# MET values specifically used to label the 5-class dataset
ACTIVITY_METS: dict[str, float] = {
    "idle":            MET_TABLE["idle"],
    "walking":         MET_TABLE["walking"],
    "jogging":         MET_TABLE["jogging"],
    "jumping_jacks":   MET_TABLE["jumping_jacks"],
    "squats":          MET_TABLE["squats"],
    "unlabeled":       MET_TABLE["unlabeled"],
}


def kcal_per_min(met: float, weight_kg: float) -> float:
    """
    Convert a MET value to kilocalories burned per minute.

    Args:
        met:       Metabolic Equivalent of Task value.
        weight_kg: Subject body mass in kilograms.

    Returns:
        Kilocalories burned per minute (float).
    """
    return met * 3.5 * weight_kg / 200.0


def kcal_per_second(met: float, weight_kg: float) -> float:
    """Same as kcal_per_min but per second."""
    return kcal_per_min(met, weight_kg) / 60.0


def label_to_kcal_per_min(label: str, weight_kg: float) -> float:
    """
    Convenience: look up MET for a named activity and return kcal/min.

    Args:
        label:     Activity string (must be a key in ACTIVITY_METS).
        weight_kg: Subject body mass in kilograms.

    Returns:
        Kilocalories burned per minute.

    Raises:
        KeyError: If label is not in ACTIVITY_METS.
    """
    met = ACTIVITY_METS.get(label, ACTIVITY_METS["unlabeled"])
    return kcal_per_min(met, weight_kg)


def get_met(label: str) -> float:
    """Return the MET value for an activity label."""
    return ACTIVITY_METS.get(label, ACTIVITY_METS["unlabeled"])


if __name__ == "__main__":
    weight = 70.0
    print(f"{'Activity':<20} {'MET':>5}  {'kcal/min @ 70 kg':>18}")
    print("-" * 50)
    for activity, met in ACTIVITY_METS.items():
        kcal = kcal_per_min(met, weight)
        print(f"{activity:<20} {met:>5.1f}  {kcal:>18.3f}")
