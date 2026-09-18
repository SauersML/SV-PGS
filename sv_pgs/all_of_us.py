from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

if TYPE_CHECKING:
    from google.cloud import bigquery

MIN_DISEASE_OCCURRENCES = 2
LOGGER = logging.getLogger(__name__)
OMOP_CATEGORICAL_COVARIATES = (
    "gender_concept_id",
    "race_concept_id",
    "ethnicity_concept_id",
)
# Quantitative traits adjust for biological sex (AoU person.sex_at_birth_concept_id),
# which drives lab physiology, instead of gender identity.
MEASUREMENT_CATEGORICAL_COVARIATES = (
    "sex_at_birth_concept_id",
    "race_concept_id",
    "ethnicity_concept_id",
)


@dataclass(frozen=True, slots=True)
class DiseaseDefinition:
    canonical_name: str
    aliases: tuple[str, ...]
    description: str
    snomed_code: str
    snomed_concept_name: str


# Top-20 chronic disease phenotypes. Each is rooted at a single SNOMED CT
# disease concept; the OMOP concept_id is resolved at query time via the
# `concept` table (vocabulary_id='SNOMED', standard_concept='S'), and the
# `concept_ancestor` join expands to every descendant disorder. This mirrors
# the OHDSI Phenotype Library's canonical-cohort logic.
DISEASE_DEFINITIONS: tuple[DiseaseDefinition, ...] = (
    DiseaseDefinition(
        canonical_name="hypertension",
        aliases=("htn", "high_blood_pressure", "high blood pressure", "essential_hypertension"),
        description="Essential hypertension phenotype from EHR conditions.",
        snomed_code="59621000",
        snomed_concept_name="Essential hypertension",
    ),
    DiseaseDefinition(
        canonical_name="type2_diabetes",
        aliases=("t2d", "type_2_diabetes", "type 2 diabetes", "diabetes_type_2", "t2dm"),
        description="Type 2 diabetes mellitus phenotype from EHR conditions.",
        snomed_code="44054006",
        snomed_concept_name="Diabetes mellitus type 2",
    ),
    DiseaseDefinition(
        canonical_name="hyperlipidemia",
        aliases=("dyslipidemia", "high_cholesterol", "hypercholesterolemia"),
        description="Hyperlipidemia phenotype from EHR conditions.",
        snomed_code="55822004",
        snomed_concept_name="Hyperlipidemia",
    ),
    DiseaseDefinition(
        canonical_name="obesity",
        aliases=("obese", "morbid_obesity"),
        description="Obesity phenotype from EHR conditions.",
        snomed_code="414916001",
        snomed_concept_name="Obesity",
    ),
    DiseaseDefinition(
        canonical_name="asthma",
        aliases=("asthma_chronic",),
        description="Asthma phenotype from EHR conditions.",
        snomed_code="195967001",
        snomed_concept_name="Asthma",
    ),
    DiseaseDefinition(
        canonical_name="depression",
        aliases=("major_depression", "major depressive disorder", "mdd"),
        description="Major depressive disorder phenotype from EHR conditions.",
        snomed_code="370143000",
        snomed_concept_name="Major depressive disorder",
    ),
    DiseaseDefinition(
        canonical_name="anxiety",
        aliases=("anxiety_disorder", "generalized_anxiety"),
        description="Anxiety phenotype from EHR conditions.",
        snomed_code="48694002",
        snomed_concept_name="Anxiety",
    ),
    DiseaseDefinition(
        canonical_name="gerd",
        aliases=("reflux", "gastroesophageal_reflux", "gastroesophageal reflux disease"),
        description="Gastroesophageal reflux disease phenotype from EHR conditions.",
        snomed_code="235595009",
        snomed_concept_name="Gastroesophageal reflux disease",
    ),
    DiseaseDefinition(
        canonical_name="chronic_kidney_disease",
        aliases=("ckd", "chronic kidney disease"),
        description="Chronic kidney disease phenotype from EHR conditions.",
        snomed_code="709044004",
        snomed_concept_name="Chronic kidney disease",
    ),
    DiseaseDefinition(
        canonical_name="coronary_artery_disease",
        aliases=("cad", "coronary artery disease", "ischemic_heart_disease", "coronary_arteriosclerosis"),
        description="Coronary artery disease / coronary arteriosclerosis phenotype from EHR conditions.",
        snomed_code="53741008",
        snomed_concept_name="Coronary arteriosclerosis",
    ),
    DiseaseDefinition(
        canonical_name="heart_failure",
        aliases=("hf", "congestive_heart_failure", "chf", "heart failure"),
        description="Heart failure phenotype from EHR conditions.",
        snomed_code="84114007",
        snomed_concept_name="Heart failure",
    ),
    DiseaseDefinition(
        canonical_name="atrial_fibrillation",
        aliases=("af", "afib", "a_fib", "atrial fibrillation"),
        description="Atrial fibrillation phenotype from EHR conditions.",
        snomed_code="49436004",
        snomed_concept_name="Atrial fibrillation",
    ),
    DiseaseDefinition(
        canonical_name="osteoarthritis",
        aliases=("oa", "degenerative_joint_disease"),
        description="Osteoarthritis phenotype from EHR conditions.",
        snomed_code="396275006",
        snomed_concept_name="Osteoarthritis",
    ),
    DiseaseDefinition(
        canonical_name="copd",
        aliases=("chronic_obstructive_pulmonary_disease", "chronic obstructive pulmonary disease", "chronic_obstructive_lung_disease"),
        description="Chronic obstructive lung disease phenotype from EHR conditions.",
        snomed_code="13645005",
        snomed_concept_name="Chronic obstructive lung disease",
    ),
    DiseaseDefinition(
        canonical_name="stroke",
        aliases=("cerebrovascular_accident", "cva", "cerebrovascular_disease"),
        description="Cerebrovascular accident (stroke) phenotype from EHR conditions.",
        snomed_code="230690007",
        snomed_concept_name="Cerebrovascular accident",
    ),
    DiseaseDefinition(
        canonical_name="hypothyroidism",
        aliases=("underactive_thyroid", "low_thyroid"),
        description="Hypothyroidism phenotype from EHR conditions.",
        snomed_code="40930008",
        snomed_concept_name="Hypothyroidism",
    ),
    DiseaseDefinition(
        canonical_name="migraine",
        aliases=("migraine_headache", "migraines"),
        description="Migraine phenotype from EHR conditions.",
        snomed_code="37796009",
        snomed_concept_name="Migraine",
    ),
    DiseaseDefinition(
        canonical_name="rheumatoid_arthritis",
        aliases=("ra",),
        description="Rheumatoid arthritis phenotype from EHR conditions.",
        snomed_code="69896004",
        snomed_concept_name="Rheumatoid arthritis",
    ),
    DiseaseDefinition(
        canonical_name="atherosclerosis",
        aliases=("ascvd", "atherosclerotic_disease"),
        description="Atherosclerosis phenotype from EHR conditions.",
        snomed_code="38716007",
        snomed_concept_name="Atherosclerosis",
    ),
    DiseaseDefinition(
        canonical_name="sleep_apnea",
        aliases=("osa", "obstructive_sleep_apnea", "sleep apnea"),
        description="Sleep apnea phenotype from EHR conditions.",
        snomed_code="73430006",
        snomed_concept_name="Sleep apnea",
    ),
)


# ---------------------------------------------------------------------------
# Quantitative traits from the OMOP `measurement` table
# ---------------------------------------------------------------------------
#
# One row of `measurement` is one reported value. A trait is built from them
# in two stages:
#
# 1. BigQuery (build_all_of_us_measurement_sql), per row: keep standard LOINC
#    concepts of the analyte; convert the UCUM unit to the canonical unit;
#    drop censored results ("<5"), unrecognized units, physiologically
#    implausible values, self-reported values, values taken before age 18,
#    values from inpatient/emergency visits (acute illness) and values inside
#    a pregnancy window; flag values taken on or after the person's first
#    exposure to the trait's medication class. Rows are then collapsed to one
#    value per person-day (same-day repeats are one occasion, as for disease
#    codes) on the analysis scale, and each person is reduced to exact
#    sufficient statistics (occasion count, mean, variance, mean age, mean
#    squared age), separately for untreated and treated occasions.
# 2. Python (build_all_of_us_measurement_targets), per person: use untreated
#    occasions when there are any, else treated occasions corrected by the
#    trait's medication convention (or none, when no validated correction
#    exists); then the target is the empirical BLUP of the person's long-run
#    mean under the random-intercept model
#        y_ij = x_i'gamma + b_i + e_ij,  b_i ~ (0, sigma_b^2), e_ij ~ (0, sigma_e^2),
#    with x_i = (1, mean age, mean squared age, sex); see
#    _estimate_person_variance_components and _person_blup.

# UCUM codes are OMOP UCUM concept_codes (vocabulary_id='UCUM'), e.g. U/L is
# '[U]/L' and uIU/mL is '10*-6.[iU]/mL' there.
ADULT_AGE_YEARS = 18
# A pregnancy record dated d can fall anywhere between conception and term, so
# the pregnancy spans at most 42 weeks (294 days) before or after d; 12 more
# weeks cover postpartum recovery of weight, lipids, hemoglobin and blood
# pressure.
PREGNANCY_WINDOW_DAYS_BEFORE = 294
PREGNANCY_WINDOW_DAYS_AFTER = 294 + 84
# Condition/observation evidence of a current or recent pregnancy: every
# descendant of "Pregnancy, childbirth and puerperium finding" (O-chapter,
# Z33.1, Z3A, postpartum and delivery findings) and "Routine antenatal care"
# (Z34, observation domain) ...
PREGNANCY_SNOMED_CODES = ("248982007", "134435003")
# ... except the descendants that do not imply one: "Not pregnant", parity and
# gravidity findings (e.g. "Gravida 0"), "Previous pregnancies" and "Birth
# details" (the patient's own birth).
NON_PREGNANCY_SNOMED_CODES = ("60001007", "118212000", "366321006", "127362006", "366343003")
# OMOP Visit vocabulary codes of Inpatient, Emergency Room, and Emergency Room
# and Inpatient visits (ERIP is not a descendant of the other two).
ACUTE_CARE_VISIT_CODES = ("IP", "ER", "ERIP")
# measurement_type_concept_id "Patient self-report" (Type Concept OMOP4976938).
SELF_REPORT_TYPE_CONCEPT_ID = 32865
# SNOMED concept_code of the '=' concept in the Meas Value Operator domain;
# any other operator ('<', '<=', '>=', '>') marks a censored result.
EQUALS_OPERATOR_SNOMED_CODE = "276136004"
# Row exclusion reasons in the order the SQL CASE assigns them.
MEASUREMENT_EXCLUSION_REASONS = (
    "censored",
    "unrecognized_unit",
    "implausible",
    "self_reported",
    "under_adult_age",
    "acute_care",
    "pregnancy",
    "sex_unknown",
)
MEASUREMENT_VALUE_FORMULAS = ("identity", "ckd_epi_2021")


@dataclass(frozen=True, slots=True)
class UnitConversion:
    """``canonical value = scale * reported value + offset`` for one UCUM unit."""

    ucum_code: str
    scale: float
    offset: float = 0.0


@dataclass(frozen=True, slots=True)
class MedicationClass:
    """Drugs whose exposure marks a measurement as treated.

    ATC class codes are expanded through concept_ancestor to every RxNorm and
    RxNorm Extension drug in drug_exposure.
    """

    name: str
    atc_codes: tuple[str, ...]


class TreatmentCorrection(Enum):
    # Untreated-equivalent value = measured value / amount.
    DIVIDE = "divide"
    # Untreated-equivalent value = measured value + amount.
    ADD = "add"
    # No validated correction exists: treated values are never used.
    EXCLUDE = "exclude"


@dataclass(frozen=True, slots=True)
class TreatmentRule:
    medication: MedicationClass
    correction: TreatmentCorrection
    amount: float | None
    citation: str

    def __post_init__(self) -> None:
        if (self.amount is None) != (self.correction is TreatmentCorrection.EXCLUDE):
            raise ValueError("a treatment correction needs an amount exactly when it is not EXCLUDE")


@dataclass(frozen=True, slots=True)
class MeasurementDefinition:
    canonical_name: str
    aliases: tuple[str, ...]
    description: str
    loinc_codes: tuple[str, ...]
    canonical_unit: str
    unit_conversions: tuple[UnitConversion, ...]
    # Inclusive bounds on the measured value in the canonical unit; values
    # outside are transcription errors or physiologically impossible.
    plausible_range: tuple[float, float]
    # Summarize and model log(value + log_offset) instead of the value.
    log_scale: bool
    # Named SV/TR biology (or its absence) behind the trait's inclusion.
    rationale: str
    treatment: TreatmentRule | None = None
    log_offset: float = 0.0
    # "identity" models the measured analyte; "ckd_epi_2021" models the
    # race-free CKD-EPI 2021 eGFR computed per measurement from creatinine.
    value_formula: str = "identity"

    def __post_init__(self) -> None:
        if not self.loinc_codes or len(set(self.loinc_codes)) != len(self.loinc_codes):
            raise ValueError(f"{self.canonical_name}: LOINC codes must be non-empty and unique")
        unit_codes = [conversion.ucum_code for conversion in self.unit_conversions]
        if len(set(unit_codes)) != len(unit_codes):
            raise ValueError(f"{self.canonical_name}: duplicate unit conversion")
        if UnitConversion(self.canonical_unit, 1.0) not in self.unit_conversions:
            raise ValueError(f"{self.canonical_name}: canonical unit must convert with scale 1, offset 0")
        low, high = self.plausible_range
        if not low < high:
            raise ValueError(f"{self.canonical_name}: empty plausible range")
        if self.value_formula not in MEASUREMENT_VALUE_FORMULAS:
            raise ValueError(f"{self.canonical_name}: unknown value formula {self.value_formula!r}")
        if self.log_scale and not low + self.log_offset > 0.0:
            raise ValueError(f"{self.canonical_name}: log scale needs plausible_range[0] + log_offset > 0")
        if not self.log_scale and self.log_offset != 0.0:
            raise ValueError(f"{self.canonical_name}: log_offset without log scale")
        if self.treatment is not None:
            if self.treatment.correction is TreatmentCorrection.ADD and self.log_scale:
                raise ValueError(f"{self.canonical_name}: an additive correction needs the linear scale")
            if self.treatment.correction is TreatmentCorrection.DIVIDE and self.log_offset != 0.0:
                raise ValueError(f"{self.canonical_name}: a ratio correction cannot pass through log(value + offset)")


LIPID_LOWERING = MedicationClass("lipid-lowering therapy", ("C10",))
ANTIHYPERTENSIVE = MedicationClass("blood-pressure-lowering therapy", ("C02", "C03", "C07", "C08", "C09"))
GLUCOSE_LOWERING = MedicationClass("glucose-lowering therapy", ("A10",))
THYROID_THERAPY = MedicationClass("thyroid therapy", ("H03",))
URATE_LOWERING = MedicationClass("urate-lowering therapy", ("M04AA", "M04AB"))
# GLP-1 analogues and tirzepatide (A10BX16) are the dominant weight-loss drugs
# and are classed under A10, not A08.
WEIGHT_LOWERING = MedicationClass("weight-lowering therapy", ("A08", "A10BJ", "A10BX16"))
HEART_RATE_LOWERING = MedicationClass(
    "heart-rate-lowering therapy",
    ("C07", "C08D", "C01EB17", "C01AA"),
)
VITAMIN_D_THERAPY = MedicationClass("vitamin D therapy", ("A11CC",))
VITAMIN_B12_THERAPY = MedicationClass("vitamin B12 therapy", ("B03BA",))
IRON_THERAPY = MedicationClass("iron therapy", ("B03A",))

_PELOSO_2014 = (
    "Peloso et al. 2014 AJHG 94:223 (after CTT 2005: lipid-lowering therapy lowers "
    "total cholesterol ~20% and LDL ~30%)"
)
_CUI_TOBIN = "Cui, Hopper & Harrap 2003 Hypertension; Tobin et al. 2005 Stat Med 24:2911"
_NO_CORRECTION = "no validated constant correction; treated values are not used"


def _exclude_treated(medication: MedicationClass) -> TreatmentRule:
    return TreatmentRule(medication, TreatmentCorrection.EXCLUDE, None, _NO_CORRECTION)


def _units(*conversions: tuple[str, float] | tuple[str, float, float]) -> tuple[UnitConversion, ...]:
    return tuple(UnitConversion(*conversion) for conversion in conversions)


_THOUSANDS_PER_MICROLITER = _units(("10*3/uL", 1.0), ("10*9/L", 1.0), ("10*3/mm3", 1.0), ("/uL", 0.001))
_MILLIONS_PER_MICROLITER = _units(("10*6/uL", 1.0), ("10*12/L", 1.0))
_GRAMS_PER_DECILITER = _units(("g/dL", 1.0), ("g/L", 0.1))
_CATALYTIC_ACTIVITY = _units(("[U]/L", 1.0), ("[iU]/L", 1.0), ("ukat/L", 60.0))
_CHOLESTEROL_MASS = _units(("mg/dL", 1.0), ("mmol/L", 38.67))
_LENGTH_CENTIMETERS = _units(("cm", 1.0), ("[in_i]", 2.54), ("[in_us]", 2.54000508), ("m", 100.0))
_PERCENT = _units(("%", 1.0))
_PRESSURE = _units(("mm[Hg]", 1.0))
_IRON_MASS = _units(("ug/dL", 1.0), ("umol/L", 5.585))
# Absolute leukocyte-subset counts are reported to 0.1 x 10^3/uL, so zero is a
# real reported value: model log(count + half the reporting resolution).
_DIFFERENTIAL_LOG_OFFSET = 0.05

_ALPHA_GLOBIN = (
    "HBA1/HBA2 -alpha3.7 deletion (3.7 kb, segmental duplication; AFR allele frequency "
    "~0.18, -5.3 fL MCV per copy, poorly SNV-tagged; Raffield 2018 PMC5891078; TOPMed "
    "PMC9732337)"
)
_MUC1_VNTR = "MUC1 60 bp VNTR (urea P=2.7e-163, urate P=4.7e-99)"
_HP_DELETION = "HP 1.7 kb Hp1/Hp2 structural allele (confirmed in panel; Boettger 2016 PMC4811681)"
_NO_NAMED_SV = "no named SV/TR driver yet"

MEASUREMENT_DEFINITIONS: tuple[MeasurementDefinition, ...] = (
    # --- Red cells: the strongest poorly-tagged SV (alpha-globin) lives here.
    MeasurementDefinition(
        canonical_name="mean_corpuscular_volume",
        aliases=("mcv",),
        description="Mean corpuscular volume of red cells (fL).",
        loinc_codes=("787-2", "30428-7"),
        canonical_unit="fL",
        unit_conversions=_units(("fL", 1.0)),
        plausible_range=(50.0, 150.0),
        log_scale=False,
        rationale=_ALPHA_GLOBIN + ".",
    ),
    MeasurementDefinition(
        canonical_name="mean_corpuscular_hemoglobin",
        aliases=("mch",),
        description="Mean corpuscular hemoglobin (pg).",
        loinc_codes=("785-6", "28539-5"),
        canonical_unit="pg",
        unit_conversions=_units(("pg", 1.0)),
        plausible_range=(12.0, 50.0),
        log_scale=False,
        rationale=_ALPHA_GLOBIN + ".",
    ),
    MeasurementDefinition(
        canonical_name="mean_corpuscular_hemoglobin_concentration",
        aliases=("mchc",),
        description="Mean corpuscular hemoglobin concentration (g/dL).",
        loinc_codes=("786-4", "28540-3"),
        canonical_unit="g/dL",
        # Older reports give MCHC in %, numerically equal to g/dL.
        unit_conversions=_GRAMS_PER_DECILITER + _units(("%", 1.0)),
        plausible_range=(24.0, 40.0),
        log_scale=False,
        rationale=_ALPHA_GLOBIN + ".",
    ),
    MeasurementDefinition(
        canonical_name="red_blood_cell_count",
        aliases=("rbc", "erythrocyte_count"),
        description="Erythrocyte count (10^6/uL).",
        loinc_codes=("789-8", "26453-1"),
        canonical_unit="10*6/uL",
        unit_conversions=_MILLIONS_PER_MICROLITER,
        plausible_range=(1.0, 9.0),
        log_scale=False,
        rationale=_ALPHA_GLOBIN + " (raises the count); ESR2 GTTT STR (Margoliash 2023).",
    ),
    MeasurementDefinition(
        canonical_name="hemoglobin",
        aliases=("hgb", "hb"),
        description="Blood hemoglobin concentration (g/dL).",
        loinc_codes=("718-7", "20509-6", "59260-0"),
        canonical_unit="g/dL",
        # 59260-0 reports the Fe-monomer concentration: 1 mmol/L = 1.611 g/dL.
        unit_conversions=_GRAMS_PER_DECILITER + _units(("mmol/L", 1.611)),
        plausible_range=(3.0, 25.0),
        log_scale=False,
        rationale=_ALPHA_GLOBIN + "; ESR2 GTTT STR (Margoliash 2023).",
    ),
    MeasurementDefinition(
        canonical_name="red_cell_distribution_width",
        aliases=("rdw", "rdw_cv"),
        description="Red cell distribution width, coefficient of variation (%); RDW-SD in fL is a different quantity.",
        # LOINC 2.8x gives these codes the DistWidth property, which also
        # carries RDW-SD (fL); the % unit table keeps only the CV. 115741-1 is
        # the LOINC 2.81 RDW-CV code, not yet in the 2025 OMOP vocabulary.
        loinc_codes=("788-0", "30385-9", "115741-1"),
        canonical_unit="%",
        unit_conversions=_PERCENT,
        plausible_range=(9.0, 35.0),
        log_scale=True,
        rationale="RHOT1 CCG STR (Margoliash 2023); MiXeR-SV RDW SV enrichment 21.5x (Nguyen 2026).",
    ),
    # --- Platelets and leukocytes.
    MeasurementDefinition(
        canonical_name="platelet_count",
        aliases=("platelets", "plt"),
        description="Platelet count (10^3/uL).",
        loinc_codes=("777-3", "26515-7", "778-1"),
        canonical_unit="10*3/uL",
        unit_conversions=_THOUSANDS_PER_MICROLITER,
        plausible_range=(10.0, 1500.0),
        log_scale=False,
        rationale=(
            "CBL promoter CGG STR (P=4e-83) and NCK2 AC STR (Margoliash 2023); MiXeR-SV enrichment 16.9x "
            "for platelet distribution width."
        ),
    ),
    MeasurementDefinition(
        canonical_name="mean_platelet_volume",
        aliases=("mpv",),
        description="Mean platelet volume (fL).",
        loinc_codes=("32623-1", "28542-9"),
        canonical_unit="fL",
        unit_conversions=_units(("fL", 1.0)),
        plausible_range=(5.0, 20.0),
        log_scale=False,
        rationale="TAOK1 poly(A) STR (P<1e-300, Margoliash 2023).",
    ),
    MeasurementDefinition(
        canonical_name="white_blood_cell_count",
        aliases=("wbc", "leukocyte_count"),
        description="Leukocyte count (10^3/uL).",
        loinc_codes=("6690-2", "26464-8", "804-5"),
        canonical_unit="10*3/uL",
        unit_conversions=_THOUSANDS_PER_MICROLITER,
        plausible_range=(0.5, 100.0),
        log_scale=True,
        rationale=_NO_NAMED_SV + "; ACKR1 Duffy-null (SNV, AFR) is an ancestry-specific positive control.",
    ),
    MeasurementDefinition(
        canonical_name="neutrophil_count",
        aliases=("anc", "absolute_neutrophil_count"),
        description="Absolute neutrophil count (10^3/uL).",
        loinc_codes=("751-8", "26499-4", "753-4"),
        canonical_unit="10*3/uL",
        unit_conversions=_THOUSANDS_PER_MICROLITER,
        plausible_range=(0.0, 60.0),
        log_scale=True,
        log_offset=_DIFFERENTIAL_LOG_OFFSET,
        rationale=_NO_NAMED_SV + "; ACKR1 Duffy-null (SNV, AFR) positive control for ancestry-aware prediction.",
    ),
    MeasurementDefinition(
        canonical_name="lymphocyte_count",
        aliases=("absolute_lymphocyte_count",),
        description="Absolute lymphocyte count (10^3/uL).",
        loinc_codes=("731-0", "26474-7", "732-8"),
        canonical_unit="10*3/uL",
        unit_conversions=_THOUSANDS_PER_MICROLITER,
        plausible_range=(0.0, 100.0),
        log_scale=True,
        log_offset=_DIFFERENTIAL_LOG_OFFSET,
        rationale=_NO_NAMED_SV + "; understudied relative to its CBC coverage.",
    ),
    MeasurementDefinition(
        canonical_name="monocyte_count",
        aliases=("absolute_monocyte_count",),
        description="Absolute monocyte count (10^3/uL).",
        loinc_codes=("742-7", "26484-6", "743-5"),
        canonical_unit="10*3/uL",
        unit_conversions=_THOUSANDS_PER_MICROLITER,
        plausible_range=(0.0, 20.0),
        log_scale=True,
        log_offset=_DIFFERENTIAL_LOG_OFFSET,
        rationale="S1PR3 602 bp deletion (AFR MAF 0.117; r2 0.996 with rs28450540, a tagged-SV control).",
    ),
    MeasurementDefinition(
        canonical_name="eosinophil_count",
        aliases=("absolute_eosinophil_count",),
        description="Absolute eosinophil count (10^3/uL).",
        loinc_codes=("711-2", "26449-9", "712-0"),
        canonical_unit="10*3/uL",
        unit_conversions=_THOUSANDS_PER_MICROLITER,
        plausible_range=(0.0, 20.0),
        log_scale=True,
        log_offset=_DIFFERENTIAL_LOG_OFFSET,
        rationale="BCL2L11 CCG STR (Margoliash 2023).",
    ),
    # --- Lipids and lipoproteins. LDL/TC/TG/HDL are heavily studied: positive
    # controls, not headline traits.
    MeasurementDefinition(
        canonical_name="ldl_cholesterol",
        aliases=("ldl", "ldl_c"),
        description="LDL cholesterol, calculated or direct (mg/dL).",
        loinc_codes=("13457-7", "18262-6", "2089-1", "96259-7"),
        canonical_unit="mg/dL",
        unit_conversions=_CHOLESTEROL_MASS,
        plausible_range=(10.0, 400.0),
        log_scale=False,
        treatment=TreatmentRule(LIPID_LOWERING, TreatmentCorrection.DIVIDE, 0.7, _PELOSO_2014),
        rationale=_HP_DELETION + "; lipids are not SV-enriched in MiXeR-SV (positive control).",
    ),
    MeasurementDefinition(
        canonical_name="total_cholesterol",
        aliases=("cholesterol", "tc"),
        description="Total cholesterol (mg/dL).",
        loinc_codes=("2093-3",),
        canonical_unit="mg/dL",
        unit_conversions=_CHOLESTEROL_MASS,
        plausible_range=(50.0, 600.0),
        log_scale=False,
        treatment=TreatmentRule(LIPID_LOWERING, TreatmentCorrection.DIVIDE, 0.8, _PELOSO_2014),
        rationale=_HP_DELETION + " (positive control).",
    ),
    MeasurementDefinition(
        canonical_name="hdl_cholesterol",
        aliases=("hdl", "hdl_c"),
        description="HDL cholesterol (mg/dL); unadjusted for therapy by convention.",
        loinc_codes=("2085-9",),
        canonical_unit="mg/dL",
        unit_conversions=_CHOLESTEROL_MASS,
        plausible_range=(5.0, 200.0),
        log_scale=True,
        rationale="GPIHBP1 VNTR (P=2.6e-41, PIP 0.99).",
    ),
    MeasurementDefinition(
        canonical_name="triglycerides",
        aliases=("tg",),
        description="Triglycerides (mg/dL); unadjusted for therapy by convention.",
        loinc_codes=("2571-8",),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("mmol/L", 88.57)),
        plausible_range=(10.0, 5000.0),
        log_scale=True,
        rationale=_NO_NAMED_SV + "; heavily studied positive control.",
    ),
    MeasurementDefinition(
        canonical_name="apolipoprotein_b",
        aliases=("apob",),
        description="Apolipoprotein B / B-100 (mg/dL).",
        loinc_codes=("1884-6", "1871-3"),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("g/L", 100.0)),
        plausible_range=(10.0, 300.0),
        log_scale=False,
        treatment=_exclude_treated(LIPID_LOWERING),
        rationale="APOB coding CTG repeat (P=1.4e-279, Margoliash 2023).",
    ),
    MeasurementDefinition(
        canonical_name="lipoprotein_a_mass",
        aliases=("lpa_mass",),
        description="Lipoprotein(a) mass concentration (mg/dL); not convertible to molar units (isoform-dependent).",
        loinc_codes=("10835-7",),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("mg/L", 0.1)),
        plausible_range=(0.1, 500.0),
        log_scale=True,
        rationale=(
            "LPA KIV-2 copy number (5.5 kb repeat units) sets most Lp(a) variance; the array exceeds the "
            "panel's 10 kb cap, so this tests SV/SNV tagging of an unimputed CNV."
        ),
    ),
    MeasurementDefinition(
        canonical_name="lipoprotein_a_molar",
        aliases=("lpa_molar",),
        description="Lipoprotein(a) molar concentration (nmol/L).",
        loinc_codes=("43583-4",),
        canonical_unit="nmol/L",
        unit_conversions=_units(("nmol/L", 1.0)),
        plausible_range=(0.5, 1000.0),
        log_scale=True,
        rationale="LPA KIV-2 copy number (see lipoprotein_a_mass).",
    ),
    # --- Glycemia.
    MeasurementDefinition(
        canonical_name="hemoglobin_a1c",
        aliases=("hba1c", "a1c"),
        description="Hemoglobin A1c (NGSP %); IFCC mmol/mol converted by the NGSP master equation.",
        loinc_codes=("4548-4", "17856-6", "4549-2", "17855-8", "59261-8"),
        canonical_unit="%",
        # NGSP % = 0.09148 * IFCC mmol/mol + 2.152 (ngsp.org/ifccngsp.asp).
        unit_conversions=_units(("%", 1.0), ("mmol/mol", 0.09148, 2.152)),
        plausible_range=(3.0, 20.0),
        log_scale=False,
        treatment=_exclude_treated(GLUCOSE_LOWERING),
        rationale=(
            _ALPHA_GLOBIN + " raises HbA1c (+0.029 ln units/copy, a biomarker artifact); JAZF1 364 bp and "
            "CTRB2 584 bp deletions (T2D)."
        ),
    ),
    MeasurementDefinition(
        canonical_name="glucose",
        aliases=("serum_glucose", "plasma_glucose"),
        description="Serum/plasma glucose, fasting or unrecorded fasting status (mg/dL).",
        loinc_codes=("2345-7", "1558-6"),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("mmol/L", 18.016)),
        plausible_range=(20.0, 1000.0),
        log_scale=True,
        treatment=_exclude_treated(GLUCOSE_LOWERING),
        rationale="JAZF1/CTRB2 deletions (T2D); free of the alpha-globin HbA1c artifact, so it cross-checks HbA1c.",
    ),
    # --- Kidney.
    MeasurementDefinition(
        canonical_name="egfr_ckd_epi_2021",
        aliases=("egfr", "egfr_creatinine"),
        description=(
            "Creatinine eGFR, race-free CKD-EPI 2021 (Inker 2021 NEJM 385:1737), mL/min/1.73m2, "
            "computed per measurement from serum creatinine, age and sex at birth."
        ),
        loinc_codes=("2160-0", "14682-9"),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("umol/L", 1.0 / 88.42)),
        plausible_range=(0.2, 20.0),
        log_scale=True,
        value_formula="ckd_epi_2021",
        rationale=(
            _MUC1_VNTR + "; the race-free equation keeps ancestry out of the phenotype; APOL1 (SNV/indel) is a "
            "confounder check."
        ),
    ),
    MeasurementDefinition(
        canonical_name="cystatin_c",
        aliases=("cysc",),
        description="Serum cystatin C (mg/L), a GFR marker independent of muscle mass.",
        loinc_codes=("33863-2",),
        canonical_unit="mg/L",
        unit_conversions=_units(("mg/L", 1.0)),
        plausible_range=(0.3, 10.0),
        log_scale=True,
        rationale=_MUC1_VNTR + " (kidney); CST3 locus; understudied in EHR biobanks.",
    ),
    MeasurementDefinition(
        canonical_name="blood_urea_nitrogen",
        aliases=("bun", "urea_nitrogen"),
        description="Serum/plasma urea nitrogen (mg/dL); urea mass (x2.14) is a different quantity.",
        loinc_codes=("3094-0", "14937-7"),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("mmol/L", 2.801)),
        plausible_range=(2.0, 200.0),
        log_scale=True,
        rationale=_MUC1_VNTR + ".",
    ),
    MeasurementDefinition(
        canonical_name="urate",
        aliases=("uric_acid",),
        description="Serum/plasma urate (mg/dL).",
        loinc_codes=("3084-1", "14933-6"),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("umol/L", 1.0 / 59.48), ("mmol/L", 1000.0 / 59.48)),
        plausible_range=(0.5, 20.0),
        log_scale=False,
        treatment=_exclude_treated(URATE_LOWERING),
        rationale=_MUC1_VNTR + "; ABCG2/SLC2A9 SNV positive controls.",
    ),
    MeasurementDefinition(
        canonical_name="urine_albumin_creatinine_ratio",
        aliases=("uacr", "acr"),
        description="Spot urine albumin/creatinine ratio (mg/g); 24-hour collections are a different system.",
        loinc_codes=("9318-7", "14959-1", "32294-1"),
        canonical_unit="mg/g",
        unit_conversions=_units(("mg/g", 1.0), ("mg/g{creat}", 1.0), ("mg/mmol", 8.84), ("mg/mmol{creat}", 8.84)),
        plausible_range=(0.1, 20000.0),
        log_scale=True,
        rationale="CUBN locus; " + _NO_NAMED_SV + "; understudied.",
    ),
    # --- Minerals.
    MeasurementDefinition(
        canonical_name="calcium",
        aliases=("serum_calcium",),
        description="Total serum/plasma calcium (mg/dL); ionized calcium is a different quantity.",
        loinc_codes=("17861-6", "2000-8"),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("mmol/L", 4.008)),
        plausible_range=(5.0, 16.0),
        log_scale=False,
        rationale="CASR/GCKR SNV loci; " + _NO_NAMED_SV + ".",
    ),
    MeasurementDefinition(
        canonical_name="phosphate",
        aliases=("phosphorus", "serum_phosphate"),
        description="Serum/plasma phosphate (mg/dL).",
        loinc_codes=("2777-1", "14879-1"),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("mmol/L", 3.097)),
        plausible_range=(0.5, 12.0),
        log_scale=False,
        rationale=_NO_NAMED_SV + "; understudied mineral trait.",
    ),
    MeasurementDefinition(
        canonical_name="magnesium",
        aliases=("serum_magnesium",),
        description="Serum/plasma magnesium (mg/dL).",
        loinc_codes=("19123-9", "2601-3"),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("mmol/L", 2.431), ("10*-3.eq/L", 1.215)),
        plausible_range=(0.5, 6.0),
        log_scale=False,
        rationale=_NO_NAMED_SV + "; understudied mineral trait.",
    ),
    # --- Plasma proteins.
    MeasurementDefinition(
        canonical_name="albumin",
        aliases=("serum_albumin",),
        description="Serum/plasma albumin (g/dL).",
        loinc_codes=("1751-7", "61151-7", "61152-5"),
        canonical_unit="g/dL",
        unit_conversions=_GRAMS_PER_DECILITER,
        plausible_range=(1.0, 6.5),
        log_scale=False,
        rationale=_NO_NAMED_SV + "; understudied relative to its CMP coverage.",
    ),
    MeasurementDefinition(
        canonical_name="total_protein",
        aliases=("serum_protein",),
        description="Serum/plasma total protein (g/dL).",
        loinc_codes=("2885-2",),
        canonical_unit="g/dL",
        unit_conversions=_GRAMS_PER_DECILITER,
        plausible_range=(2.0, 14.0),
        log_scale=False,
        rationale=_NO_NAMED_SV + "; immunoglobulin-driven, understudied.",
    ),
    MeasurementDefinition(
        canonical_name="globulin",
        aliases=("serum_globulin",),
        description="Serum globulin, measured or calculated (g/dL).",
        loinc_codes=("10834-0", "2336-6"),
        canonical_unit="g/dL",
        unit_conversions=_GRAMS_PER_DECILITER,
        plausible_range=(0.5, 10.0),
        log_scale=True,
        rationale=_NO_NAMED_SV + "; immunoglobulin-driven, understudied.",
    ),
    MeasurementDefinition(
        canonical_name="haptoglobin",
        aliases=("hp",),
        description="Serum/plasma haptoglobin (mg/dL).",
        loinc_codes=("4542-7",),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("g/L", 100.0), ("mg/L", 0.1)),
        plausible_range=(1.0, 700.0),
        log_scale=True,
        rationale=_HP_DELETION + " sets the haptoglobin level itself: the most direct SV-to-protein trait.",
    ),
    MeasurementDefinition(
        canonical_name="angiotensin_converting_enzyme",
        aliases=("ace", "serum_ace"),
        description="Serum/plasma angiotensin-converting enzyme activity (U/L).",
        loinc_codes=("2742-5",),
        canonical_unit="[U]/L",
        unit_conversions=_CATALYTIC_ACTIVITY,
        plausible_range=(1.0, 300.0),
        log_scale=True,
        rationale="ACE 287 bp Alu insertion/deletion (confirmed in panel) sets serum ACE activity.",
    ),
    # --- Liver.
    MeasurementDefinition(
        canonical_name="alanine_aminotransferase",
        aliases=("alt", "sgpt"),
        description="Serum/plasma ALT (U/L).",
        loinc_codes=("1742-6", "1743-4", "1744-2"),
        canonical_unit="[U]/L",
        unit_conversions=_CATALYTIC_ACTIVITY,
        plausible_range=(1.0, 3000.0),
        log_scale=True,
        rationale=_NO_NAMED_SV + "; PNPLA3 (SNV) positive control.",
    ),
    MeasurementDefinition(
        canonical_name="aspartate_aminotransferase",
        aliases=("ast", "sgot"),
        description="Serum/plasma AST (U/L).",
        loinc_codes=("1920-8", "30239-8", "88112-8"),
        canonical_unit="[U]/L",
        unit_conversions=_CATALYTIC_ACTIVITY,
        plausible_range=(1.0, 3000.0),
        log_scale=True,
        rationale="MiXeR-SV AST SV enrichment 23.4x (Nguyen 2026).",
    ),
    MeasurementDefinition(
        canonical_name="alkaline_phosphatase",
        aliases=("alp", "alk_phos"),
        description="Serum/plasma alkaline phosphatase (U/L).",
        loinc_codes=("6768-6",),
        canonical_unit="[U]/L",
        unit_conversions=_CATALYTIC_ACTIVITY,
        plausible_range=(5.0, 3000.0),
        log_scale=True,
        rationale=_NO_NAMED_SV + "; placental ALP makes the pregnancy exclusion essential.",
    ),
    MeasurementDefinition(
        canonical_name="gamma_glutamyl_transferase",
        aliases=("ggt",),
        description="Serum/plasma gamma-glutamyl transferase (U/L).",
        loinc_codes=("2324-2",),
        canonical_unit="[U]/L",
        unit_conversions=_CATALYTIC_ACTIVITY,
        plausible_range=(1.0, 3000.0),
        log_scale=True,
        rationale="GGT1 VNTR-7 (P=5.0e-241, Bai 2026 PMC13263140).",
    ),
    MeasurementDefinition(
        canonical_name="total_bilirubin",
        aliases=("bilirubin",),
        description="Serum/plasma total bilirubin (mg/dL).",
        loinc_codes=("1975-2", "14631-6"),
        canonical_unit="mg/dL",
        unit_conversions=_units(("mg/dL", 1.0), ("umol/L", 1.0 / 17.104)),
        plausible_range=(0.05, 30.0),
        log_scale=True,
        rationale="UGT1A1 promoter (TA)n repeat (Gilbert TA7), an indel-sized TR: the classic TR positive control.",
    ),
    # --- Iron and vitamins.
    MeasurementDefinition(
        canonical_name="ferritin",
        aliases=("serum_ferritin",),
        description="Serum/plasma ferritin (ng/mL).",
        loinc_codes=("2276-4", "20567-4"),
        canonical_unit="ng/mL",
        unit_conversions=_units(("ng/mL", 1.0), ("ug/L", 1.0)),
        plausible_range=(1.0, 10000.0),
        log_scale=True,
        treatment=_exclude_treated(IRON_THERAPY),
        rationale="HFE/TMPRSS6/TF SNV loci; " + _NO_NAMED_SV + ".",
    ),
    MeasurementDefinition(
        canonical_name="serum_iron",
        aliases=("iron",),
        description="Serum/plasma iron (ug/dL).",
        loinc_codes=("2498-4",),
        canonical_unit="ug/dL",
        unit_conversions=_IRON_MASS,
        plausible_range=(5.0, 600.0),
        log_scale=True,
        treatment=_exclude_treated(IRON_THERAPY),
        rationale="HFE/TMPRSS6 SNV loci; " + _NO_NAMED_SV + ".",
    ),
    MeasurementDefinition(
        canonical_name="total_iron_binding_capacity",
        aliases=("tibc",),
        description="Serum/plasma total iron binding capacity (ug/dL).",
        loinc_codes=("2500-7",),
        canonical_unit="ug/dL",
        unit_conversions=_IRON_MASS,
        plausible_range=(80.0, 800.0),
        log_scale=False,
        treatment=_exclude_treated(IRON_THERAPY),
        rationale="TF/HFE SNV loci; " + _NO_NAMED_SV + ".",
    ),
    MeasurementDefinition(
        canonical_name="transferrin_saturation",
        aliases=("tsat", "iron_saturation"),
        description="Iron (transferrin) saturation (%).",
        loinc_codes=("2502-3",),
        canonical_unit="%",
        unit_conversions=_PERCENT,
        plausible_range=(1.0, 100.0),
        log_scale=True,
        treatment=_exclude_treated(IRON_THERAPY),
        rationale="HFE C282Y (SNV) positive control; " + _NO_NAMED_SV + ".",
    ),
    MeasurementDefinition(
        canonical_name="vitamin_d_25_hydroxy",
        aliases=("vitamin_d", "25_oh_vitamin_d"),
        description="Serum/plasma 25-hydroxyvitamin D (ng/mL).",
        loinc_codes=("62292-8", "1989-3", "83070-3"),
        canonical_unit="ng/mL",
        unit_conversions=_units(("ng/mL", 1.0), ("ug/L", 1.0), ("nmol/L", 1.0 / 2.496)),
        plausible_range=(2.0, 200.0),
        log_scale=True,
        treatment=_exclude_treated(VITAMIN_D_THERAPY),
        rationale="GC locus; " + _NO_NAMED_SV + "; seasonal within-person noise is absorbed by the person BLUP.",
    ),
    MeasurementDefinition(
        canonical_name="vitamin_b12",
        aliases=("cobalamin", "b12"),
        description="Serum/plasma cobalamin (pg/mL).",
        loinc_codes=("2132-9", "14685-2"),
        canonical_unit="pg/mL",
        unit_conversions=_units(("pg/mL", 1.0), ("ng/L", 1.0), ("pmol/L", 1.355)),
        plausible_range=(50.0, 5000.0),
        log_scale=True,
        treatment=_exclude_treated(VITAMIN_B12_THERAPY),
        rationale="FUT2/TCN1/TCN2 loci; " + _NO_NAMED_SV + "; understudied.",
    ),
    # --- Thyroid and inflammation.
    MeasurementDefinition(
        canonical_name="thyrotropin",
        aliases=("tsh",),
        description="Serum/plasma thyrotropin (mIU/L).",
        loinc_codes=("3016-3", "11580-8", "11579-0"),
        canonical_unit="10*-3.[iU]/L",
        unit_conversions=_units(("10*-3.[iU]/L", 1.0), ("10*-6.[iU]/mL", 1.0)),
        plausible_range=(0.005, 200.0),
        log_scale=True,
        treatment=_exclude_treated(THYROID_THERAPY),
        rationale=_NO_NAMED_SV + "; highly heritable and understudied for SVs.",
    ),
    MeasurementDefinition(
        canonical_name="free_thyroxine",
        aliases=("free_t4", "ft4"),
        description="Serum/plasma free T4 (ng/dL).",
        loinc_codes=("3024-7", "14920-3"),
        canonical_unit="ng/dL",
        unit_conversions=_units(("ng/dL", 1.0), ("pmol/L", 1.0 / 12.871)),
        plausible_range=(0.1, 8.0),
        log_scale=False,
        treatment=_exclude_treated(THYROID_THERAPY),
        rationale=_NO_NAMED_SV + "; understudied for SVs.",
    ),
    MeasurementDefinition(
        canonical_name="c_reactive_protein",
        aliases=("hs_crp", "crp"),
        description=(
            "High-sensitivity C-reactive protein (mg/L), capped at 10 mg/L: higher values indicate acute "
            "inflammation (AHA/CDC 2003), and the standard CRP assay censors the baseline range."
        ),
        loinc_codes=("30522-7",),
        canonical_unit="mg/L",
        unit_conversions=_units(("mg/L", 1.0), ("mg/dL", 10.0)),
        plausible_range=(0.01, 10.0),
        log_scale=True,
        rationale="CRP/HNF1A/IL6R/APOE SNV loci; " + _NO_NAMED_SV + ".",
    ),
    # --- Physical measurements (AoU enrollment protocol + EHR vitals).
    MeasurementDefinition(
        canonical_name="height",
        aliases=("body_height", "standing_height"),
        description="Measured body height (cm); stated (self-reported) height LOINCs are excluded.",
        loinc_codes=("8302-2", "3137-7"),
        canonical_unit="cm",
        unit_conversions=_LENGTH_CENTIMETERS,
        plausible_range=(120.0, 230.0),
        log_scale=False,
        rationale="ACAN 57 bp VNTR (0.60% of variance in AFR) and TENT5A VNTR; heavily studied sanity check.",
    ),
    MeasurementDefinition(
        canonical_name="body_mass_index",
        aliases=("bmi",),
        description="Body mass index (kg/m2).",
        loinc_codes=("39156-5",),
        canonical_unit="kg/m2",
        unit_conversions=_units(("kg/m2", 1.0)),
        plausible_range=(12.0, 90.0),
        log_scale=True,
        treatment=_exclude_treated(WEIGHT_LOWERING),
        rationale="NEGR1 45 kb deletion is outside the panel; heavily studied sanity check.",
    ),
    MeasurementDefinition(
        canonical_name="waist_circumference",
        aliases=("waist",),
        description="Waist circumference (cm); 56086-2 is the AoU enrollment protocol concept.",
        loinc_codes=("8280-0", "56115-9", "56086-2"),
        canonical_unit="cm",
        unit_conversions=_LENGTH_CENTIMETERS,
        plausible_range=(40.0, 250.0),
        log_scale=False,
        treatment=_exclude_treated(WEIGHT_LOWERING),
        rationale=_NO_NAMED_SV + "; understudied relative to BMI.",
    ),
    MeasurementDefinition(
        canonical_name="hip_circumference",
        aliases=("hip",),
        description="Hip circumference (cm); 62409-8 is the AoU enrollment protocol concept.",
        loinc_codes=("62409-8", "56063-1"),
        canonical_unit="cm",
        unit_conversions=_LENGTH_CENTIMETERS,
        plausible_range=(50.0, 250.0),
        log_scale=False,
        treatment=_exclude_treated(WEIGHT_LOWERING),
        rationale=_NO_NAMED_SV + "; understudied relative to BMI.",
    ),
    MeasurementDefinition(
        canonical_name="systolic_blood_pressure",
        aliases=("sbp",),
        description="Systolic blood pressure (mmHg), sitting or unspecified position.",
        loinc_codes=("8480-6", "8459-0", "76534-7"),
        canonical_unit="mm[Hg]",
        unit_conversions=_PRESSURE,
        plausible_range=(60.0, 270.0),
        log_scale=False,
        treatment=TreatmentRule(ANTIHYPERTENSIVE, TreatmentCorrection.ADD, 15.0, _CUI_TOBIN),
        rationale="CHMP1A VNTR (PIP 1.00); HRCT1 poly-His repeat; ACE Alu; MiXeR-SV DBP enrichment 13.7x.",
    ),
    MeasurementDefinition(
        canonical_name="diastolic_blood_pressure",
        aliases=("dbp",),
        description="Diastolic blood pressure (mmHg), sitting or unspecified position.",
        loinc_codes=("8462-4", "8453-3", "76535-4"),
        canonical_unit="mm[Hg]",
        unit_conversions=_PRESSURE,
        plausible_range=(30.0, 160.0),
        log_scale=False,
        treatment=TreatmentRule(ANTIHYPERTENSIVE, TreatmentCorrection.ADD, 10.0, _CUI_TOBIN),
        rationale="CHMP1A VNTR; HRCT1 poly-His repeat; MiXeR-SV DBP SV enrichment 13.7x.",
    ),
    MeasurementDefinition(
        canonical_name="heart_rate",
        aliases=("pulse", "resting_heart_rate"),
        description="Heart rate (beats/min).",
        loinc_codes=("8867-4",),
        canonical_unit="/min",
        unit_conversions=_units(("/min", 1.0)),
        plausible_range=(25.0, 220.0),
        log_scale=False,
        treatment=_exclude_treated(HEART_RATE_LOWERING),
        rationale=_NO_NAMED_SV + "; moderately studied.",
    ),
)

_NamedDefinition = TypeVar("_NamedDefinition", DiseaseDefinition, MeasurementDefinition)


@dataclass(slots=True)
class AllOfUsDiseaseRequest:
    disease: str

    def __post_init__(self) -> None:
        if not self.disease.strip():
            raise ValueError("disease cannot be blank.")


@dataclass(slots=True)
class AllOfUsPreparedPhenotype:
    sample_table_path: Path
    sql_path: Path
    metadata_path: Path


def available_disease_names() -> list[str]:
    return sorted(disease_definition.canonical_name for disease_definition in DISEASE_DEFINITIONS)


def resolve_disease_definition(disease: str) -> DiseaseDefinition:
    disease_definition = _find_named_definition(DISEASE_DEFINITIONS, disease)
    if disease_definition is None:
        raise ValueError(
            "Unsupported disease: "
            + disease
            + ". Available diseases: "
            + ", ".join(available_disease_names())
        )
    return disease_definition


def build_all_of_us_disease_sql(disease_definition: DiseaseDefinition) -> str:
    dataset = _require_env("WORKSPACE_CDR")
    return f"""
WITH primary_consent AS (
  SELECT
    observation.person_id,
    MIN(observation.observation_date) AS primary_consent_date
  FROM `{dataset}.concept` AS concept
  JOIN `{dataset}.concept_ancestor` AS concept_ancestor
    ON concept.concept_id = concept_ancestor.ancestor_concept_id
  JOIN `{dataset}.observation` AS observation
    ON concept_ancestor.descendant_concept_id = observation.observation_concept_id
  WHERE concept.concept_name = 'Consent PII'
    AND concept.concept_class_id = 'Module'
  GROUP BY observation.person_id
),
ehr_participants AS (
  SELECT
    person_id,
    MIN(observation_period_start_date) AS observation_start_date,
    MAX(observation_period_end_date) AS observation_end_date
  FROM `{dataset}.observation_period`
  GROUP BY person_id
),
visit_counts AS (
  SELECT
    person_id,
    COUNT(*) AS n_visits
  FROM `{dataset}.visit_occurrence`
  GROUP BY person_id
),
disease_root AS (
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE vocabulary_id = 'SNOMED'
    AND standard_concept = 'S'
    AND concept_code = @snomed_code
),
disease_concepts AS (
  SELECT DISTINCT concept_ancestor.descendant_concept_id AS concept_id
  FROM `{dataset}.concept_ancestor` AS concept_ancestor
  JOIN disease_root ON disease_root.concept_id = concept_ancestor.ancestor_concept_id
),
matched_conditions AS (
  SELECT
    condition_occurrence.person_id,
    condition_occurrence.condition_start_date
  FROM `{dataset}.condition_occurrence` AS condition_occurrence
  WHERE condition_occurrence.condition_concept_id IN (
    SELECT concept_id FROM disease_concepts
  )
),
aggregated_conditions AS (
  -- Occurrences are distinct diagnosis dates: several condition rows on one
  -- day (one visit coded twice) are one occurrence, not a confirmation.
  SELECT
    person_id,
    COUNT(DISTINCT condition_start_date) AS phenotype_occurrence_count,
    MIN(condition_start_date) AS first_condition_date
  FROM matched_conditions
  GROUP BY person_id
)
SELECT
  CAST(ehr_participants.person_id AS STRING) AS sample_id,
  CAST(ehr_participants.person_id AS STRING) AS person_id,
  CASE
    WHEN COALESCE(aggregated_conditions.phenotype_occurrence_count, 0) >= {MIN_DISEASE_OCCURRENCES} THEN 1
    WHEN COALESCE(aggregated_conditions.phenotype_occurrence_count, 0) = 0 THEN 0
    ELSE NULL
  END AS target,
  COALESCE(aggregated_conditions.phenotype_occurrence_count, 0) AS phenotype_occurrence_count,
  aggregated_conditions.first_condition_date,
  ehr_participants.observation_start_date,
  ehr_participants.observation_end_date,
  primary_consent.primary_consent_date,
  CASE
    WHEN person.year_of_birth IS NULL OR ehr_participants.observation_start_date IS NULL THEN NULL
    ELSE EXTRACT(YEAR FROM ehr_participants.observation_start_date) - person.year_of_birth
  END AS age_at_observation_start,
  CASE
    WHEN person.year_of_birth IS NULL OR ehr_participants.observation_start_date IS NULL THEN NULL
    ELSE POW(EXTRACT(YEAR FROM ehr_participants.observation_start_date) - person.year_of_birth, 2)
  END AS age_at_observation_start_squared,
  DATE_DIFF(ehr_participants.observation_end_date, ehr_participants.observation_start_date, DAY) AS observation_duration_days,
  LN(1 + COALESCE(visit_counts.n_visits, 0)) AS log1p_n_visits,
  person.gender_concept_id,
  person.race_concept_id,
  person.ethnicity_concept_id
FROM ehr_participants
JOIN `{dataset}.person` AS person
  ON person.person_id = ehr_participants.person_id
LEFT JOIN visit_counts
  ON visit_counts.person_id = ehr_participants.person_id
LEFT JOIN aggregated_conditions
  ON aggregated_conditions.person_id = ehr_participants.person_id
LEFT JOIN primary_consent
  ON primary_consent.person_id = ehr_participants.person_id
ORDER BY ehr_participants.person_id
""".strip()


def build_all_of_us_disease_query_config(disease_definition: DiseaseDefinition) -> "bigquery.QueryJobConfig":
    from google.cloud import bigquery

    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("snomed_code", "STRING", disease_definition.snomed_code),
        ]
    )


def fetch_all_of_us_disease_rows(
    request: AllOfUsDiseaseRequest,
    client: bigquery.Client | None = None,
) -> list[dict[str, Any]]:
    disease_definition = resolve_disease_definition(request.disease)
    active_client = _active_bigquery_client(client)
    return _query_rows(
        active_client,
        build_all_of_us_disease_sql(disease_definition),
        build_all_of_us_disease_query_config(disease_definition),
    )


def prepare_all_of_us_disease_sample_table(
    request: AllOfUsDiseaseRequest,
    output_path: str | Path,
    *,
    client: bigquery.Client | None = None,
) -> AllOfUsPreparedPhenotype:
    disease_definition = resolve_disease_definition(request.disease)
    rows = fetch_all_of_us_disease_rows(request=request, client=client)
    billing_project = _resolve_billing_project(client)
    sample_table_path = Path(output_path)
    sample_table_path.parent.mkdir(parents=True, exist_ok=True)
    training_rows, encoded_categorical_columns, phenotype_counts = _prepare_training_rows(rows)
    LOGGER.info(
        "Prepared All of Us phenotype rows: n_cases=%d n_controls=%d n_excluded_ambiguous=%d",
        phenotype_counts["n_cases"],
        phenotype_counts["n_controls"],
        phenotype_counts["n_excluded_ambiguous"],
    )

    header = (
        "sample_id",
        "person_id",
        "target",
        "phenotype_occurrence_count",
        "first_condition_date",
        "observation_start_date",
        "observation_end_date",
        "primary_consent_date",
        "age_at_observation_start",
        "age_at_observation_start_squared",
        "observation_duration_days",
        "log1p_n_visits",
        *encoded_categorical_columns,
    )
    _write_sample_table(sample_table_path, header, training_rows)

    sql_path = sample_table_path.with_suffix(sample_table_path.suffix + ".sql")
    sql_path.write_text(build_all_of_us_disease_sql(disease_definition) + "\n", encoding="utf-8")

    metadata_path = sample_table_path.with_suffix(sample_table_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "disease": disease_definition.canonical_name,
                "description": disease_definition.description,
                "snomed_code": disease_definition.snomed_code,
                "snomed_concept_name": disease_definition.snomed_concept_name,
                "min_occurrences": MIN_DISEASE_OCCURRENCES,
                "billing_project_env": "GOOGLE_PROJECT",
                "cdr_dataset_env": "WORKSPACE_CDR",
                "billing_project": billing_project,
                "cdr_dataset": _require_env("WORKSPACE_CDR"),
                "raw_row_count": len(rows),
                "row_count": len(training_rows),
                "n_cases": phenotype_counts["n_cases"],
                "n_controls": phenotype_counts["n_controls"],
                "n_excluded_ambiguous": phenotype_counts["n_excluded_ambiguous"],
                "excluded_training_definition": "phenotype_occurrence_count == 1",
                "encoded_categorical_covariates": encoded_categorical_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return AllOfUsPreparedPhenotype(
        sample_table_path=sample_table_path,
        sql_path=sql_path,
        metadata_path=metadata_path,
    )


def available_measurement_names() -> list[str]:
    return sorted(definition.canonical_name for definition in MEASUREMENT_DEFINITIONS)


def resolve_measurement_definition(trait: str) -> MeasurementDefinition:
    definition = _find_named_definition(MEASUREMENT_DEFINITIONS, trait)
    if definition is None:
        raise ValueError(
            "Unsupported trait: "
            + trait
            + ". Available traits: "
            + ", ".join(available_measurement_names())
        )
    return definition


def resolve_all_of_us_phenotype(phenotype: str) -> DiseaseDefinition | MeasurementDefinition:
    """Resolve a disease or quantitative trait name (the two name sets are disjoint)."""
    disease_definition = _find_named_definition(DISEASE_DEFINITIONS, phenotype)
    if disease_definition is not None:
        return disease_definition
    measurement_definition = _find_named_definition(MEASUREMENT_DEFINITIONS, phenotype)
    if measurement_definition is not None:
        return measurement_definition
    raise ValueError(
        "Unsupported phenotype: "
        + phenotype
        + ". Available diseases: "
        + ", ".join(available_disease_names())
        + ". Available traits: "
        + ", ".join(available_measurement_names())
    )


# CKD-EPI 2021 creatinine equation (Inker et al. 2021 NEJM 385:1737), serum
# creatinine Scr in mg/dL:
#   eGFR = 142 * min(Scr/kappa, 1)^alpha * max(Scr/kappa, 1)^-1.200 * 0.9938^age * sex_factor
# with (kappa, alpha, sex_factor) by sex at birth.
CKD_EPI_2021_SEX_COEFFICIENTS: dict[str, tuple[float, float, float]] = {
    "female": (0.7, -0.241, 1.012),
    "male": (0.9, -0.302, 1.0),
}
_OCCASION_GROUPS = (
    ("untreated", "retained_row_count > 0 AND NOT on_treatment"),
    ("treated", "retained_row_count > 0 AND on_treatment"),
)
_OCCASION_STATISTICS = ("occasion_count", "mean", "variance", "mean_age", "mean_age_squared")


def build_all_of_us_measurement_sql() -> str:
    """One parameterized query for every trait; the trait enters only through
    build_all_of_us_measurement_query_parameters. Returns one row per person
    with at least one analyte row (see the module comment above
    MEASUREMENT_DEFINITIONS for the per-row rules).

    Every CTE is referenced once, so BigQuery scans `measurement` once (it
    re-executes a non-recursive CTE at each reference).
    """
    dataset = _require_env("WORKSPACE_CDR")
    # exclusion_reason is NULL on retained rows; the COALESCE keeps the
    # predicate two-valued so the count is 0, never NULL, on every engine.
    day_exclusion_counts = ",\n".join(
        f"    COUNTIF(COALESCE(exclusion_reason = '{reason}', FALSE)) AS {reason}_row_count"
        for reason in MEASUREMENT_EXCLUSION_REASONS
    )
    person_exclusion_counts = ",\n".join(
        f"    SUM({reason}_row_count) AS {reason}_row_count" for reason in MEASUREMENT_EXCLUSION_REASONS
    )
    occasion_summary_columns = ",\n".join(
        f"    COUNTIF({condition}) AS {group}_occasion_count,\n"
        f"    AVG(IF({condition}, occasion_value, NULL)) AS {group}_mean,\n"
        f"    VAR_POP(IF({condition}, occasion_value, NULL)) AS {group}_variance,\n"
        f"    AVG(IF({condition}, age_years, NULL)) AS {group}_mean_age,\n"
        f"    AVG(IF({condition}, age_years * age_years, NULL)) AS {group}_mean_age_squared"
        for group, condition in _OCCASION_GROUPS
    )
    selected_columns = ",\n".join(
        [f"  person_summaries.{reason}_row_count" for reason in MEASUREMENT_EXCLUSION_REASONS]
        + [
            f"  person_summaries.{group}_{statistic}"
            for group, _condition in _OCCASION_GROUPS
            for statistic in _OCCASION_STATISTICS
        ]
    )
    egfr_branches = "\n".join(
        f"          WHEN '{sex}' THEN 142 * POW(LEAST(canonical_value / {kappa}, 1), {alpha})"
        f" * POW(GREATEST(canonical_value / {kappa}, 1), -1.2) * POW(0.9938, age_years) * {sex_factor}"
        for sex, (kappa, alpha, sex_factor) in CKD_EPI_2021_SEX_COEFFICIENTS.items()
    )
    return f"""
WITH analyte_concepts AS (
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE vocabulary_id = 'LOINC'
    AND standard_concept = 'S'
    AND concept_code IN UNNEST(@loinc_codes)
),
unit_conversions AS (
  SELECT
    unit_code,
    @unit_scales[OFFSET(unit_position)] AS unit_scale,
    @unit_offsets[OFFSET(unit_position)] AS unit_offset
  FROM UNNEST(@unit_codes) AS unit_code WITH OFFSET AS unit_position
),
acute_care_visit_concepts AS (
  {_descendant_concepts_sql(dataset, "Visit", "acute_care_visit_codes")}
),
pregnancy_concepts AS (
  {_descendant_concepts_sql(dataset, "SNOMED", "pregnancy_snomed_codes")}
  EXCEPT DISTINCT
  {_descendant_concepts_sql(dataset, "SNOMED", "non_pregnancy_snomed_codes")}
),
pregnancy_evidence_dates AS (
  SELECT person_id, condition_start_date AS evidence_date
  FROM `{dataset}.condition_occurrence`
  WHERE condition_concept_id IN (SELECT concept_id FROM pregnancy_concepts)
  UNION DISTINCT
  SELECT person_id, observation_date AS evidence_date
  FROM `{dataset}.observation`
  WHERE observation_concept_id IN (SELECT concept_id FROM pregnancy_concepts)
),
treatment_concepts AS (
  {_descendant_concepts_sql(dataset, "ATC", "treatment_atc_codes")}
),
treatment_starts AS (
  SELECT person_id, MIN(drug_exposure_start_date) AS first_treatment_date
  FROM `{dataset}.drug_exposure`
  WHERE drug_concept_id IN (SELECT concept_id FROM treatment_concepts)
  GROUP BY person_id
),
analyte_rows AS (
  SELECT
    measurement.person_id,
    measurement.measurement_date,
    -- year_of_birth is the only birth field OMOP requires; a mid-year birthday
    -- makes the age unbiased with error at most half a year.
    DATE_DIFF(measurement.measurement_date, DATE(person.year_of_birth, 7, 1), DAY) / 365.25 AS age_years,
    CASE LOWER(sex_concept.concept_name)
      WHEN 'female' THEN 'female'
      WHEN 'male' THEN 'male'
    END AS biological_sex,
    measurement.value_as_number * unit_conversions.unit_scale + unit_conversions.unit_offset AS canonical_value,
    COALESCE(
      unit.concept_code,
      CONCAT('unit_concept_id=', COALESCE(CAST(measurement.unit_concept_id AS STRING), 'NULL'))
    ) AS unit_label,
    unit_conversions.unit_code IS NOT NULL AS unit_recognized,
    (operator.concept_id IS NOT NULL AND operator.concept_code != '{EQUALS_OPERATOR_SNOMED_CODE}')
      OR COALESCE(measurement.value_source_value, '') LIKE '%<%'
      OR COALESCE(measurement.value_source_value, '') LIKE '%>%' AS censored,
    measurement.measurement_type_concept_id = {SELF_REPORT_TYPE_CONCEPT_ID} AS self_reported,
    acute_care_visit_concepts.concept_id IS NOT NULL AS acute_care,
    EXISTS (
      SELECT 1
      FROM pregnancy_evidence_dates
      WHERE pregnancy_evidence_dates.person_id = measurement.person_id
        AND measurement.measurement_date
          BETWEEN DATE_SUB(pregnancy_evidence_dates.evidence_date, INTERVAL {PREGNANCY_WINDOW_DAYS_BEFORE} DAY)
          AND DATE_ADD(pregnancy_evidence_dates.evidence_date, INTERVAL {PREGNANCY_WINDOW_DAYS_AFTER} DAY)
    ) AS in_pregnancy_window,
    COALESCE(measurement.measurement_date >= treatment_starts.first_treatment_date, FALSE) AS on_treatment
  FROM `{dataset}.measurement` AS measurement
  JOIN analyte_concepts
    ON analyte_concepts.concept_id = measurement.measurement_concept_id
  JOIN `{dataset}.person` AS person
    ON person.person_id = measurement.person_id
  LEFT JOIN `{dataset}.concept` AS sex_concept
    ON sex_concept.concept_id = person.sex_at_birth_concept_id
  LEFT JOIN `{dataset}.concept` AS unit
    ON unit.concept_id = measurement.unit_concept_id
    AND unit.vocabulary_id = 'UCUM'
  LEFT JOIN unit_conversions
    ON unit_conversions.unit_code = unit.concept_code
  LEFT JOIN `{dataset}.concept` AS operator
    ON operator.concept_id = measurement.operator_concept_id
    AND operator.concept_id != 0
  LEFT JOIN `{dataset}.visit_occurrence` AS visit_occurrence
    ON visit_occurrence.visit_occurrence_id = measurement.visit_occurrence_id
  LEFT JOIN acute_care_visit_concepts
    ON acute_care_visit_concepts.concept_id = visit_occurrence.visit_concept_id
  LEFT JOIN treatment_starts
    ON treatment_starts.person_id = measurement.person_id
  WHERE measurement.value_as_number IS NOT NULL
),
classified_rows AS (
  SELECT
    *,
    CASE
      WHEN censored THEN 'censored'
      WHEN NOT unit_recognized THEN 'unrecognized_unit'
      WHEN canonical_value < @plausible_low OR canonical_value > @plausible_high THEN 'implausible'
      WHEN self_reported THEN 'self_reported'
      WHEN age_years < {ADULT_AGE_YEARS} THEN 'under_adult_age'
      WHEN acute_care THEN 'acute_care'
      WHEN in_pregnancy_window THEN 'pregnancy'
      WHEN @value_formula = 'ckd_epi_2021' AND biological_sex IS NULL THEN 'sex_unknown'
    END AS exclusion_reason
  FROM analyte_rows
),
valued_rows AS (
  -- BigQuery never evaluates an untaken CASE branch, so POW and LN below
  -- only ever see retained (plausible, positive) values.
  SELECT
    *,
    CASE
      WHEN exclusion_reason IS NOT NULL THEN NULL
      WHEN @value_formula = 'identity' THEN canonical_value
      WHEN @value_formula = 'ckd_epi_2021' THEN
        CASE biological_sex
{egfr_branches}
        END
    END AS linear_value
  FROM classified_rows
),
person_days AS (
  -- One occasion per person-day: same-day repeats are averaged on the
  -- analysis scale. age_years and on_treatment are functions of the day.
  SELECT
    person_id,
    measurement_date,
    age_years,
    on_treatment,
    COUNT(*) AS row_count,
{day_exclusion_counts},
    COUNTIF(exclusion_reason IS NULL) AS retained_row_count,
    AVG(
      CASE
        WHEN exclusion_reason IS NOT NULL THEN NULL
        WHEN @log_scale THEN LN(linear_value + @log_offset)
        ELSE linear_value
      END
    ) AS occasion_value,
    ARRAY_AGG(
      DISTINCT IF(exclusion_reason = 'unrecognized_unit', unit_label, NULL) IGNORE NULLS
    ) AS unrecognized_unit_labels
  FROM valued_rows
  GROUP BY person_id, measurement_date, age_years, on_treatment
),
person_summaries AS (
  SELECT
    person_id,
    SUM(row_count) AS measurement_row_count,
{person_exclusion_counts},
{occasion_summary_columns},
    ARRAY_CONCAT_AGG(unrecognized_unit_labels) AS unrecognized_unit_labels
  FROM person_days
  GROUP BY person_id
)
SELECT
  CAST(person_summaries.person_id AS STRING) AS sample_id,
  CAST(person_summaries.person_id AS STRING) AS person_id,
  person_summaries.measurement_row_count,
{selected_columns},
  person_summaries.unrecognized_unit_labels,
  person.sex_at_birth_concept_id,
  person.race_concept_id,
  person.ethnicity_concept_id
FROM person_summaries
JOIN `{dataset}.person` AS person
  ON person.person_id = person_summaries.person_id
ORDER BY person_summaries.person_id
""".strip()


def build_all_of_us_measurement_query_parameters(
    definition: MeasurementDefinition,
) -> dict[str, tuple[str, Any]]:
    """Query parameter name -> (GoogleSQL type, value); list values are ARRAY<type>."""
    treatment_atc_codes = () if definition.treatment is None else definition.treatment.medication.atc_codes
    plausible_low, plausible_high = definition.plausible_range
    return {
        "loinc_codes": ("STRING", list(definition.loinc_codes)),
        "unit_codes": ("STRING", [conversion.ucum_code for conversion in definition.unit_conversions]),
        "unit_scales": ("FLOAT64", [conversion.scale for conversion in definition.unit_conversions]),
        "unit_offsets": ("FLOAT64", [conversion.offset for conversion in definition.unit_conversions]),
        "plausible_low": ("FLOAT64", plausible_low),
        "plausible_high": ("FLOAT64", plausible_high),
        "log_scale": ("BOOL", definition.log_scale),
        "log_offset": ("FLOAT64", definition.log_offset),
        "value_formula": ("STRING", definition.value_formula),
        "treatment_atc_codes": ("STRING", list(treatment_atc_codes)),
        "acute_care_visit_codes": ("STRING", list(ACUTE_CARE_VISIT_CODES)),
        "pregnancy_snomed_codes": ("STRING", list(PREGNANCY_SNOMED_CODES)),
        "non_pregnancy_snomed_codes": ("STRING", list(NON_PREGNANCY_SNOMED_CODES)),
    }


def build_all_of_us_measurement_query_config(definition: MeasurementDefinition) -> "bigquery.QueryJobConfig":
    from google.cloud import bigquery

    query_parameters: list[bigquery.ArrayQueryParameter | bigquery.ScalarQueryParameter] = []
    for name, (parameter_type, value) in build_all_of_us_measurement_query_parameters(definition).items():
        if isinstance(value, list):
            query_parameters.append(bigquery.ArrayQueryParameter(name, parameter_type, value))
        else:
            query_parameters.append(bigquery.ScalarQueryParameter(name, parameter_type, value))
    return bigquery.QueryJobConfig(query_parameters=query_parameters)


def fetch_all_of_us_measurement_rows(
    definition: MeasurementDefinition,
    client: bigquery.Client | None = None,
) -> list[dict[str, Any]]:
    active_client = _active_bigquery_client(client)
    return _query_rows(
        active_client,
        build_all_of_us_measurement_sql(),
        build_all_of_us_measurement_query_config(definition),
    )


def build_all_of_us_measurement_targets(
    definition: MeasurementDefinition,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...], dict[str, Any]]:
    """Turn per-person query rows into one training row per person.

    Returns (training rows, one-hot covariate columns, summary for the metadata).
    A person's occasions are the untreated ones when there are any; otherwise
    the treated ones corrected to their untreated equivalent by the trait's
    TreatmentRule; persons with neither are dropped. The target is the
    empirical BLUP of the person's long-run mean (_person_blup, with variance
    components from _estimate_person_variance_components);
    target_inverse_normal is its rank-based inverse normal transform.
    """
    selected_rows: list[dict[str, Any]] = []
    summaries: list[_OccasionSummary] = []
    sources: list[str] = []
    n_without_retained_occasion = 0
    n_treated_only_excluded = 0
    for row in rows:
        untreated = _occasion_summary(row, "untreated")
        treated = _occasion_summary(row, "treated")
        if treated is not None and definition.treatment is None:
            raise ValueError(
                f"{definition.canonical_name}: treated occasions returned for a trait without a medication rule"
            )
        if untreated is not None:
            summaries.append(untreated)
            sources.append("untreated")
        elif treated is None:
            n_without_retained_occasion += 1
            continue
        elif definition.treatment is None or definition.treatment.correction is TreatmentCorrection.EXCLUDE:
            n_treated_only_excluded += 1
            continue
        else:
            summaries.append(_untreated_equivalent(treated, definition))
            sources.append("treated_corrected")
        selected_rows.append(row)
    if not summaries:
        raise ValueError(f"{definition.canonical_name}: no person has a retained measurement occasion")

    occasion_counts = np.array([summary.count for summary in summaries], dtype=np.float64)
    person_means = np.array([summary.mean for summary in summaries], dtype=np.float64)
    within_sum_squares = occasion_counts * np.array([summary.variance for summary in summaries], dtype=np.float64)
    mean_ages = np.array([summary.mean_age for summary in summaries], dtype=np.float64)
    mean_squared_ages = np.array([summary.mean_age_squared for summary in summaries], dtype=np.float64)
    design = _person_design(mean_ages, mean_squared_ages, [row.get("sex_at_birth_concept_id") for row in selected_rows])
    between_variance, within_variance = _estimate_person_variance_components(
        occasion_counts, person_means, within_sum_squares, design
    )
    targets, reliabilities = _person_blup(occasion_counts, person_means, design, between_variance, within_variance)
    inverse_normal_targets = _rank_inverse_normal(targets)

    training_rows = [
        {
            "sample_id": row["sample_id"],
            "person_id": row["person_id"],
            "target": float(target),
            "target_inverse_normal": float(inverse_normal_target),
            "occasion_count": occasion_summary.count,
            "target_reliability": float(reliability),
            "measurement_source": source,
            "age_at_measurement": occasion_summary.mean_age,
            "age_at_measurement_squared": occasion_summary.mean_age_squared,
            **{column: row.get(column) for column in MEASUREMENT_CATEGORICAL_COVARIATES},
        }
        for row, occasion_summary, source, target, inverse_normal_target, reliability in zip(
            selected_rows, summaries, sources, targets, inverse_normal_targets, reliabilities, strict=True
        )
    ]
    encoded_categorical_columns = _add_one_hot_omop_categorical_covariates(
        training_rows, MEASUREMENT_CATEGORICAL_COVARIATES
    )
    unrecognized_unit_persons: Counter[str] = Counter()
    for row in rows:
        unrecognized_unit_persons.update(set(row.get("unrecognized_unit_labels") or ()))
    summary = {
        "n_persons_with_analyte_rows": len(rows),
        "n_persons": len(training_rows),
        "n_persons_untreated": sources.count("untreated"),
        "n_persons_treated_corrected": sources.count("treated_corrected"),
        "n_persons_without_retained_occasion": n_without_retained_occasion,
        "n_persons_treated_only_excluded": n_treated_only_excluded,
        "n_measurement_rows": sum(int(row["measurement_row_count"]) for row in rows),
        "excluded_row_counts": {
            reason: sum(int(row[f"{reason}_row_count"]) for row in rows)
            for reason in MEASUREMENT_EXCLUSION_REASONS
        },
        "unrecognized_unit_person_counts": dict(unrecognized_unit_persons.most_common()),
        "n_occasions": int(occasion_counts.sum()),
        "occasions_per_person_quartiles": [float(value) for value in np.quantile(occasion_counts, (0.25, 0.5, 0.75))],
        "between_person_variance": between_variance,
        "within_person_variance": within_variance,
        "repeatability": between_variance / (between_variance + within_variance),
        "mean_target_reliability": float(reliabilities.mean()),
        "target_mean": float(targets.mean()),
        "target_sd": float(targets.std()),
    }
    return training_rows, encoded_categorical_columns, summary


def prepare_all_of_us_measurement_sample_table(
    trait: str,
    output_path: str | Path,
    *,
    client: bigquery.Client | None = None,
) -> AllOfUsPreparedPhenotype:
    definition = resolve_measurement_definition(trait)
    rows = fetch_all_of_us_measurement_rows(definition, client=client)
    billing_project = _resolve_billing_project(client)
    sample_table_path = Path(output_path)
    sample_table_path.parent.mkdir(parents=True, exist_ok=True)
    training_rows, encoded_categorical_columns, summary = build_all_of_us_measurement_targets(definition, rows)
    LOGGER.info(
        "Prepared All of Us trait %s: n_persons=%d repeatability=%.3f mean_reliability=%.3f",
        definition.canonical_name,
        summary["n_persons"],
        summary["repeatability"],
        summary["mean_target_reliability"],
    )

    header = (
        "sample_id",
        "person_id",
        "target",
        "target_inverse_normal",
        "occasion_count",
        "target_reliability",
        "measurement_source",
        "age_at_measurement",
        "age_at_measurement_squared",
        *encoded_categorical_columns,
    )
    _write_sample_table(sample_table_path, header, training_rows)

    sql_path = sample_table_path.with_suffix(sample_table_path.suffix + ".sql")
    sql_path.write_text(build_all_of_us_measurement_sql() + "\n", encoding="utf-8")

    treatment = definition.treatment
    metadata_path = sample_table_path.with_suffix(sample_table_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "trait": definition.canonical_name,
                "description": definition.description,
                "rationale": definition.rationale,
                "loinc_codes": list(definition.loinc_codes),
                "canonical_unit": definition.canonical_unit,
                "analysis_scale": (
                    f"log(value + {definition.log_offset:g})" if definition.log_scale else "linear"
                ),
                "value_formula": definition.value_formula,
                "treatment": None if treatment is None else {
                    "medication": treatment.medication.name,
                    "atc_codes": list(treatment.medication.atc_codes),
                    "correction": treatment.correction.value,
                    "amount": treatment.amount,
                    "citation": treatment.citation,
                },
                "query_parameters": {
                    name: value
                    for name, (_parameter_type, value) in build_all_of_us_measurement_query_parameters(definition).items()
                },
                "adult_age_years": ADULT_AGE_YEARS,
                "pregnancy_window_days": [PREGNANCY_WINDOW_DAYS_BEFORE, PREGNANCY_WINDOW_DAYS_AFTER],
                "self_report_type_concept_id": SELF_REPORT_TYPE_CONCEPT_ID,
                "target_definition": (
                    "empirical BLUP of the person's long-run mean on the analysis scale (random-intercept "
                    "model, mean model: intercept, mean age, mean squared age, sex at birth); "
                    "target_inverse_normal is its Blom rank inverse normal transform"
                ),
                "billing_project_env": "GOOGLE_PROJECT",
                "cdr_dataset_env": "WORKSPACE_CDR",
                "billing_project": billing_project,
                "cdr_dataset": _require_env("WORKSPACE_CDR"),
                **summary,
                "encoded_categorical_covariates": encoded_categorical_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return AllOfUsPreparedPhenotype(
        sample_table_path=sample_table_path,
        sql_path=sql_path,
        metadata_path=metadata_path,
    )


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _find_named_definition(
    definitions: tuple[_NamedDefinition, ...],
    name: str,
) -> _NamedDefinition | None:
    normalized_name = _normalize_name(name)
    for definition in definitions:
        candidate_names = (definition.canonical_name, *definition.aliases)
        if normalized_name in {_normalize_name(candidate_name) for candidate_name in candidate_names}:
            return definition
    return None


def _active_bigquery_client(client: bigquery.Client | None) -> bigquery.Client:
    from google.cloud import bigquery

    return client if client is not None else bigquery.Client(project=_require_env("GOOGLE_PROJECT"))


def _query_rows(
    client: bigquery.Client,
    sql: str,
    job_config: bigquery.QueryJobConfig,
) -> list[dict[str, Any]]:
    query_job = client.query(sql, job_config=job_config)
    return [dict(row.items()) for row in query_job.result()]


def _write_sample_table(path: Path, header: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_format_value(row.get(column_name)) for column_name in header])


# BigQuery project IDs and dataset IDs must match a restrictive identifier
# grammar (letters, digits, underscores, hyphens; dataset IDs additionally
# allow a single dot for project.dataset qualification). Validating the
# WORKSPACE_CDR env var here closes a SQL-injection vector: the value is
# interpolated unescaped into the GoogleSQL inside backticks (e.g.
# `{dataset}.concept`), so a hostile WORKSPACE_CDR like
# `proj.dset`.foo`; DROP TABLE x; --` could otherwise break out of the
# identifier. AoU workbenches set this to something like
# `fc-aou-cdr-prod.R2024Q3R3`, which matches the regex; anything that
# doesn't match is rejected loudly.
_BQ_QUALIFIED_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*$")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError("Missing required All of Us environment variable: " + name)
    stripped = value.strip()
    if name in ("WORKSPACE_CDR", "GOOGLE_PROJECT"):
        if not _BQ_QUALIFIED_NAME_RE.match(stripped):
            raise ValueError(
                f"Refusing to use {name}={stripped!r}: BigQuery project / dataset "
                "identifiers may contain only letters, digits, underscore, hyphen, "
                "and a single '.' separator."
            )
    return stripped


def _resolve_billing_project(client: bigquery.Client | None) -> str:
    if client is not None:
        client_project = getattr(client, "project", None)
        if isinstance(client_project, str) and client_project.strip():
            return client_project.strip()
    return _require_env("GOOGLE_PROJECT")


def _prepare_training_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...], dict[str, int]]:
    training_rows: list[dict[str, Any]] = []
    n_cases = 0
    n_controls = 0
    n_excluded_ambiguous = 0

    for row in rows:
        phenotype_occurrence_count = _parse_phenotype_occurrence_count(row)
        if phenotype_occurrence_count >= MIN_DISEASE_OCCURRENCES:
            training_row = dict(row)
            training_row["target"] = 1
            training_rows.append(training_row)
            n_cases += 1
        elif phenotype_occurrence_count == 0:
            training_row = dict(row)
            training_row["target"] = 0
            training_rows.append(training_row)
            n_controls += 1
        else:
            # One-code participants are neither clean controls nor cases, so exclude
            # them from training instead of diluting controls with ambiguous records.
            n_excluded_ambiguous += 1

    encoded_categorical_columns = _add_one_hot_omop_categorical_covariates(training_rows, OMOP_CATEGORICAL_COVARIATES)
    return (
        training_rows,
        encoded_categorical_columns,
        {
            "n_cases": n_cases,
            "n_controls": n_controls,
            "n_excluded_ambiguous": n_excluded_ambiguous,
        },
    )


def _add_one_hot_omop_categorical_covariates(
    rows: list[dict[str, Any]],
    categorical_columns: tuple[str, ...],
) -> tuple[str, ...]:
    encoded_column_names: list[str] = []
    for categorical_column in categorical_columns:
        # Every observed level gets a column. The reference level is chosen
        # once, downstream, by aou_runner._expand_one_hot_covariates (it drops
        # the majority level); dropping one here as well merged two levels
        # into the reference.
        encoded_concept_ids = sorted(
            {
                _parse_concept_id(categorical_column, row.get(categorical_column))
                for row in rows
                if row.get(categorical_column) not in (None, "")
            }
        )
        column_names = tuple(
            f"{categorical_column}_{concept_id}"
            for concept_id in encoded_concept_ids
        )
        encoded_column_names.extend(column_names)
        for row in rows:
            row_concept_id = row.get(categorical_column)
            parsed_concept_id = (
                _parse_concept_id(categorical_column, row_concept_id)
                if row_concept_id not in (None, "")
                else None
            )
            for concept_id, column_name in zip(encoded_concept_ids, column_names, strict=True):
                row[column_name] = 1 if parsed_concept_id == concept_id else 0
            row.pop(categorical_column, None)
    return tuple(encoded_column_names)


def _parse_phenotype_occurrence_count(row: dict[str, Any]) -> int:
    value = row.get("phenotype_occurrence_count")
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid phenotype_occurrence_count: " + str(value)) from error


def _parse_concept_id(column_name: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid " + column_name + ": " + str(value)) from error


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _descendant_concepts_sql(dataset: str, vocabulary_id: str, codes_parameter: str) -> str:
    return f"""SELECT DISTINCT concept_ancestor.descendant_concept_id AS concept_id
  FROM `{dataset}.concept` AS concept
  JOIN `{dataset}.concept_ancestor` AS concept_ancestor
    ON concept_ancestor.ancestor_concept_id = concept.concept_id
  WHERE concept.vocabulary_id = '{vocabulary_id}'
    AND concept.concept_code IN UNNEST(@{codes_parameter})"""


@dataclass(frozen=True, slots=True)
class _OccasionSummary:
    """Sufficient statistics of one person's occasions on the analysis scale."""

    count: int
    mean: float
    # Population variance of the occasion values (0 for a single occasion).
    variance: float
    mean_age: float
    mean_age_squared: float


def _occasion_summary(row: dict[str, Any], group: str) -> _OccasionSummary | None:
    count = row.get(f"{group}_occasion_count")
    if count is None or int(count) == 0:
        return None
    return _OccasionSummary(
        count=int(count),
        mean=float(row[f"{group}_mean"]),
        variance=float(row[f"{group}_variance"]),
        mean_age=float(row[f"{group}_mean_age"]),
        mean_age_squared=float(row[f"{group}_mean_age_squared"]),
    )


def _untreated_equivalent(summary: _OccasionSummary, definition: MeasurementDefinition) -> _OccasionSummary:
    """Apply the trait's treatment correction to every treated occasion.

    The correction is affine in the occasion value on the analysis scale, so it
    maps the sufficient statistics exactly: y + a shifts the mean; y / a scales
    the mean by 1/a and the variance by 1/a^2 on the linear scale, and shifts
    the mean by -log(a) on the log scale (log_offset is 0 there).
    """
    treatment = definition.treatment
    if treatment is None or treatment.amount is None:
        raise ValueError(f"{definition.canonical_name}: no treatment correction to apply")
    if treatment.correction is TreatmentCorrection.ADD:
        mean, variance = summary.mean + treatment.amount, summary.variance
    elif definition.log_scale:
        mean, variance = summary.mean - math.log(treatment.amount), summary.variance
    else:
        mean, variance = summary.mean / treatment.amount, summary.variance / treatment.amount**2
    return _OccasionSummary(summary.count, mean, variance, summary.mean_age, summary.mean_age_squared)


def _person_design(
    mean_ages: np.ndarray,
    mean_squared_ages: np.ndarray,
    sex_concept_ids: list[Any],
) -> np.ndarray:
    """Person-level mean model: intercept, centered mean age and mean squared
    age (exact for a quadratic age trend at the measurement level), and one
    indicator per sex-at-birth level other than the most common one (a missing
    value is its own level)."""
    sex_levels = ["missing" if value in (None, "") else str(value) for value in sex_concept_ids]
    level_counts = Counter(sex_levels)
    reference_level = max(level_counts, key=lambda level: (level_counts[level], level))
    columns = [
        np.ones_like(mean_ages),
        mean_ages - mean_ages.mean(),
        mean_squared_ages - mean_squared_ages.mean(),
    ]
    for level in sorted(level_counts):
        if level != reference_level:
            columns.append(np.array([1.0 if sex_level == level else 0.0 for sex_level in sex_levels]))
    return np.column_stack(columns)


def _estimate_person_variance_components(
    occasion_counts: np.ndarray,
    person_means: np.ndarray,
    within_sum_squares: np.ndarray,
    design: np.ndarray,
) -> tuple[float, float]:
    """Closed-form, exactly unbiased moment estimators of (sigma_b^2, sigma_e^2).

    Model for person i with k_i occasions and person-level covariates x_i:
    ybar_i = x_i'gamma + b_i + ebar_i, so Var(ybar_i) = sigma_b^2 + sigma_e^2 / k_i.
    Within persons, E[sum_i SSW_i] = sigma_e^2 (N - m) for N occasions and m
    persons. Between persons, the OLS residuals r = (I - H) ybar of ybar on X
    satisfy E[r'r] = tr((I - H) V) = sigma_b^2 (m - p) + sigma_e^2 sum_i (1 - h_ii) / k_i,
    with h_ii the leverages and p = rank(X). Solving both moment equations gives
    the estimators below (Henderson's method III for the one-way model).
    """
    person_count, parameter_count = design.shape
    within_degrees_of_freedom = float(occasion_counts.sum()) - person_count
    if within_degrees_of_freedom <= 0:
        raise ValueError("within-person variance is not identifiable: no person has a repeated occasion")
    if np.linalg.matrix_rank(design) < parameter_count or person_count <= parameter_count:
        raise ValueError("person-level mean model is rank deficient or has no residual degrees of freedom")
    within_variance = float(within_sum_squares.sum()) / within_degrees_of_freedom
    orthonormal_basis, _ = np.linalg.qr(design)
    leverages = np.sum(orthonormal_basis**2, axis=1)
    residuals = person_means - orthonormal_basis @ (orthonormal_basis.T @ person_means)
    between_variance = (
        float(residuals @ residuals) - within_variance * float(np.sum((1.0 - leverages) / occasion_counts))
    ) / (person_count - parameter_count)
    if not between_variance > 0.0:
        raise ValueError(
            "no between-person variance: the moment estimate of sigma_b^2 is "
            f"{between_variance:.6g}, so person means carry no signal beyond occasion noise"
        )
    return between_variance, within_variance


def _person_blup(
    occasion_counts: np.ndarray,
    person_means: np.ndarray,
    design: np.ndarray,
    between_variance: float,
    within_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """BLUP of each person's long-run mean T_i = x_i'gamma + b_i and its reliability.

    With V_i = sigma_b^2 + sigma_e^2 / k_i, gamma is the GLS estimate (weights
    1/V_i) and the reliability w_i = sigma_b^2 / V_i, the shrinkage of
    ybar_i - x_i'gamma, equals k_i rho / (1 + (k_i - 1) rho) with the
    repeatability rho. This is Henderson's mixed-model-equation solution
    collapsed to per-person means (exact: ybar_i is sufficient for b_i).
    Regressing the BLUP on genotypes reproduces the numerator of the efficient
    weighted regression of ybar_i with weights 1/V_i (w_i is proportional to
    1/V_i), with effects scaled by about the mean reliability.
    """
    mean_variances = between_variance + within_variance / occasion_counts
    root_weights = 1.0 / np.sqrt(mean_variances)
    fixed_effects, *_ = np.linalg.lstsq(design * root_weights[:, None], person_means * root_weights, rcond=None)
    fitted_means = design @ fixed_effects
    reliabilities = between_variance / mean_variances
    return fitted_means + reliabilities * (person_means - fitted_means), reliabilities


def _rank_inverse_normal(values: np.ndarray) -> np.ndarray:
    """Blom rank inverse normal transform, ties at their average rank."""
    ranks = rankdata(values, method="average")
    return ndtri((ranks - 0.375) / (len(values) + 0.25))
