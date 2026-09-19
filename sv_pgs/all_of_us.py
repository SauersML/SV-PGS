from __future__ import annotations

import csv
import hashlib
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
# Symmetric EHR-depth floor (pgsEngine C06): cases and controls alike need at
# least this many distinct condition dates on or before the landmark, the
# first observation-period start plus EHR_DEPTH_LANDMARK_DAYS. The landmark
# does not depend on the outcome, so the floor selects on pre-outcome EHR use
# only.
MIN_PRE_LANDMARK_CONDITION_DATES = 5
EHR_DEPTH_LANDMARK_DAYS = 365
LOGGER = logging.getLogger(__name__)
# Every phenotype adjusts for biological sex (All of Us
# person.sex_at_birth_concept_id), not gender identity; ancestry enters
# through genetic PCs, not self-reported race or ethnicity.
PHENOTYPE_CATEGORICAL_COVARIATES = ("sex_at_birth_concept_id",)


@dataclass(frozen=True, slots=True)
class MedicationClass:
    """A drug class; ATC class codes are expanded through concept_ancestor to
    every RxNorm and RxNorm Extension drug in drug_exposure."""

    name: str
    atc_codes: tuple[str, ...]


# Statins and statin combinations, ezetimibe, PCSK9 inhibitors (evolocumab,
# alirocumab), bempedoic acid and inclisiran: the drugs the LDL/0.7 convention
# is about. Fibrates, niacin and omega-3 (elsewhere in C10) barely move LDL.
LDL_LOWERING = MedicationClass(
    "LDL-lowering therapy",
    ("C10AA", "C10BA", "C10BX", "C10AX09", "C10AX13", "C10AX14", "C10AX15", "C10AX16"),
)
LIPID_MODIFYING = MedicationClass("lipid-modifying therapy", ("C10",))
ANTIHYPERTENSIVE = MedicationClass("blood-pressure-lowering therapy", ("C02", "C03", "C07", "C08", "C09"))
GLUCOSE_LOWERING = MedicationClass("glucose-lowering therapy", ("A10",))
# GLP-1 analogues and tirzepatide (A10BX16) are the dominant weight-loss drugs
# and are classed under A10, not A08.
WEIGHT_LOWERING = MedicationClass("weight-lowering therapy", ("A08", "A10BJ", "A10BX16"))
HEART_RATE_LOWERING = MedicationClass(
    "heart-rate-lowering therapy",
    ("C07", "C08D", "C01EB17", "C01AA"),
)


@dataclass(frozen=True, slots=True)
class DiseaseDefinition:
    canonical_name: str
    aliases: tuple[str, ...]
    description: str
    snomed_code: str
    snomed_concept_name: str
    # Any record of these SNOMED roots (or their descendants) removes a person
    # from the controls (phecode-style control exclusions) ...
    control_exclusion_snomed_codes: tuple[str, ...] = ()
    # ... and of these from cases and controls alike.
    ambiguous_snomed_codes: tuple[str, ...] = ()
    # A treatment of the disease: one diagnosis date plus an exposure makes a
    # case, and an exposure without case status removes a person from the
    # controls.
    disease_medication: MedicationClass | None = None


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
        # Controls exclude any hypertensive disorder and antihypertensive users;
        # one code plus an antihypertensive is a case.
        control_exclusion_snomed_codes=("38341003",),
        disease_medication=ANTIHYPERTENSIVE,
    ),
    DiseaseDefinition(
        canonical_name="type2_diabetes",
        aliases=("t2d", "type_2_diabetes", "type 2 diabetes", "diabetes_type_2", "t2dm"),
        description="Type 2 diabetes mellitus phenotype from EHR conditions.",
        snomed_code="44054006",
        snomed_concept_name="Diabetes mellitus type 2",
        # Type 1 diabetes is ambiguous for both groups; one code plus a
        # glucose-lowering drug is a case (as in eMERGE's T2D algorithm); controls
        # exclude any diabetes mellitus (gestational included) and those drugs.
        control_exclusion_snomed_codes=("73211009", "11687002"),
        ambiguous_snomed_codes=("46635009",),
        disease_medication=GLUCOSE_LOWERING,
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
        # Controls exclude COPD.
        control_exclusion_snomed_codes=("13645005",),
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
        # Controls exclude any ischemic heart disease, angina or myocardial
        # infarction.
        control_exclusion_snomed_codes=("414545008", "194828000", "22298006"),
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
# 1. BigQuery (build_all_of_us_measurement_sql), per row: keep rows whose
#    standard or source concept is one of the analyte's LOINC concepts or All
#    of Us physical-measurement (PPI) concepts; convert the row's unit, named
#    as All of Us names it, to the canonical unit; drop censored results
#    ("<5"), unrecognized units, physiologically implausible values,
#    self-reported values, values taken below the trait's minimum age, values
#    within 30 days of an inpatient or emergency stay, values inside a
#    pregnancy window and values inside the trait's clinical exclusion windows;
#    flag values taken on or after the person's first exposure to the trait's
#    medication class. Rows are then collapsed to one value per person-day
#    (same-day repeats are one occasion, as for disease codes) on the analysis
#    scale, and each person is reduced to exact sufficient statistics (occasion
#    count, mean, variance, mean age, mean squared age), separately for
#    untreated and treated occasions.
# 2. Python (build_all_of_us_measurement_targets), per person: combine the two
#    groups by the trait's TreatmentRule, then the target is the empirical
#    BLUP of the person's long-run mean under the random-intercept model
#        y_ij = x_i'gamma + b_i + e_ij,  b_i ~ (0, sigma_b^2), e_ij ~ (0, sigma_e^2),
#    with x_i = (1, mean age, mean squared age, sex, mean age x female); see
#    _estimate_person_variance_components and _person_blup.
#
# The catalogue is the 20-trait panel's 16 quantitative traits (design-traits
# panel20_v1). Codes, units and counts were read off the public All of Us Data
# Browser (CDR 2025Q4R5, 747,040 participants, counts binned to 20); the
# participant counts in the comments are that release's.

ADULT_AGE_YEARS = 18
# Values from an inpatient or emergency stay, or up to 30 days either side of
# it, reflect acute illness (infection, AKI, bleeding, transfusion, stress).
ACUTE_CARE_WINDOW_DAYS = 30
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
# Unit label of a row with no unit concept (unit_concept_id NULL or 0), as
# the All of Us Data Browser labels it.
NO_UNIT_LABEL = "no unit"
# A clinical window side with this many days is unbounded.
UNBOUNDED_WINDOW_DAYS = -1
# Row exclusion reasons in the order the SQL CASE assigns them.
MEASUREMENT_EXCLUSION_REASONS = (
    "censored",
    "unrecognized_unit",
    "implausible",
    "self_reported",
    "under_minimum_age",
    "acute_care",
    "pregnancy",
    "clinical_exclusion",
    "sex_unknown",
)
MEASUREMENT_VALUE_FORMULAS = ("identity", "ckd_epi_2021")
WINDOW_DOMAINS = ("condition", "procedure", "drug")


@dataclass(frozen=True, slots=True)
class UnitConversion:
    """``canonical value = scale * reported value + offset`` for one unit label.

    The label is the lower-cased OMOP concept_name of the row's unit concept
    (NO_UNIT_LABEL when there is none), which is how All of Us reports it.
    """

    unit_label: str
    scale: float
    offset: float = 0.0


class TreatmentCorrection(Enum):
    # Untreated-equivalent value = measured value / amount.
    DIVIDE = "divide"
    # Untreated-equivalent value = measured value + amount.
    ADD = "add"
    # No validated correction exists: treated values are never used.
    EXCLUDE = "exclude"
    # Treated values are used as measured; the fraction of the person's
    # occasions on treatment becomes a covariate.
    INDICATOR = "indicator"


@dataclass(frozen=True, slots=True)
class TreatmentRule:
    medication: MedicationClass
    correction: TreatmentCorrection
    amount: float | None
    citation: str

    def __post_init__(self) -> None:
        corrects = self.correction in (TreatmentCorrection.DIVIDE, TreatmentCorrection.ADD)
        if (self.amount is not None) != corrects:
            raise ValueError("a treatment rule has an amount exactly when it divides or adds")


@dataclass(frozen=True, slots=True)
class WindowConcept:
    """One root concept of a clinical window; its concept_ancestor descendants count too."""

    # "condition" (condition_occurrence), "procedure" (procedure_occurrence)
    # or "drug" (drug_exposure).
    domain: str
    vocabulary_id: str
    concept_code: str


@dataclass(frozen=True, slots=True)
class ClinicalWindow:
    """Measurements dated from days_before before to days_after after any
    record of these concepts are excluded (UNBOUNDED_WINDOW_DAYS: no limit on
    that side, so a window unbounded on both sides excludes the person)."""

    name: str
    concepts: tuple[WindowConcept, ...]
    days_before: int
    days_after: int

    def __post_init__(self) -> None:
        if not self.concepts:
            raise ValueError(f"{self.name}: a clinical window needs concepts")
        if any(concept.domain not in WINDOW_DOMAINS for concept in self.concepts):
            raise ValueError(f"{self.name}: window domains must be one of {WINDOW_DOMAINS}")
        if min(self.days_before, self.days_after) < UNBOUNDED_WINDOW_DAYS:
            raise ValueError(f"{self.name}: window days must be >= 0 or UNBOUNDED_WINDOW_DAYS")


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
    # All of Us physical-measurement (PPI) concepts matched on either concept
    # column, e.g. the enrollment protocol's computed blood-pressure mean ...
    physical_measurement_concept_ids: tuple[int, ...] = ()
    # ... and PPI source concepts whose rows are dropped (the individual
    # readings that protocol mean already summarizes).
    excluded_source_concept_ids: tuple[int, ...] = ()
    treatment: TreatmentRule | None = None
    clinical_windows: tuple[ClinicalWindow, ...] = ()
    minimum_age_years: float = ADULT_AGE_YEARS
    log_offset: float = 0.0
    # "identity" models the measured analyte; "ckd_epi_2021" models the
    # race-free CKD-EPI 2021 eGFR computed per measurement from creatinine.
    value_formula: str = "identity"

    def __post_init__(self) -> None:
        if not self.loinc_codes or len(set(self.loinc_codes)) != len(self.loinc_codes):
            raise ValueError(f"{self.canonical_name}: LOINC codes must be non-empty and unique")
        unit_labels = [conversion.unit_label for conversion in self.unit_conversions]
        if len(set(unit_labels)) != len(unit_labels) or any(label != label.lower() for label in unit_labels):
            raise ValueError(f"{self.canonical_name}: unit labels must be unique and lower case")
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


def _exclude_treated(medication: MedicationClass) -> TreatmentRule:
    return TreatmentRule(medication, TreatmentCorrection.EXCLUDE, None, _NO_CORRECTION)


def _treatment_indicator(medication: MedicationClass) -> TreatmentRule:
    return TreatmentRule(medication, TreatmentCorrection.INDICATOR, None, _NO_CORRECTION_BY_CONVENTION)


def _units(*conversions: tuple[str, float] | tuple[str, float, float]) -> tuple[UnitConversion, ...]:
    return tuple(UnitConversion(*conversion) for conversion in conversions)


def _conditions(*snomed_codes: str) -> tuple[WindowConcept, ...]:
    return tuple(WindowConcept("condition", "SNOMED", code) for code in snomed_codes)


def _drugs(*atc_codes: str) -> tuple[WindowConcept, ...]:
    return tuple(WindowConcept("drug", "ATC", code) for code in atc_codes)


_PELOSO_2014 = (
    "Peloso et al. 2014 AJHG 94:223 (after CTT 2005: lipid-lowering therapy lowers "
    "total cholesterol ~20% and LDL ~30%)"
)
_CUI_TOBIN = "Cui, Hopper & Harrap 2003 Hypertension; Tobin et al. 2005 Stat Med 24:2911"
_NO_CORRECTION = "no validated constant correction; treated values are not used"
_NO_CORRECTION_BY_CONVENTION = "not corrected by convention (Peloso 2014); on-treatment fraction is a covariate"

# Statins and statin combinations, ezetimibe, PCSK9 inhibitors (evolocumab,
# alirocumab), bempedoic acid and inclisiran: the drugs the LDL/0.7 convention
# is about. Fibrates, niacin and omega-3 (elsewhere in C10) barely move LDL.
# Clinical exclusion windows (SNOMED codes checked against the OMOP vocabulary).
# Leukemia, malignant lymphoma, multiple myeloma, myelodysplastic syndrome and
# myeloproliferative disorders, from 180 days before the first record on.
HEMATOLOGIC_MALIGNANCY = ClinicalWindow(
    "hematologic_malignancy",
    _conditions("93143009", "118600007", "109989006", "109995007", "425333006"),
    180,
    UNBOUNDED_WINDOW_DAYS,
)
# Antineoplastic agents: cytopenias and macrocytosis over the following 90 days.
CHEMOTHERAPY = ClinicalWindow("chemotherapy", _drugs("L01"), 0, 90)
# "Administration of blood product" (SNOMED; ICD10PCS transfusion codes are
# its descendants) and CPT 36430: transfused red cells circulate ~120 days.
TRANSFUSION = ClinicalWindow(
    "transfusion",
    (WindowConcept("procedure", "SNOMED", "116762002"), WindowConcept("procedure", "CPT4", "36430")),
    0,
    120,
)
# A systemic antibacterial course marks an infection from a week before to
# two weeks after its start.
ANTIBACTERIAL_COURSE = ClinicalWindow("antibacterial_course", _drugs("J01"), 7, 14)
# Systemic corticosteroids suppress circulating eosinophils for weeks.
SYSTEMIC_CORTICOSTEROID = ClinicalWindow("systemic_corticosteroid", _drugs("H02"), 0, 30)
# Obstruction of bile duct, cholangitis, cholecystitis, calculus of bile duct.
ACUTE_HEPATOBILIARY = ClinicalWindow(
    "acute_hepatobiliary", _conditions("30144000", "82403002", "76581006", "30093007"), 30, 30
)
CIRRHOSIS = ClinicalWindow("cirrhosis", _conditions("19943007"), 0, UNBOUNDED_WINDOW_DAYS)
# Paget's disease of bone (osteitis deformans) raises bone ALP for good ...
PAGET_DISEASE = ClinicalWindow("paget_disease_of_bone", _conditions("2089002"), 0, UNBOUNDED_WINDOW_DAYS)
# ... and fracture healing for about three months.
FRACTURE = ClinicalWindow("fracture", _conditions("125605004"), 0, 90)
# End-stage renal disease, a transplanted kidney, dialysis or kidney transplant
# (SNOMED; CPT hemodialysis/dialysis evaluations and renal allotransplantation).
KIDNEY_REPLACEMENT = ClinicalWindow(
    "kidney_replacement",
    _conditions("46177005", "737295003")
    + (WindowConcept("procedure", "SNOMED", "108241001"), WindowConcept("procedure", "SNOMED", "70536003"))
    + tuple(WindowConcept("procedure", "CPT4", code) for code in ("90935", "90937", "90945", "90947", "50360", "50365")),
    0,
    UNBOUNDED_WINDOW_DAYS,
)
# Non-diabetic HbA1c: any diabetes mellitus record or glucose-lowering drug
# removes the person.
DIABETES = ClinicalWindow(
    "diabetes",
    _conditions("73211009") + _drugs("A10"),
    UNBOUNDED_WINDOW_DAYS,
    UNBOUNDED_WINDOW_DAYS,
)
_BLOOD_COUNT_WINDOWS = (HEMATOLOGIC_MALIGNANCY, CHEMOTHERAPY, TRANSFUSION)
_CHOLESTASIS_WINDOWS = (ACUTE_HEPATOBILIARY, CIRRHOSIS)

# Unit labels All of Us attaches to each analyte whose value histograms (Data
# Browser) match the canonical unit, including mislabeled units: e.g. counts
# reported as "Kelvin per microliter" (K/uL read as kelvin), "nL" or
# "percent"; ALP in "gram per liter"; blood pressure and counts with no unit.
_BLOOD_COUNT_THOUSANDS_PER_MICROLITER = _units(
    ("thousand per microliter", 1.0),
    (NO_UNIT_LABEL, 1.0),
    ("thousand per cubic millimeter", 1.0),
    ("billion per liter", 1.0),
    ("kelvin per microliter", 1.0),
    ("kelvin per cubic millimeter", 1.0),
    ("nl", 1.0),
    ("thousand", 1.0),
    ("number ten", 1.0),
)
_MILLIGRAMS_PER_DECILITER = _units(("milligram per deciliter", 1.0), (NO_UNIT_LABEL, 1.0))
_MILLIMETERS_OF_MERCURY = _units(("millimeter mercury column", 1.0), (NO_UNIT_LABEL, 1.0), ("unit", 1.0))

MEASUREMENT_DEFINITIONS: tuple[MeasurementDefinition, ...] = (
    # --- Anthropometrics and vital signs: enrollment physical measurements
    # (protocol means) plus EHR vitals as repeats.
    MeasurementDefinition(
        canonical_name="height",
        aliases=("body_height", "standing_height"),
        description="Measured adult body height (cm); stated (self-reported) height is excluded.",
        # 8302-2 395,480 participants; 3137-7 10,260; PM 903133 600,520.
        loinc_codes=("8302-2", "3137-7"),
        physical_measurement_concept_ids=(903133,),
        canonical_unit="centimeter",
        # EHR heights with no unit, "per minute" or "pound (US)" are inch
        # values (histograms 59-77); a cm value read as inches is implausible.
        unit_conversions=_units(
            ("centimeter", 1.0),
            ("inch (us)", 2.54000508),
            ("inch (international)", 2.54),
            ("inches", 2.54),
            (NO_UNIT_LABEL, 2.54),
            ("per minute", 2.54),
            ("pound (us)", 2.54),
            ("meter", 100.0),
            ("foot (international)", 30.48),
            ("foot (us)", 30.4800610),
        ),
        plausible_range=(120.0, 230.0),
        log_scale=False,
        minimum_age_years=20.0,
    ),
    MeasurementDefinition(
        canonical_name="body_mass_index",
        aliases=("bmi",),
        description="Body mass index (kg/m2).",
        # 39156-5 337,440 participants; the enrollment BMI (PPI 903124) is not
        # in the CDR (<20). Every unit label on 39156-5 carries kg/m2 values.
        loinc_codes=("39156-5",),
        canonical_unit="kilogram per square meter",
        unit_conversions=_units(
            ("kilogram per square meter", 1.0),
            (NO_UNIT_LABEL, 1.0),
            ("ratio", 1.0),
            ("percent", 1.0),
            ("link for gunter's chain (us)", 1.0),
            ("square meter", 1.0),
            ("unit", 1.0),
        ),
        plausible_range=(12.0, 90.0),
        log_scale=True,
        treatment=_exclude_treated(WEIGHT_LOWERING),
    ),
    MeasurementDefinition(
        canonical_name="systolic_blood_pressure",
        aliases=("sbp",),
        description="Systolic blood pressure (mmHg); the enrollment value is the mean of the 2nd and 3rd readings.",
        # 8480-6 341,580; 8459-0 37,940; 76534-7 400; PM mean 903118 515,400.
        loinc_codes=("8480-6", "8459-0", "76534-7"),
        physical_measurement_concept_ids=(903118,),
        excluded_source_concept_ids=(903109, 903114, 903130),
        canonical_unit="millimeter mercury column",
        unit_conversions=_MILLIMETERS_OF_MERCURY,
        plausible_range=(60.0, 270.0),
        log_scale=False,
        treatment=TreatmentRule(ANTIHYPERTENSIVE, TreatmentCorrection.ADD, 15.0, _CUI_TOBIN),
    ),
    MeasurementDefinition(
        canonical_name="diastolic_blood_pressure",
        aliases=("dbp",),
        description="Diastolic blood pressure (mmHg); the enrollment value is the mean of the 2nd and 3rd readings.",
        # 8462-4 341,620; 8453-3 38,040; 76535-4 340; PM mean 903115 515,400.
        loinc_codes=("8462-4", "8453-3", "76535-4"),
        physical_measurement_concept_ids=(903115,),
        excluded_source_concept_ids=(903110, 903129, 903106),
        canonical_unit="millimeter mercury column",
        unit_conversions=_MILLIMETERS_OF_MERCURY,
        plausible_range=(30.0, 160.0),
        log_scale=False,
        treatment=TreatmentRule(ANTIHYPERTENSIVE, TreatmentCorrection.ADD, 10.0, _CUI_TOBIN),
    ),
    MeasurementDefinition(
        canonical_name="heart_rate",
        aliases=("pulse", "resting_heart_rate"),
        description="Heart rate (beats/min); the enrollment value is the mean of the 2nd and 3rd readings.",
        # 8867-4 365,260; PM mean 903126 515,340. Pulse oximetry (8889-8) is
        # mostly monitored care and is left out.
        loinc_codes=("8867-4",),
        physical_measurement_concept_ids=(903126,),
        excluded_source_concept_ids=(903112, 903105, 903108),
        canonical_unit="per minute",
        unit_conversions=_units(
            ("per minute", 1.0),
            (NO_UNIT_LABEL, 1.0),
            ("counts per minute", 1.0),
            ("% ref", 1.0),
            ("minute", 1.0),
            ("unit", 1.0),
            ("heartbeat", 1.0),
        ),
        plausible_range=(25.0, 220.0),
        log_scale=False,
        treatment=_exclude_treated(HEART_RATE_LOWERING),
    ),
    # --- Blood counts.
    MeasurementDefinition(
        canonical_name="mean_corpuscular_volume",
        aliases=("mcv",),
        description="Mean corpuscular volume of red cells (fL).",
        # 787-2 345,360; 30428-7 48,380. "u/m3" is cubic micrometers (= fL).
        loinc_codes=("787-2", "30428-7"),
        canonical_unit="femtoliter",
        unit_conversions=_units(("femtoliter", 1.0), (NO_UNIT_LABEL, 1.0), ("u/m3", 1.0)),
        plausible_range=(50.0, 150.0),
        log_scale=False,
        clinical_windows=_BLOOD_COUNT_WINDOWS,
    ),
    MeasurementDefinition(
        canonical_name="platelet_count",
        aliases=("platelets", "plt"),
        description="Platelet count (10^3/uL).",
        # 777-3 341,900; 26515-7 68,400; 778-1 1,080 (manual).
        loinc_codes=("777-3", "26515-7", "778-1"),
        canonical_unit="thousand per microliter",
        unit_conversions=_BLOOD_COUNT_THOUSANDS_PER_MICROLITER
        + _units(
            ("microliter", 1.0),
            ("percent", 1.0),
            ("per cubic millimeter", 1.0),
            ("per microliter", 1.0),
            ("thousand per milliliter", 1.0),
            ("billion per microliter", 1.0),
            ("cells per microliter", 1.0),
        ),
        plausible_range=(10.0, 1500.0),
        log_scale=True,
        clinical_windows=_BLOOD_COUNT_WINDOWS,
    ),
    MeasurementDefinition(
        canonical_name="white_blood_cell_count",
        aliases=("wbc", "leukocyte_count"),
        description="Leukocyte count (10^3/uL).",
        # 6690-2 333,820; 26464-8 37,660; 33256-9 39,160 (corrected for
        # nucleated red cells); 804-5 1,340 (manual).
        loinc_codes=("6690-2", "26464-8", "33256-9", "804-5"),
        canonical_unit="thousand per microliter",
        unit_conversions=_BLOOD_COUNT_THOUSANDS_PER_MICROLITER
        + _units(
            ("percent", 1.0),
            ("per microliter", 1.0),
            ("thousand per milliliter", 1.0),
            ("cubic millimeter", 1.0),
            ("ul", 1.0),
            ("billion per microliter", 1.0),
        ),
        plausible_range=(0.5, 100.0),
        log_scale=True,
        clinical_windows=_BLOOD_COUNT_WINDOWS + (ANTIBACTERIAL_COURSE,),
    ),
    MeasurementDefinition(
        canonical_name="eosinophil_count",
        aliases=("absolute_eosinophil_count",),
        description="Absolute eosinophil count (10^3/uL); percentages are a different quantity.",
        # 711-2 284,840; 26449-9 67,600; 712-0 26,900. "percent" rows under
        # these codes are eosinophil percentages and are not accepted.
        loinc_codes=("711-2", "26449-9", "712-0"),
        canonical_unit="thousand per microliter",
        unit_conversions=_BLOOD_COUNT_THOUSANDS_PER_MICROLITER
        + _units(("cells per microliter", 0.001), ("per microliter", 0.001)),
        plausible_range=(0.0, 20.0),
        log_scale=True,
        # Counts are reported to 0.1 x 10^3/uL, so zero is a real reported
        # value: model log(count + half the reporting resolution).
        log_offset=0.05,
        clinical_windows=_BLOOD_COUNT_WINDOWS + (SYSTEMIC_CORTICOSTEROID,),
    ),
    # --- Liver.
    MeasurementDefinition(
        canonical_name="total_bilirubin",
        aliases=("bilirubin",),
        description="Serum/plasma total bilirubin (mg/dL).",
        # 1975-2 336,120.
        loinc_codes=("1975-2",),
        canonical_unit="milligram per deciliter",
        unit_conversions=_MILLIGRAMS_PER_DECILITER,
        plausible_range=(0.05, 30.0),
        log_scale=True,
        clinical_windows=_CHOLESTASIS_WINDOWS,
    ),
    MeasurementDefinition(
        canonical_name="alkaline_phosphatase",
        aliases=("alp", "alk_phos"),
        description="Serum/plasma alkaline phosphatase (U/L).",
        # 6768-6 340,340; "gram per liter" (199,160 participants) and the
        # other labels below carry U/L values.
        loinc_codes=("6768-6",),
        canonical_unit="unit per liter",
        unit_conversions=_units(
            ("unit per liter", 1.0),
            ("gram per liter", 1.0),
            (NO_UNIT_LABEL, 1.0),
            ("international unit per liter", 1.0),
            ("arbitrary unit per liter", 1.0),
            ("% ref", 1.0),
            ("nl", 1.0),
        ),
        plausible_range=(5.0, 3000.0),
        log_scale=True,
        clinical_windows=_CHOLESTASIS_WINDOWS + (PAGET_DISEASE, FRACTURE),
    ),
    # --- Kidney.
    MeasurementDefinition(
        canonical_name="egfr_ckd_epi_2021",
        aliases=("egfr", "egfr_creatinine", "creatinine"),
        description=(
            "Creatinine eGFR, race-free CKD-EPI 2021 (Inker 2021 NEJM 385:1737), mL/min/1.73m2, "
            "computed per measurement from serum creatinine, age and sex at birth."
        ),
        # 2160-0 359,640; 14682-9 10,900. Whole-blood point-of-care creatinine
        # (38483-4) is left out. The "micromole per liter" rows of 14682-9
        # carry mg/dL values (0.3-2.1); a true umol/L value is implausible here.
        loinc_codes=("2160-0", "14682-9"),
        canonical_unit="milligram per deciliter",
        unit_conversions=_MILLIGRAMS_PER_DECILITER + _units(("micromole per liter", 1.0)),
        plausible_range=(0.2, 20.0),
        log_scale=True,
        value_formula="ckd_epi_2021",
        clinical_windows=(KIDNEY_REPLACEMENT,),
    ),
    # --- Lipids.
    MeasurementDefinition(
        canonical_name="hdl_cholesterol",
        aliases=("hdl", "hdl_c"),
        description="HDL cholesterol (mg/dL).",
        # 2085-9 264,760.
        loinc_codes=("2085-9",),
        canonical_unit="milligram per deciliter",
        unit_conversions=_MILLIGRAMS_PER_DECILITER,
        plausible_range=(5.0, 200.0),
        log_scale=True,
        treatment=_treatment_indicator(LIPID_MODIFYING),
    ),
    MeasurementDefinition(
        canonical_name="triglycerides",
        aliases=("tg",),
        description="Serum/plasma triglycerides, fasting or unrecorded fasting status (mg/dL).",
        # 2571-8 268,460; 3048-6 2,340 (fasting); 14927-8 3,180 (its rows
        # carry mg/dL values).
        loinc_codes=("2571-8", "3048-6", "14927-8"),
        canonical_unit="milligram per deciliter",
        unit_conversions=_MILLIGRAMS_PER_DECILITER,
        plausible_range=(10.0, 5000.0),
        log_scale=True,
        treatment=_treatment_indicator(LIPID_MODIFYING),
    ),
    MeasurementDefinition(
        canonical_name="ldl_cholesterol",
        aliases=("ldl", "ldl_c"),
        description="LDL cholesterol, calculated or measured (mg/dL); pre-treatment values preferred.",
        # 13457-7 214,480; 2089-1 90,880; 18262-6 42,500; 49132-4 8,760;
        # 12773-8 2,960.
        loinc_codes=("13457-7", "2089-1", "18262-6", "49132-4", "12773-8"),
        canonical_unit="milligram per deciliter",
        unit_conversions=_MILLIGRAMS_PER_DECILITER + _units(("milligram per deciliter calculated", 1.0)),
        plausible_range=(10.0, 400.0),
        log_scale=False,
        treatment=TreatmentRule(LDL_LOWERING, TreatmentCorrection.DIVIDE, 0.7, _PELOSO_2014),
    ),
    # --- Glycemia.
    MeasurementDefinition(
        canonical_name="hemoglobin_a1c",
        aliases=("hba1c", "a1c"),
        description="Hemoglobin A1c in people without diabetes (NGSP %); IFCC mmol/mol converted.",
        # 4548-4 212,720; 17856-6 65,120; 17855-8 2,860; 4549-2 980; 59261-8 80.
        loinc_codes=("4548-4", "17856-6", "17855-8", "4549-2", "59261-8"),
        canonical_unit="percent",
        # All of Us percent labels (some mislabeled; all carry % values); NGSP
        # % = 0.09148 * IFCC mmol/mol + 2.152 (ngsp.org/ifccngsp.asp).
        unit_conversions=_units(
            ("percent", 1.0),
            (NO_UNIT_LABEL, 1.0),
            ("of total hemoglobin", 1.0),
            ("of total h", 1.0),
            ("percent total protein", 1.0),
            ("percent hemoglobin a1c", 1.0),
            ("percent hemoglobin", 1.0),
            ("percentage of total", 1.0),
            (
                "retired snomed uk drug extension concept, do not use, use concept indicated by the "
                "concept_relationship table, if any",
                1.0,
            ),
            ("millimole per mole", 0.09148, 2.152),
        ),
        plausible_range=(3.0, 20.0),
        log_scale=False,
        clinical_windows=(DIABETES,),
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
    """One row per EHR participant: diagnosis dates, exclusion flags, the
    pre-landmark EHR depth and the ages the liability target needs. The case
    and control rules are applied in _prepare_training_rows."""
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
control_exclusion_concepts AS (
  {_descendant_concepts_sql(dataset, "SNOMED", "control_exclusion_snomed_codes")}
),
ambiguous_concepts AS (
  {_descendant_concepts_sql(dataset, "SNOMED", "ambiguous_snomed_codes")}
),
medication_concepts AS (
  {_descendant_concepts_sql(dataset, "ATC", "disease_medication_atc_codes")}
),
condition_summaries AS (
  -- One pass over condition_occurrence. Occurrences are distinct diagnosis
  -- dates: several condition rows on one day (one visit coded twice) are one
  -- occurrence, not a confirmation. The EHR depth counts distinct condition
  -- dates of any kind on or before the landmark.
  SELECT
    condition_occurrence.person_id,
    COUNT(
      DISTINCT IF(disease_concepts.concept_id IS NOT NULL, condition_occurrence.condition_start_date, NULL)
    ) AS phenotype_occurrence_count,
    MIN(IF(disease_concepts.concept_id IS NOT NULL, condition_occurrence.condition_start_date, NULL))
      AS first_condition_date,
    LOGICAL_OR(control_exclusion_concepts.concept_id IS NOT NULL) AS has_control_exclusion_code,
    LOGICAL_OR(ambiguous_concepts.concept_id IS NOT NULL) AS has_ambiguous_code,
    COUNT(
      DISTINCT IF(
        condition_occurrence.condition_start_date
          <= DATE_ADD(ehr_participants.observation_start_date, INTERVAL {EHR_DEPTH_LANDMARK_DAYS} DAY),
        condition_occurrence.condition_start_date,
        NULL
      )
    ) AS pre_landmark_condition_dates
  FROM `{dataset}.condition_occurrence` AS condition_occurrence
  JOIN ehr_participants
    ON ehr_participants.person_id = condition_occurrence.person_id
  LEFT JOIN disease_concepts
    ON disease_concepts.concept_id = condition_occurrence.condition_concept_id
  LEFT JOIN control_exclusion_concepts
    ON control_exclusion_concepts.concept_id = condition_occurrence.condition_concept_id
  LEFT JOIN ambiguous_concepts
    ON ambiguous_concepts.concept_id = condition_occurrence.condition_concept_id
  GROUP BY condition_occurrence.person_id
),
medicated_persons AS (
  SELECT DISTINCT person_id
  FROM `{dataset}.drug_exposure`
  WHERE drug_concept_id IN (SELECT concept_id FROM medication_concepts)
)
SELECT
  CAST(ehr_participants.person_id AS STRING) AS sample_id,
  CAST(ehr_participants.person_id AS STRING) AS person_id,
  COALESCE(condition_summaries.phenotype_occurrence_count, 0) AS phenotype_occurrence_count,
  condition_summaries.first_condition_date,
  ehr_participants.observation_start_date,
  ehr_participants.observation_end_date,
  primary_consent.primary_consent_date,
  -- year_of_birth is the only birth field OMOP requires; a mid-year birthday
  -- makes the ages unbiased with error at most half a year.
  DATE_DIFF(condition_summaries.first_condition_date, DATE(person.year_of_birth, 7, 1), DAY) / 365.25
    AS age_at_first_condition,
  DATE_DIFF(ehr_participants.observation_end_date, DATE(person.year_of_birth, 7, 1), DAY) / 365.25
    AS age_at_observation_end,
  COALESCE(condition_summaries.pre_landmark_condition_dates, 0) AS pre_landmark_condition_dates,
  COALESCE(condition_summaries.has_control_exclusion_code, FALSE) AS has_control_exclusion_code,
  COALESCE(condition_summaries.has_ambiguous_code, FALSE) AS has_ambiguous_code,
  medicated_persons.person_id IS NOT NULL AS has_disease_medication,
  person.sex_at_birth_concept_id,
  LOWER(sex_concept.concept_name) AS sex_at_birth_name
FROM ehr_participants
JOIN `{dataset}.person` AS person
  ON person.person_id = ehr_participants.person_id
LEFT JOIN `{dataset}.concept` AS sex_concept
  ON sex_concept.concept_id = person.sex_at_birth_concept_id
LEFT JOIN condition_summaries
  ON condition_summaries.person_id = ehr_participants.person_id
LEFT JOIN medicated_persons
  ON medicated_persons.person_id = ehr_participants.person_id
LEFT JOIN primary_consent
  ON primary_consent.person_id = ehr_participants.person_id
ORDER BY ehr_participants.person_id
""".strip()


def build_all_of_us_disease_query_parameters(disease_definition: DiseaseDefinition) -> dict[str, tuple[str, Any]]:
    """Query parameter name -> (GoogleSQL type, value); list values are ARRAY<type>."""
    medication = disease_definition.disease_medication
    return {
        "snomed_code": ("STRING", disease_definition.snomed_code),
        "control_exclusion_snomed_codes": ("STRING", list(disease_definition.control_exclusion_snomed_codes)),
        "ambiguous_snomed_codes": ("STRING", list(disease_definition.ambiguous_snomed_codes)),
        "disease_medication_atc_codes": ("STRING", [] if medication is None else list(medication.atc_codes)),
    }


def build_all_of_us_disease_query_config(disease_definition: DiseaseDefinition) -> "bigquery.QueryJobConfig":
    return _query_config(build_all_of_us_disease_query_parameters(disease_definition))


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


def disease_covariate_columns() -> tuple[str, ...]:
    """Covariates of a disease fit, before one-hot expansion of sex at birth."""
    return (
        "age_at_observation_end",
        "age_at_observation_end_squared",
        "age_at_observation_end_x_female",
        "sex_at_birth_concept_id",
        "log1p_pre_landmark_condition_dates",
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
    training_rows, encoded_categorical_columns, phenotype_counts = _prepare_training_rows(disease_definition, rows)
    LOGGER.info(
        "Prepared All of Us phenotype rows: n_cases=%d n_controls=%d",
        phenotype_counts["n_cases"],
        phenotype_counts["n_controls"],
    )

    header = (
        "sample_id",
        "person_id",
        "target",
        "liability_target",
        "phenotype_occurrence_count",
        "first_condition_date",
        "observation_start_date",
        "observation_end_date",
        "primary_consent_date",
        "age_at_first_condition",
        "age_at_observation_end",
        "age_at_observation_end_squared",
        "age_at_observation_end_x_female",
        "log1p_pre_landmark_condition_dates",
        *encoded_categorical_columns,
    )
    _write_tsv(sample_table_path, header, training_rows)

    sql_path = sample_table_path.with_suffix(sample_table_path.suffix + ".sql")
    sql_path.write_text(build_all_of_us_disease_sql(disease_definition) + "\n", encoding="utf-8")

    medication = disease_definition.disease_medication
    metadata_path = sample_table_path.with_suffix(sample_table_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "disease": disease_definition.canonical_name,
                "description": disease_definition.description,
                "phenotype_fingerprint": phenotype_fingerprint(disease_definition),
                "covariate_columns": list(disease_covariate_columns()),
                "snomed_code": disease_definition.snomed_code,
                "snomed_concept_name": disease_definition.snomed_concept_name,
                "control_exclusion_snomed_codes": list(disease_definition.control_exclusion_snomed_codes),
                "ambiguous_snomed_codes": list(disease_definition.ambiguous_snomed_codes),
                "disease_medication": None if medication is None else {
                    "name": medication.name,
                    "atc_codes": list(medication.atc_codes),
                },
                "min_occurrences": MIN_DISEASE_OCCURRENCES,
                "min_pre_landmark_condition_dates": MIN_PRE_LANDMARK_CONDITION_DATES,
                "ehr_depth_landmark_days": EHR_DEPTH_LANDMARK_DAYS,
                "liability_target_definition": (
                    "E[liability | status, age] under the age-of-onset threshold model; cumulative "
                    "incidence by age from the cohort's own Kaplan-Meier curve within sex at birth"
                ),
                "billing_project_env": "GOOGLE_PROJECT",
                "cdr_dataset_env": "WORKSPACE_CDR",
                "billing_project": billing_project,
                "cdr_dataset": _require_env("WORKSPACE_CDR"),
                "raw_row_count": len(rows),
                "row_count": len(training_rows),
                **phenotype_counts,
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

    Each CTE downstream of the measurement scan is referenced once, so
    BigQuery scans `measurement` once (it re-executes a non-recursive CTE at
    each reference).
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
    AND concept_code IN UNNEST(@loinc_codes)
  UNION DISTINCT
  SELECT concept_id
  FROM UNNEST(@physical_measurement_concept_ids) AS concept_id
),
unit_conversions AS (
  SELECT
    unit_label,
    @unit_scales[OFFSET(unit_position)] AS unit_scale,
    @unit_offsets[OFFSET(unit_position)] AS unit_offset
  FROM UNNEST(@unit_labels) AS unit_label WITH OFFSET AS unit_position
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
window_roots AS (
  SELECT
    window_domain,
    @window_vocabularies[OFFSET(window_position)] AS vocabulary_id,
    @window_concept_codes[OFFSET(window_position)] AS concept_code,
    @window_days_before[OFFSET(window_position)] AS days_before,
    @window_days_after[OFFSET(window_position)] AS days_after
  FROM UNNEST(@window_domains) AS window_domain WITH OFFSET AS window_position
),
window_concepts AS (
  SELECT DISTINCT
    window_roots.window_domain,
    window_roots.days_before,
    window_roots.days_after,
    concept_ancestor.descendant_concept_id AS concept_id
  FROM window_roots
  JOIN `{dataset}.concept` AS concept
    ON concept.vocabulary_id = window_roots.vocabulary_id
    AND concept.concept_code = window_roots.concept_code
  JOIN `{dataset}.concept_ancestor` AS concept_ancestor
    ON concept_ancestor.ancestor_concept_id = concept.concept_id
),
window_evidence AS (
  SELECT clinical_events.person_id, clinical_events.evidence_date, window_concepts.days_before, window_concepts.days_after
  FROM (
    SELECT person_id, 'condition' AS window_domain, condition_concept_id AS concept_id, condition_start_date AS evidence_date
    FROM `{dataset}.condition_occurrence`
    UNION ALL
    SELECT person_id, 'procedure', procedure_concept_id, procedure_date
    FROM `{dataset}.procedure_occurrence`
    UNION ALL
    SELECT person_id, 'drug', drug_concept_id, drug_exposure_start_date
    FROM `{dataset}.drug_exposure`
  ) AS clinical_events
  JOIN window_concepts
    ON window_concepts.concept_id = clinical_events.concept_id
    AND window_concepts.window_domain = clinical_events.window_domain
),
exclusion_windows AS (
  -- Priority 1: acute care; 2: pregnancy; 3: the trait's clinical windows.
  SELECT
    person_id,
    1 AS window_priority,
    DATE_SUB(visit_start_date, INTERVAL {ACUTE_CARE_WINDOW_DAYS} DAY) AS window_start,
    DATE_ADD(COALESCE(visit_end_date, visit_start_date), INTERVAL {ACUTE_CARE_WINDOW_DAYS} DAY) AS window_end
  FROM `{dataset}.visit_occurrence`
  WHERE visit_concept_id IN (SELECT concept_id FROM acute_care_visit_concepts)
  UNION ALL
  SELECT
    person_id,
    2,
    DATE_SUB(evidence_date, INTERVAL {PREGNANCY_WINDOW_DAYS_BEFORE} DAY),
    DATE_ADD(evidence_date, INTERVAL {PREGNANCY_WINDOW_DAYS_AFTER} DAY)
  FROM pregnancy_evidence_dates
  UNION ALL
  SELECT
    person_id,
    3,
    IF(days_before = {UNBOUNDED_WINDOW_DAYS}, DATE '0001-01-01', DATE_SUB(evidence_date, INTERVAL days_before DAY)),
    IF(days_after = {UNBOUNDED_WINDOW_DAYS}, DATE '9999-12-31', DATE_ADD(evidence_date, INTERVAL days_after DAY))
  FROM window_evidence
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
measurement_rows AS (
  SELECT
    measurement.person_id,
    measurement.measurement_date,
    measurement.value_as_number,
    -- year_of_birth is the only birth field OMOP requires; a mid-year birthday
    -- makes the age unbiased with error at most half a year.
    DATE_DIFF(measurement.measurement_date, DATE(person.year_of_birth, 7, 1), DAY) / 365.25 AS age_years,
    CASE LOWER(sex_concept.concept_name)
      WHEN 'female' THEN 'female'
      WHEN 'male' THEN 'male'
    END AS biological_sex,
    CASE
      WHEN measurement.unit_concept_id IS NULL OR measurement.unit_concept_id = 0 THEN '{NO_UNIT_LABEL}'
      ELSE COALESCE(
        LOWER(unit.concept_name),
        CONCAT('unit_concept_id=', CAST(measurement.unit_concept_id AS STRING))
      )
    END AS unit_label,
    (operator.concept_id IS NOT NULL AND operator.concept_code != '{EQUALS_OPERATOR_SNOMED_CODE}')
      OR COALESCE(measurement.value_source_value, '') LIKE '%<%'
      OR COALESCE(measurement.value_source_value, '') LIKE '%>%' AS censored,
    measurement.measurement_type_concept_id = {SELF_REPORT_TYPE_CONCEPT_ID} AS self_reported,
    (
      SELECT MIN(exclusion_windows.window_priority)
      FROM exclusion_windows
      WHERE exclusion_windows.person_id = measurement.person_id
        AND measurement.measurement_date BETWEEN exclusion_windows.window_start AND exclusion_windows.window_end
    ) AS exclusion_window,
    COALESCE(measurement.measurement_date >= treatment_starts.first_treatment_date, FALSE) AS on_treatment
  FROM `{dataset}.measurement` AS measurement
  JOIN `{dataset}.person` AS person
    ON person.person_id = measurement.person_id
  LEFT JOIN `{dataset}.concept` AS sex_concept
    ON sex_concept.concept_id = person.sex_at_birth_concept_id
  LEFT JOIN `{dataset}.concept` AS unit
    ON unit.concept_id = measurement.unit_concept_id
  LEFT JOIN `{dataset}.concept` AS operator
    ON operator.concept_id = measurement.operator_concept_id
    AND operator.concept_id != 0
  LEFT JOIN treatment_starts
    ON treatment_starts.person_id = measurement.person_id
  WHERE measurement.value_as_number IS NOT NULL
    AND (
      measurement.measurement_concept_id IN (SELECT concept_id FROM analyte_concepts)
      OR measurement.measurement_source_concept_id IN (SELECT concept_id FROM analyte_concepts)
    )
    AND COALESCE(measurement.measurement_source_concept_id, 0) NOT IN UNNEST(@excluded_source_concept_ids)
),
analyte_rows AS (
  SELECT
    measurement_rows.*,
    measurement_rows.value_as_number * unit_conversions.unit_scale + unit_conversions.unit_offset AS canonical_value,
    unit_conversions.unit_label IS NOT NULL AS unit_recognized
  FROM measurement_rows
  LEFT JOIN unit_conversions
    ON unit_conversions.unit_label = measurement_rows.unit_label
),
classified_rows AS (
  SELECT
    *,
    CASE
      WHEN censored THEN 'censored'
      WHEN NOT unit_recognized THEN 'unrecognized_unit'
      WHEN canonical_value < @plausible_low OR canonical_value > @plausible_high THEN 'implausible'
      WHEN self_reported THEN 'self_reported'
      WHEN age_years < @minimum_age_years THEN 'under_minimum_age'
      WHEN exclusion_window = 1 THEN 'acute_care'
      WHEN exclusion_window = 2 THEN 'pregnancy'
      WHEN exclusion_window = 3 THEN 'clinical_exclusion'
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
  LOWER(sex_concept.concept_name) AS sex_at_birth_name
FROM person_summaries
JOIN `{dataset}.person` AS person
  ON person.person_id = person_summaries.person_id
LEFT JOIN `{dataset}.concept` AS sex_concept
  ON sex_concept.concept_id = person.sex_at_birth_concept_id
ORDER BY person_summaries.person_id
""".strip()


def build_all_of_us_measurement_query_parameters(
    definition: MeasurementDefinition,
) -> dict[str, tuple[str, Any]]:
    """Query parameter name -> (GoogleSQL type, value); list values are ARRAY<type>."""
    treatment_atc_codes = () if definition.treatment is None else definition.treatment.medication.atc_codes
    window_roots = [
        (concept, window) for window in definition.clinical_windows for concept in window.concepts
    ]
    plausible_low, plausible_high = definition.plausible_range
    return {
        "loinc_codes": ("STRING", list(definition.loinc_codes)),
        "physical_measurement_concept_ids": ("INT64", list(definition.physical_measurement_concept_ids)),
        "excluded_source_concept_ids": ("INT64", list(definition.excluded_source_concept_ids)),
        "unit_labels": ("STRING", [conversion.unit_label for conversion in definition.unit_conversions]),
        "unit_scales": ("FLOAT64", [conversion.scale for conversion in definition.unit_conversions]),
        "unit_offsets": ("FLOAT64", [conversion.offset for conversion in definition.unit_conversions]),
        "plausible_low": ("FLOAT64", plausible_low),
        "plausible_high": ("FLOAT64", plausible_high),
        "minimum_age_years": ("FLOAT64", definition.minimum_age_years),
        "log_scale": ("BOOL", definition.log_scale),
        "log_offset": ("FLOAT64", definition.log_offset),
        "value_formula": ("STRING", definition.value_formula),
        "treatment_atc_codes": ("STRING", list(treatment_atc_codes)),
        "acute_care_visit_codes": ("STRING", list(ACUTE_CARE_VISIT_CODES)),
        "pregnancy_snomed_codes": ("STRING", list(PREGNANCY_SNOMED_CODES)),
        "non_pregnancy_snomed_codes": ("STRING", list(NON_PREGNANCY_SNOMED_CODES)),
        "window_domains": ("STRING", [concept.domain for concept, _window in window_roots]),
        "window_vocabularies": ("STRING", [concept.vocabulary_id for concept, _window in window_roots]),
        "window_concept_codes": ("STRING", [concept.concept_code for concept, _window in window_roots]),
        "window_days_before": ("INT64", [window.days_before for _concept, window in window_roots]),
        "window_days_after": ("INT64", [window.days_after for _concept, window in window_roots]),
    }


def build_all_of_us_measurement_query_config(definition: MeasurementDefinition) -> "bigquery.QueryJobConfig":
    return _query_config(build_all_of_us_measurement_query_parameters(definition))


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
    A person's occasions are combined by the trait's TreatmentRule: untreated
    ones when there are any, otherwise treated ones corrected to their
    untreated equivalent (DIVIDE/ADD) or dropped (EXCLUDE); INDICATOR pools
    both and records the on-treatment fraction. Persons with no occasion left
    are dropped. The target is the empirical BLUP of the person's long-run mean
    (_person_blup, with variance components from
    _estimate_person_variance_components); target_inverse_normal is its
    rank-based inverse normal transform.
    """
    selected_rows: list[dict[str, Any]] = []
    summaries: list[_OccasionSummary] = []
    sources: list[str] = []
    treated_fractions: list[float] = []
    n_without_retained_occasion = 0
    n_treated_only_excluded = 0
    for row in rows:
        untreated = _occasion_summary(row, "untreated")
        treated = _occasion_summary(row, "treated")
        if treated is not None and definition.treatment is None:
            raise ValueError(
                f"{definition.canonical_name}: treated occasions returned for a trait without a medication rule"
            )
        if untreated is None and treated is None:
            n_without_retained_occasion += 1
            continue
        if definition.treatment is not None and definition.treatment.correction is TreatmentCorrection.INDICATOR:
            summary = _pooled_summary([group for group in (untreated, treated) if group is not None])
            treated_fractions.append(0.0 if treated is None else treated.count / summary.count)
            summaries.append(summary)
            sources.append("pooled")
        elif untreated is not None:
            treated_fractions.append(0.0)
            summaries.append(untreated)
            sources.append("untreated")
        elif definition.treatment is None or definition.treatment.correction is TreatmentCorrection.EXCLUDE:
            n_treated_only_excluded += 1
            continue
        else:
            treated_fractions.append(1.0)
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
    female = np.array([row.get("sex_at_birth_name") == "female" for row in selected_rows], dtype=np.float64)
    design = _person_design(
        mean_ages, mean_squared_ages, female, [row.get("sex_at_birth_concept_id") for row in selected_rows]
    )
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
            "age_at_measurement_x_female": occasion_summary.mean_age * is_female,
            "log_occasion_count": math.log(occasion_summary.count),
            "treated_occasion_fraction": treated_fraction,
            "sex_at_birth_concept_id": row.get("sex_at_birth_concept_id"),
        }
        for row, occasion_summary, source, treated_fraction, is_female, target, inverse_normal_target, reliability
        in zip(
            selected_rows,
            summaries,
            sources,
            treated_fractions,
            female,
            targets,
            inverse_normal_targets,
            reliabilities,
            strict=True,
        )
    ]
    encoded_categorical_columns = _add_one_hot_omop_categorical_covariates(training_rows, PHENOTYPE_CATEGORICAL_COVARIATES)
    unrecognized_unit_persons: Counter[str] = Counter()
    for row in rows:
        unrecognized_unit_persons.update(set(row.get("unrecognized_unit_labels") or ()))
    summary = {
        "n_persons_with_analyte_rows": len(rows),
        "n_persons": len(training_rows),
        "n_persons_by_source": dict(Counter(sources)),
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


def measurement_covariate_columns(definition: MeasurementDefinition) -> tuple[str, ...]:
    """Covariates of a trait's fit, before one-hot expansion of sex at birth."""
    indicator = (
        definition.treatment is not None and definition.treatment.correction is TreatmentCorrection.INDICATOR
    )
    return (
        "age_at_measurement",
        "age_at_measurement_squared",
        "age_at_measurement_x_female",
        "sex_at_birth_concept_id",
        "log_occasion_count",
        *(("treated_occasion_fraction",) if indicator else ()),
    )


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
        "age_at_measurement_x_female",
        "log_occasion_count",
        "treated_occasion_fraction",
        *encoded_categorical_columns,
    )
    _write_tsv(sample_table_path, header, training_rows)

    sql_path = sample_table_path.with_suffix(sample_table_path.suffix + ".sql")
    sql_path.write_text(build_all_of_us_measurement_sql() + "\n", encoding="utf-8")

    treatment = definition.treatment
    metadata_path = sample_table_path.with_suffix(sample_table_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "trait": definition.canonical_name,
                "description": definition.description,
                "phenotype_fingerprint": phenotype_fingerprint(definition),
                "covariate_columns": list(measurement_covariate_columns(definition)),
                "loinc_codes": list(definition.loinc_codes),
                "physical_measurement_concept_ids": list(definition.physical_measurement_concept_ids),
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
                "clinical_windows": [
                    {
                        "name": window.name,
                        "concepts": [
                            [concept.domain, concept.vocabulary_id, concept.concept_code]
                            for concept in window.concepts
                        ],
                        "days_before": window.days_before,
                        "days_after": window.days_after,
                    }
                    for window in definition.clinical_windows
                ],
                "query_parameters": {
                    name: value
                    for name, (_parameter_type, value) in build_all_of_us_measurement_query_parameters(definition).items()
                },
                "minimum_age_years": definition.minimum_age_years,
                "acute_care_window_days": ACUTE_CARE_WINDOW_DAYS,
                "pregnancy_window_days": [PREGNANCY_WINDOW_DAYS_BEFORE, PREGNANCY_WINDOW_DAYS_AFTER],
                "self_report_type_concept_id": SELF_REPORT_TYPE_CONCEPT_ID,
                "target_definition": (
                    "empirical BLUP of the person's long-run mean on the analysis scale (random-intercept "
                    "model, mean model: intercept, mean age, mean squared age, sex at birth, mean age x "
                    "female); target_inverse_normal is its Blom rank inverse normal transform"
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


# All of Us may release no count below 20 participants.
MINIMUM_REPORTED_PARTICIPANTS = 20


def build_all_of_us_measurement_census_sql() -> str:
    """Participants and rows per trait, matched concept and unit label, over
    every catalogue trait at once: the first query to run on a new CDR, to
    replace estimated availability and extend the unit tables. A row counts
    under its standard concept when that is one of the trait's, else under
    its source concept."""
    dataset = _require_env("WORKSPACE_CDR")
    return f"""
WITH analyte_codes AS (
  SELECT
    trait_name,
    @census_loinc_codes[OFFSET(code_position)] AS concept_code
  FROM UNNEST(@census_loinc_traits) AS trait_name WITH OFFSET AS code_position
),
analyte_concepts AS (
  SELECT analyte_codes.trait_name, concept.concept_id, concept.vocabulary_id, concept.concept_code
  FROM analyte_codes
  JOIN `{dataset}.concept` AS concept
    ON concept.vocabulary_id = 'LOINC'
    AND concept.concept_code = analyte_codes.concept_code
  UNION ALL
  SELECT
    trait_name,
    @census_physical_measurement_concept_ids[OFFSET(concept_position)],
    'PPI',
    CAST(@census_physical_measurement_concept_ids[OFFSET(concept_position)] AS STRING)
  FROM UNNEST(@census_physical_measurement_traits) AS trait_name WITH OFFSET AS concept_position
),
matched_rows AS (
  SELECT
    measurement.person_id,
    COALESCE(standard_match.trait_name, source_match.trait_name) AS trait_name,
    COALESCE(standard_match.vocabulary_id, source_match.vocabulary_id) AS vocabulary_id,
    COALESCE(standard_match.concept_code, source_match.concept_code) AS concept_code,
    standard_match.concept_id IS NOT NULL AS matched_on_standard_concept,
    CASE
      WHEN measurement.unit_concept_id IS NULL OR measurement.unit_concept_id = 0 THEN '{NO_UNIT_LABEL}'
      ELSE COALESCE(
        LOWER(unit.concept_name),
        CONCAT('unit_concept_id=', CAST(measurement.unit_concept_id AS STRING))
      )
    END AS unit_label
  FROM `{dataset}.measurement` AS measurement
  LEFT JOIN analyte_concepts AS standard_match
    ON standard_match.concept_id = measurement.measurement_concept_id
  LEFT JOIN analyte_concepts AS source_match
    ON source_match.concept_id = measurement.measurement_source_concept_id
  LEFT JOIN `{dataset}.concept` AS unit
    ON unit.concept_id = measurement.unit_concept_id
  WHERE measurement.value_as_number IS NOT NULL
    AND (standard_match.concept_id IS NOT NULL OR source_match.concept_id IS NOT NULL)
)
SELECT
  trait_name,
  vocabulary_id,
  concept_code,
  matched_on_standard_concept,
  unit_label,
  COUNT(DISTINCT person_id) AS participant_count,
  COUNT(*) AS row_count
FROM matched_rows
GROUP BY trait_name, vocabulary_id, concept_code, matched_on_standard_concept, unit_label
ORDER BY trait_name, concept_code, participant_count DESC
""".strip()


def build_all_of_us_measurement_census_query_parameters() -> dict[str, tuple[str, Any]]:
    loinc_codes = [
        (definition.canonical_name, code) for definition in MEASUREMENT_DEFINITIONS for code in definition.loinc_codes
    ]
    physical_measurements = [
        (definition.canonical_name, concept_id)
        for definition in MEASUREMENT_DEFINITIONS
        for concept_id in definition.physical_measurement_concept_ids
    ]
    return {
        "census_loinc_traits": ("STRING", [trait for trait, _code in loinc_codes]),
        "census_loinc_codes": ("STRING", [code for _trait, code in loinc_codes]),
        "census_physical_measurement_traits": ("STRING", [trait for trait, _concept_id in physical_measurements]),
        "census_physical_measurement_concept_ids": (
            "INT64",
            [concept_id for _trait, concept_id in physical_measurements],
        ),
    }


def prepare_all_of_us_measurement_census(
    output_path: str | Path,
    *,
    client: bigquery.Client | None = None,
) -> Path:
    """Write the census as TSV. Cells below MINIMUM_REPORTED_PARTICIPANTS
    participants keep their labels but report neither count."""
    active_client = _active_bigquery_client(client)
    rows = _query_rows(
        active_client,
        build_all_of_us_measurement_census_sql(),
        _query_config(build_all_of_us_measurement_census_query_parameters()),
    )
    for row in rows:
        suppressed = int(row["participant_count"]) < MINIMUM_REPORTED_PARTICIPANTS
        row["suppressed_below_minimum"] = suppressed
        if suppressed:
            row["participant_count"] = None
            row["row_count"] = None
    census_path = Path(output_path)
    census_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        census_path,
        (
            "trait_name",
            "vocabulary_id",
            "concept_code",
            "matched_on_standard_concept",
            "unit_label",
            "participant_count",
            "row_count",
            "suppressed_below_minimum",
        ),
        rows,
    )
    return census_path


def phenotype_fingerprint(definition: DiseaseDefinition | MeasurementDefinition) -> str:
    """SHA-256 over the phenotype's query, its parameters and its definition;
    a prepared sample table is current exactly when its metadata carries it."""
    if isinstance(definition, MeasurementDefinition):
        sql = build_all_of_us_measurement_sql()
        parameters = build_all_of_us_measurement_query_parameters(definition)
    else:
        sql = build_all_of_us_disease_sql(definition)
        parameters = build_all_of_us_disease_query_parameters(definition)
    payload = json.dumps({"sql": sql, "parameters": parameters, "definition": repr(definition)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _query_config(parameters: dict[str, tuple[str, Any]]) -> "bigquery.QueryJobConfig":
    from google.cloud import bigquery

    query_parameters: list[bigquery.ArrayQueryParameter | bigquery.ScalarQueryParameter] = []
    for name, (parameter_type, value) in parameters.items():
        if isinstance(value, list):
            query_parameters.append(bigquery.ArrayQueryParameter(name, parameter_type, value))
        else:
            query_parameters.append(bigquery.ScalarQueryParameter(name, parameter_type, value))
    return bigquery.QueryJobConfig(query_parameters=query_parameters)


def _prepare_training_rows(
    disease_definition: DiseaseDefinition,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...], dict[str, int]]:
    """Cases: >= MIN_DISEASE_OCCURRENCES distinct diagnosis dates, or one date
    plus the disease's medication. Controls: no diagnosis date, no control
    exclusion code and no disease medication. Everyone else (one date without
    medication, excluded or medicated non-cases, an ambiguous code, or EHR
    depth below MIN_PRE_LANDMARK_CONDITION_DATES) is left out."""
    training_rows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    for row in rows:
        occurrence_count = int(row["phenotype_occurrence_count"])
        medicated = bool(row["has_disease_medication"])
        if int(row["pre_landmark_condition_dates"]) < MIN_PRE_LANDMARK_CONDITION_DATES:
            exclusion_counts["n_excluded_sparse_ehr"] += 1
        elif bool(row["has_ambiguous_code"]):
            exclusion_counts["n_excluded_ambiguous_code"] += 1
        elif occurrence_count >= MIN_DISEASE_OCCURRENCES or (occurrence_count == 1 and medicated):
            training_rows.append(dict(row, target=1))
        elif occurrence_count == 1:
            # One-code participants are neither clean controls nor cases.
            exclusion_counts["n_excluded_one_date"] += 1
        elif bool(row["has_control_exclusion_code"]) or medicated:
            exclusion_counts["n_excluded_control_exclusion"] += 1
        else:
            training_rows.append(dict(row, target=0))
    targets = np.array([row["target"] for row in training_rows], dtype=np.float64)
    event_ages = np.array(
        [
            float(row["age_at_first_condition"] if row["target"] == 1 else row["age_at_observation_end"])
            for row in training_rows
        ]
    )
    sex_names = [row.get("sex_at_birth_name") for row in training_rows]
    liability_targets = _liability_targets(targets, event_ages, sex_names)
    for row, liability_target, sex_name in zip(training_rows, liability_targets, sex_names, strict=True):
        age = float(row["age_at_observation_end"])
        row["liability_target"] = float(liability_target)
        row["age_at_observation_end_squared"] = age * age
        row["age_at_observation_end_x_female"] = age if sex_name == "female" else 0.0
        row["log1p_pre_landmark_condition_dates"] = math.log1p(int(row["pre_landmark_condition_dates"]))
    encoded_categorical_columns = _add_one_hot_omop_categorical_covariates(
        training_rows, PHENOTYPE_CATEGORICAL_COVARIATES
    )
    return (
        training_rows,
        encoded_categorical_columns,
        {
            "n_cases": int(targets.sum()),
            "n_controls": int(len(targets) - targets.sum()),
            **{name: exclusion_counts[name] for name in (
                "n_excluded_sparse_ehr",
                "n_excluded_ambiguous_code",
                "n_excluded_one_date",
                "n_excluded_control_exclusion",
            )},
        },
    )


def _liability_targets(targets: np.ndarray, event_ages: np.ndarray, sex_names: list[Any]) -> np.ndarray:
    """E[liability | status, age] under the age-of-onset liability threshold model.

    Liability l ~ N(0, 1); a person becomes a case at the first age a with
    l >= t(a) = Phi^-1(1 - K(a)), where K is the cumulative incidence by age,
    so a case diagnosed at age a has l = t(a) and a person unaffected at age c
    has l < t(c), whose mean is E[l | l < t] = -phi(t) / Phi(t) = -phi(t) / S(c).
    K is the Kaplan-Meier cumulative incidence of first diagnosis by age
    (cases: event at the first diagnosis age; controls: censored at the end of
    observation) within sex at birth, female and male separately and everyone
    else on the pooled curve. A case's K is the mid-point of the survival step
    at its own age, so the transform never reaches K = 1. Approximations:
    no left truncation at EHR entry (first diagnosis dates of prevalent cases
    are late), and the cohort's own incidence, which a disease-enriched cohort
    inflates, stands in for the population's (LT-FH++, Pedersen 2022, without
    family history).
    """
    liabilities = np.empty(len(targets))
    female = np.array([name == "female" for name in sex_names], dtype=bool)
    male = np.array([name == "male" for name in sex_names], dtype=bool)
    everyone = np.ones(len(targets), dtype=bool)
    for members, curve_members in ((female, female), (male, male), (~(female | male), everyone)):
        event_times, survival = _kaplan_meier_survival(event_ages[curve_members], targets[curve_members])
        cases = members & (targets == 1)
        controls = members & (targets == 0)
        after = _survival_at(event_times, survival, event_ages[cases], strictly_before=False)
        before = _survival_at(event_times, survival, event_ages[cases], strictly_before=True)
        liabilities[cases] = ndtri(0.5 * (after + before))
        control_survival = _survival_at(event_times, survival, event_ages[controls], strictly_before=False)
        thresholds = ndtri(control_survival)
        liabilities[controls] = -np.exp(-0.5 * thresholds**2) / math.sqrt(2.0 * math.pi) / control_survival
    return liabilities


def _kaplan_meier_survival(ages: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct event ages and the Kaplan-Meier survival just after each; a
    person censored at an event age is still at risk there."""
    event_ages = np.sort(ages[events == 1])
    event_times = np.unique(event_ages)
    at_risk = len(ages) - np.searchsorted(np.sort(ages), event_times, side="left")
    diagnosed = np.searchsorted(event_ages, event_times, side="right") - np.searchsorted(
        event_ages, event_times, side="left"
    )
    return event_times, np.cumprod(1.0 - diagnosed / at_risk)


def _survival_at(
    event_times: np.ndarray,
    survival: np.ndarray,
    ages: np.ndarray,
    *,
    strictly_before: bool,
) -> np.ndarray:
    """S(age) (events at the age included) or S(age-) (excluded)."""
    steps = np.searchsorted(event_times, ages, side="left" if strictly_before else "right")
    return np.concatenate([[1.0], survival])[steps]


def _pooled_summary(summaries: list[_OccasionSummary]) -> _OccasionSummary:
    """Exact sufficient statistics of the union of disjoint occasion groups."""
    count = sum(summary.count for summary in summaries)
    mean = sum(summary.count * summary.mean for summary in summaries) / count
    variance = sum(summary.count * (summary.variance + (summary.mean - mean) ** 2) for summary in summaries) / count
    return _OccasionSummary(
        count=count,
        mean=mean,
        variance=variance,
        mean_age=sum(summary.count * summary.mean_age for summary in summaries) / count,
        mean_age_squared=sum(summary.count * summary.mean_age_squared for summary in summaries) / count,
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


def _write_tsv(path: Path, header: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
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
    female: np.ndarray,
    sex_concept_ids: list[Any],
) -> np.ndarray:
    """Person-level mean model: intercept, centered mean age and mean squared
    age (exact for a quadratic age trend at the measurement level), one
    indicator per sex-at-birth level other than the most common one (a missing
    value is its own level) and centered mean age x female. Columns that are
    identically zero (one sex only) are left out."""
    sex_levels = ["missing" if value in (None, "") else str(value) for value in sex_concept_ids]
    level_counts = Counter(sex_levels)
    reference_level = max(level_counts, key=lambda level: (level_counts[level], level))
    centered_ages = mean_ages - mean_ages.mean()
    columns = [
        np.ones_like(mean_ages),
        centered_ages,
        mean_squared_ages - mean_squared_ages.mean(),
        *(
            np.array([1.0 if sex_level == level else 0.0 for sex_level in sex_levels])
            for level in sorted(level_counts)
            if level != reference_level
        ),
        centered_ages * female,
    ]
    design = np.column_stack(columns)
    return design[:, np.any(design != 0.0, axis=0)]


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
