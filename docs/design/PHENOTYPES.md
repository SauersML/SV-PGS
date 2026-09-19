# Phenotypes

The All of Us phenotypes live in `sv_pgs/all_of_us.py`.
- **Commands:** `list-all-of-us-diseases`, `prepare-all-of-us-disease`, `list-all-of-us-traits`, `prepare-all-of-us-trait` and `census-all-of-us-traits`.
- **Tests,** on synthetic OMOP tables: `tests/test_all_of_us_phenotypes.py`, and `tests/test_all_of_us_sql.py`, which runs the BigQuery SQL in DuckDB.
- **Execution:** everything runs inside the workspace. Only counts of at least 21 participants leave it.

## The panel
21 traits, chosen by power and phenotype quality, never by SV biology (DECISIONS.md):
- **11 quantitative:** height, body mass index, systolic blood pressure, heart rate, mean corpuscular volume, platelet count, white blood cell count, total bilirubin, creatinine eGFR (CKD-EPI 2021), LDL cholesterol, and HbA1c in people without diabetes.
- **10 diseases:** type 2 diabetes, atrial fibrillation, hypothyroidism, COPD, depression, gout, cataract, chronic kidney disease, psoriasis, prostate cancer.

## Diseases
- **Codes.** Every rule reads codes by root concept, and each root's `concept_ancestor` descendants count too:
  - diagnoses by SNOMED root;
  - drugs by ATC class;
  - procedures by SNOMED root;
  - labs through the measurement query below.
- **A case,** at an onset age of at least the disease's minimum case age, and of the required sex if the disease has one, has any of:
  - diagnoses on 2 or more distinct dates;
  - a diagnosis on one date, plus the case medication on at least the disease's minimum number of dates;
  - a case procedure;
  - a met lab criterion: 2 or more qualifying occasions, the first and last at least the criterion's span apart.
- **A control** has none of those evidence types. It also has no control-exclusion code or drug, and no qualifying lab occasion. Its age at the end of observation is at least the disease's minimum control age.
- **Ambiguous codes** remove a person from both groups.
- **EHR depth.** Cases and controls alike need 5 or more distinct condition dates by the first observation-period start plus 365 days.
- **Targets:**
  - the binary status;
  - the liability target E[l | status, age] under the age-of-onset threshold model, with each person's age at first evidence or at censoring, and the cohort's own Kaplan–Meier cumulative incidence within sex at birth.
- **Covariates:** age at the end of observation, its square, its product with female sex, sex at birth, and log(1 + pre-landmark condition dates).

## Quantitative traits
- **Per measurement row, in SQL:**
  - match the analyte's LOINC and physical-measurement concepts;
  - convert the unit to the canonical unit;
  - drop a row that is any of:
    - censored;
    - in an unrecognized unit;
    - implausible;
    - self-reported;
    - taken below the minimum age;
    - taken within 30 days of an inpatient or emergency stay;
    - inside a pregnancy window;
    - inside one of the trait's clinical exclusion windows;
  - flag a row as treated on or after the first exposure to the trait's medication class.
- **Occasions.** Rows collapse to one occasion per person-day. Each person is reduced to exact sufficient statistics, separately for untreated and treated occasions:
  - the occasion count;
  - the mean and the variance;
  - the mean age and the mean squared age.
- **Treatment.** A person's untreated occasions are used when there are any.
  - Otherwise the treated occasions are corrected: LDL is divided by 0.7, and systolic blood pressure has 15 mmHg added.
  - For BMI and heart rate the treated occasions are dropped.
- **Target.** The empirical BLUP of the person's long-run mean under a random-intercept model:
  - the variance components come from closed-form, unbiased Henderson III moment estimators;
  - each target carries its reliability.
- **Covariates:** mean age, its square, its product with female sex, and sex at birth. The occasion count is not a covariate: how often a trait is measured depends on its level, so adjusting for it attenuates every genetic effect (novel-pheno Theorem 5). Each person's precision enters through the target's reliability instead.

## Which values are standards
Kept, and cited where they are defined:
- **Codes and concept ids:** SNOMED, LOINC, ATC and CPT codes, OMOP concept ids, and the All of Us physical-measurement concepts.
- **Conversions and equations:**
  - unit conversions;
  - the NGSP–IFCC master equation for HbA1c;
  - the CKD-EPI 2021 creatinine equation (Inker 2021).
- **Clinical criteria:**
  - KDIGO 2012 chronic kidney disease: GFR < 60 or ACR ≥ 30 mg/g for more than 3 months;
  - the ADA HbA1c diagnosis: ≥ 6.5%, confirmed by a second test;
  - the GOLD report's age of 40 for a COPD diagnosis.
- **Physiology and definitions:**
  - growth complete by 20, where the CDC growth charts end;
  - a 42-week pregnancy bound, where post-term begins;
  - the 12-week postpartum period (ACOG Committee Opinion 736).
- **All of Us rules:** the Data and Statistics Dissemination Policy's minimum reported count of 21.
- **Derived ages:** only the birth year is released, so the mid-year birthday is the unbiased age, and a year is 365.25 days.

## Rules that are not standards, and their replacements
None of these values is a published standard for these data. They must be replaced before the case definitions are frozen (EVALUATION.md), and they have no owner yet.

1. **Disease evidence rules.** The values:
   - 2 diagnosis dates;
   - 2 levothyroxine dates for hypothyroidism;
   - the exclusion of a person with one diagnosis date;
   - the EHR-depth floor (5 dates, 365 days);
   - the minimum control ages (COPD 40, prostate cancer 50).

   **Replacement:** a latent-class measurement model of EHR evidence.
   - Diagnosis dates, case-medication dates, case procedures and qualifying lab occasions are counts, conditionally independent given the true status Z and the person's EHR exposure.
   - Per-channel rates and the prevalence are learned by marginal likelihood for each disease. Three or more conditionally independent channels identify the model.
   - The target becomes E[liability | all evidence, age]. Sparse EHR then yields a posterior near the prior, instead of an exclusion.
2. **Quantitative row rules.** The values:
   - the plausible ranges;
   - the 30-day acute-care window;
   - the clinical exclusion windows: 180 days before a hematologic malignancy, 90 days after chemotherapy, 120 days after a transfusion, 7 before and 14 after an antibacterial course, and 30 either side of an acute hepatobiliary event;
   - the adult minimum age of 18.

   **Replacement:** a per-occasion measurement model, y_ij = x_ij'γ + b_i + Σ_w f_w(t_ij − t_event) + e_ij.
   - Each f_w is a learned smooth of time since an event of class w.
   - e_ij is a two-component scale mixture: regular noise plus contamination.
   - Everything is learned by marginal likelihood, and the target is the BLUP of x_i'γ + b_i.
   - It needs per-occasion rows from the SQL instead of per-person sufficient statistics.
3. **Treatment corrections.** The values:
   - LDL / 0.7 (Peloso 2014, after the CTT meta-analysis);
   - systolic blood pressure + 15 mmHg (Cui 2003; Tobin 2005);
   - dropping treated-only BMI and heart-rate persons.

   These are external average effects, not properties of these data.

   **Replacement:** a treatment effect with a heterogeneity variance, learned within person inside the per-occasion model from people with both untreated and treated occasions.
   - Treatment tends to start after a high reading, so the pre-treatment mean is inflated. That regression-to-the-mean bias has to be removed by construction.
4. **The analysis scale.** Log vs linear is chosen by hand for each trait.

   **Replacement:** a Box–Cox exponent learned by the marginal likelihood of the per-occasion model.
5. **Covariate functional forms.** Age, age², age × female and log(1 + EHR depth).

   **Replacement:** smooths with learned smoothness in the cohort builder. They are projected out exactly (MODEL.md §1).
