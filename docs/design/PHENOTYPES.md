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
  - a met lab criterion: 2 or more qualifying occasions, the first and last at least the criterion's span apart. A criterion counts only rows inside its plausible range (HbA1c 3–20%, creatinine 0.2–20 mg/dL, UACR 0.1–30,000 mg/g), a hand-set rule kept on this path alone until the disease measurement model replaces it (item 1 below).
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
    - not positive (an analyte is a positive quantity);
    - self-reported;
    - taken below the minimum age;
    - taken within 30 days of an inpatient or emergency stay;
    - inside a pregnancy window;
    - inside one of the trait's clinical exclusion windows;
  - flag a row as treated on or after the first exposure to the trait's medication class.
  - No plausible range: a gross error (a typo, a unit confusion) is downweighted by the learned noise density below, not cut.
- **Occasions.** Rows collapse to one occasion per person-day, the same-day mean of its retained rows. The SQL returns every occasion with its age and treatment flag, and each person's row counts by exclusion reason.
- **Treatment.** A person's untreated occasions are used when there are any.
  - Otherwise the treated occasions are corrected: LDL is divided by 0.7, and systolic blood pressure has 15 mmHg added.
  - For BMI and heart rate the treated occasions are dropped.
- **The measurement model** (`sv_pgs/phenotype_measurement.py`; novel-pheno.md §3): h_λ(y_ij) = d_ij'γ + T_i + e_ij, with T_i ~ N(0, τ²) the person's long-run level.
  - h_λ is the Box–Cox transform, with λ learned by the profile evidence. That replaces the hand-chosen log or linear scale.
  - e_ij follows a learned continuous scale mixture: the engine's mixing density over log variance, with a third-difference roughness penalty whose weight maximizes the evidence. Its lattice starts at the variance of the recording resolution, (δ·y^{λ−1})²/12, the smallest resolvable noise. **Pending derivation:** its extent past the largest within-person squared deviation (currently that value plus the lattice's own width, generous rather than derived) is to follow math-density's R2 tail-mass rule (mixing_density.md §2); owner deslop-hygiene.
  - d_ij holds an intercept, age and age² at the occasion, sex at birth, and age × female.
  - The fit is exact in T_i: EM whose E-step integrates each person's level by a certified trapezoid rule (mixing_density.md §11); ECME with PX-EM for γ and τ², Newton with Louis' information for the density, and SQUAREM where that curvature is indefinite. The evidence's curvature comes from Louis' identity.
  - Validation so far is synthetic: exactness against brute-force quadrature and against the dense Gaussian (Henderson) solution, and injected ×10 typos and mmol/L-as-mg/dL readings moving no person's level by a posterior standard deviation [sim-only]. On within-exam replicates the learned noise density gained +0.110 ± 0.022 (SBP) and +0.191 ± 0.022 (pulse) nats per held-out reading over the current rules [real, NHANES replicates] (novel-pheno.md §10); those are not EHR occasions, so the in-workspace checks V-B and V-D (novel-pheno.md §8) must pass before the case definitions are frozen (HANDOFF.md, pilot).
- **Target.** E[T_i | occasions], with its reliability 1 − Var(T_i | occasions)/τ².
  - Only within-person replicates separate the level variance τ² from the noise: without them only τ² + E[s] is identified. So the per-occasion model is used only when some person has repeated occasions and the fit certifies the split. The certificate is Louis' observed information for (log τ², a common noise scale), positive definite beyond its derived rounding bound (Higham, Weyl).
  - Otherwise the target is the mean of each person's Box–Cox-transformed readings, the exponent learned from their Gaussian marginal, with reliability 1. The noise then stays in the genetic model's residual. The run logs this, and the metadata's `measurement_model` records it (lead ruling).
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
   - the plausible ranges, now on the lab-criterion path only (the traits' gross errors go to the learned noise density);
   - the 30-day acute-care window;
   - the clinical exclusion windows: 180 days before a hematologic malignancy, 90 days after chemotherapy, 120 days after a transfusion, 7 before and 14 after an antibacterial course, and 30 either side of an acute hepatobiliary event;
   - the adult minimum age of 18.

   **Replacement:** the measurement model above, extended with Σ_w f_w(t_ij − t_event), a learned smooth of time since an event of class w, for each window. Its per-occasion rows and learned noise density are built; the event-time smooths are part 2.
3. **Treatment corrections.** The values:
   - LDL / 0.7 (Peloso 2014, after the CTT meta-analysis);
   - systolic blood pressure + 15 mmHg (Cui 2003; Tobin 2005);
   - dropping treated-only BMI and heart-rate persons.

   These are external average effects, not properties of these data.

   **Replacement:** a treatment effect with a heterogeneity variance, learned within person inside the per-occasion model from people with both untreated and treated occasions.
   - Treatment tends to start after a high reading, so the pre-treatment mean is inflated. That regression-to-the-mean bias has to be removed by construction.
4. **The analysis scale: replaced.** Log vs linear was chosen by hand for each trait; the measurement model now learns a Box–Cox exponent by its evidence.
5. **Covariate functional forms.** Age, age², age × female and log(1 + EHR depth).

   **Replacement:** smooths with learned smoothness in the cohort builder. They are projected out exactly (MODEL.md §1).
