"""
Curated Drug Interaction Database
----------------------------------
Covers biologics / mAbs (anti-TNF, IL-6, IL-17, JAK inhibitors) against:
  - Antibiotics & antifungals  (CYP3A4/PK interactions, infection risk stacking)
  - NSAIDs & analgesics        (GI, renal, haematologic risk in immunocompromised)

Each record contains:
  immunosuppressant : canonical drug name
  co_therapy        : interacting drug name
  severity          : "Contraindicated" | "Major" | "Moderate" | "Minor"
  confidence        : "High" | "Medium" | "Low"
  mechanism         : plain-language pharmacokinetic / pharmacodynamic explanation
  alert             : one-sentence clinical alert shown to the prescriber
  monitoring        : recommended labs / parameters
  alternatives      : safer substitutions where applicable
  populations       : list of relevant patient populations
  references        : evidence sources (FDA label, guideline, publication)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class InteractionRecord:
    immunosuppressant: str
    co_therapy: str
    severity: str           # Contraindicated / Major / Moderate / Minor
    confidence: str         # High / Medium / Low
    mechanism: str
    alert: str
    monitoring: List[str]
    alternatives: List[str]
    populations: List[str]
    references: List[str]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: normalise names for look-up
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
#  Synonym map  →  canonical name
# ─────────────────────────────────────────────────────────────────────────────

SYNONYMS: dict[str, str] = {
    # Anti-TNF biologics
    "remicade":          "infliximab",
    "humira":            "adalimumab",
    "enbrel":            "etanercept",
    "cimzia":            "certolizumab",
    "simponi":           "golimumab",
    # IL-6 inhibitors
    "actemra":           "tocilizumab",
    "kevzara":           "sarilumab",
    # IL-17 inhibitors
    "cosentyx":          "secukinumab",
    "taltz":             "ixekizumab",
    # IL-12/23
    "stelara":           "ustekinumab",
    # JAK inhibitors
    "xeljanz":           "tofacitinib",
    "olumiant":          "baricitinib",
    "rinvoq":            "upadacitinib",
    "jakafi":            "ruxolitinib",
    # Antifungals
    "diflucan":          "fluconazole",
    "vfend":             "voriconazole",
    "sporanox":          "itraconazole",
    "nizoral":           "ketoconazole",
    # Antibiotics
    "biaxin":            "clarithromycin",
    "zithromax":         "azithromycin",
    "cipro":             "ciprofloxacin",
    "levaquin":          "levofloxacin",
    "rifadin":           "rifampin",
    # NSAIDs
    "advil":             "ibuprofen",
    "motrin":            "ibuprofen",
    "aleve":             "naproxen",
    "voltaren":          "diclofenac",
    "celebrex":          "celecoxib",
    # Analgesics
    "tylenol":           "acetaminophen",
    "paracetamol":       "acetaminophen",
}


def canonical(name: str) -> str:
    """Return the canonical drug name, resolving brand-name synonyms."""
    key = _norm(name)
    return SYNONYMS.get(key, key)


# ─────────────────────────────────────────────────────────────────────────────
#  Drug class membership
# ─────────────────────────────────────────────────────────────────────────────

DRUG_CLASSES: dict[str, list[str]] = {
    "anti_tnf": [
        "infliximab", "adalimumab", "etanercept",
        "certolizumab", "golimumab",
    ],
    "il6_inhibitor": [
        "tocilizumab", "sarilumab", "siltuximab",
    ],
    "il17_inhibitor": [
        "secukinumab", "ixekizumab",
    ],
    "il12_23_inhibitor": [
        "ustekinumab",
    ],
    "jak_inhibitor": [
        "tofacitinib", "baricitinib", "upadacitinib", "ruxolitinib",
    ],
    "azole_antifungal": [
        "fluconazole", "voriconazole", "itraconazole", "ketoconazole",
        "posaconazole",
    ],
    "macrolide_antibiotic": [
        "clarithromycin", "erythromycin",
    ],
    "fluoroquinolone": [
        "ciprofloxacin", "levofloxacin", "moxifloxacin",
    ],
    "rifamycin": [
        "rifampin", "rifabutin",
    ],
    "nsaid": [
        "ibuprofen", "naproxen", "diclofenac", "celecoxib",
        "meloxicam", "indomethacin", "ketorolac",
    ],
    "analgesic": [
        "acetaminophen", "tramadol", "codeine", "oxycodone",
    ],
    "macrolide_azithromycin": [
        "azithromycin",
    ],
}


def drug_class(name: str) -> list[str]:
    """Return all class labels for a canonical drug name."""
    c = canonical(name)
    return [cls for cls, members in DRUG_CLASSES.items() if c in members]


# ─────────────────────────────────────────────────────────────────────────────
#  Curated interaction records
# ─────────────────────────────────────────────────────────────────────────────

INTERACTIONS: list[InteractionRecord] = [

    # ── JAK inhibitors × Azole antifungals ──────────────────────────────────
    InteractionRecord(
        immunosuppressant="tofacitinib",
        co_therapy="fluconazole",
        severity="Major",
        confidence="High",
        mechanism=(
            "Fluconazole is a potent CYP3A4 and CYP2C19 inhibitor. "
            "Tofacitinib is a CYP3A4 substrate; concurrent use raises tofacitinib "
            "AUC by ~65%, increasing risk of immunosuppression, infections, and "
            "dose-dependent adverse effects (anaemia, lymphopenia)."
        ),
        alert=(
            "Reduce tofacitinib dose by 50% when co-administered with fluconazole; "
            "monitor CBC and hepatic function closely."
        ),
        monitoring=["CBC with differential", "LFTs", "Signs of infection"],
        alternatives=["Topical antifungal agents where feasible", "Micafungin (no CYP3A4 interaction)"],
        populations=["Autoimmune disease", "Transplant"],
        references=[
            "Pfizer tofacitinib (Xeljanz) prescribing information 2023",
            "FDA Drug Interactions – CYP3A4/2C19 inhibitors",
        ],
    ),

    InteractionRecord(
        immunosuppressant="tofacitinib",
        co_therapy="voriconazole",
        severity="Major",
        confidence="High",
        mechanism=(
            "Voriconazole is a potent CYP3A4 and CYP2C19 inhibitor. "
            "Combined use increases tofacitinib exposure (AUC ↑~65%) and "
            "adds pharmacodynamic immunosuppressive burden, raising infection risk."
        ),
        alert=(
            "Reduce tofacitinib dose by 50% and monitor for opportunistic infections; "
            "voriconazole-level monitoring also recommended."
        ),
        monitoring=["Tofacitinib drug level if available", "CBC", "Voriconazole trough", "LFTs"],
        alternatives=["Micafungin", "Anidulafungin (no CYP3A4 interaction)"],
        populations=["Autoimmune disease", "Oncology"],
        references=[
            "Pfizer Xeljanz prescribing information 2023",
            "ECMM/ISHAM guidelines on antifungal therapy in immunocompromised",
        ],
    ),

    InteractionRecord(
        immunosuppressant="baricitinib",
        co_therapy="fluconazole",
        severity="Moderate",
        confidence="Medium",
        mechanism=(
            "Baricitinib is primarily eliminated renally (OAT3 transporter) with "
            "minor CYP3A4 metabolism. Fluconazole may modestly raise baricitinib "
            "exposure. Additive immunosuppression increases infection risk."
        ),
        alert=(
            "Monitor for increased baricitinib adverse effects; "
            "consider dose reduction in renal impairment."
        ),
        monitoring=["CBC", "SCr / eGFR", "Signs of infection"],
        alternatives=["Micafungin for systemic fungal infection"],
        populations=["Autoimmune disease"],
        references=["Eli Lilly Olumiant prescribing information 2023"],
    ),

    InteractionRecord(
        immunosuppressant="upadacitinib",
        co_therapy="itraconazole",
        severity="Major",
        confidence="High",
        mechanism=(
            "Upadacitinib is a sensitive CYP3A4 substrate. Itraconazole (strong "
            "CYP3A4 inhibitor) raises upadacitinib AUC by ~75%, markedly increasing "
            "the risk of lymphopenia, hepatotoxicity, and serious infections."
        ),
        alert=(
            "Avoid itraconazole with upadacitinib; if unavoidable, reduce "
            "upadacitinib to 15 mg once daily and monitor closely."
        ),
        monitoring=["CBC with differential", "LFTs", "Opportunistic infection surveillance"],
        alternatives=["Micafungin", "Anidulafungin"],
        populations=["Autoimmune disease"],
        references=["AbbVie Rinvoq prescribing information 2023"],
    ),

    InteractionRecord(
        immunosuppressant="ruxolitinib",
        co_therapy="ketoconazole",
        severity="Major",
        confidence="High",
        mechanism=(
            "Ruxolitinib is a CYP3A4 substrate. Strong CYP3A4 inhibitors like "
            "ketoconazole increase ruxolitinib AUC ~91%, substantially raising "
            "the risk of haematologic toxicity."
        ),
        alert=(
            "Reduce ruxolitinib dose by 50% and monitor CBC twice weekly for the "
            "first 4 weeks when ketoconazole is required."
        ),
        monitoring=["CBC twice weekly initially", "LFTs"],
        alternatives=["Topical ketoconazole", "Fluconazole at lowest effective dose with monitoring"],
        populations=["Oncology", "Autoimmune disease"],
        references=["Incyte Jakafi prescribing information 2023"],
    ),

    # ── JAK inhibitors × Rifamycins ─────────────────────────────────────────
    InteractionRecord(
        immunosuppressant="tofacitinib",
        co_therapy="rifampin",
        severity="Major",
        confidence="High",
        mechanism=(
            "Rifampin is a potent CYP3A4 inducer. Co-administration reduces "
            "tofacitinib AUC by ~84%, likely resulting in sub-therapeutic "
            "immunosuppression and disease flare."
        ),
        alert=(
            "Avoid rifampin with tofacitinib; consider alternative anti-TB agents "
            "such as rifabutin (weaker inducer) with dose adjustment."
        ),
        monitoring=["Clinical disease activity", "Drug levels if available"],
        alternatives=["Rifabutin (dose-adjust tofacitinib by 50% increase)", "Clarithromycin-based regimen"],
        populations=["Transplant", "Autoimmune disease"],
        references=["Pfizer Xeljanz prescribing information 2023", "ATS/IDSA TB treatment guidelines"],
    ),

    # ── Anti-TNF × Fluoroquinolones ──────────────────────────────────────────
    InteractionRecord(
        immunosuppressant="infliximab",
        co_therapy="ciprofloxacin",
        severity="Moderate",
        confidence="Medium",
        mechanism=(
            "Pharmacodynamic interaction: infliximab suppresses TNF-α–mediated "
            "immune clearance. Concurrent fluoroquinolone use does not alter "
            "infliximab PK but additive immunosuppression raises risk of "
            "Clostridioides difficile colitis and tendinopathy."
        ),
        alert=(
            "Use the shortest effective course of ciprofloxacin; monitor for C. diff "
            "and tendinopathy symptoms in patients on anti-TNF therapy."
        ),
        monitoring=["Stool culture if diarrhoea", "Tendon pain assessment"],
        alternatives=["Trimethoprim/sulfamethoxazole for appropriate indications", "Amoxicillin-clavulanate"],
        populations=["Autoimmune disease", "Transplant"],
        references=[
            "ACR Guideline for IBD biologic use 2022",
            "ECCO Guidelines on infections in IBD 2021",
        ],
    ),

    InteractionRecord(
        immunosuppressant="adalimumab",
        co_therapy="levofloxacin",
        severity="Moderate",
        confidence="Medium",
        mechanism=(
            "Additive pharmacodynamic immunosuppression. Levofloxacin extends the "
            "QTc interval; adalimumab-associated cardiac events are rare but the "
            "combination warrants ECG monitoring in susceptible patients."
        ),
        alert=(
            "Obtain baseline ECG and monitor QTc if levofloxacin is required in "
            "patients receiving adalimumab, particularly with pre-existing cardiac disease."
        ),
        monitoring=["ECG / QTc", "Signs of serious infection"],
        alternatives=["Amoxicillin-clavulanate", "Doxycycline"],
        populations=["Autoimmune disease"],
        references=["FDA levofloxacin label – QT prolongation warning"],
    ),

    # ── Anti-TNF × NSAIDs ────────────────────────────────────────────────────
    InteractionRecord(
        immunosuppressant="infliximab",
        co_therapy="ibuprofen",
        severity="Moderate",
        confidence="High",
        mechanism=(
            "Pharmacodynamic: NSAIDs inhibit prostaglandin-mediated renal afferent "
            "arteriolar dilation. Anti-TNF therapy may independently impair renal "
            "reserve in inflammatory disease. Combined use raises risk of acute "
            "kidney injury, especially in elderly or dehydrated patients."
        ),
        alert=(
            "Avoid regular NSAID use in patients on anti-TNF therapy; use the lowest "
            "effective NSAID dose for the shortest duration and monitor renal function."
        ),
        monitoring=["SCr / eGFR at baseline and after NSAID initiation", "BP"],
        alternatives=["Acetaminophen for analgesia", "Topical diclofenac gel"],
        populations=["Autoimmune disease", "Transplant"],
        references=["ACR Recommendations for NSAID use in rheumatic disease 2020"],
    ),

    InteractionRecord(
        immunosuppressant="adalimumab",
        co_therapy="naproxen",
        severity="Moderate",
        confidence="High",
        mechanism=(
            "Same renal prostaglandin mechanism as ibuprofen + infliximab. "
            "Naproxen's longer half-life (12–17 h) sustains COX-2 inhibition, "
            "compounding fluid-retention risk in patients with cardiac co-morbidities."
        ),
        alert=(
            "Limit naproxen to short courses; monitor BP, weight, and eGFR. "
            "Avoid if eGFR < 30 mL/min."
        ),
        monitoring=["eGFR", "Blood pressure", "Oedema assessment"],
        alternatives=["Acetaminophen", "Celecoxib at lowest dose if COX-2 selectivity needed"],
        populations=["Autoimmune disease"],
        references=["BSR guidelines on NSAID use in biologic-treated patients 2021"],
    ),

    # ── IL-6 inhibitors × NSAIDs ─────────────────────────────────────────────
    InteractionRecord(
        immunosuppressant="tocilizumab",
        co_therapy="ibuprofen",
        severity="Moderate",
        confidence="Medium",
        mechanism=(
            "Tocilizumab (IL-6R blockade) suppresses CYP450 enzyme normalisation "
            "that occurs during active inflammation. Initiating tocilizumab may "
            "alter NSAID metabolism and heighten GI/renal risk as systemic "
            "inflammation subsides."
        ),
        alert=(
            "Review NSAID dosing when starting tocilizumab; reduced inflammation may "
            "change drug metabolism and increase GI bleeding risk."
        ),
        monitoring=["GI symptoms", "SCr", "LFTs"],
        alternatives=["Acetaminophen", "PPI co-prescription if NSAID continuation is necessary"],
        populations=["Autoimmune disease"],
        references=[
            "Roche Actemra prescribing information 2023",
            "ACR tocilizumab safety review 2021",
        ],
    ),

    # ── JAK inhibitors × NSAIDs ──────────────────────────────────────────────
    InteractionRecord(
        immunosuppressant="tofacitinib",
        co_therapy="diclofenac",
        severity="Moderate",
        confidence="Medium",
        mechanism=(
            "Diclofenac inhibits COX-2 and is a CYP2C9 substrate/inhibitor. "
            "Tofacitinib is metabolised by CYP3A4 and CYP2C19; additive GI "
            "and renal toxicity risk, with possible modest elevation of "
            "diclofenac exposure."
        ),
        alert=(
            "Use diclofenac at lowest effective dose; monitor GI symptoms and "
            "renal function in patients on tofacitinib."
        ),
        monitoring=["GI symptoms / occult blood", "SCr", "BP"],
        alternatives=["Acetaminophen", "Topical diclofenac"],
        populations=["Autoimmune disease"],
        references=["Pfizer Xeljanz safety profile 2023"],
    ),

    InteractionRecord(
        immunosuppressant="baricitinib",
        co_therapy="celecoxib",
        severity="Minor",
        confidence="Medium",
        mechanism=(
            "Celecoxib is a selective COX-2 inhibitor with lower GI risk than "
            "non-selective NSAIDs. Pharmacodynamic additive cardiovascular risk "
            "(BP elevation, fluid retention) exists with JAK inhibitors at higher doses."
        ),
        alert=(
            "Celecoxib is preferred over non-selective NSAIDs with baricitinib "
            "but is not risk-free; monitor cardiovascular parameters."
        ),
        monitoring=["Blood pressure", "Lipid panel (JAK inhibitors raise LDL)"],
        alternatives=["Acetaminophen"],
        populations=["Autoimmune disease"],
        references=["Eli Lilly Olumiant prescribing information 2023"],
    ),

    # ── Anti-TNF × Macrolides ────────────────────────────────────────────────
    InteractionRecord(
        immunosuppressant="infliximab",
        co_therapy="clarithromycin",
        severity="Major",
        confidence="Medium",
        mechanism=(
            "Pharmacodynamic: clarithromycin has significant immunomodulatory "
            "effects (suppresses neutrophil oxidative burst). Combined with "
            "anti-TNF therapy, this markedly increases risk of serious and "
            "opportunistic infections."
        ),
        alert=(
            "Avoid clarithromycin in patients on anti-TNF therapy where possible; "
            "use azithromycin or amoxicillin as safer alternatives."
        ),
        monitoring=["Signs of serious or opportunistic infection", "CRP/ESR"],
        alternatives=["Azithromycin", "Amoxicillin-clavulanate", "Doxycycline"],
        populations=["Autoimmune disease", "Transplant"],
        references=["IDSA immunocompromised host infection guidelines 2022"],
    ),

    # ── IL-17 inhibitors × NSAIDs ────────────────────────────────────────────
    InteractionRecord(
        immunosuppressant="secukinumab",
        co_therapy="ibuprofen",
        severity="Minor",
        confidence="Low",
        mechanism=(
            "No direct PK interaction; pharmacodynamic concern only. "
            "IL-17 blockade slightly impairs mucosal barrier integrity in the GI "
            "tract; NSAID-induced mucosal injury may be additive."
        ),
        alert=(
            "Prefer acetaminophen over NSAIDs in patients on IL-17 inhibitors; "
            "if NSAID use is necessary, add a PPI."
        ),
        monitoring=["GI symptoms"],
        alternatives=["Acetaminophen", "Topical NSAID"],
        populations=["Autoimmune disease"],
        references=["Novartis Cosentyx prescribing information 2023"],
    ),

    # ── Acetaminophen – generally low risk context ───────────────────────────
    InteractionRecord(
        immunosuppressant="tofacitinib",
        co_therapy="acetaminophen",
        severity="Minor",
        confidence="High",
        mechanism=(
            "No clinically significant PK interaction. Both drugs undergo hepatic "
            "metabolism but via separate pathways (CYP3A4 vs glucuronidation). "
            "Acetaminophen is the preferred analgesic in immunocompromised patients."
        ),
        alert=(
            "Acetaminophen is the recommended first-line analgesic with tofacitinib; "
            "do not exceed 3 g/day and monitor LFTs with prolonged use."
        ),
        monitoring=["LFTs with long-term use"],
        alternatives=[],
        populations=["Autoimmune disease", "Transplant", "Oncology"],
        references=["ACR Guideline on analgesic use in rheumatic disease 2022"],
    ),

    # ── Ustekinumab × Fluoroquinolones ───────────────────────────────────────
    InteractionRecord(
        immunosuppressant="ustekinumab",
        co_therapy="ciprofloxacin",
        severity="Minor",
        confidence="Low",
        mechanism=(
            "Ustekinumab (IL-12/23 inhibitor) has no significant CYP450 involvement. "
            "Additive pharmacodynamic concern: fluoroquinolones + biologic "
            "immunosuppression increases C. difficile risk."
        ),
        alert=(
            "Limit fluoroquinolone duration to the minimum necessary; "
            "consider probiotics and monitor for C. difficile colitis."
        ),
        monitoring=["Stool culture if GI symptoms develop"],
        alternatives=["Trimethoprim/sulfamethoxazole for appropriate indications"],
        populations=["Autoimmune disease"],
        references=["Janssen Stelara prescribing information 2023"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Class-level interaction templates
#  Used as fallback when an exact pair is not in INTERACTIONS
# ─────────────────────────────────────────────────────────────────────────────

CLASS_INTERACTIONS: list[dict] = [
    {
        "immunosuppressant_class": "jak_inhibitor",
        "co_therapy_class": "azole_antifungal",
        "severity": "Major",
        "confidence": "High",
        "mechanism": (
            "JAK inhibitors are CYP3A4 substrates. Azole antifungals inhibit "
            "CYP3A4 to varying degrees (ketoconazole > itraconazole > voriconazole "
            "> fluconazole), elevating JAK inhibitor plasma levels and risk of "
            "haematologic, hepatic, and infectious adverse effects."
        ),
        "alert": (
            "Reduce JAK inhibitor dose per prescribing information and monitor "
            "CBC, LFTs when any azole antifungal is co-prescribed."
        ),
        "monitoring": ["CBC with differential", "LFTs", "Signs of serious infection"],
        "alternatives": ["Micafungin", "Anidulafungin"],
        "populations": ["Autoimmune disease", "Transplant", "Oncology"],
        "references": ["FDA JAK inhibitor class labelling update 2022"],
    },
    {
        "immunosuppressant_class": "jak_inhibitor",
        "co_therapy_class": "rifamycin",
        "severity": "Major",
        "confidence": "High",
        "mechanism": (
            "Rifamycins are potent CYP3A4 inducers. Co-administration with JAK "
            "inhibitors reduces JAK inhibitor AUC by 50–84%, risking sub-therapeutic "
            "immunosuppression and disease relapse."
        ),
        "alert": (
            "Avoid rifamycins with JAK inhibitors; if TB treatment is necessary, "
            "consider rifabutin with a supervised dose increase."
        ),
        "monitoring": ["Disease activity scores", "Drug levels"],
        "alternatives": ["Rifabutin (weaker CYP3A4 inducer)"],
        "populations": ["Autoimmune disease", "Transplant"],
        "references": ["ATS/CDC/IDSA TB treatment guidelines 2022"],
    },
    {
        "immunosuppressant_class": "anti_tnf",
        "co_therapy_class": "nsaid",
        "severity": "Moderate",
        "confidence": "High",
        "mechanism": (
            "Pharmacodynamic: NSAIDs reduce prostaglandin-mediated renal blood flow "
            "protection. Anti-TNF agents may independently impair renal haemodynamics "
            "in inflammatory states. Additive risk of AKI, GI bleeding, and "
            "fluid retention."
        ),
        "alert": (
            "Use lowest effective NSAID dose for shortest duration; "
            "monitor renal function and GI symptoms."
        ),
        "monitoring": ["eGFR", "GI symptoms", "Blood pressure"],
        "alternatives": ["Acetaminophen", "Topical NSAIDs"],
        "populations": ["Autoimmune disease", "Transplant"],
        "references": ["ACR NSAID safety recommendations 2020"],
    },
    {
        "immunosuppressant_class": "il6_inhibitor",
        "co_therapy_class": "nsaid",
        "severity": "Moderate",
        "confidence": "Medium",
        "mechanism": (
            "IL-6 blockade normalises CYP450 activity suppressed by inflammation, "
            "potentially altering NSAID metabolism. Additive GI and renal risk."
        ),
        "alert": (
            "Review NSAID dosing when initiating IL-6 inhibitors; "
            "add GI protection (PPI) if NSAID is continued."
        ),
        "monitoring": ["GI symptoms", "SCr", "LFTs"],
        "alternatives": ["Acetaminophen"],
        "populations": ["Autoimmune disease"],
        "references": ["Roche Actemra prescribing information 2023"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  Public look-up API
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_ORDER = {
    "Contraindicated": 4,
    "Major": 3,
    "Moderate": 2,
    "Minor": 1,
    "None": 0,
}

CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1}


def lookup_interactions(
    immunosuppressant: str,
    co_therapies: list[str],
) -> list[dict]:
    """
    Return all interaction records between one immunosuppressant and a list
    of co-therapy drugs.  Falls back to class-level rules when no exact
    match exists.

    Returns a list of result dicts sorted by descending severity.
    """
    is_canon = canonical(immunosuppressant)
    results: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for drug in co_therapies:
        co_canon = canonical(drug)

        # 1. Exact match
        for rec in INTERACTIONS:
            if (canonical(rec.immunosuppressant) == is_canon
                    and canonical(rec.co_therapy) == co_canon):
                pair = (is_canon, co_canon)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    results.append({
                        "immunosuppressant": is_canon,
                        "co_therapy": co_canon,
                        "query_co_therapy": drug,   # preserve user's original input
                        "severity": rec.severity,
                        "confidence": rec.confidence,
                        "mechanism": rec.mechanism,
                        "alert": rec.alert,
                        "monitoring": rec.monitoring,
                        "alternatives": rec.alternatives,
                        "populations": rec.populations,
                        "references": rec.references,
                        "match_type": "exact",
                    })

        # 2. Class-level fallback
        if (is_canon, co_canon) not in seen_pairs:
            is_classes = drug_class(is_canon)
            co_classes = drug_class(co_canon)
            for template in CLASS_INTERACTIONS:
                if (template["immunosuppressant_class"] in is_classes
                        and template["co_therapy_class"] in co_classes):
                    pair = (is_canon, co_canon)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        results.append({
                            "immunosuppressant": is_canon,
                            "co_therapy": co_canon,
                            "query_co_therapy": drug,
                            "severity": template["severity"],
                            "confidence": template["confidence"],
                            "mechanism": template["mechanism"],
                            "alert": template["alert"],
                            "monitoring": template["monitoring"],
                            "alternatives": template["alternatives"],
                            "populations": template["populations"],
                            "references": template["references"],
                            "match_type": "class-level",
                        })
                    break

        # 3. No interaction found
        if (is_canon, co_canon) not in seen_pairs:
            seen_pairs.add((is_canon, co_canon))
            results.append({
                "immunosuppressant": is_canon,
                "co_therapy": co_canon,
                "query_co_therapy": drug,
                "severity": "None",
                "confidence": "High",
                "mechanism": "No clinically significant interaction identified in curated database.",
                "alert": "No known interaction. Standard monitoring applies.",
                "monitoring": [],
                "alternatives": [],
                "populations": [],
                "references": [],
                "match_type": "none",
            })

    results.sort(
        key=lambda r: (SEVERITY_ORDER[r["severity"]],
                       CONFIDENCE_ORDER[r["confidence"]]),
        reverse=True,
    )
    return results


def all_immunosuppressants() -> list[str]:
    """Return a sorted list of all immunosuppressants in the database."""
    names: set[str] = set()
    for cls in ["anti_tnf", "il6_inhibitor", "il17_inhibitor",
                "il12_23_inhibitor", "jak_inhibitor"]:
        names.update(DRUG_CLASSES.get(cls, []))
    return sorted(names)


def all_co_therapies() -> list[str]:
    """Return a sorted list of all co-therapy drugs in the database."""
    names: set[str] = set()
    for cls in ["azole_antifungal", "macrolide_antibiotic", "fluoroquinolone",
                "rifamycin", "nsaid", "analgesic", "macrolide_azithromycin"]:
        names.update(DRUG_CLASSES.get(cls, []))
    return sorted(names)
