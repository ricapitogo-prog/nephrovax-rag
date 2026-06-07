"""UI helpers used only by the Streamlit app — not part of the library logic."""


def _shorten_prior_vaccine(prior: str) -> str:
    """Compress long prior-vaccine descriptions for use in compact UI badges.

    The breadcrumbs preserve the full source headings (sometimes verbose for
    audit purposes), but in a one-line badge we need shorter labels.
    """
    # Penbraya/Penmenvy combination — collapse to just the brand names
    if "Penbraya" in prior and "Penmenvy" in prior:
        return "Penbraya or Penmenvy (combined)"
    # Strip generic parenthetical brand names when the abbreviation is also present
    # e.g. "PCV13 (Prevnar 13) only" → "PCV13 only"
    # e.g. "PCV20 (Prevnar 20)" → "PCV20"
    # e.g. "PPSV23 (Pneumovax 23)" → "PPSV23"
    import re as _re
    prior = _re.sub(r"\s*\([^)]+\)", "", prior).strip()
    return prior


def humanize_breadcrumb(doc_id: str, breadcrumb: list[str]) -> str:
    """Convert a structural breadcrumb into a short human-readable label.

    Used for both the system-interpretation badge and the footnote references.

    Examples:
        ("nephrovax-pneumococcal-2026-02",
         ["Pathway B — Prior pneumococcal vaccination (known vaccine)",
          "Prior vaccine: PCV20 (Prevnar 20)",
          "Age ≥19 years"])
        →  "Pneumococcal · Prior PCV20 · Adults ≥19"

        ("nephrovax-meningococcal-2026-02",
         ["Pathway A — No or unknown prior meningococcal vaccination",
          "Age 2–9 years"])
        →  "Meningococcal · No prior vaccination · Ages 2–9"
    """
    # Vaccine class from doc_id
    vaccine_class = ""
    if "pneumococcal" in doc_id:
        vaccine_class = "Pneumococcal"
    elif "meningococcal" in doc_id:
        vaccine_class = "Meningococcal"
    elif "hib" in doc_id:
        vaccine_class = "Hib"

    # Pathway short label
    pathway_label = ""
    if breadcrumb and "Pathway A" in breadcrumb[0]:
        pathway_label = "No prior vaccination"
    elif breadcrumb and "Pathway B" in breadcrumb[0]:
        # Look for the "Prior vaccine: X" element
        prior = ""
        for crumb in breadcrumb[1:-1]:
            if crumb.startswith("Prior vaccine:"):
                prior = crumb.replace("Prior vaccine:", "").strip()
                break
        if prior:
            prior = _shorten_prior_vaccine(prior)
            pathway_label = f"Prior {prior}"
        else:
            pathway_label = "Prior vaccination"

    # Age band — last element
    age_label = ""
    if breadcrumb:
        last = breadcrumb[-1]
        # Trim "Age" prefix for compactness
        age_label = last.replace("Age ", "").strip()
        # Add "adults" / "children" hint for the common bands
        age_lower = age_label.lower()
        if "≥19" in age_label:
            age_label = "Adults ≥19"
        elif "≥10" in age_label:
            age_label = "Ages ≥10"
        elif "6–18" in age_label or "6-18" in age_label:
            age_label = "Ages 6–18"
        elif "6–19" in age_label or "6-19" in age_label:
            age_label = "Ages 6–19"
        elif "2–9" in age_label or "2-9" in age_label:
            age_label = "Ages 2–9"
        elif "<2" in age_label:
            age_label = "Infants <2"
        elif "≤5" in age_label:
            age_label = "Children ≤5"
        elif "<5" in age_label:
            age_label = "Children <5"
        elif "≥5" in age_label:
            age_label = "Ages ≥5"
        elif "all ages" in age_lower:
            age_label = "All ages"

    parts = [p for p in [vaccine_class, pathway_label, age_label] if p]
    return " · ".join(parts)
