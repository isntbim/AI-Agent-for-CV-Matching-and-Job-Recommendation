"""Label-blind serializers. Read dictionaries before schema defaults are applied."""

TEXT_VERSION = "retrieval-text-v1"
_MISSING = {"n/a", "none", "null", "not specified", "not mentioned", "unknown"}


def _value(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in _MISSING else text


def _add(parts, label, values):
    if not isinstance(values, (list, tuple)):
        values = [values]
    values = list(dict.fromkeys(v for value in values if (v := _value(value))))
    if values:
        parts.append(f"{label}: " + "; ".join(values))


def resume_text(record: dict) -> str:
    parts = []
    basics = record.get("basics") or {}
    _add(parts, "Title", basics.get("label"))
    _add(parts, "Summary", basics.get("summary"))
    for skill in record.get("skills") or []:
        _add(parts, "Skills", [skill.get("name"), *(skill.get("keywords") or [])])
    for work in record.get("work") or []:
        _add(parts, "Experience", work.get("position"))
        _add(parts, "Dates", [work.get("startDate", work.get("start_date")),
                              work.get("endDate", work.get("end_date"))])
        _add(parts, "Responsibilities", [work.get("summary"), *(work.get("highlights") or [])])
    for edu in record.get("education") or []:
        _add(parts, "Education", [edu.get("studyType", edu.get("study_type")), edu.get("area"),
                                  *(edu.get("courses") or [])])
    for project in record.get("projects") or []:
        _add(parts, "Project", [project.get("name"), project.get("description"),
                                *(project.get("highlights") or []), *(project.get("keywords") or [])])
    _add(parts, "Certificates", [c.get("name") for c in record.get("certificates") or []])
    return "\n".join(parts)


def job_text(record: dict) -> str:
    parts = []
    _add(parts, "Job title", record.get("title"))
    seniority = record.get("seniority") or {}
    _add(parts, "Seniority", seniority.get("level"))
    if seniority.get("min_years") is not None:
        _add(parts, "Minimum experience (years)", seniority["min_years"])
    if seniority.get("max_years") is not None:
        _add(parts, "Maximum experience (years)", seniority["max_years"])
    skills = record.get("skills") or {}
    for bucket in ("required", "preferred"):
        _add(parts, bucket.capitalize() + " skills", [s.get("name") for s in skills.get(bucket) or []])
    description = record.get("description") or {}
    _add(parts, "Summary", description.get("summary"))
    _add(parts, "Responsibilities", description.get("responsibilities") or [])
    _add(parts, "Requirements", description.get("requirements") or [])
    return "\n".join(parts)
