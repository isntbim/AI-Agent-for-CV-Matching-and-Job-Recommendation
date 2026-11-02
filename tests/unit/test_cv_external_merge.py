from scripts.cv_external_merge import blind, union_skills


def label(groups):
    return {"document": {"skills": [{"name": g, "level": None, "keywords": [n for n in names],
                                     "details": [{"name": n} for n in names]} for g, names in groups.items()]}}


def test_union_adds_only_missing_skills_and_keeps_keywords_in_sync():
    base = label({"Tools": ["Python", "Excel"]})
    added = union_skills(base, label({"tools": ["python", "SQL"], "Soft": ["Negotiation"]}))
    assert added == 2
    tools, soft = base["document"]["skills"]
    assert [d["name"] for d in tools["details"]] == tools["keywords"] == ["Python", "Excel", "SQL"]
    assert soft["name"] == "Soft" and soft["keywords"] == ["Negotiation"]
    assert union_skills(base, label({"Languages": ["French"]})) == 0  # languages follow document.languages


def test_blinding_is_deterministic_and_hides_the_teacher():
    first, teacher = blind("ex_hr_001", "facts.domains", ["hr"], ["sales"])
    assert blind("ex_hr_001", "facts.domains", ["hr"], ["sales"]) == (first, teacher)
    assert first["A"] == (["hr"] if teacher == "haiku" else ["sales"])
