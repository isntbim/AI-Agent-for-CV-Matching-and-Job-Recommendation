from scripts.cv_external_apply import apply, drop_bad_levels


def test_drop_bad_levels_keeps_the_skill_and_clears_only_named_levels():
    lab = {"document": {"skills": [{"name": "Tools", "details": [
        {"name": "Excel", "proficiency": "expert", "raw_proficiency": "Expert", "proficiency_quote": "Expert in Office"},
        {"name": "SQL", "proficiency": "advanced", "raw_proficiency": "Advanced", "proficiency_quote": "Advanced SQL"}]}]}}
    err = "Invalid v3 skill evidence: skills.Excel.proficiency:quote_not_linked_to_skill"
    assert drop_bad_levels(lab, err) == 1
    excel, sql = lab["document"]["skills"][0]["details"]
    assert (excel["proficiency"], excel["raw_proficiency"], excel["proficiency_quote"]) == ("normal", None, None)
    assert sql["proficiency"] == "advanced"


def label():
    return {"document": {"work": [{"name": "Acme", "position": "Clerk"}, {"name": "Pfizer", "position": "Rep"}],
                         "education": [], "languages": [{"language": "French", "fluency": None}],
                         "skills": [{"name": "Languages", "level": None, "keywords": ["French"],
                                     "details": [{"name": "French", "skill_quote": "French"}]}]},
            "facts": {"domains": ["sales"], "completed_degrees": []}}


def test_apply_maps_a_b_value_and_employer_items():
    items = [{"item": "x|facts.domains", "kind": "fact", "field": "facts.domains", "A": ["hr"], "B": ["sales"]},
             {"item": "x|employer", "kind": "employer", "field": "document.work[].name", "names": ["Acme", "Pfizer"]},
             {"item": "x|document.languages", "kind": "records", "field": "document.languages",
              "A": [], "B": [{"language": "French", "fluency": None}]}]
    decisions = [{"item": "x|facts.domains", "decision": "A"},
                 {"item": "x|employer", "decision": "value", "value": {"null": ["Pfizer"]}},
                 {"item": "x|document.languages", "decision": "A"}]
    out = apply(label(), items, decisions)
    assert out["facts"]["domains"] == ["hr"]
    assert [w["name"] for w in out["document"]["work"]] == ["Acme", None]
    assert out["document"]["languages"] == [] and not any(g["name"] == "Languages" for g in out["document"]["skills"])
    assert label()["facts"]["domains"] == ["sales"]  # input untouched


def test_drop_bad_levels_removes_a_skill_whose_own_quote_fails():
    lab = {"document": {"skills": [{"name": "Soft", "keywords": ["Complex problem solving"], "details": [
        {"name": "Complex problem solving", "proficiency": "normal", "raw_proficiency": None,
         "proficiency_quote": None, "skill_quote": "managementComplex problem solvingSuperior"}]}]}}
    assert drop_bad_levels(lab, "skills.Complex problem solving:quote_not_linked_to_skill") == 1
    assert lab["document"]["skills"] == []
