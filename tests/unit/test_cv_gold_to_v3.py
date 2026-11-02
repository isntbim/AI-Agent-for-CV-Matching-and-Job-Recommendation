from scripts.cv_gold_to_v3 import convert, literal


def test_literal_recovers_written_dates_and_rejects_absent_ones():
    text = "Analyst 11/1991to 03/1995\nIntern Feb-2007 to Present\nGraduated May 2010"
    assert literal("1991-11", text) == "11/1991"
    assert literal("2007-02", text) == "Feb-2007"
    assert literal("2010-05", text) == "May 2010"
    assert literal("Present", text) == "Present"
    assert literal("Analyst  11/1991to", text) == "Analyst 11/1991to"
    assert literal("1991-12", text) is None
    assert literal("Senior Analyst", text) is None


def test_convert_renames_keys_and_applies_mechanical_rules():
    text = "Accountant at Acme 2011-07. Languages: English, French (fluent). Austin, TX"
    envelope = {"document": {
        "basics": {"name": "Candidate", "location": {"postal_code": None, "country_code": None, "city": "Austin"}},
        "work": [{"name": "Company Name", "start_date": "2011-07", "end_date": None}],
        "education": [], "projects": [{"name": "P", "start_date": None, "end_date": None, "project_type": "volunteer"}],
        "certificates": [{"name": "C", "certificate_type": "certification"}],
        "languages": [{"language": "English", "fluency": None}, {"language": "French (fluent)", "fluency": None}],
        "skills": [{"name": "Source reviewed skills", "level": None, "keywords": ["accounting", "english"],
                    "details": [{"name": "accounting", "skill_quote": "Accountant", "provenance": {"source_id": "x"}},
                                {"name": "english", "skill_quote": "English"}]}]},
        "facts": {"evidence": {"location": [{"quote": "US"}], "experience": [{"quote": "2011-07"}]},
                  "domains": [], "country": "US"}}
    label, log = convert(envelope, text)
    doc, facts = label["document"], label["facts"]
    assert doc["basics"]["location"] == {"postalCode": None, "countryCode": None, "city": "Austin"}
    assert doc["work"][0] == {"name": None, "startDate": "2011-07", "endDate": None}
    assert doc["projects"][0]["type"] == "project" and doc["certificates"][0]["type"] == "certificate"
    general, languages = doc["skills"]
    assert general["name"] == "General" and general["keywords"] == ["accounting"]
    assert "provenance" not in general["details"][0]
    assert languages["name"] == "Languages" and languages["keywords"] == ["english", "French (fluent)"]
    assert facts["country"] is None and "location" not in facts["evidence"]
    assert facts["evidence"]["experience"] == [{"quote": "2011-07"}]
    assert "location: US" in log["dropped"] and "facts.domains empty" in log["review"]


def test_convert_applies_strict_finals_and_rejects_stale_ones():
    import pytest
    text = "Skills: CRM Platforms (Salesforce, Zoho), financial statements"
    envelope = {"document": {
        "basics": {"location": None}, "work": [], "education": [], "projects": [], "certificates": [], "languages": [],
        "skills": [{"name": "General", "level": None, "keywords": [], "details": [
            {"name": "crm platforms (salesforce, zoho)", "skill_quote": "CRM Platforms (Salesforce, Zoho)"},
            {"name": "financial statements", "skill_quote": "financial statements"}]}]},
        "facts": {"evidence": {}, "domains": ["sales"], "country": None}}
    zoho = {"name": "zoho", "group": "General", "skill_quote": "CRM Platforms (Salesforce, Zoho)"}
    label, _ = convert(envelope, text, strict=[{"drop": "Financial Statements"},
                                               {"drop": "crm platforms (salesforce, zoho)"}, {"set": zoho}])
    assert label["document"]["skills"][0]["keywords"] == ["zoho"]
    with pytest.raises(ValueError, match="strict drop not in label"):
        convert(envelope, text, strict=[{"drop": "budgets"}])


def test_strict_drop_removes_only_the_quoted_copy_of_a_twice_listed_skill():
    text = "Advanced Excel\nWord Excel Access"
    envelope = {"document": {
        "basics": {"location": None}, "work": [], "education": [], "projects": [], "certificates": [], "languages": [],
        "skills": [{"name": "General", "level": None, "keywords": [], "details": [
            {"name": "excel", "skill_quote": "Advanced Excel"}, {"name": "excel", "skill_quote": "Word Excel Access"}]}]},
        "facts": {"evidence": {}, "domains": ["finance"], "country": None}}
    label, _ = convert(envelope, text, strict=[{"drop": "excel", "skill_quote": "Word Excel Access"}])
    assert label["document"]["skills"][0]["details"] == [{"name": "excel", "skill_quote": "Advanced Excel"}]
    import pytest
    label, _ = convert(envelope, text, strict=[{"drop": "excel", "skill_quote": "Word  Excel Access"}])  # respaced quote
    assert label["document"]["skills"][0]["details"] == [{"name": "excel", "skill_quote": "Advanced Excel"}]
    with pytest.raises(ValueError, match="strict drop quote not in label"):  # a stale quote never deletes the other copy
        convert(envelope, text, strict=[{"drop": "excel", "skill_quote": "Excel Outlook"}])
    envelope["document"]["skills"][0]["details"][1].pop("skill_quote")  # nor a copy that has no quote
    with pytest.raises(ValueError, match="strict drop quote not in label"):
        convert(envelope, text, strict=[{"drop": "excel", "skill_quote": "Excel Outlook"}])


def test_strict_set_accepts_a_name_broken_by_a_pdf_line_break():
    text = "Tools: Microsoft SQL\nServer"
    envelope = {"document": {
        "basics": {"location": None}, "work": [], "education": [], "projects": [], "certificates": [], "languages": [],
        "skills": []}, "facts": {"evidence": {}, "domains": ["software"], "country": None}}
    detail = {"name": "Microsoft SQL Server", "group": "Tools", "skill_quote": "Microsoft SQL\nServer"}
    label, _ = convert(envelope, text, strict=[{"set": detail}])
    assert label["document"]["skills"][0]["details"][0]["skill_quote"] == "Microsoft SQL\nServer"


def test_field_final_sets_one_record_field_and_rejects_a_stale_index():
    import pytest
    envelope = {"document": {
        "basics": {"location": None}, "education": [], "projects": [], "certificates": [], "languages": [], "skills": [],
        "work": [{"name": "Acme", "position": "Clerk", "start_date": "May 2015", "end_date": "Current"}]},
        "facts": {"evidence": {}, "domains": [], "country": None}}
    acme = {"name": "Acme", "position": "Clerk"}
    label, _ = convert(envelope, "Clerk May 2015 - Current", strict=[
        {"field": "work[0].startDate", "record": acme, "from": "May 2015", "value": "2015-05"},
        {"field": "work[0].endDate", "record": acme, "from": "Current", "value": "Present"}])
    assert label["document"]["work"][0]["startDate"] == "2015-05" and label["document"]["work"][0]["endDate"] == "Present"
    with pytest.raises(ValueError, match="expected"):
        convert(envelope, "Clerk", strict=[{"field": "work[0].startDate", "record": acme, "from": "June 2015", "value": "2015-06"}])
    with pytest.raises(ValueError, match="expected record"):  # same old value on another record: the identity catches it
        convert(envelope, "Clerk", strict=[{"field": "work[0].startDate", "record": {"name": "Acme", "position": "Intern"},
                                            "from": "May 2015", "value": "2015-05"}])


def test_split_turns_one_record_into_its_written_periods_after_the_field_finals():
    envelope = {"document": {
        "basics": {"location": None}, "education": [], "projects": [], "certificates": [], "languages": [], "skills": [],
        "work": [{"name": "Uni", "position": "TA", "start_date": "2004; 2005", "end_date": None},
                 {"name": "Lab", "position": "RA", "start_date": "2003", "end_date": "2005"}]},
        "facts": {"evidence": {}, "domains": [], "country": None}}
    ta = {"name": "Uni", "position": "TA"}
    label, _ = convert(envelope, "TA Spring 2004, Spring 2005", strict=[
        {"split": "work[0]", "record": ta, "into": [{"startDate": "2004", "endDate": "2004"}, {"startDate": "2005", "endDate": "2005"}]},
        {"field": "work[0].startDate", "record": ta, "from": "2004; 2005", "value": "2004"},
        {"field": "work[1].endDate", "record": {"name": "Lab", "position": "RA"}, "from": "2005", "value": "2005-10"}])
    assert [(w["position"], w["startDate"], w["endDate"]) for w in label["document"]["work"]] == [
        ("TA", "2004", "2004"), ("TA", "2005", "2005"), ("RA", "2003", "2005-10")]
