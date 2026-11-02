from scripts.cv_text_refresh import respace


def test_respace_fixes_glued_and_line_broken_names_without_losing_letters():
    assert respace("ContentCreation", "Content Creation, SEO") == "Content Creation"
    assert respace("FAIRdata\nprinciples", "with FAIR data\nprinciples") == "FAIR data principles"
    assert respace("MicrosoftSQL\nServer", "Android Studio, Microsoft SQL\nServer, Postman") == "Microsoft SQL Server"
    assert respace("machine\nlearning", "machine\nlearning") == "machine learning"
    assert respace("cross-\ntraining", "cross-\ntraining") == "cross-training"
    assert respace("Python", "Skills: Python") == "Python"
    assert respace("fault tolerance", "fault toler\nance") == "fault tolerance"
    assert respace("Problem - Solving", "Problem -\nSolving") == "Problem - Solving"
    assert respace("Docker", "no match here") == "Docker"
