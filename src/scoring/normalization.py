"""Project-authored English mappings; aliases are equivalences, not fuzzy matches.

Method references: scorer_references.md (R1 skill aliases, R3 domain tags).
Unlisted skills retain their normalized name instead of disappearing.
"""

import re
import unicodedata


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


SKILL_ALIASES = {
    "python 3": "python", "python3": "python", "py": "python",
    "javascript": "javascript", "js": "javascript", "typescript": "typescript", "ts": "typescript",
    "postgres": "postgresql", "postgres sql": "postgresql", "node.js": "nodejs", "node js": "nodejs",
    "react.js": "react", "reactjs": "react", "scikit learn": "scikit-learn", "sklearn": "scikit-learn",
    "amazon web services": "aws", "google cloud platform": "gcp", "ms excel": "excel", "microsoft excel": "excel",
    "c sharp": "c#", "cpp": "c++", "c plus plus": "c++",
    "natural language processing": "nlp", "machine learning": "ml", "structured query language": "sql",
    "search engine optimization": "seo", "customer relationship management": "crm",
}


def skill_name(text: str) -> str:
    key = normalize(text)
    return SKILL_ALIASES.get(key, key)


def skill_set(values: list[str]) -> set[str]:
    return {skill_name(v) for v in values if normalize(v)}


# Each family may contain several distinct canonical roles.
ROLE_FAMILIES = {
    "software": {
        "software engineer": ["software developer", "programmer", "application developer"],
        "backend engineer": ["backend developer", "back end developer", "back-end developer"],
        "frontend engineer": ["frontend developer", "front end developer", "front-end developer"],
        "fullstack engineer": ["full stack developer", "full-stack developer", "fullstack developer"],
        "mobile developer": ["android developer", "ios developer"],
        "qa engineer": ["quality assurance engineer", "software tester", "test engineer"],
        "devops engineer": [], "site reliability engineer": ["sre"],
    },
    "data": {"data scientist": [], "machine learning engineer": ["ml engineer"], "ai engineer": [],
             "data analyst": ["business intelligence analyst", "bi analyst"], "data engineer": []},
    "finance": {"accountant": ["staff accountant", "bookkeeper"], "financial analyst": ["finance analyst"],
                "auditor": ["internal auditor"], "banker": ["bank officer"]},
    "sales": {"sales associate": ["sales representative", "sales rep", "salesperson"],
              "sales manager": [], "account executive": [], "business development manager": []},
    "marketing": {"marketing specialist": ["marketing coordinator", "marketing executive"],
                  "digital marketing specialist": ["seo specialist", "social media specialist"]},
    "human_resources": {"hr specialist": ["human resources specialist", "hr generalist", "recruiter"],
                        "hr manager": ["human resources manager"]},
    "healthcare": {"nurse": ["registered nurse", "rn"], "physician": ["doctor"],
                   "medical assistant": [], "pharmacist": []},
    "education": {"teacher": ["school teacher", "educator"], "lecturer": ["instructor"], "professor": []},
    "operations": {"operations manager": [], "project manager": [], "administrative assistant": ["office assistant"],
                   "customer service representative": ["customer support representative", "customer service associate"]},
    "engineering": {"mechanical engineer": [], "electrical engineer": [], "civil engineer": [], "industrial engineer": []},
    "design": {"graphic designer": [], "ux designer": ["user experience designer"], "ui designer": ["user interface designer"],
               "product designer": [], "ux/ui designer": []},
    "legal": {"lawyer": ["attorney"], "paralegal": ["legal assistant"]},
    "hospitality": {"chef": ["cook"], "server": ["waiter", "waitress"], "hotel manager": []},
    "logistics": {"warehouse associate": ["warehouse worker"], "logistics coordinator": [], "driver": ["truck driver"]},
}
ROLE_INDEX = {normalize(alias): (role, family) for family, roles in ROLE_FAMILIES.items()
              for role, aliases in roles.items() for alias in [role, *aliases]}


def role(text: str | None) -> tuple[str, str | None]:
    if not text:
        return "", None
    key = normalize(text).replace("_", " ")
    key = re.sub(r"\b(senior|junior|sr|jr|lead|principal|entry level|mid level)\b\.?", "", key)
    key = normalize(key).strip(" -")
    return ROLE_INDEX.get(key, (key, None))


def domain_set(facts: list[str], titles: list[str]) -> set[str]:
    domains = {normalize(v).replace(" ", "_") for v in facts if normalize(v).replace(" ", "_") in ROLE_FAMILIES}
    domains.update(family for title in titles if (family := role(title)[1]))
    return domains


COUNTRIES = {"us": "us", "usa": "us", "united states": "us", "united states of america": "us",
             "uk": "gb", "gb": "gb", "united kingdom": "gb", "vn": "vn", "vietnam": "vn",
             "ca": "ca", "canada": "ca", "au": "au", "australia": "au", "india": "in", "in": "in"}


def country_name(text: str | None) -> str:
    key = normalize(text or "")
    return COUNTRIES.get(key, key)
