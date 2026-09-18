"""
CareerViet Importer and Standardizer
Transforms scraped CareerViet jobs from data/raw/jobs/careerviet_*.xlsx
into the standardized JobDescriptionSchema (Pydantic v2).

Partitions JDs based on user direction:
- Primary Corpus (EN + Mixed): data/processed/structured_jobs/jd_{id}.json & data/raw/jobs/jd_{id}.txt
- Secondary Corpus (VI): data/processed/structured_jobs_vi/jd_vi_{id}.json & data/raw/jobs_vi/jd_vi_{id}.txt
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import openpyxl
from src.parser.job_schemas import (
    Benefits,
    Company,
    Compensation,
    DescriptionSection,
    JobDescriptionSchema,
    JobLocation,
    JobMetadata,
    Seniority,
    SkillItem,
    SkillsSection,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Vietnamese diacritic regex pattern
VN_CHARS_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)

EN_STOPWORDS: Set[str] = {
    "the", "and", "to", "of", "in", "for", "with", "a", "as", "is", "that", "on",
    "you", "are", "be", "this", "have", "from", "at", "by", "an", "we", "will",
    "experience", "work", "skills", "team", "development", "ability", "knowledge",
    "requirements", "responsibilities", "years", "business", "support", "management",
    "candidate", "working", "degree", "required", "preferred", "communication"
}

VN_STOPWORDS: Set[str] = {
    "và", "của", "các", "có", "được", "cho", "trong", "để", "về", "người", "những",
    "là", "với", "này", "một", "không", "từ", "đã", "sẽ", "tại", "theo", "khi",
    "hay", "như", "công", "ty", "việc", "làm", "kinh", "nghiệm", "yêu", "cầu",
    "mô", "tả", "thực", "hiện", "chức", "vụ", "quyền", "lợi", "chính", "sách"
}

DOMAIN_MAPPING: Dict[str, str] = {
    "careerviet_ACCOUNTANT.xlsx": "accountant",
    "careerviet_Data_Scientist_Analystr.xlsx": "data_scientist_analyst",
    "careerviet_FrontendWebDeveloper.xlsx": "frontend_web_developer",
    "careerviet_HR.xlsx": "hr",
    "careerviet_Marketing_Executive.xlsx": "marketing_executive",
    "careerviet_Software_Engineer.xlsx": "software_engineer",
}

DOMAIN_SKILLS: Dict[str, List[str]] = {
    "accountant": [
        "Financial Reporting", "Tax Reporting", "General Ledger", "Accounts Payable",
        "Accounts Receivable", "Bank Reconciliation", "Internal Audit", "Cost Accounting",
        "ERP", "SAP", "MISA", "Fast", "Excel", "IFRS", "VAS", "Financial Statement",
        "Kế toán thuế", "Kế toán tổng hợp", "Báo cáo tài chính", "Hóa đơn", "Quyết toán thuế",
    ],
    "data_scientist_analyst": [
        "Python", "SQL", "R", "Machine Learning", "Deep Learning", "Data Analysis",
        "Data Visualization", "Power BI", "Tableau", "Pandas", "NumPy", "Scikit-Learn",
        "TensorFlow", "PyTorch", "BigQuery", "Spark", "ETL", "Statistics", "Data Modeling",
    ],
    "frontend_web_developer": [
        "JavaScript", "TypeScript", "React", "Vue", "Angular", "Next.js", "HTML5",
        "CSS3", "Sass", "Tailwind CSS", "Redux", "Webpack", "Vite", "REST API",
        "GraphQL", "UI/UX", "Responsive Design", "Git",
    ],
    "hr": [
        "Recruitment", "Talent Acquisition", "Onboarding", "Employee Relations",
        "Payroll", "C&B", "Performance Management", "Training & Development",
        "Labor Law", "HRIS", "Compensation & Benefits", "KPI", "Tuyển dụng",
        "Bảo hiểm xã hội", "Tiền lương", "Đào tạo nhân sự",
    ],
    "marketing_executive": [
        "Digital Marketing", "SEO", "SEM", "Google Ads", "Facebook Ads", "Content Marketing",
        "Social Media Marketing", "Copywriting", "Market Research", "Brand Management",
        "Email Marketing", "Google Analytics", "Campaign Management", "TikTok Marketing",
    ],
    "software_engineer": [
        "Python", "Java", "C++", "C#", ".NET", "Go", "Golang", "Rust", "Spring Boot",
        "Microservices", "REST API", "Docker", "Kubernetes", "Git", "CI/CD", "PostgreSQL",
        "MySQL", "MongoDB", "Redis", "Kafka", "AWS", "Azure", "Linux", "SQL",
    ],
}


def detect_language(text: str) -> str:
    """Classify text language into 'en', 'vi', or 'mixed'."""
    if not text or len(text.strip()) < 30:
        return "en"

    text_clean = text.lower()
    vn_char_count = len(VN_CHARS_RE.findall(text_clean))
    words = re.findall(
        r"\b[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+\b",
        text_clean,
    )

    if not words:
        return "en"

    en_stop = sum(1 for w in words if w in EN_STOPWORDS)
    vn_stop = sum(1 for w in words if w in VN_STOPWORDS)

    if vn_char_count <= 4 and en_stop >= 5:
        return "en"
    if vn_char_count > 25 and vn_stop > en_stop * 2:
        return "vi"
    if en_stop >= 5 and vn_char_count > 15:
        return "mixed"
    if vn_char_count > 10:
        return "vi"
    elif en_stop > 3:
        return "en"
    else:
        return "mixed"


def parse_careerviet_salary(salary_str: Optional[str]) -> Tuple[str, Optional[int], Optional[int], bool, str]:
    """Parse salary into currency, min_monthly, max_monthly, is_negotiable, raw_text."""
    if not salary_str or str(salary_str).lower() in [
        "none", "cạnh tranh", "thương lượng", "thỏa thuận", "competitive", "thoả thuận"
    ]:
        raw = str(salary_str or "Cạnh tranh").strip()
        return "VND", None, None, True, raw

    s = str(salary_str).strip()
    is_neg = "cạnh tranh" in s.lower() or "thương lượng" in s.lower() or "thỏa thuận" in s.lower()

    # USD: e.g. "1,000 - 2,000 USD" or "$1000 - $2000"
    if "usd" in s.lower() or "$" in s:
        nums = [float(x.replace(",", "")) for x in re.findall(r"(\d[\d,]*(?:\.\d+)?)", s)]
        min_m = int(nums[0]) if len(nums) >= 1 else None
        max_m = int(nums[1]) if len(nums) >= 2 else None
        return "USD", min_m, max_m, is_neg, s

    # VND range: e.g. "12 Tr - 15 Tr VND" or "12 - 15 Triệu VND"
    m_range = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:Tr|Triệu)?\s*-\s*(\d+(?:\.\d+)?)\s*(?:Tr|Triệu)?\s*(?:VND|VNĐ)?",
        s,
        re.IGNORECASE,
    )
    if m_range:
        min_m = int(float(m_range.group(1)) * 1_000_000)
        max_m = int(float(m_range.group(2)) * 1_000_000)
        return "VND", min_m, max_m, False, s

    # VND above: e.g. "Trên 30 Tr VND"
    m_above = re.search(r"trên\s*(\d+(?:\.\d+)?)\s*(?:Tr|Triệu)?", s, re.IGNORECASE)
    if m_above:
        min_m = int(float(m_above.group(1)) * 1_000_000)
        return "VND", min_m, None, False, s

    # VND below: e.g. "Dưới 10 Tr VND"
    m_below = re.search(r"dưới\s*(\d+(?:\.\d+)?)\s*(?:Tr|Triệu)?", s, re.IGNORECASE)
    if m_below:
        max_m = int(float(m_below.group(1)) * 1_000_000)
        return "VND", None, max_m, False, s

    return "VND", None, None, True, s


def parse_careerviet_experience(exp_str: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse experience years into min_years, max_years."""
    if not exp_str or str(exp_str).lower() in ["none", "không yêu cầu", "chưa có kinh nghiệm"]:
        return 0, 0
    s = str(exp_str).strip()
    m_range = re.search(r"(\d+)\s*-\s*(\d+)\s*(?:Năm|years?)?", s, re.IGNORECASE)
    if m_range:
        return int(m_range.group(1)), int(m_range.group(2))
    m_above = re.search(r"(?:trên|tối thiểu|at least|\+)\s*(\d+)", s, re.IGNORECASE)
    if m_above:
        return int(m_above.group(1)), None
    m_single = re.search(r"(\d+)\s*(?:Năm|years?)", s, re.IGNORECASE)
    if m_single:
        return int(m_single.group(1)), None
    return None, None


def parse_careerviet_level(level_str: Optional[str], min_years: Optional[int] = None) -> Tuple[str, str]:
    """Determine normalized seniority level and raw label."""
    raw_level = str(level_str).strip() if level_str else "Nhân viên"
    s = raw_level.lower()

    if any(w in s for w in ["giám đốc", "director", "head", "trưởng phòng"]):
        return "Director", raw_level
    if any(w in s for w in ["quản lý", "manager"]):
        return "Manager", raw_level
    if any(w in s for w in ["trưởng nhóm", "lead", "leader", "giám sát", "supervisor"]):
        return "Lead", raw_level
    if any(w in s for w in ["sinh viên", "thực tập", "intern"]):
        return "Intern", raw_level
    if any(w in s for w in ["mới đi làm", "fresher", "junior", "entry"]):
        return "Junior", raw_level

    if min_years is not None:
        if min_years >= 5:
            return "Senior", raw_level
        elif min_years >= 2:
            return "Mid-level", raw_level
        elif min_years >= 1:
            return "Junior", raw_level
        else:
            return "Junior", raw_level

    return "Mid-level", raw_level


def parse_careerviet_location(wp_str: Optional[str], loc_str: Optional[str], title_text: Optional[str]) -> Tuple[str, str, str, str]:
    """Parse city, region, country, and countryCode."""
    full = f"{wp_str or ''} {loc_str or ''} {title_text or ''}".lower()

    if any(k in full for k in ["hồ chí minh", "tp.hcm", "tp hcm", "tp. hcm", "hcm", "saigon", "sài gòn"]):
        city, region = "Ho Chi Minh City", "South"
    elif any(k in full for k in ["hà nội", "ha noi", "hanoi"]):
        city, region = "Ha Noi", "North"
    elif any(k in full for k in ["đà nẵng", "da nang"]):
        city, region = "Da Nang", "Central"
    elif any(k in full for k in ["bình dương", "binh duong"]):
        city, region = "Binh Duong", "South"
    elif any(k in full for k in ["đồng nai", "dong nai"]):
        city, region = "Dong Nai", "South"
    elif any(k in full for k in ["hải phòng", "hai phong"]):
        city, region = "Hai Phong", "North"
    elif any(k in full for k in ["bắc ninh", "bac ninh"]):
        city, region = "Bac Ninh", "North"
    elif any(k in full for k in ["cần thơ", "can tho"]):
        city, region = "Can Tho", "South"
    elif any(k in full for k in ["long an"]):
        city, region = "Long An", "South"
    elif any(k in full for k in ["vũng tàu", "bà rịa"]):
        city, region = "Ba Ria - Vung Tau", "South"
    elif any(k in full for k in ["an giang"]):
        city, region = "An Giang", "South"
    elif any(k in full for k in ["quảng nam", "quảng ngãi", "thừa thiên huế", "huế"]):
        city, region = "Central Vietnam", "Central"
    else:
        city, region = "Vietnam", "National"

    return city, region, "Vietnam", "VN"


GENERAL_SKILLS_EN: List[str] = [
    "Project Management", "Account Management", "Sales", "B2B", "B2C", "Customer Service",
    "Leadership", "Communication", "Problem Solving", "Teamwork", "Operations", "Compliance",
    "Budgeting", "Contract Negotiation", "Quality Assurance", "Logistics", "Supply Chain",
    "AutoCAD", "BIM", "C#", "C++", "Java", "Python", "SQL", "Excel", "PowerPoint",
    "ERP", "CRM", "English", "Business Development", "Risk Management", "HSE",
    "Talent Development", "Market Research", "Consulting",
]

GENERAL_SKILLS_VI: List[str] = [
    "Quản lý dự án", "Quản lý khách hàng", "Bán hàng", "Chăm sóc khách hàng", "Giao tiếp",
    "Kỹ năng lãnh đạo", "Giải quyết vấn đề", "Làm việc nhóm", "Vận hành", "Tuân thủ",
    "Quản lý ngân sách", "Đàm phán", "Đảm bảo chất lượng", "Logistics", "Chuỗi cung ứng",
    "Kế toán", "Nhân sự", "Tuyển dụng", "Tiếng Anh", "Tin học văn phòng", "Thuyết trình",
    "Phân tích dữ liệu", "Nghiên cứu thị trường", "Tư vấn khách hàng",
]


def extract_skills(tags_str: Optional[str], full_text: str, domain: str, title: str = "") -> List[str]:
    """Extract distinct skills from tags, domain list, cross-domain lists, and general skills."""
    skills: List[str] = []
    seen: Set[str] = set()

    if tags_str and str(tags_str).strip() and str(tags_str).lower() != "none":
        raw_tags = re.split(r"[\|,;]", str(tags_str))
        for t in raw_tags:
            cl = t.strip()
            if cl and len(cl) > 1 and cl.lower() not in seen:
                seen.add(cl.lower())
                skills.append(cl)

    text_lower = (full_text or "").lower()

    # 1. Augment with domain-specific keywords
    if len(skills) < 5 and text_lower:
        for kw in DOMAIN_SKILLS.get(domain, []):
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text_lower) and kw.lower() not in seen:
                seen.add(kw.lower())
                skills.append(kw)

    # 2. Cross-domain keyword check if still under 3 skills
    if len(skills) < 3 and text_lower:
        for other_dom, other_kws in DOMAIN_SKILLS.items():
            if other_dom == domain:
                continue
            for kw in other_kws:
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                if re.search(pattern, text_lower) and kw.lower() not in seen:
                    seen.add(kw.lower())
                    skills.append(kw)

    # 3. General professional skills check
    if len(skills) < 3 and text_lower:
        for kw in GENERAL_SKILLS_EN + GENERAL_SKILLS_VI:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text_lower) and kw.lower() not in seen:
                seen.add(kw.lower())
                skills.append(kw)

    # 4. Fallback: extract title keywords if still empty
    if not skills and title:
        title_words = [w.strip() for w in re.split(r"[\s\-\/\(\)\|]+", title) if len(w.strip()) >= 3]
        for w in title_words:
            if w.lower() not in seen and w.lower() not in EN_STOPWORDS and w.lower() not in VN_STOPWORDS:
                seen.add(w.lower())
                skills.append(w)
                if len(skills) >= 3:
                    break

    return skills


def parse_job_sections(desc_str: Optional[str], req_str: Optional[str]) -> Tuple[List[str], List[str], Optional[str]]:
    """Clean bullet lines and split responsibilities vs requirements intelligently."""
    def clean_lines(text: Optional[str]) -> List[str]:
        if not text:
            return []
        res = []
        for line in str(text).split("\n"):
            line = line.strip()
            if not line:
                continue
            cl = re.sub(r"^[•\-\*\+\d+\.\)\·\–\—\s]+", "", line).strip()
            if len(cl) > 5 and not cl.lower().endswith(
                ("mô tả công việc:", "yêu cầu công việc:", "quyền lợi:", "phúc lợi:", "job description:", "requirements:")
            ):
                res.append(cl)
        return res

    desc_bullets = clean_lines(desc_str)
    req_bullets = clean_lines(req_str)

    if not req_bullets and desc_bullets:
        split_idx = -1
        req_headers = [
            "yêu cầu công việc", "yêu cầu ứng viên", "tiêu chuẩn tuyển dụng",
            "job requirements", "requirements", "qualifications"
        ]
        for idx, b in enumerate(desc_bullets):
            if any(h in b.lower() for h in req_headers):
                split_idx = idx
                break
        if split_idx != -1:
            req_bullets = desc_bullets[split_idx + 1:]
            desc_bullets = desc_bullets[:split_idx]
        else:
            new_desc = []
            for b in desc_bullets:
                if any(k in b.lower() for k in [
                    "bachelor", "degree", "tốt nghiệp", "năm kinh nghiệm", "years of experience",
                    "proficient in", "thành thạo", "kinh nghiệm tối thiểu"
                ]):
                    req_bullets.append(b)
                else:
                    new_desc.append(b)
            if req_bullets:
                desc_bullets = new_desc

    summary = desc_bullets[0] if desc_bullets else None
    return desc_bullets, req_bullets, summary


def transform_careerviet_row(row_dict: Dict, job_id: str, domain: str, lang: str) -> JobDescriptionSchema:
    """Transform a single CareerViet Excel row into JobDescriptionSchema."""
    title = str(row_dict.get("job_title") or "Unknown Title").strip()
    company_name = str(row_dict.get("company_name") or "Confidential").strip()
    job_url = str(row_dict.get("job_url") or "https://careerviet.vn").strip()
    industries = str(row_dict.get("industries") or "").strip() or None
    salary_raw = row_dict.get("salary")
    wp_detail = row_dict.get("workplace_detail")
    locations = row_dict.get("locations")
    emp_type = str(row_dict.get("employment_type") or "Full-time").strip()
    exp_str = row_dict.get("experience")
    level_str = row_dict.get("level")
    desc_str = row_dict.get("description") or ""
    req_str = row_dict.get("requirements") or ""
    benefits_str = row_dict.get("benefits_detail") or row_dict.get("benefits") or ""
    tags_str = row_dict.get("tags")
    pub_date = str(row_dict.get("updated_display") or row_dict.get("published_at") or "").strip() or None
    deadline = str(row_dict.get("deadline") or "").strip() or None

    # Normalization & parsing
    min_exp, max_exp = parse_careerviet_experience(exp_str)
    level_norm, level_raw = parse_careerviet_level(level_str, min_exp)
    currency, min_m, max_m, is_neg, sal_display = parse_careerviet_salary(salary_raw)
    city, region, country, country_code = parse_careerviet_location(wp_detail, locations, title)

    full_text_for_skills = f"{title}\n{desc_str}\n{req_str}"
    extracted_skill_names = extract_skills(tags_str, full_text_for_skills, domain, title=title)
    skill_items = [SkillItem(name=s, proficiency="required") for s in extracted_skill_names]

    desc_bullets, req_bullets, summary = parse_job_sections(desc_str, req_str)

    # Clean combined raw text
    raw_desc_text = f"{desc_str.strip()}\n\n{req_str.strip()}".strip()

    # Remote policy check
    full_lower = f"{title} {desc_str} {emp_type}".lower()
    if "remote" in full_lower or "làm việc từ xa" in full_lower:
        remote_policy = "Remote"
    elif "hybrid" in full_lower or "linh hoạt" in full_lower:
        remote_policy = "Hybrid"
    else:
        remote_policy = "Onsite"

    # Employment type mapping
    if "chính thức" in emp_type.lower():
        norm_emp_type = "Full-time"
    elif "bán thời gian" in emp_type.lower() or "part-time" in emp_type.lower():
        norm_emp_type = "Part-time"
    elif "thực tập" in emp_type.lower() or "intern" in emp_type.lower():
        norm_emp_type = "Internship"
    elif "hợp đồng" in emp_type.lower() or "dự án" in emp_type.lower() or "contract" in emp_type.lower():
        norm_emp_type = "Contract"
    else:
        norm_emp_type = "Full-time"

    title_normalized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:50] or domain

    schema = JobDescriptionSchema(
        id=job_id,
        source="careerviet",
        url=job_url,
        scraped_at=datetime.now(timezone.utc).isoformat(),
        title=title,
        title_normalized=title_normalized,
        category=domain,
        company=Company(
            name=company_name,
            size=None,
            industry=industries,
        ),
        location=JobLocation(
            city=city,
            region=region,
            country=country,
            countryCode=country_code,
            remote_policy=remote_policy,
            remote_policy_raw=emp_type,
        ),
        seniority=Seniority(
            level=level_norm,
            level_raw=level_raw,
            min_years=min_exp,
            max_years=max_exp,
        ),
        skills=SkillsSection(
            required=skill_items,
            preferred=[],
            raw_text=", ".join(extracted_skill_names) if extracted_skill_names else None,
        ),
        description=DescriptionSection(
            summary=summary,
            responsibilities=desc_bullets,
            requirements=req_bullets,
            raw_text=raw_desc_text,
        ),
        compensation=Compensation(
            currency=currency,
            min_monthly=min_m,
            max_monthly=max_m,
            min_annual=min_m * 12 if min_m else None,
            max_annual=max_m * 12 if max_m else None,
            is_negotiable=is_neg,
            raw_text=sal_display,
        ),
        benefits=Benefits(
            raw_text=str(benefits_str).strip() if benefits_str else None,
        ),
        metadata=JobMetadata(
            posted_date=pub_date,
            deadline=deadline,
            job_type=norm_emp_type,
            work_arrangement=remote_policy,
            language=lang,
        ),
    )

    return schema


def generate_raw_text(job: JobDescriptionSchema) -> str:
    """Generate human-readable raw text file content."""
    lines = [
        f"Title: {job.title}",
        f"Company: {job.company.name}",
        f"Location: {job.location.city}, {job.location.country}",
        f"Category: {job.category}",
        f"Seniority: {job.seniority.level or 'Not specified'}",
        f"Compensation: {job.compensation.raw_text or 'Negotiable'}",
        "",
        "--- JOB DESCRIPTION & REQUIREMENTS ---",
        job.description.raw_text or "No detailed description provided.",
    ]
    if job.benefits and job.benefits.raw_text:
        lines.extend([
            "",
            "--- BENEFITS & PERKS ---",
            job.benefits.raw_text,
        ])
    return "\n".join(lines).strip() + "\n"


def import_all_careerviet_jobs(
    data_dir: Path = Path("data"),
    en_start_id: int = 47,
    vi_start_id: int = 1,
) -> Tuple[List[JobDescriptionSchema], List[JobDescriptionSchema]]:
    """
    Import and process all CareerViet Excel spreadsheets.
    Partitions into primary EN/Mixed dataset and secondary VI dataset.
    """
    raw_dir = data_dir / "raw" / "jobs"
    raw_vi_dir = data_dir / "raw" / "jobs_vi"
    proc_en_dir = data_dir / "processed" / "structured_jobs"
    proc_vi_dir = data_dir / "processed" / "structured_jobs_vi"

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_vi_dir.mkdir(parents=True, exist_ok=True)
    proc_en_dir.mkdir(parents=True, exist_ok=True)
    proc_vi_dir.mkdir(parents=True, exist_ok=True)

    en_current_id = en_start_id
    vi_current_id = vi_start_id

    imported_en: List[JobDescriptionSchema] = []
    imported_vi: List[JobDescriptionSchema] = []

    stats = {
        "total_read": 0,
        "en_mixed_count": 0,
        "vi_count": 0,
        "by_domain_en": {},
        "by_domain_vi": {},
    }

    for filename, domain in DOMAIN_MAPPING.items():
        fpath = raw_dir / filename
        if not fpath.exists():
            logger.warning(f"File not found: {fpath}")
            continue

        logger.info(f"Processing {filename} for domain [{domain}]...")
        wb = openpyxl.load_workbook(fpath, read_only=True)
        sheet = wb.active
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            continue

        header = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(rows[0])]

        for row_vals in rows[1:]:
            stats["total_read"] += 1
            row_dict = dict(zip(header, row_vals))

            title = str(row_dict.get("job_title") or "")
            desc = str(row_dict.get("description") or "")
            req = str(row_dict.get("requirements") or "")
            text_for_lang = f"{title}\n{desc}\n{req}"

            lang = detect_language(text_for_lang)

            if lang in ("en", "mixed"):
                job_id = f"jd_{en_current_id:03d}"
                schema = transform_careerviet_row(row_dict, job_id, domain, lang)
                
                # Test embedding text generation
                _ = schema.to_embedding_text()

                # Save JSON
                out_json = proc_en_dir / f"{job_id}.json"
                with open(out_json, "w", encoding="utf-8") as fp:
                    json.dump(schema.model_dump(by_alias=True), fp, ensure_ascii=False, indent=2)

                # Save raw TXT
                out_txt = raw_dir / f"{job_id}.txt"
                with open(out_txt, "w", encoding="utf-8") as fp:
                    fp.write(generate_raw_text(schema))

                imported_en.append(schema)
                en_current_id += 1
                stats["en_mixed_count"] += 1
                stats["by_domain_en"][domain] = stats["by_domain_en"].get(domain, 0) + 1

            else:  # Vietnamese ('vi')
                job_id = f"jd_vi_{vi_current_id:03d}"
                schema = transform_careerviet_row(row_dict, job_id, domain, "vi")

                # Test embedding text generation
                _ = schema.to_embedding_text()

                # Save JSON
                out_json = proc_vi_dir / f"{job_id}.json"
                with open(out_json, "w", encoding="utf-8") as fp:
                    json.dump(schema.model_dump(by_alias=True), fp, ensure_ascii=False, indent=2)

                # Save raw TXT
                out_txt = raw_vi_dir / f"{job_id}.txt"
                with open(out_txt, "w", encoding="utf-8") as fp:
                    fp.write(generate_raw_text(schema))

                imported_vi.append(schema)
                vi_current_id += 1
                stats["vi_count"] += 1
                stats["by_domain_vi"][domain] = stats["by_domain_vi"].get(domain, 0) + 1

    # Generate consolidated index files
    logger.info("Generating consolidated index files...")

    # Load existing ITviec jobs to merge into all_jobs_en.json
    all_en_schemas = []
    for f in sorted(proc_en_dir.glob("jd_*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            all_en_schemas.append(data)

    with open(proc_en_dir / "all_jobs_en.json", "w", encoding="utf-8") as fp:
        json.dump(all_en_schemas, fp, ensure_ascii=False, indent=2)

    all_vi_schemas = [s.model_dump(by_alias=True) for s in imported_vi]
    with open(proc_vi_dir / "jobs_vi_all.json", "w", encoding="utf-8") as fp:
        json.dump(all_vi_schemas, fp, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY:")
    logger.info(f"Total CareerViet rows processed: {stats['total_read']}")
    logger.info(f"Primary Corpus (EN + Mixed) imported: {stats['en_mixed_count']} (IDs jd_{en_start_id:03d} to jd_{en_current_id-1:03d})")
    logger.info(f"Total Primary Corpus now: {len(all_en_schemas)} JDs in data/processed/structured_jobs/")
    logger.info(f"Secondary Corpus (VI) imported: {stats['vi_count']} (IDs jd_vi_{vi_start_id:03d} to jd_vi_{vi_current_id-1:03d})")
    logger.info("Primary Corpus by Domain (CareerViet only):")
    for d, c in stats["by_domain_en"].items():
        logger.info(f"  - {d}: {c}")
    logger.info("Secondary Corpus by Domain (Vietnamese):")
    for d, c in stats["by_domain_vi"].items():
        logger.info(f"  - {d}: {c}")
    logger.info("=" * 60)

    return imported_en, imported_vi


if __name__ == "__main__":
    import_all_careerviet_jobs()
