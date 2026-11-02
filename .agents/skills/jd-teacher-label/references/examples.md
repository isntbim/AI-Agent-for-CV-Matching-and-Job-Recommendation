# JD teacher label examples (synthetic, job-v3)

Both texts are invented. They show the decisions, not a template to copy word for word.

## 1. English (ITviec style)

Source text:
```
Title: Senior Data Engineer
Company: Example Tech

Requirements
- 4+ years of experience in data engineering
- Must have strong Python and SQL
- Experience with Airflow or Dagster
- Bachelor's degree in Computer Science or related field
Nice to have
- Kafka
- Advanced Spark
```

Decisions:
- Title "Senior Data Engineer" → `title_normalized: data_engineer`, `category: data_scientist_analyst`,
  `seniority.level: Senior`, `level_raw: Senior`, `min_years: 4` (quote "4+ years of experience in data engineering").
- "Must have strong Python and SQL" → two units (AND), both `core`, and both with
  `quote` = `importance_quote` = "Must have strong Python and SQL" (the importance quote must name the skill).
  "strong" is not a mapped level: `minimum_proficiency: normal`,
  `raw_proficiency: "strong"`, `proficiency_quote: "strong Python"` on Python only (it sits next to Python; SQL
  gets no raw level).
- "Airflow or Dagster" → ONE unit `options: ["Airflow", "Dagster"]`, `standard`.
- Degree → `facts.minimum_degree: bachelor`, evidence.education "Bachelor's degree in Computer Science". Not a skill.
- Under "Nice to have": "Kafka" → `preferred`, `quote: "Kafka"`, `importance_quote: "Nice to have\n- Kafka"`
  (the span from the heading to the skill, with the line break and dash as written; in the JSON file that is one
  string containing `\n`). "Advanced Spark" → `preferred`, `importance_quote: "Nice to have\n- Kafka\n- Advanced Spark"`,
  `minimum_proficiency: advanced`, `raw_proficiency: "Advanced"`, `proficiency_quote: "Advanced Spark"`.
  "Proficient in Power BI" would map to `intermediate` (raw "Proficient"); "strong", "good", "thành thạo" stay normal.
- No location in the text → `location.city: null`, `facts.country: null`. No salary → numbers null, `is_negotiable: true`.
- `facts.domains: ["data"]`, evidence.industry "Senior Data Engineer".
- Legacy: `required` = Python, SQL, Airflow, Dagster; `preferred` = Kafka, Spark.

```json
{"id": "req_02", "options": ["SQL"], "importance": "core", "quote": "Must have strong Python and SQL",
 "importance_quote": "Must have strong Python and SQL", "review_status": "pending", "issues": [],
 "option_details": [{"name": "SQL", "minimum_proficiency": "normal", "raw_proficiency": null, "proficiency_quote": null, "experience": []}]}
```

## 2. Vietnamese (CareerViet field layout)

Source text:
```
job_title:
Kế toán tổng hợp

company_name:
Công ty ABC

salary:
15 Tr - 20 Tr VND

level:
Nhân viên

experience:
2 - 5 Năm

education:
Cao đẳng

workplace_detail:
Hồ Chí Minh Quận 1

requirements:
Tốt nghiệp Cao đẳng/Đại học chuyên ngành Kế toán
Thành thạo Excel, phần mềm MISA
Bắt buộc có chứng chỉ kế toán trưởng
Ưu tiên ứng viên có tiếng Anh giao tiếp
Năng động, chăm chỉ
```

Decisions:
- `title: "Kế toán tổng hợp"`, `title_normalized: accountant`, `category: accountant`, `facts.domains: ["finance"]`
  (evidence.industry "Kế toán tổng hợp"). "Kế toán tổng hợp" is the job title, never a requirement.
- `level_raw: "Nhân viên"`, `level: null`. Years 2/5, evidence.experience "2 - 5 Năm".
- `education: Cao đẳng` and "Cao đẳng/Đại học" → `minimum_degree: associate`, evidence.education "Cao đẳng".
- Salary → `min_monthly: 15000000`, `max_monthly: 20000000`, currency "VND" (document and facts),
  `is_negotiable: false`, evidence.salary "15 Tr - 20 Tr VND".
- "Hồ Chí Minh" → `city: "Ho Chi Minh City"`, evidence.location "Hồ Chí Minh". No country is written, so
  `facts.country: null` (a city alone does not state a country).
- "Thành thạo Excel, phần mềm MISA" → two units `Excel` and `MISA`, standard; "Thành thạo" is not a mapped level:
  Excel keeps `minimum_proficiency: normal`, `raw_proficiency: "Thành thạo"`, `proficiency_quote: "Thành thạo Excel"`.
- "Bắt buộc có chứng chỉ kế toán trưởng" → one unit `chứng chỉ kế toán trưởng`, `core`,
  `importance_quote: "Bắt buộc có chứng chỉ kế toán trưởng"`. The name stays Vietnamese, exactly as in the quote.
- "Ưu tiên ứng viên có tiếng Anh giao tiếp" → `tiếng Anh`, `preferred`, `quote` = `importance_quote` =
  "Ưu tiên ứng viên có tiếng Anh giao tiếp".
- "Năng động, chăm chỉ" → personality, no unit. "Kế toán" as the major is part of the degree line, not a unit.
- `metadata.language: vi`, `job_type: Full-time`.

```json
{"id": "req_04", "options": ["tiếng Anh"], "importance": "preferred", "quote": "Ưu tiên ứng viên có tiếng Anh giao tiếp",
 "importance_quote": "Ưu tiên ứng viên có tiếng Anh giao tiếp", "review_status": "pending", "issues": [],
 "option_details": [{"name": "tiếng Anh", "minimum_proficiency": "normal", "raw_proficiency": null, "proficiency_quote": null, "experience": []}]}
```

## 3. Vietnamese harvest drill (CareerViet; judge rounds 3-4, 2026-10-26)

The commonest batch-3 error is a missed clause. Source `requirements:` / `other_info:` cells:
```
requirements:
Có kinh nghiệm triển khai ERP, CRM hoặc HRM
Am hiểu tích hợp hệ thống (REST API, SOAP)
Có khả năng xác thực và phân quyền người dùng
Làm việc độc lập và làm việc nhóm tốt
Biết sử dụng Excel, PowerPoint và các công cụ mạng xã hội
Phối hợp với nhà cung cấp và người dùng cuối
other_info:
Có tinh thần học hỏi công nghệ mới
```
Tick every clause; the label has 11 units:
| Clause | Units | Why |
|---|---|---|
| ERP, CRM hoặc HRM | 1 unit `["ERP", "CRM", "HRM"]` | "hoặc": all three alternatives, none dropped |
| tích hợp hệ thống (REST API, SOAP) | 1 unit `["REST API", "SOAP"]` | examples of one capability; no extra `tích hợp hệ thống` unit, no category option |
| xác thực và phân quyền | 2 units `xác thực`, `phân quyền` | "và" joins two different capabilities, both required |
| độc lập và làm việc nhóm | 2 units `làm việc độc lập`, `làm việc nhóm` | both capabilities are units; each name is a literal substring of the quote |
| Excel, PowerPoint và công cụ mạng xã hội | 3 units | the third item is easy to drop; keep it |
| nhà cung cấp và người dùng cuối | 1 unit `phối hợp với nhà cung cấp và người dùng cuối` | one coordination capability, literal name |
| học hỏi công nghệ mới (`other_info:`) | 1 unit | `other_info:` lines are requirements too |

Not units: "đã làm việc tại sàn TMĐT" (employer background, not the skill of operating a platform), "cẩn thận",
"sáng tạo" (traits). "Ít nhất hai trong số FI, CO, MM, SD" → one unit with all modules as options and the issue
`"source asks for at least two of these; schema has no k-of-n"`.
Also split compound competencies ("quản lý và đào tạo nhân sự" → `quản lý nhân sự`, `đào tạo nhân sự` only when both
names occur literally; otherwise keep the literal phrase). Overall years ("5-7 năm kinh nghiệm Marketing") are
`min_years`, not a shorter managerial subset ("trong đó 3 năm quản lý" stays skill-specific).
After writing, re-read the cells: every requirement clause has a unit or one of the reasons above for none.
