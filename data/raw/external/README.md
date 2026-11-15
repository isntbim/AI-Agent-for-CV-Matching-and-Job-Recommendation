# External raw datasets (downloaded 2026-10-25, git-ignored)

Untrusted downloads: read them only with `python -I`, never run code from these folders.

| Folder | Source | Files | Licence on Hugging Face |
|---|---|---|---|
| `saba06_resume_classification/` | https://huggingface.co/datasets/Saba06huggingface/Resume_classification_dataset | `resume_dataset_huggingface.csv` (2,484 rows: text, label; 24 job categories) | none declared |
| `med2425_resume_job_fit_merged_v1/` | https://huggingface.co/datasets/med2425/resume-job-fit-merged-v1 | `data/train-*.parquet` (80,017 pairs), `data/test-*.parquet` (13,716 pairs): resume, jd, label, source, resume_domain, jd_domain | none declared |

No licence is declared for either dataset: the paper must state this, and the licence of the upstream sources
(LiveCareer resumes for Saba; cnamuangtoun/resume-job-description-fit for med2425) must be checked before use.

Audit (2026-10-25, normalised-text matching):
- Saba: 2,481 unique CVs; 212 are already in the local CV pool or gold set (same LiveCareer source) -> about 2,269 new CVs.
- med2425 is the extended version of cnamuangtoun/resume-job-description-fit; user decision 2026-10-25: new CV/JD
  training data comes from med2425 only (cnamuangtoun is not used separately).
- med2425: 93,733 pairs but only 638 unique CVs and 350 unique JDs (labels are synthetic: source `generated_smart` /
  `synthetic_test`). 606 CVs and 341 JDs are the cnamuangtoun CVs/JDs that the scorer benchmark uses
  (output/ats/prepared/pairs.json); only 32 CVs and 9 JDs are new.
- LEAKAGE RULE: 141 CVs and 86 JDs sit on the scorer's held-out TEST side. They must never enter Qwen training.
  Usable for Qwen training: 465 + 32 = 497 CVs, 255 + 9 = 264 JDs.
None of these are labelled yet; nothing from here is in data/datasets/qwen_training/manifest.json.

| `vanetik_vacancy_resume/` | https://github.com/NataliaVanetik/vacancy-resume-matching-dataset (Vanetik & Kogan 2023, Information 14(8):468, doi 10.3390/info14080468) | `5_vacancies.csv` (5 real vacancies), `CV/1-30.docx` (anonymised real resumes), `annotations-for-the-first-30-vacancies.txt` (2 human annotators rank the 5 vacancies per CV, senior-HR instructions) | GPL-3.0 |

Downloaded 2026-11-04: the only REAL human-label resume–vacancy set found (Perplexity search); only CVs 1-30 are ranked.
Annotator 1 rows 9 and 28 repeat a vacancy id (not a permutation): excluded for annotator 1.
