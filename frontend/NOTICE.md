# Third-party notice

This frontend is adapted from **ResuMatch** by Gaurav Singh: https://github.com/gaurav-singh7092/ResuMatch
(commit dc0a17f, cloned 2026-11-04). Its README states it is licensed under the MIT License; the repository has no
LICENSE file of its own, so the original copyright and that statement are kept here.

Changes for this capstone: the original Python backend is not used. The pages, API client and result views were
rewritten to call this project's FastAPI service (`src/api/app.py`), which ranks CareerViet jobs for one uploaded CV
and explains each match with the V3.1 rule scorer.
