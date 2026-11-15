---
name: o5-explanation-vi
description: Use whenever writing Vietnamese O5 match explanations as a teacher (writing output/o5/teacher_vi/<pair_id>.json from output/o5/inputs/<pair_id>.json). Gives the Vietnamese verdict labels, length, wording and naming rules on top of the English o5-explanation skill, and the self-check command to run after every file.
---

# O5 explanation teacher, Vietnamese

Read `.agents/skills/o5-explanation/SKILL.md` first. **Everything in it still applies** (input format, what a good
explanation does, the judgement rules, the hard rules, the strict self-review), except the changes below.
Do NOT spawn subagents; do one pair at a time. Repo root: `D:/SS9 - FinalSeason`.

## Output: `output/o5/teacher_vi/<pair_id>.json`
```json
{"pair_id": "<pair_id>", "explanation": "<your Vietnamese text>"}
```
Write the file as UTF-8. Never write into `output/o5/teacher/` (that folder is the English set).

## What changes in Vietnamese
1. **Language.** Natural, plain Vietnamese as a Vietnamese recruiter would write it. Address the candidate as "bạn".
   Not a word-for-word translation of an English explanation.
2. **Verdict label, exactly, at the very start**, then the fit in brackets and a colon:
   - Good Fit → `Phù hợp tốt (độ phù hợp 86/100): ...`
   - Potential Fit → `Có tiềm năng (độ phù hợp 47/100): ...`
   - No Fit → `Chưa phù hợp (độ phù hợp 3/100): ...`
   Never write the English words "Good Fit", "Potential Fit" or "No Fit" anywhere.
3. **Length: 3 to 5 sentences, 100 to 160 tiếng** (space-separated syllables; about the same content as 90-120
   English words).
4. **Keep names exactly as given, in English:** skill names, requirement names, job titles, CV role titles, degree
   names and dates ("2013-06 to Present" may become "từ 2013-06 đến nay", but keep the date itself as given).
   Do not translate or respell them: write "Power BI", "GAAP", "Senior Accountant", "Bachelor of Science Accounting".
   Wrap them naturally: "kỹ năng GAAP", "vị trí Senior Accountant", "bằng Bachelor of Science Accounting".
5. **Unscored item:** when you mention one (at most one, as in the English skill), use the phrase
   "không thể đối chiếu", e.g. "kinh nghiệm của bạn không thể đối chiếu với yêu cầu 5 năm".
   Apart from that one sentence, never use an unscored item as evidence or as a gap (see the English skill): no degree
   ("bằng", "học vấn") when education is unscored, no "gần với / sát với / phù hợp với vị trí" about the CV's roles when
   title is unscored, no "lĩnh vực / ngành" when industry is unscored, no "kinh nghiệm phù hợp / thiếu / lâu năm" when
   experience is unscored.
6. **No system words:** never "điểm quy tắc", "chiều đánh giá", "mô hình", "thuật toán", "hệ thống chấm",
   "dữ liệu đầu vào". **No overclaims or harsh words:** never "dễ dàng", "hoàn hảo", "chắc chắn", "đảm bảo",
   "ứng viên lý tưởng", "ứng viên mạnh", "yếu kém", "không đủ năng lực". Never mention "địa điểm" or "lương".
7. **Say met when it is met.** If experience or education scores 100, write "đáp ứng" (meets), never "gần với"
   (close to) or "gần đạt". Only use "gần" or "một phần" for a requirement whose status is "partly met".
8. **One next step only** (one sentence of advice), as in the English skill.
9. **Different work (No Fit):** say the candidate's work is a different kind of work ("công việc khác hẳn", "nghề khác"), not a different "ngành" or "lĩnh vực" when `dimensions.industry` is scored 100: the app shows that Industry score next to the text.
10. **Numbers** follow the English hard rule 2 exactly. Write them as digits ("5 năm", "3 kỹ năng"), never as words.

## Example (Good Fit, fit 86; facts: Senior Electrical Engineer, 6 years required, met: project management;
## not met: industrial electrical design, national electric code, autocad)
Phù hợp tốt (độ phù hợp 86/100): bạn có nhiều năm làm Electrical Engineer, đúng lĩnh vực và đủ thâm niên cho vị trí
Senior Electrical Engineer này, dù CV còn thiếu vài kỹ năng thiết kế cốt lõi. Vai trò Customer Advocate/Electrical
Engineer từ 1999-11 đến 2012-01 cùng bằng Bachelor of Science Electrical Engineering đáp ứng yêu cầu 6 năm kinh
nghiệm, và bạn có Project Management. Tuy nhiên, CV chưa thể hiện Industrial Electrical Design, National Electric Code
và AutoCAD. Nếu bạn từng làm với AutoCAD hoặc National Electric Code, hãy ghi rõ vào CV kèm một dự án cụ thể trước
khi ứng tuyển.

**Do not copy this example's sentences.** It shows the shape only. In particular, vary the last sentence: fit it to
this pair (which gap to show first, whether to apply now, which kind of role suits the CV better, what to learn first
for a career change). Never end two explanations in a row with the same wording, e.g. not every one with
"hãy ghi rõ vào CV kèm một dự án cụ thể trước khi ứng tuyển". Vary the opening clause after the label too.

## Self-check after every file
```
.venv/Scripts/python.exe -m scripts.o5_faithfulness check <pair_id> --dir teacher_vi
```
It runs the faithfulness check (verdict label at the start, numbers, skill names) and the Vietnamese style rules
(100 to 160 tiếng, system words, overclaims, location/salary, generic gaps, more than one unscored item).
Fix your file until it prints `faithful 1/1`.
