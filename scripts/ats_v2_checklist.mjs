/** Create the ATS v2 reviewer checklist using the supported artifact-tool API. */
import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.resolve(process.argv[2] || process.cwd());
const packetPath = path.join(root, "human-verify/scorer-v2/review_package.json");
const outputPath = path.join(root, "human-verify/scorer-v2/checklist.xlsx");
const qaDirectory = path.join(root, "output/scorer_v2/checklist");
const itemDirectory = path.join(root, "human-verify/scorer-v2/items");
const sourceDirectory = path.join(root, "human-verify/scorer-v2/sources");
const originalDirectory = path.join(root, "human-verify/scorer-v2/originals");
const workbook = Workbook.create();

if (process.argv.includes("--help-hyperlink")) {
  console.log(workbook.help("fx.HYPERLINK", { include: "index,examples,notes", maxChars: 3000 }).ndjson);
  process.exit(0);
}

// The workbook must not erase decisions from an already-delivered checklist.
if (!process.argv.includes("--rebuild-fresh")) {
  try {
    await fs.access(outputPath);
    throw new Error("checklist.xlsx already exists. Use --rebuild-fresh only before human editing.");
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

const packet = JSON.parse(await fs.readFile(packetPath, "utf8"));
const items = packet.items;
if (!Array.isArray(items) || !packet.corpus_identity_sha256) throw new Error("Invalid review package.");
await fs.mkdir(qaDirectory, { recursive: true });
await fs.mkdir(itemDirectory, { recursive: true });
await fs.mkdir(sourceDirectory, { recursive: true });
await fs.mkdir(originalDirectory, { recursive: true });

// Relative links and byte-for-byte source copies make the review folder portable.
const sourceLinks = new Map();
const copiedOriginals = new Set();
for (const item of items) {
  if (!item.document_id || sourceLinks.has(item.document_id) || !item.source_path || !item.text_path) continue;
  const id = item.document_id;
  if (!/^[A-Za-z0-9_.-]+$/.test(id)) throw new Error(`Unsafe source ID: ${id}`);
  const textTarget = `sources/${id}.txt`;
  await fs.copyFile(path.resolve(root, item.text_path), path.join(sourceDirectory, `${id}.txt`));
  const textFileSha = createHash("sha256").update(await fs.readFile(path.join(sourceDirectory, `${id}.txt`))).digest("hex");
  const originalName = item.source_sha256.slice(0, 24) + path.extname(item.source_path);
  if (!copiedOriginals.has(originalName)) {
    await fs.copyFile(path.resolve(root, item.source_path), path.join(originalDirectory, originalName));
    copiedOriginals.add(originalName);
  }
  sourceLinks.set(id, { source: textTarget, original: `originals/${originalName}`,
    original_name: path.basename(item.source_path), source_path: item.source_path,
    text_path: item.text_path, source_sha256: item.source_sha256, text_sha256: item.text_sha256,
    text_file_sha256: textFileSha, text_checksum_normalization: "universal-newlines" });
}
await fs.writeFile(path.join(root, "human-verify/scorer-v2/source_links.json"), JSON.stringify({
  corpus_identity_sha256: packet.corpus_identity_sha256, documents: Object.fromEntries(sourceLinks)
}, null, 2) + "\n");

const catalog = [
  ["Documents", "documents", "Document annotations"],
  ["CV proficiency", "cv_proficiency", "CV skills and proficiency"],
  ["JD requirements", "jd_requirements", "JD requirements"],
  ["Choices", "choices", "JD choices"],
  ["Relations", "relations", "Skill relations"],
  ["Ambiguities", "ambiguities", "Unresolved evidence"],
];

const headers = ["item_id", "document_id", "decision", "reviewer", "reason", "replacement", "source", "quote", "previous_value", "proposed_value", "proposal_reason", "split", "source_sha256", "text_sha256", "document_checksum", "proposal_sha256", "item_sha256", "full_proposal", "original_source"];
const widths = [26, 20, 12, 17, 36, 56, 24, 70, 44, 60, 48, 15, 28, 28, 28, 28, 28, 24, 24];

function column(index) {
  let name = "";
  for (let value = index + 1; value; value = Math.floor((value - 1) / 26)) name = String.fromCharCode(65 + (value - 1) % 26) + name;
  return name;
}

function literal(value) {
  const raw = value == null ? "" : typeof value === "string" ? value : JSON.stringify(value);
  // XML cannot store PDF layout control characters. Full evidence stays intact
  // in the linked JSON proposal; only the spreadsheet display uses spaces.
  const text = raw.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\uFFFE\uFFFF]/g, " ");
  return text.startsWith("=") ? "'" + text : text;
}

function preview(value, limit = 1100) {
  const text = literal(value);
  if (text.length <= limit) return text;
  return text.slice(0, limit) + "\n[Excerpt; open full_proposal for the complete value.]";
}

function proposedPreview(proposal) {
  if (proposal?.document && proposal?.facts) {
    const doc = proposal.document;
    const skills = Array.isArray(doc.skills)
        ? doc.skills.flatMap(group => group.details?.map(detail => detail.name) ?? group.keywords ?? [])
        : doc.skills?.requirements?.map(unit => ({ options: unit.options, importance: unit.importance }));
    return JSON.stringify({ schema_version: doc.schema_version,
      title: doc.title ?? doc.basics?.label, skill_count: skills?.length ?? 0,
      first_skills: skills?.slice(0, 8), skills_status: proposal.annotation?.skills_status,
      skills_review: proposal.annotation?.review_status,
      full_envelope_review: proposal.annotation?.envelope_review_status }, null, 2);
  }
  return literal(proposal);
}

function quoteFor(proposal) {
  if (!proposal || typeof proposal !== "object") return "";
  const quotes = [proposal.skill_quote, proposal.proficiency_quote, proposal.quote, proposal.importance_quote, proposal.source_quote, proposal.evidence_quote];
  return [...new Set(quotes.filter(value => typeof value === "string" && value.trim()))].join("\n\n");
}

function display(item, readOnly) {
  const proposal = item.proposal;
  const before = item.previous_value ?? item.previous ?? item.before ?? proposal?.previous_value ?? proposal?.before;
  const defaultReasons = {
    cv_proficiency: proposal?.proficiency === "normal" ? "Unspecified per-skill proficiency defaults to normal." : "Explicit source proficiency retained.",
    jd_requirements: proposal?.importance === "core" ? "Source identifies a mandatory skill requirement." : proposal?.importance === "preferred" ? "Source identifies a preferred skill requirement." : "Required skills default to Standard without Core evidence.",
    documents: "Document annotation remains pending human review.",
    ambiguities: proposal?.detail || "Source interpretation requires a reviewer decision.",
  };
  const reasoning = item.proposal_reason ?? proposal?.reason ?? item.reason ?? (proposal?.issues?.length ? proposal.issues : defaultReasons[item.table]);
  const source = item.source_path || item.text_path || proposal?.source || "";
  return [
    literal(item.item_id), literal(item.document_id), readOnly ? "See JD requirements" : "pending", "", "", "",
    literal(source), preview(quoteFor(proposal), 1400), before === undefined ? "Not captured in review package" : preview(before), preview(proposedPreview(proposal)), preview(reasoning), literal(item.split),
    literal(item.source_sha256), literal(item.text_sha256), literal(item.document_checksum), literal(item.proposal_sha256), literal(item.item_sha256),
    "Open full proposal",
    sourceLinks.has(item.document_id) ? "Open original source" : "",
  ];
}

const qa = { version: packet.version, corpus_identity_sha256: packet.corpus_identity_sha256, input_rows: items.length, sheets: [] };
const seen = new Set();
for (const [name, kind, title] of catalog) {
  const rows = items.filter(item => kind === "choices" ? item.table === "jd_requirements" && (item.proposal?.options?.length || 0) > 1 : item.table === kind)
    .sort((left, right) => String(left.document_id).localeCompare(String(right.document_id)) || String(left.item_id).localeCompare(String(right.item_id)));
  const readOnly = kind === "choices";
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = { name: "Arial", size: 16, bold: true, color: "#172033" };
  sheet.getRange("A2").values = [["corpus_identity_sha256"]];
  sheet.getRange("B2").values = [[packet.corpus_identity_sha256]];
  sheet.getRange("A3").values = [[readOnly ? "Decide these items on JD requirements. Each choice has one requirement and one decision." : "Set decision and reviewer. For edit, provide full replacement JSON. Blank decisions remain pending."]];
  sheet.getRange("A2:S3").format.font = { name: "Arial", size: 10, color: "#526075" };
  sheet.getRange("A1:S1").format.borders = { bottom: { style: "thin", color: "#CBD3DF" } };
  sheet.getRange("A5:S5").values = [headers];
  sheet.getRange("A5:S5").format = { fill: "#24344B", font: { name: "Arial", size: 11, color: "#FFFFFF", bold: true }, wrapText: true, verticalAlignment: "center", horizontalAlignment: "center", rowHeightPx: 44 };
  const count = rows.length;
  if (count) {
    const matrix = rows.map(item => display(item, readOnly));
    sheet.getRange(`A6:S${count + 5}`).values = matrix;
    sheet.getRange(`A6:S${count + 5}`).format = { font: { name: "Arial", size: 11, color: "#172033" }, wrapText: true, verticalAlignment: "top", rowHeightPx: 88 };
    if (!readOnly) {
      sheet.getRange(`C6:F${count + 5}`).format.fill = "#FFF2CC";
      sheet.getRange(`C6:C${count + 5}`).dataValidation = { rule: { type: "list", values: ["accept", "edit", "reject", "pending"] } };
      sheet.getRange(`C6:C${count + 5}`).conditionalFormats.add("containsText", { text: "accept", format: { fill: "#E2F0D9", font: { color: "#285A21" } } });
      sheet.getRange(`C6:C${count + 5}`).conditionalFormats.add("containsText", { text: "reject", format: { fill: "#FBE2E2", font: { color: "#8A2929" } } });
    }
    const table = sheet.tables.add(`A5:S${count + 5}`, true, `ATS_${kind}`);
    table.showFilterButton = true;
    for (let index = 0; index < matrix.length; index++) {
      const lineCounts = matrix[index].map((value, col) => String(value).split("\n").reduce((sum, line) => sum + Math.max(1, Math.ceil(line.length / Math.max(10, widths[col] - 3))), 0));
      const height = Math.max(88, ...lineCounts.map(lines => lines * 16 + 16));
      sheet.getRange(`A${index + 6}:S${index + 6}`).format.rowHeightPx = Math.min(546, height);
      const detailFilename = rows[index].item_sha256.slice(0, 32) + ".json";
      if (!readOnly) await fs.writeFile(path.join(itemDirectory, detailFilename), JSON.stringify({
        corpus_identity_sha256: packet.corpus_identity_sha256, ...rows[index] }, null, 2) + "\n");
      sheet.getRange(`R${index + 6}`).formulas = [[`=IFERROR(HYPERLINK("items/${detailFilename}","Open full proposal"),"Open full proposal")`]];
      sheet.getRange(`R${index + 6}`).format.font = { name: "Arial", size: 11, color: "#215E9B" };
      const portable = sourceLinks.get(rows[index].document_id);
      const sourcePath = portable?.source || rows[index].source_path || rows[index].text_path;
      if (sourcePath) {
        const url = portable?.source || pathToFileURL(path.resolve(root, sourcePath)).href;
        const label = path.basename(sourcePath);
        const linkLabel = label.replaceAll('"', '""');
        sheet.getRange(`G${index + 6}`).formulas = [[`=IFERROR(HYPERLINK("${url.replaceAll('"', '""')}","${linkLabel}"),"${linkLabel}")`]];
        sheet.getRange(`G${index + 6}`).format.font = { name: "Arial", size: 11, color: "#215E9B" };
      }
      if (portable) sheet.getRange(`S${index + 6}`).formulas = [[`=IFERROR(HYPERLINK("${portable.original}","Open original source"),"Open original source")`]];
      if (!readOnly) {
        if (seen.has(rows[index].item_id)) throw new Error(`Duplicate review item: ${rows[index].item_id}`);
        seen.add(rows[index].item_id);
      }
    }
  } else {
    sheet.getRange("A6").values = [["No items in this category."]];
    sheet.getRange("A6").format.font = { name: "Arial", size: 11, color: "#526075" };
  }
  for (let index = 0; index < widths.length; index++) sheet.getRange(`${column(index)}5:${column(index)}${Math.max(6, count + 5)}`).format.columnWidth = widths[index];
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(2);
  const inspected = await workbook.inspect({ kind: "table", range: `'${name}'!A5:K${Math.min(count + 5, 8)}`, include: "values,formulas", tableMaxRows: 4, tableMaxCols: 11, maxChars: 3500 });
  await fs.writeFile(path.join(qaDirectory, `${kind}-inspect.ndjson`), inspected.ndjson);
  const preview = await workbook.render({ sheetName: name, range: `A1:H${Math.max(6, Math.min(8, count + 5))}`, scale: 1, format: "png" });
  await fs.writeFile(path.join(qaDirectory, `${kind}.png`), new Uint8Array(await preview.arrayBuffer()));
  const evidencePreview = await workbook.render({ sheetName: name, range: `G5:K${Math.max(6, Math.min(8, count + 5))}`, scale: 1, format: "png" });
  await fs.writeFile(path.join(qaDirectory, `${kind}-evidence.png`), new Uint8Array(await evidencePreview.arrayBuffer()));
  const identityPreview = await workbook.render({ sheetName: name, range: `L5:Q${Math.max(6, Math.min(7, count + 5))}`, scale: 1, format: "png" });
  await fs.writeFile(path.join(qaDirectory, `${kind}-identity.png`), new Uint8Array(await identityPreview.arrayBuffer()));
  qa.sheets.push({ name, kind, rows: count, read_only: readOnly, preview: `${kind}.png` });
}
if (seen.size !== items.length) throw new Error(`Checklist covers ${seen.size} of ${items.length} review items.`);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, maxChars: 3000, summary: "Formula errors" });
await fs.writeFile(path.join(qaDirectory, "formula-errors.ndjson"), errors.ndjson);
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(outputPath);
await fs.writeFile(path.join(qaDirectory, "manifest.json"), JSON.stringify(qa, null, 2) + "\n");
console.log(JSON.stringify({ workbook: outputPath, review_items: seen.size, sheets: qa.sheets }, null, 2));
