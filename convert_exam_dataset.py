from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# =========================
# Helpers
# =========================
KANJI_DIGITS = {"〇":0,"零":0,"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
CIRCLED_MAP = {"①":"1","②":"2","③":"3","④":"4","⑤":"5","⑥":"6","⑦":"7","⑧":"8","⑨":"9","⑩":"10",
               "➀":"1","➁":"2","➂":"3","➃":"4","➄":"5","➅":"6","➆":"7","➇":"8","➈":"9","➉":"10"}


def kanji_to_int_small(s: str) -> int | None:
    """Convert simple Japanese numerals (roughly up to 20)."""
    s = "".join(ch for ch in s if ch in KANJI_DIGITS or ch == "十")
    if not s:
        return None
    if s == "十":
        return 10
    if s.startswith("十"):
        tail = KANJI_DIGITS.get(s[1:], 0)
        return 10 + tail
    if s.endswith("十"):
        head = KANJI_DIGITS.get(s[0], 0)
        return head * 10
    return KANJI_DIGITS.get(s, None)


def to_halfwidth_digits(s: str) -> str:
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def normalize_choice_label_variants(label: str) -> list[str]:
    variants = [label, to_halfwidth_digits(label)]
    if label in CIRCLED_MAP:
        variants.append(CIRCLED_MAP[label])
    return list(dict.fromkeys(variants))


BRACKETS_L = r"[【\[\(（〔［〈<]?"
BRACKETS_R = r"[】\]\)）〕］〉>]?"
core1 = r"(?:第?\s*[一二三四五六七八九十〇零]+|第?\s*[0-9０-９]+(?:\s*問)?)"
SECTION_PATTERNS = [
    re.compile(r"^#\s*" + BRACKETS_L + "(" + core1 + r")" + BRACKETS_R + r"\s*$", re.MULTILINE),
    re.compile(r"^#\s*第\s*([0-9０-９一二三四五六七八九十〇零]+)\s*問\s*$", re.MULTILINE),
    re.compile(r"^#\s*([0-9０-９一二三四五六七八九十〇零]+)\s*$", re.MULTILINE),
]
SUB_PATTERNS = [
    re.compile(r"^##\s*[（(]?\s*([一二三四五六七八九十〇零0-9０-９①②③④⑤⑥⑦⑧⑨⑩➀➁➂➃➄➅➆➇➈➉]+)\s*[)）]?\s*$", re.MULTILINE),
]
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
COMMENT_RE = re.compile(r"<!---?[\s\S]*?--->?")
ANSWER_TYPE_TRAIL_RE = re.compile(r"[ 　]*[（(]\s*解答類型\s*[０-９0-9]+[）)]\s*$")
CHOICE_HEAD = r"(?:[ア-ンＡ-ＺA-Z0-9０-９①②③④⑤⑥⑦⑧⑨⑩➀➁➂➃➄➅➆➇➈➉])"
CHOICE_LINE_RE = re.compile(r"^[\s　]*(" + CHOICE_HEAD + r")[\.\)）]?\s+(.+)$")
QN_RE = re.compile(
    r"^\s*([0-9０-９]+)"
    r"(?:\s*\(\s*([一二三四五六七八九十〇零0-9０-９]+)\s*\))?"
    r"(?:\s*([①②③④⑤⑥⑦⑧⑨⑩➀➁➂➃➄➅➆➇➈➉]))?"
    r"\s*$"
)


def strip_comments(text: str) -> str:
    return COMMENT_RE.sub("", text)


def extract_images(text: str) -> list[str]:
    return IMG_RE.findall(text)


def extract_choices_labeled(text: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        match = CHOICE_LINE_RE.match(raw)
        if not match:
            continue
        label = match.group(1).strip()
        content = match.group(2).strip()
        if content.startswith("図") or content.startswith("#"):
            continue
        if label not in seen:
            items.append((label, content))
            seen.add(label)
    return items


def normalize_fw_letters(s: str) -> str:
    fw = "ＡＢＣＤＸＹＺ"
    hw = "ABCDXYZ"
    return s.translate({ord(f): h for f, h in zip(fw, hw)})


def parse_gold(answer_type: str) -> dict[str, str]:
    s = (answer_type or "").strip()
    pairs = re.findall(r"([A-Za-zＡ-ＺＸＹＺX-Z])\s*[：:]\s*([ア-ンＡ-ＺA-Z0-9０-９])", s)
    if pairs:
        mapped = {normalize_fw_letters(k): to_halfwidth_digits(v) for k, v in pairs}
        return {"type": "labeled_tuple", "label": ", ".join([f"{k}:{v}" for k, v in mapped.items()])}
    labels = re.findall(r"[ア-ンＡ-ＺA-Z0-9０-９①②③④⑤⑥⑦⑧⑨⑩➀➁➂➃➄➅➆➇➈➉]", s)
    if len(labels) == 1:
        return {"type": "single_choice", "label": to_halfwidth_digits(labels[0])}
    if len(labels) > 1:
        return {"type": "multi_choice", "label": ",".join(to_halfwidth_digits(x) for x in labels)}
    return {"type": "free_text", "label": s}


def find_sections(md_text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for pattern in SECTION_PATTERNS:
        for match in pattern.finditer(md_text):
            core = next((g for g in match.groups() if g), match.group(0))
            core = str(core)
            if re.fullmatch(r"[一二三四五六七八九十〇零]+", core):
                num = kanji_to_int_small(core)
            else:
                num = None
                digit_match = re.search(r"([0-9０-９]+)", core)
                if digit_match:
                    num = int(to_halfwidth_digits(digit_match.group(1)))
            if num is None:
                continue
            hits.append({"no": num, "start": match.start(), "end": match.end()})
    hits.sort(key=lambda d: (d["start"], d["end"]))
    cleaned: list[dict[str, Any]] = []
    seen: set[int] = set()
    for hit in hits:
        if hit["start"] in seen:
            continue
        seen.add(hit["start"])
        cleaned.append(hit)
    for i, section in enumerate(cleaned):
        start = section["end"]
        end = cleaned[i + 1]["start"] if i + 1 < len(cleaned) else len(md_text)
        section["content"] = md_text[start:end].strip()
    return cleaned


def find_subs(content: str) -> list[tuple[int, int, str]]:
    subs: list[tuple[int, int, str]] = []
    for pattern in SUB_PATTERNS:
        for match in pattern.finditer(content):
            subs.append((match.start(), match.end(), match.group(1)))
    subs.sort(key=lambda x: x[0])
    return subs


def parse_md_split_images_keep_markdown(md_text: str) -> dict[str, dict[str, Any]]:
    qmap: dict[str, dict[str, Any]] = {}
    sections = find_sections(md_text)
    for section in sections:
        content = section["content"]
        subs = find_subs(content)
        if subs:
            main_stem_raw = content[:subs[0][0]].strip()
            main_text = strip_comments(main_stem_raw).strip()
            main_imgs = list(dict.fromkeys(extract_images(main_text)))
            for i, (sub_start, sub_end, token) in enumerate(subs):
                heading = token.strip()
                body_start = sub_end
                body_end = subs[i + 1][0] if i + 1 < len(subs) else len(content)
                sub_body_raw = content[body_start:body_end].strip()
                sub_text_raw = f"（{heading}）\n{sub_body_raw}".strip()
                sub_text = strip_comments(sub_text_raw).strip()
                sub_imgs_exact = list(dict.fromkeys(extract_images(sub_text)))
                choices = extract_choices_labeled(sub_text) or extract_choices_labeled(sub_body_raw)
                qmap[f"{section['no']}({heading})"] = {
                    "main_text": main_text,
                    "main_image_files": main_imgs,
                    "sub_text": sub_text,
                    "sub_image_files": sub_imgs_exact,
                    "choices_labeled": choices,
                }
        else:
            main_text = strip_comments(content).strip()
            images = list(dict.fromkeys(extract_images(main_text)))
            choices = extract_choices_labeled(main_text) or extract_choices_labeled(content)
            rec = {
                "main_text": main_text,
                "main_image_files": images,
                "sub_text": "",
                "sub_image_files": [],
                "choices_labeled": choices,
            }
            qmap[f"{section['no']}(1)"] = rec
            qmap[f"{section['no']}"] = dict(rec)
    return qmap


def normalize_qn_candidates(qn: str) -> list[str]:
    qn = qn.strip()
    match = QN_RE.match(qn)
    if not match:
        qn2 = re.sub(r"^\s*([0-9０-９]+)\s*-\s*([0-9０-９]+)\s*$", r"\1(\2)", qn)
        if qn2 != qn:
            return normalize_qn_candidates(qn2)
        return [qn]
    main = to_halfwidth_digits(match.group(1))
    sub = match.group(2)
    subsub = match.group(3)
    candidates: list[str] = []
    if sub:
        candidates.append(f"{main}({sub})")
        if subsub:
            candidates.insert(0, f"{main}({sub}){subsub}")
    else:
        candidates.extend([f"{main}", f"{main}(1)"])
    return list(dict.fromkeys(candidates))


def remap_main_number_to_json(md_qmap: dict[str, Any], answers: list[dict[str, Any]]) -> dict[str, Any]:
    def main_of(key: str) -> str:
        match = re.match(r"^\s*([0-9]+)\s*(?:\(.+\))?", key)
        return match.group(1) if match else ""

    md_mains = {main_of(k) for k in md_qmap.keys() if main_of(k)}
    if len(md_mains) != 1:
        return md_qmap

    counts: dict[str, int] = {}
    for item in answers:
        qn = str(item.get("question_number", ""))
        match = re.match(r"^\s*([0-9０-９]+)", qn)
        if not match:
            continue
        main = to_halfwidth_digits(match.group(1))
        counts[main] = counts.get(main, 0) + 1
    if not counts:
        return md_qmap

    target_main = max(counts.items(), key=lambda x: x[1])[0]
    current_main = next(iter(md_mains))
    if not target_main or target_main == current_main:
        return md_qmap

    remapped: dict[str, Any] = {}
    for key, value in md_qmap.items():
        new_key = re.sub(
            rf"^{re.escape(current_main)}(\(.+\))?$",
            lambda m: f"{target_main}{m.group(1) or ''}",
            key,
        )
        remapped[new_key] = value
    return remapped


def build_records(md_qmap: dict[str, dict[str, Any]], answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in answers:
        qn_raw = str(item.get("question_number", "")).strip()
        if not qn_raw:
            continue

        mdrec = None
        for candidate in normalize_qn_candidates(qn_raw):
            mdrec = md_qmap.get(candidate)
            if mdrec:
                break
        if not mdrec:
            fallback = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩➀➁➂➃➄➅➆➇➈➉]$", "", qn_raw).strip()
            mdrec = md_qmap.get(fallback)
            if not mdrec:
                continue

        choices_labeled = mdrec.get("choices_labeled", [])
        choices: dict[str, str] = {}
        label_to_index: dict[str, str] = {}
        choices_labels: list[str] = []
        for idx, (label, text) in enumerate(choices_labeled, start=1):
            choice_key = f"choice{idx}"
            choices[choice_key] = text
            choices_labels.append(label)
            for alt in normalize_choice_label_variants(label):
                label_to_index.setdefault(alt, choice_key)

        answer_entries = item.get("answers", []) or []
        correct_entries = [e for e in answer_entries if e.get("correct") is True]
        gold_label = ""
        if correct_entries:
            gold_label = (parse_gold(correct_entries[0].get("answer_type", "")).get("label") or "").strip()

        raw_examples = item.get("correct_examples", [])
        if isinstance(raw_examples, str):
            raw_examples = [raw_examples]
        first_example = next((s.strip() for s in raw_examples if isinstance(s, str) and s.strip()), "")
        correct_answer = re.sub(ANSWER_TYPE_TRAIL_RE, "", first_example if first_example else gold_label).strip()

        correct_choice_id = None
        if gold_label and ":" not in gold_label:
            labels = [token.strip() for token in gold_label.split(",") if token.strip()]
            if len(labels) == 1:
                for alt in normalize_choice_label_variants(labels[0]):
                    if alt in label_to_index:
                        correct_choice_id = label_to_index[alt]
                        break

        match = QN_RE.match(qn_raw)
        if match:
            main = to_halfwidth_digits(match.group(1))
            sub = match.group(2) or ""
            subsub = match.group(3) or ""
            view_label = f"問{main}（{sub}）{subsub}" if sub else f"問{main}"
        else:
            view_label = f"問{qn_raw}"

        main_imgs = mdrec.get("main_image_files", []) or []
        sub_imgs = mdrec.get("sub_image_files", []) or []
        records.append(
            {
                "source": item.get("source") or "JHS (compiled)",
                "subject": item.get("subject") or "JHS",
                "year": item.get("year") or "",
                "question_id": f"Q{qn_raw.replace('(', '-').replace(')', '')}",
                "label": view_label,
                "main_text": mdrec.get("main_text", ""),
                "sub_text": mdrec.get("sub_text", ""),
                "main_image_files": main_imgs,
                "sub_image_files": sub_imgs,
                "image_files": list(dict.fromkeys([*main_imgs, *sub_imgs])),
                "choices": choices,
                "choices_labels": choices_labels,
                "answer_style": "multipleChoice" if choices else "openEnded",
                "correct_answer": correct_answer,
                "correct_answer_choice_id": correct_choice_id,
                "answer_distribution": [
                    {
                        "type_id": e.get("type_id"),
                        "answer_type": e.get("answer_type"),
                        "response_rate_percent": e.get("response_rate_percent") or e.get("rate"),
                        "correct": bool(e.get("correct")),
                    }
                    for e in answer_entries
                ],
                "correct_condition": item.get("correct_condition"),
                "correct_examples": raw_examples or [],
                "incorrect_examples": item.get("incorrect_examples"),
            }
        )
    return records


def convert_dataset(problem_md: Path, answers_json: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    md_text = problem_md.read_text(encoding="utf-8").lstrip("\ufeff")
    answers = json.loads(answers_json.read_text(encoding="utf-8"))
    sections = find_sections(md_text)
    diagnostics = {
        "sections_found": len(sections),
        "total_subheadings_found": sum(len(find_subs(s["content"])) for s in sections),
        "first_sections": [{"no": s["no"], "start": s["start"], "end": s["end"]} for s in sections[:5]],
    }
    md_qmap = parse_md_split_images_keep_markdown(md_text)
    md_qmap = remap_main_number_to_json(md_qmap, answers)
    records = build_records(md_qmap, answers)
    diagnostics["records_built"] = len(records)
    return records, diagnostics


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_readme(problem_md: Path, answers_json: Path, output_jsonl: Path, diagnostics: dict[str, Any]) -> str:
    return f"""# Markdown + JSON to JSONL converter

## Files
- Input Markdown: `{problem_md.name}`
- Input answers JSON: `{answers_json.name}`
- Output JSONL: `{output_jsonl.name}`

## Diagnostics
- Sections found: {diagnostics.get('sections_found')}
- Subheadings found: {diagnostics.get('total_subheadings_found')}
- Records built: {diagnostics.get('records_built')}

## Notes
- Supports common heading variants such as `# 第1問`, `# 【一】`, `# 1`, `## （三）`, `## 三`, `## 1`, and `## ①`.
- Normalizes `question_number` variants such as `1-1` to `1(1)` for matching.
- Preserves original `question_number` when writing `question_id`.
- Falls back from `2(二)①` to parent `2(二)` when needed.
- Automatically remaps the main section number when the Markdown contains only one main section but the JSON uses a different dominant main number.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert exam Markdown + answers JSON into JSONL for public release.")
    parser.add_argument("--problem-md", type=Path, required=True, help="Path to the source Markdown file.")
    parser.add_argument("--answers-json", type=Path, required=True, help="Path to the answers JSON file.")
    parser.add_argument("--out-jsonl", type=Path, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--out-readme", type=Path, default=None, help="Optional README path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, diagnostics = convert_dataset(args.problem_md, args.answers_json)
    write_jsonl(records, args.out_jsonl)
    if args.out_readme:
        args.out_readme.write_text(
            build_readme(args.problem_md, args.answers_json, args.out_jsonl, diagnostics),
            encoding="utf-8",
        )
    print(json.dumps({"diagnostics": diagnostics, "out_jsonl": str(args.out_jsonl)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
