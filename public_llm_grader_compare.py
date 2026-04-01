# -*- coding: utf-8 -*-
"""
Public version for GitHub release.

What this script does
- Compare multiple LLM graders on existing benchmark run directories.
- Evaluate open-ended and multiple-choice outputs separately.
- Inject rubric-derived metadata such as student_correct_rate.
- Save grader-specific JSONL/JSON summaries.

Required environment variables
- OPENAI_API_KEY
- GEMINI_API_KEY
- ANTHROPIC_API_KEY
"""

# ============================================
# 複数の“採点用LLM”を一気に評価・比較するスクリプト（Gemini=GEMINI_API_KEY 版）
# - OpenAI / Anthropic / Gemini を同時に評価可
# - 採点結果は grader_id ごとに open_grades__{id}.jsonl / mc_grades__{id}.jsonl
# - 予測モデル（RUNS_ROOT配下の各フォルダ）× 採点LLM の比較表を表示
# - OpenAI: max_tokens / max_completion_tokens 自動切替、temperature 非対応時は自動で外す
# - Gemini: GEMINI_API_KEY を使用、モデル名のエイリアス解決 + list_models による自動代替選択
# - Anthropic: temperature 非対応時は自動で外す
# ============================================

from pathlib import Path
import os, json, re, time, ast, math, unicodedata
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm

# ====== 設定 ======
RUNS_ROOT = Path(os.getenv("RUNS_ROOT", "./benchmark_runs"))
RUBRIC_JSON = Path(os.getenv("RUBRIC_JSON", "./rubric.json"))

# ★ “採点用LLM（grader）” を並べる（Geminiは 1.5-latest を推奨）
#   temperature は None なら API 既定（温度非対応モデルでのエラー回避にも有効）
GRADERS = [
    {"id": "gpt4o", "provider": "openai", "model": "gpt-4o", "base_url": None, "max_tokens": 256, "temperature": 0.0},
    {"id": "gemini25flash", "provider": "gemini", "model": "gemini-2.5-flash", "max_tokens": 256, "temperature": 0.0},
    {"id": "claude_sonnet", "provider": "anthropic", "model": "claude-sonnet-4-5", "max_tokens": 256, "temperature": 0.0},
]

LIMIT_PER_MODEL = 0               # 0=制限なし
INCLUDE_PROBLEM_TEXT_IN_PROMPT = False

# ====== APIキー（環境に既にあれば不要）======
# OpenAI / Anthropic はそのまま。Gemini は GEMINI_API_KEY を使用（GOOGLE_API_KEY ではありません）
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("ANTHROPIC_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")   # ← ここを使用



def load_graders_from_json(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return GRADERS
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Graders config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Graders config must be a JSON list.")
    return data

# ============================================
# 小物ユーティリティ
# ============================================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def parse_list_like(val) -> List[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)): return []
    if isinstance(val, list): return [str(x) for x in val]
    s = str(val).strip()
    if not s: return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list): return [str(x) for x in obj]
    except Exception:
        pass
    if ";" in s: return [x.strip() for x in s.split(";") if x.strip()]
    if "," in s: return [x.strip() for x in s.split(",") if x.strip()]
    return [s]

def with_retries(func, *, max_tries=5, base_delay=1.0):
    for i in range(max_tries):
        try:
            return func()
        except Exception:
            if i == max_tries - 1: raise
            time.sleep(base_delay * (2 ** i))

# ============================================
# ルーブリックJSONの読込 & インデックス
# ============================================
def _norm(s: Optional[str]) -> str:
    if s is None: return ""
    return re.sub(r"\s+", "", str(s)).strip()

def _to_year_any(s: Any) -> str:
    if s is None: return ""
    txt = str(s)
    m4 = re.search(r"(20\d{2})", txt)
    if m4: return m4.group(1)
    m2 = re.search(r"(?<!\d)(\d{2})(?=msci|[_\-\W])", txt, flags=re.IGNORECASE)
    if m2: return f"20{m2.group(1)}"
    return ""

def _to_qno_any(s: Any) -> str:
    if not s: return ""
    t = str(s)
    t = t.replace("（", "(").replace("）", ")")
    t = re.sub(r"^(?:Q|問)\s*", "", t, flags=re.IGNORECASE)
    m = re.search(r"(\d+\([^)]+\))", t)
    if m:
        return _norm(m.group(1))
    m2 = re.search(r"(?<!\d)(\d+)\s*[-_ ]\s*(\d+)(?!\d)", t)
    if m2:
        return f"{m2.group(1)}({m2.group(2)})"
    return _norm(t)

def _extract_year_qno_from_qid_any(qid: str) -> Tuple[str, str]:
    if not qid: return "", ""
    s = str(qid).replace("（","(").replace("）",")")
    y = _to_year_any(s)
    m_qno = re.search(r"(\d+\([^)]+\))", s)
    if m_qno:
        q = m_qno.group(1)
    else:
        m_alt = re.search(r"(?<!\d)(\d+)\s*[-_ ]\s*(\d+)(?!\d)", s)
        q = f"{m_alt.group(1)}({m_alt.group(2)})" if m_alt else ""
    return y, _to_qno_any(q)

def load_rubric_items(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Rubric JSON が見つかりません: {json_path}")
    items = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Rubric JSON は配列である必要があります。")
    return items

def _parse_rate_to_percent(rate) -> Optional[float]:
    """'12.3%' / '12.3' / 12.3 / 0.123 → 12.3（%）。不正なら None。"""
    if rate is None: return None
    try:
        if isinstance(rate, str):
            s = rate.strip().replace("%","")
            if not s:
                return None
            v = float(s)
        else:
            v = float(rate)
        if 0.0 <= v <= 1.0:
            return v * 100.0
        return v
    except Exception:
        return None

def _answers_to_distribution(answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dist = []
    for a in answers or []:
        rate = a.get("response_rate_percent")
        if rate is None: rate = a.get("rate")
        dist.append({
            "type_id": a.get("type_id"),
            "answer_type": a.get("answer_type"),
            "rate": rate,
            "correct": bool(a.get("correct", False)),
        })
    return dist

def _sum_correct_rate_from_answers(answers: List[Dict[str, Any]]) -> Optional[float]:
    if not answers: return None
    s, used = 0.0, False
    for a in answers:
        if not bool(a.get("correct", False)):
            continue
        rv = _parse_rate_to_percent(a.get("response_rate_percent", a.get("rate")))
        if rv is None:
            continue
        s += rv
        used = True
    return s if used else None

def build_rubric_index(items: List[Dict[str, Any]]):
    by_year_q = {}
    by_subj_year_q = {}
    by_qid = {}
    rate_by_year_q = {}  # (year,qno) → sum(correct rates)
    for it in items:
        subj = _norm(it.get("subject",""))
        year = _to_year_any(it.get("year",""))
        qno  = _to_qno_any(it.get("question_number",""))
        qid  = str(it.get("question_id","")).strip()
        if year and qno:
            by_year_q.setdefault((year,qno), []).append(it)
            if subj:
                by_subj_year_q.setdefault((subj,year,qno), []).append(it)
            r = _sum_correct_rate_from_answers(it.get("answers") or [])
            if r is not None:
                rate_by_year_q[(year,qno)] = r
        if qid:
            by_qid.setdefault(qid, []).append(it)
    return by_year_q, by_subj_year_q, by_qid, rate_by_year_q

_RUBRIC_ITEMS = load_rubric_items(RUBRIC_JSON)
_RUBRIC_BY_YEAR_Q, _RUBRIC_BY_SUBJ_YEAR_Q, _RUBRIC_BY_QID, _RUBRIC_RATE_BY_YEAR_Q = build_rubric_index(_RUBRIC_ITEMS)
print(f"[Rubric] items loaded: {len(_RUBRIC_ITEMS)}")

def inject_rubric_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(rec)
    subj = _norm(rec.get("subject",""))
    year = _to_year_any(rec.get("year",""))
    qno  = _to_qno_any(rec.get("question_number") or rec.get("q_number") or rec.get("qno") or rec.get("item_number"))
    qid  = str(rec.get("question_id","")).strip()
    if not qno:
        y_guess, q_guess = _extract_year_qno_from_qid_any(qid)
        year = year or y_guess
        qno  = qno or _to_qno_any(q_guess)
    item = None
    if subj and year and qno and (subj,year,qno) in _RUBRIC_BY_SUBJ_YEAR_Q:
        item = _RUBRIC_BY_SUBJ_YEAR_Q[(subj,year,qno)][0]
    elif year and qno and (year,qno) in _RUBRIC_BY_YEAR_Q:
        item = _RUBRIC_BY_YEAR_Q[(year,qno)][0]
    elif qid and qid in _RUBRIC_BY_QID:
        item = _RUBRIC_BY_QID[qid][0]
    if item:
        rec.setdefault("correct_condition", item.get("correct_condition") or "")
        rec.setdefault("correct_examples",  item.get("correct_examples") or [])
        rec.setdefault("incorrect_examples",item.get("incorrect_examples") or [])
        if not rec.get("answer_distribution"):
            rec["answer_distribution"] = _answers_to_distribution(item.get("answers") or [])
    return rec

def resolve_student_correct_rate(rec: Dict[str, Any]) -> Optional[float]:
    dist = rec.get("answer_distribution") or []
    if dist:
        s, used = 0.0, False
        for d in dist:
            if not d.get("correct"): 
                continue
            rv = _parse_rate_to_percent(d.get("rate", d.get("response_rate_percent")))
            if rv is None:
                continue
            s += rv; used = True
        if used:
            return s
    year = _to_year_any(rec.get("year",""))
    qno  = _to_qno_any(rec.get("question_number") or rec.get("q_number") or rec.get("qno") or rec.get("item_number"))
    if not qno:
        y2, q2 = _extract_year_qno_from_qid_any(str(rec.get("question_id","")))
        year = year or y2
        qno  = qno  or q2
    return _RUBRIC_RATE_BY_YEAR_Q.get((year,qno))

# ============================================
# ラベル正規化
# ============================================
_LABEL_CANON = {"ア":"ア","A":"ア","1":"ア","①":"ア","イ":"イ","B":"イ","2":"イ","②":"イ",
                "ウ":"ウ","C":"ウ","3":"ウ","③":"ウ","エ":"エ","D":"エ","4":"エ","④":"エ",
                "オ":"オ","E":"オ","5":"オ","⑤":"オ"}
_MULTI_TOKEN = {"(1)":"1","(2)":"2","(3)":"3","(4)":"4","(5)":"5",
                "（1）":"1","（2）":"2","（3）":"3","（4）":"4","（5）":"5"}

def _nfkc_upper(s: str) -> str:
    if s is None: return ""
    return unicodedata.normalize("NFKC", str(s)).upper().strip()

def _label_from_any(s: Optional[str]) -> Optional[str]:
    if not s: return None
    t = _nfkc_upper(s)
    for k,v in _MULTI_TOKEN.items(): t = t.replace(k, v)
    if re.fullmatch(r"[A-E]", t): return _LABEL_CANON.get(t)
    if re.fullmatch(r"[アイウエオ]", t): return _LABEL_CANON.get(t)
    if re.fullmatch(r"[1-5]", t): return _LABEL_CANON.get(t)
    if re.fullmatch(r"[①②③④⑤]", t): return _LABEL_CANON.get(t)
    return None

# ============================================
# LLM クライアント: OpenAI / Anthropic / Gemini（共通 complete IF）
# ============================================
class _MinimalOpenAIClient:
    """OpenAI Chat Completions 専用。max_*tokens と temperature を自動で切替。"""
    def __init__(self, model: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai v1 SDK が必要です。pip install openai") from e
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY が未設定です。")
        self.model = model
        self._openai = OpenAI(api_key=key, base_url=base_url)

    @staticmethod
    def _is_temp_unsupported_msg(msg: str) -> bool:
        m = msg.lower()
        return ("temperature" in m) and ("unsupported" in m or "does not support" in m)

    @staticmethod
    def _wants_max_tokens_msg(msg: str) -> bool:
        m = msg.lower()
        return ("use 'max_tokens'" in m) or ('use "max_tokens"' in m)

    @staticmethod
    def _wants_max_completion_tokens_msg(msg: str) -> bool:
        m = msg.lower()
        return ("use 'max_completion_tokens'" in m) or ('use "max_completion_tokens"' in m) or \
               ("unsupported parameter" in m and "max_tokens" in m)

    @staticmethod
    def _kw_error_for_mctokens(msg: str) -> bool:
        m = msg.lower()
        return ("max_completion_tokens" in m) and ("unexpected" in m or "unexpected keyword" in m)

    def _chat_create(self, *, messages, max_tokens: int, param_name: str, temperature: Optional[float]):
        kwargs = {"model": self.model, "messages": messages}
        kwargs[param_name] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        return self._openai.chat.completions.create(**kwargs)

    def complete(self, *, system_prompt: Optional[str], user_text: str,
                 max_tokens: int, temperature: Optional[float] = 0.0) -> str:
        messages = []
        if system_prompt: messages.append({"role":"system","content":system_prompt})
        messages.append({"role":"user","content":user_text})

        def try_sequence(seq):
            last_exc = None
            for param_name, use_temp in seq:
                t = temperature if use_temp else None
                try:
                    r = self._chat_create(messages=messages, max_tokens=max_tokens,
                                          param_name=param_name, temperature=t)
                    return (r.choices[0].message.content or "").strip()
                except Exception as e:
                    last_exc = e
                    if self._is_temp_unsupported_msg(str(e)):  # 温度禁止 → 次へ
                        continue
                    continue
            if last_exc: raise last_exc
            return ""

        seq = [("max_completion_tokens", True), ("max_completion_tokens", False),
               ("max_tokens", True), ("max_tokens", False)]
        try:
            return try_sequence(seq)
        except Exception as e1:
            msg = str(e1)
            if self._wants_max_tokens_msg(msg) or self._kw_error_for_mctokens(msg):
                return try_sequence([("max_tokens", True), ("max_tokens", False)])
            if self._wants_max_completion_tokens_msg(msg):
                return try_sequence([("max_completion_tokens", True), ("max_completion_tokens", False)])
            raise

class _MinimalAnthropicClient:
    def __init__(self, model: str):
        try:
            import anthropic
        except Exception as e:
            raise RuntimeError("anthropic SDK が必要です。pip install anthropic") from e
        key = os.getenv("ANTHROPIC_API_KEY","")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY が未設定です。")
        import anthropic
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model

    def _blocks_to_text(self, content) -> str:
        parts = []
        try:
            for blk in content or []:
                t = getattr(blk, "text", None)
                if t: parts.append(t)
        except Exception:
            if isinstance(content, list):
                for blk in content:
                    t = blk.get("text") if isinstance(blk, dict) else None
                    if t: parts.append(t)
        return "\n".join(parts).strip()

    def complete(self, *, system_prompt: Optional[str], user_text: str,
                 max_tokens: int, temperature: Optional[float] = 0.0) -> str:
        def _call(temp: Optional[float]):
            return self._client.messages.create(
                model=self.model, max_tokens=max_tokens,
                temperature=(temp if temp is not None else 1),
                system=system_prompt or "", messages=[{"role":"user","content": user_text}],
            )
        try:
            msg = _call(temperature)
        except Exception as e:
            if "temperature" in str(e).lower() and ("unsupported" in str(e).lower() or "does not support" in str(e).lower()):
                msg = _call(None)
            else:
                raise
        return self._blocks_to_text(getattr(msg, "content", None))

class _MinimalGeminiClient:
    """
    Gemini: モデル名のエイリアス解決 + list_models による自動代替選択。
    API キーは GEMINI_API_KEY を使用します。
    """
    _ALIASES = {
        "gemini-2.0-pro": "gemini-1.5-pro-latest",
        "gemini-pro":     "gemini-1.5-pro-latest",
        "gemini-pro-1.5": "gemini-1.5-pro-latest",
        "gemini-2.5":     "gemini-1.5-flash-latest",
        "gemini-2.5-pro": "gemini-1.5-pro-latest",
        "gemini-2.5flash":"gemini-1.5-flash-latest",
        "gemini-2.5-flash":"gemini-1.5-flash-latest",
        "gemini-2.0-flash":"gemini-1.5-flash-latest",
    }

    def __init__(self, model: str):
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError("google-generativeai が必要です。pip install google-generativeai") from e
        key = os.getenv("GEMINI_API_KEY","")   # ← GEMINI_API_KEY を使用
        if not key:
            raise RuntimeError("GEMINI_API_KEY が未設定です。")
        import google.generativeai as genai
        genai.configure(api_key=key)
        self._genai = genai
        self.requested = model
        self.model_name = self._resolve_model_name(model)

    def _list_text_models(self) -> list:
        names = []
        try:
            for m in self._genai.list_models():
                methods = getattr(m, "supported_generation_methods", []) or getattr(m, "generation_methods", [])
                if methods and any("generate" in str(x).lower() for x in methods):
                    names.append(getattr(m, "name", ""))
        except Exception:
            pass
        return [n for n in names if n]

    def _resolve_model_name(self, model: str) -> str:
        m = self._ALIASES.get(model, model)
        models = self._list_text_models()
        if models:
            short_set = {n.split("/")[-1]: n for n in models}
            if m in short_set:
                return short_set[m]
            want_pro = ("pro" in m.lower()) and ("flash" not in m.lower())
            preferred = [n for n in short_set if ("pro" in n if want_pro else "flash" in n)]
            preferred = [n for n in preferred if "latest" in n] + [n for n in preferred if "latest" not in n]
            if preferred:
                return short_set[preferred[0]]
            return models[0]
        return f"models/{('gemini-1.5-pro-latest' if 'pro' in m and 'flash' not in m else 'gemini-1.5-flash-latest')}"

    # --- 追加: 安全なテキスト抽出 ---
    def _extract_text(self, resp) -> str:
        # 1) まず quick accessor（例外が出たら握りつぶして次へ）
        try:
            t = getattr(resp, "text", None)
            if t:
                return t.strip()
        except Exception:
            pass
        # 2) candidates -> content.parts[].text を順に拾う
        try:
            cands = getattr(resp, "candidates", None) or []
            for c in cands:
                content = getattr(c, "content", None) or {}
                parts = getattr(content, "parts", None) or []
                for p in parts:
                    # p は dict or SDK の Part オブジェクト想定
                    t = (p.get("text") if isinstance(p, dict) else getattr(p, "text", None))
                    if t and str(t).strip():
                        return str(t).strip()
        except Exception:
            pass
        # 3) 何も取れなければ空文字（上位でデフォルトJSONにフォールバック）
        return ""

    def _call_once(self, *, sys_prompt: str | None, text: str,
                   max_tokens: int, temperature: float | None, use_sys: bool) -> str:
        genai = self._genai
        cfg = {"max_output_tokens": max_tokens}
        if temperature is not None:
            cfg["temperature"] = temperature

        if use_sys:
            gm = genai.GenerativeModel(self.model_name, system_instruction=sys_prompt or "")
            resp = gm.generate_content(text, generation_config=cfg)
        else:
            gm = genai.GenerativeModel(self.model_name)
            prefixed = (sys_prompt.strip() + "\n\n" + text) if sys_prompt else text
            resp = gm.generate_content(prefixed, generation_config=cfg)

        # ← ここで例外を出さず、必ず文字列を返す
        return self._extract_text(resp)

    def complete(self, *, system_prompt: Optional[str], user_text: str,
                 max_tokens: int, temperature: Optional[float] = 0.0) -> str:
        # 1) system_instruction を使う
        try:
            return self._call_once(sys_prompt=system_prompt, text=user_text,
                                   max_tokens=max_tokens, temperature=temperature, use_sys=True)
        except Exception as e1:
            # モデル未検出/非対応など → エイリアス再解決してやり直し
            msg1 = str(e1).lower()
            if "not found" in msg1 or "is not found" in msg1 or "unsupported" in msg1:
                self.model_name = self._resolve_model_name(self.requested)
        # 2) system_instruction を外す（安全ブロック時や互換性のため）
        try:
            return self._call_once(sys_prompt=system_prompt, text=user_text,
                                   max_tokens=max_tokens, temperature=temperature, use_sys=False)
        except Exception as e2:
            msg2 = str(e2).lower()
            # 3) 温度非対応 → temperature 無しで再試行
            if "temperature" in msg2 and ("unsupported" in msg2 or "does not support" in msg2):
                try:
                    return self._call_once(sys_prompt=None, text=user_text,
                                           max_tokens=max_tokens, temperature=None, use_sys=False)
                except Exception:
                    pass
        # 4) 最後の最後は system も temperature も外して最小構成
        try:
            return self._call_once(sys_prompt=None, text=user_text,
                                   max_tokens=max_tokens, temperature=None, use_sys=False)
        except Exception:
            return ""  # ここまで来たら空文字（上位の既定処理に任せる）

def _get_client(provider: str, model: str, base_url: Optional[str]):
    p = (provider or "").lower()
    if p == "openai":
        return _MinimalOpenAIClient(model=model, base_url=base_url)
    if p == "anthropic":
        return _MinimalAnthropicClient(model=model)
    if p == "gemini":
        return _MinimalGeminiClient(model=model)
    raise RuntimeError(f"未知の provider: {provider}")

def complete_flex(client, *, system_prompt: str, user_text: str,
                  max_tokens: int = 256, temperature: Optional[float] = 0.0) -> str:
    fn = getattr(client, "complete", None)
    if not callable(fn):
        raise RuntimeError("client に .complete がありません。")
    return fn(system_prompt=system_prompt, user_text=user_text,
              max_tokens=max_tokens, temperature=temperature)

# ============================================
# 採点プロンプト（open / MC）
# ============================================
SYSTEM_PROMPT = """あなたは採点担当者です。与えられた情報だけで厳格に採点します。
結論と短い根拠、正規化した回答だけをJSONで返してください。"""

def _coerce_answer_distribution(val) -> List[Dict[str, Any]]:
    if isinstance(val, list):
        return [x for x in val if isinstance(x, dict)]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            return []
    return []

def _parse_json_list(val) -> List[Any]:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
        except Exception:
            return [s]
    return []

def _mc_expected_label(rec: Dict[str, Any]) -> str:
    gl = rec.get("gold_label")
    if gl:
        lab = _label_from_any(gl)
        return lab if lab else str(gl).strip()
    cid = rec.get("correct_answer_choice_id")
    labels = rec.get("choices_labels") or []
    if cid and labels:
        m = re.search(r"choice(\d+)", str(cid))
        if m:
            k = int(m.group(1))
            if 1 <= k <= len(labels):
                lab = _label_from_any(labels[k-1])
                return lab if lab else str(labels[k-1]).strip()
    dist = _coerce_answer_distribution(rec.get("answer_distribution"))
    for d in dist:
        if d.get("correct") is True:
            lab = _label_from_any(d.get("answer_type"))
            if lab: return lab
    ca = rec.get("correct_answer")
    lab = _label_from_any(ca)
    if lab: return lab
    return ""

def build_grading_user_text_open(rec: Dict[str, Any]) -> str:
    prediction_raw    = (rec.get("prediction_raw") or rec.get("prediction") or "").strip()
    gold_text         = (rec.get("gold_text") or rec.get("gold_text_dup") or rec.get("correct_answer") or "").strip()
    correct_condition = (rec.get("correct_condition") or "").strip()
    answer_distribution = _coerce_answer_distribution(rec.get("answer_distribution"))
    correct_examples    = _parse_json_list(rec.get("correct_examples"))
    incorrect_examples  = _parse_json_list(rec.get("incorrect_examples"))
    text = (rec.get("text") or rec.get("main_text") or "").strip()
    sub_text = (rec.get("sub_text") or "").strip()
    if sub_text:
        text = (text + ("\n\n" if text else "") + sub_text).strip()
    schema_desc = {
        "is_correct": "boolean",
        "normalized_answer": "string",
        "expected_answer": "string",
        "category_type_id": "integer|null",
        "category_label": "string",
        "reason_short": "string (1文以内)",
        "confidence": "number (0.0-1.0)"
    }
    lines = []
    if INCLUDE_PROBLEM_TEXT_IN_PROMPT and text:
        lines += ["【問題文】", text, ""]
    lines += [
        "【受験者の回答（モデル予測）】", prediction_raw or "(空)", "",
        "【模範回答（教師データ）】", gold_text or "(空)", "",
        "【採点規準・注意】", correct_condition or "特記事項なし", "",
        "【分布（参考）】", json.dumps(answer_distribution, ensure_ascii=False), "",
        "【正解例（任意）】", json.dumps(correct_examples, ensure_ascii=False), "",
        "【誤答例（任意）】", json.dumps(incorrect_examples, ensure_ascii=False), "",
        "出力は厳密なJSONのみ。次のキーを必ず含めてください：", json.dumps(schema_desc, ensure_ascii=False), "",
        "採点方針：模範に合致すれば is_correct=true。複数ラベルは全軸一致で正解。式等は同値なら正解。理由は一文以内。"
    ]
    return "\n".join(lines)

def build_grading_user_text_mc(rec: Dict[str, Any]) -> str:
    prediction_raw = (rec.get("prediction_raw") or rec.get("prediction") or "").strip()
    expected_label = _mc_expected_label(rec)
    labels = rec.get("choices_labels") or []
    text = (rec.get("text") or rec.get("main_text") or "").strip()
    sub_text = (rec.get("sub_text") or "").strip()
    if sub_text:
        text = (text + ("\n\n" if text else "") + sub_text).strip()
    answer_distribution = _coerce_answer_distribution(rec.get("answer_distribution"))
    schema_desc = {
        "is_correct": "boolean",
        "normalized_answer": "string",
        "expected_answer": "string",
        "reason_short": "string (1文以内)",
        "confidence": "number (0.0-1.0)"
    }
    label_set_hint = labels if labels else ["ア","イ","ウ","エ","オ","A","B","C","D","E","①","②","③","④","1","2","3","4"]
    lines = []
    if INCLUDE_PROBLEM_TEXT_IN_PROMPT and text:
        lines += ["【問題文（参考）】", text, ""]
    lines += [
        "【受験者の回答（モデル予測/ラベル表記）】", prediction_raw or "(空)", "",
        "【正解ラベル（教師データ）】", expected_label or "(空)", "",
        "【想定されるラベル集合（参考）】", ", ".join(map(str, label_set_hint)), "",
        "【分布（参考）】", json.dumps(answer_distribution, ensure_ascii=False), "",
        "出力は厳密なJSONのみ。次のキーを必ず含めてください：", json.dumps(schema_desc, ensure_ascii=False), "",
        "採点方針：両者が同値なら is_correct=true。",
        "正規化ルール：全角/半角/大小/仮名の表記ゆれ、'1→ア','2→イ','3→ウ','4→エ' 等の対応、'(1)→1','①→1' などは同値扱い。"
    ]
    return "\n".join(lines)

# ============================================
# 収集: preds で open/MC を判定 → canonical 構築（1回だけ）
# ============================================
def _load_preds_for_run(run_dir: Path) -> List[Dict[str, Any]]:
    report  = run_dir / "report.json"
    predsjl = run_dir / "preds.jsonl"
    if report.exists():
        try:
            obj = json.loads(report.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                rows = obj.get("preds") or []
                if isinstance(rows, list):
                    return rows
        except Exception:
            pass
    if predsjl.exists():
        return load_jsonl(predsjl)
    return []

def _load_details_index(run_dir: Path):
    details = run_dir / "details.csv"
    idx = {}
    if not details.exists(): return idx
    df = pd.read_csv(details)
    for _, r in df.iterrows():
        rec = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r.to_dict().items()}
        qid = rec.get("question_id")
        if not qid: continue
        rec["choices_labels"] = parse_list_like(rec.get("choices_labels"))
        for k in ("correct_examples","incorrect_examples","answer_distribution"):
            v = rec.get(k)
            if isinstance(v, str):
                try:
                    rec[k] = json.loads(v)
                except Exception:
                    pass
        idx[qid] = rec
    return idx

def build_canonical_sets_and_preds(runs_root: Path):
    qtype: Dict[str, str] = {}
    for child in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        preds = _load_preds_for_run(child)
        if not preds: continue
        for p in preds:
            qid = p.get("question_id")
            if not qid: continue
            ca  = p.get("correct_answer")
            lab = _label_from_any(ca)
            if lab:
                qtype[qid] = "mc"; continue
            if qid in qtype:  # 既に確定していれば尊重
                continue
            f1v = p.get("f1_chara", None)
            is_open = (f1v is not None) and not (isinstance(f1v, float) and pd.isna(f1v))
            qtype[qid] = "open" if is_open else "mc"

    open_qids = {qid for qid,t in qtype.items() if t == "open"}
    mc_qids   = {qid for qid,t in qtype.items() if t == "mc"}

    preds_by_run_open: Dict[str, Dict[str, str]] = {}
    preds_by_run_mc:   Dict[str, Dict[str, str]] = {}
    for child in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        preds = _load_preds_for_run(child)
        if not preds: continue
        o_map, m_map = {}, {}
        for p in preds:
            qid = p.get("question_id"); pred = (p.get("prediction") or "").strip()
            if not qid: continue
            if qid in open_qids: o_map[qid] = pred
            if qid in mc_qids:   m_map[qid] = pred
        preds_by_run_open[child.name] = o_map
        preds_by_run_mc[child.name]   = m_map

    canonical_open: Dict[str, Dict[str, Any]] = {qid: {"question_id": qid} for qid in sorted(open_qids)}
    canonical_mc:   Dict[str, Dict[str, Any]] = {qid: {"question_id": qid} for qid in sorted(mc_qids)}

    for child in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        d_idx = _load_details_index(child)
        for qid, rec in d_idx.items():
            base_open = {
                "question_id": qid, "subject": rec.get("subject"), "year": rec.get("year"),
                "question_number": rec.get("question_number") or rec.get("q_number") or rec.get("qno") or rec.get("item_number"),
                "text": rec.get("text") or rec.get("main_text"), "sub_text": rec.get("sub_text"),
                "gold_text": rec.get("gold_text"), "gold_text_dup": rec.get("gold_text_dup"),
                "correct_answer": rec.get("correct_answer"), "correct_condition": rec.get("correct_condition"),
                "correct_examples": rec.get("correct_examples"), "incorrect_examples": rec.get("incorrect_examples"),
                "answer_distribution": rec.get("answer_distribution"),
            }
            base_mc = {
                "question_id": qid, "subject": rec.get("subject"), "year": rec.get("year"),
                "question_number": rec.get("question_number") or rec.get("q_number") or rec.get("qno") or rec.get("item_number"),
                "choices_labels": rec.get("choices_labels"), "correct_answer_choice_id": rec.get("correct_answer_choice_id"),
                "gold_label": rec.get("gold_label"), "gold_choice_text": rec.get("gold_choice_text"),
                "text": rec.get("text") or rec.get("main_text"), "sub_text": rec.get("sub_text"),
                "answer_distribution": rec.get("answer_distribution"), "correct_answer": rec.get("correct_answer"),
            }
            if qid in canonical_open: canonical_open[qid].update({k:v for k,v in base_open.items() if v not in (None,"")})
            if qid in canonical_mc:   canonical_mc[qid].update({k:v for k,v in base_mc.items() if v not in (None,"")})

    # preds.jsonl の correct_answer を補完
    for child in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        preds = _load_preds_for_run(child)
        for p in preds:
            qid = p.get("question_id"); ca  = (p.get("correct_answer") or "").strip()
            if not qid or not ca: continue
            if qid in canonical_open: canonical_open[qid]["correct_answer"] = ca
            if qid in canonical_mc:   canonical_mc[qid]["correct_answer"]   = ca

    return (
        [canonical_open[qid] for qid in sorted(open_qids)],
        preds_by_run_open,
        [canonical_mc[qid] for qid in sorted(mc_qids)],
        preds_by_run_mc,
    )

# ============================================
# 採点: open（LLM） / MC（LLM）  ※出力先に grader_id を付与
# ============================================
def grade_open_run_using_canonical(
    run_dir: Path, client, canonical_recs: List[Dict[str, Any]],
    preds_by_run_open: Dict[str, Dict[str, str]],
    limit: int, max_tokens: int, grader_id: str, temperature: Optional[float]
) -> Dict[str, Any]:
    run_name = run_dir.name
    pred_index = preds_by_run_open.get(run_name, {})
    recs = canonical_recs[:limit] if (limit and limit > 0) else list(canonical_recs)

    grades, correct_cnt, n_eff = [], 0, 0
    rate_sum, rate_cnt = 0.0, 0
    for base in tqdm(recs, desc=f"[{grader_id}] Grading(open) @ {run_name}"):
        qid = base.get("question_id"); n_eff += 1
        rec = dict(base); rec["prediction_raw"] = pred_index.get(qid, "")
        rec = inject_rubric_fields(rec)
        student_rate = resolve_student_correct_rate(rec)
        if student_rate is not None and not (isinstance(student_rate, float) and math.isnan(student_rate)):
            rate_sum += float(student_rate); rate_cnt += 1
        user_text = build_grading_user_text_open(rec)
        def _call():
            return complete_flex(client, system_prompt=SYSTEM_PROMPT, user_text=user_text,
                                 max_tokens=max_tokens, temperature=temperature)
        content = with_retries(_call)
        parsed = None
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if m:
            try: parsed = json.loads(m.group(0))
            except Exception: parsed = None
        if not parsed:
            parsed = {
                "is_correct": None, "normalized_answer": "",
                "expected_answer": rec.get("gold_text") or rec.get("gold_text_dup") or rec.get("correct_answer") or "",
                "reason_short": "LLM出力のJSON解析に失敗", "confidence": 0.0
            }
        if isinstance(parsed.get("is_correct"), bool) and parsed["is_correct"]:
            correct_cnt += 1
        grades.append({"question_id": qid, "llm_grade": parsed, "raw": content, "student_correct_rate": student_rate})

    save_jsonl(run_dir / f"open_grades__{grader_id}.jsonl", grades)
    acc = (correct_cnt / n_eff) if n_eff else None
    scr_mean = (rate_sum / rate_cnt) if rate_cnt else None
    return {"grader_id": grader_id, "pred_model_dir": run_name, "n": n_eff, "correct": correct_cnt,
            "accuracy": acc, "student_correct_rate_mean": scr_mean}

def grade_mc_run_using_canonical(
    run_dir: Path, client, canonical_mc_recs: List[Dict[str, Any]],
    preds_by_run_mc: Dict[str, Dict[str, str]],
    limit: int, max_tokens: int, grader_id: str, temperature: Optional[float]
) -> Dict[str, Any]:
    run_name = run_dir.name
    pred_index = preds_by_run_mc.get(run_name, {})
    recs = canonical_mc_recs[:limit] if (limit and limit > 0) else list(canonical_mc_recs)

    grades, correct_cnt, n_eff = [], 0, 0
    rate_sum, rate_cnt = 0.0, 0
    for base in tqdm(recs, desc=f"[{grader_id}] Grading(MC)  @ {run_name}"):
        qid = base.get("question_id"); n_eff += 1
        rec = dict(base); rec["prediction_raw"] = pred_index.get(qid, "")
        rec = inject_rubric_fields(rec)
        student_rate = resolve_student_correct_rate(rec)
        if student_rate is not None and not (isinstance(student_rate, float) and math.isnan(student_rate)):
            rate_sum += float(student_rate); rate_cnt += 1
        user_text = build_grading_user_text_mc(rec)
        def _call():
            return complete_flex(client, system_prompt=SYSTEM_PROMPT, user_text=user_text,
                                 max_tokens=max_tokens, temperature=temperature)
        content = with_retries(_call)
        parsed = None
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if m:
            try: parsed = json.loads(m.group(0))
            except Exception: parsed = None
        if not parsed:
            parsed = {
                "is_correct": None, "normalized_answer": "",
                "expected_answer": _mc_expected_label(rec),
                "reason_short": "LLM出力のJSON解析に失敗", "confidence": 0.0
            }
        if isinstance(parsed.get("is_correct"), bool) and parsed["is_correct"]:
            correct_cnt += 1
        grades.append({"question_id": qid, "llm_grade": parsed, "raw": content, "student_correct_rate": student_rate})

    save_jsonl(run_dir / f"mc_grades__{grader_id}.jsonl", grades)
    acc = (correct_cnt / n_eff) if n_eff else None
    scr_mean = (rate_sum / rate_cnt) if rate_cnt else None
    return {"grader_id": grader_id, "pred_model_dir": run_name, "n": n_eff, "correct": correct_cnt,
            "accuracy": acc, "student_correct_rate_mean": scr_mean}

# ============================================
# 実行: canonical を1回構築 → 各“採点用LLM”で一気に採点・比較
# ============================================
canonical_open, preds_by_run_open, canonical_mc, preds_by_run_mc = build_canonical_sets_and_preds(RUNS_ROOT)
print(f"[Canonical] open items: {len(canonical_open)}  |  mc items: {len(canonical_mc)}")

all_open_summaries: List[Dict[str, Any]] = []
all_mc_summaries:   List[Dict[str, Any]] = []

# 予測モデルフォルダ一覧（空フォルダ除外）
pred_model_dirs = [p for p in sorted(RUNS_ROOT.iterdir()) if p.is_dir() and _load_preds_for_run(p)]

for g in GRADERS:
    gid   = g["id"];   prov  = g["provider"]; model = g["model"]
    base  = g.get("base_url"); mtok  = int(g.get("max_tokens", 256)); gtemp = g.get("temperature", 0.0)

    print(f"\n=== Grader: {gid}  ({prov} / {model}) ===")
    client = _get_client(prov, model, base)

    open_summaries, mc_summaries = [], []
    for child in pred_model_dirs:
        s_open = grade_open_run_using_canonical(
            run_dir=child, client=client,
            canonical_recs=canonical_open, preds_by_run_open=preds_by_run_open,
            limit=LIMIT_PER_MODEL, max_tokens=mtok, grader_id=gid, temperature=gtemp
        )
        open_summaries.append(s_open)

        s_mc = grade_mc_run_using_canonical(
            run_dir=child, client=client,
            canonical_mc_recs=canonical_mc, preds_by_run_mc=preds_by_run_mc,
            limit=LIMIT_PER_MODEL, max_tokens=mtok, grader_id=gid, temperature=gtemp
        )
        mc_summaries.append(s_mc)

    # ルートに採点者別サマリーを書き出し
    open_total_n = sum(s["n"] for s in open_summaries if s["n"] is not None)
    open_total_correct = sum(s["correct"] for s in open_summaries if s["correct"] is not None)
    open_total_acc = (open_total_correct / open_total_n) if open_total_n else None
    open_summary_obj = {"grader_id": gid, "by_pred_model": open_summaries,
                        "total_n": open_total_n, "total_correct": open_total_correct, "total_accuracy": open_total_acc}
    with (RUNS_ROOT / f"open_llm_summary__{gid}.json").open("w", encoding="utf-8") as f:
        json.dump(open_summary_obj, f, ensure_ascii=False, indent=2)

    mc_total_n = sum(s["n"] for s in mc_summaries if s["n"] is not None)
    mc_total_correct = sum(s["correct"] for s in mc_summaries if s["correct"] is not None)
    mc_total_acc = (mc_total_correct / mc_total_n) if mc_total_n else None
    mc_summary_obj = {"grader_id": gid, "by_pred_model": mc_summaries,
                      "total_n": mc_total_n, "total_correct": mc_total_correct, "total_accuracy": mc_total_acc}
    with (RUNS_ROOT / f"mc_summary__{gid}.json").open("w", encoding="utf-8") as f:
        json.dump(mc_summary_obj, f, ensure_ascii=False, indent=2)

    all_open_summaries.extend(open_summaries)
    all_mc_summaries.extend(mc_summaries)

# ============= 採点モデル比較テーブル（Open / MC 別） =============
def _round_or_none(x, ndigits):
    if x is None:
        return None
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return round(float(x), ndigits)
    except Exception:
        return None

df_open = pd.DataFrame(all_open_summaries)
df_mc   = pd.DataFrame(all_mc_summaries)

if not df_open.empty:
    print("\n== OPEN: accuracy by [pred_model_dir] × [grader_id] ==")
    piv_open = (df_open.assign(accuracy=lambda d: d["accuracy"].map(lambda v: _round_or_none(v, 6)))
                      .pivot_table(index="pred_model_dir", columns="grader_id", values="accuracy", aggfunc="first"))
    print(piv_open.to_markdown())

if not df_mc.empty:
    print("\n== MC: accuracy by [pred_model_dir] × [grader_id] ==")
    piv_mc = (df_mc.assign(accuracy=lambda d: d["accuracy"].map(lambda v: _round_or_none(v, 6)))
                    .pivot_table(index="pred_model_dir", columns="grader_id", values="accuracy", aggfunc="first"))
    print(piv_mc.to_markdown())

def _one_row_student_rate(df_any: pd.DataFrame, title: str):
    print(f"\n== {title} : student_correct_rate_mean (代表値) ==")
    if df_any.empty or "student_correct_rate_mean" not in df_any.columns:
        print(pd.DataFrame([{"metric":"student_correct_rate_mean","value":"(no value)"}]).to_markdown(index=False)); return
    series = df_any["student_correct_rate_mean"].dropna()
    vals = []
    for v in series.tolist():
        try:
            if v is None: continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): continue
            vals.append(float(v))
        except Exception:
            continue
    if not vals:
        print(pd.DataFrame([{"metric":"student_correct_rate_mean","value":"(no value)"}]).to_markdown(index=False)); return
    disp = f"{round(vals[0],1):.1f}%"
    print(pd.DataFrame([{"metric":"student_correct_rate_mean","value": disp}]).to_markdown(index=False))

_one_row_student_rate(df_open, "OPEN")
_one_row_student_rate(df_mc,   "MC")
def main():
    global RUNS_ROOT, RUBRIC_JSON, GRADERS, LIMIT_PER_MODEL, INCLUDE_PROBLEM_TEXT_IN_PROMPT
    import argparse

    parser = argparse.ArgumentParser(description="Compare multiple LLM graders on benchmark run outputs.")
    parser.add_argument("--runs-root", default=str(RUNS_ROOT), help="Directory containing one subdirectory per prediction model.")
    parser.add_argument("--rubric-json", default=str(RUBRIC_JSON), help="Rubric JSON path.")
    parser.add_argument("--graders-json", default=None, help="Optional JSON file describing grader models.")
    parser.add_argument("--limit-per-model", type=int, default=LIMIT_PER_MODEL, help="Limit number of items per prediction model. 0 means no limit.")
    parser.add_argument("--include-problem-text", action="store_true", help="Include problem text in grader prompts.")
    args = parser.parse_args()

    RUNS_ROOT = Path(args.runs_root)
    RUBRIC_JSON = Path(args.rubric_json)
    GRADERS = load_graders_from_json(args.graders_json)
    LIMIT_PER_MODEL = args.limit_per_model
    INCLUDE_PROBLEM_TEXT_IN_PROMPT = args.include_problem_text




if __name__ == "__main__":
    main()
