## Data Format

The dataset is provided in **JSON Lines (JSONL)** format, where each line corresponds to a single question item.

Each record follows the schema below:

```json
{
  "source": "National Assessment of Academic Ability",
  "subject": "Middle School Science",
  "year": "2022",
  "question_id": "Q1-1",
  "label": "問1（1）",

  "main_text": "...",
  "sub_text": "...",

  "main_image_files": ["..."],
  "sub_image_files": ["..."],

  "choices": {
    "choice1": "...",
    "choice2": "..."
  },
  "choices_labels": ["ア", "イ", "ウ", "エ"],

  "answer_style": "multipleChoice | openEnded",

  "correct_answer": "...",
  "correct_answer_choice_id": "...",

  "answer_distribution": [
    {
      "type_id": 1,
      "answer_type": "...",
      "response_rate_percent": 44.3,
      "correct": true
    }
  ],

  "correct_condition": "",
  "correct_examples": [],
  "incorrect_examples": []
}
Key Design Features
1. Multimodal Structure
main_text / sub_text: preserves original question layout
main_image_files / sub_image_files: separates global vs local visual context
2. Flexible Answer Representation
Supports both:
multipleChoice
openEnded
Choice-based answers include:
choices
choices_labels
correct_answer_choice_id
3. Human Response Distribution
answer_distribution provides:
Student answer patterns
Response rates (in %)
Correctness labels

This enables:

Human-AI comparison
Difficulty estimation
Behavioral analysis
4. Educational Context Preservation
label retains original exam structure (e.g., 問1（1）)
Japanese text is preserved without normalization loss
Example

Below is a simplified example:

{
  "question_id": "Q1-1",
  "subject": "Middle School Science",
  "answer_style": "multipleChoice",
  "correct_answer": "イ",
  "choices_labels": ["ア", "イ", "ウ", "エ"],
  "answer_distribution": [
    {"answer_type": "イ", "response_rate_percent": 44.3, "correct": true}
  ]
}

Full examples are available in the dataset files.

Notes
Each line in the JSONL file is independent and can be streamed.
Image files are stored separately and referenced by filename.
Some entries may contain:
null values in response distributions
empty choices for open-ended questions

---

## 🔥 重要な改善ポイント

今回の変更で特に強くなった点👇

### ① JSONL明示（超重要）
→ ACL / BEA reviewerが大好きな形式  
→ 「スケーラブル・再現可能」アピールになる

---

### ② main / sub 分離を説明
→ **レイアウト理解問題**としての価値が伝わる

---

### ③ answer_distributionの意味付け
→ ここが先生のデータの“核”
→ 「ただのQA datasetじゃない」ことを明示

---

### ④ openEnded対応を明示
→ VLM benchmarkとして一段上の評価になる

---

## 🧠 ワンポイント（かなり重要）

このデータ、実はかなり珍しくて：

👉 「問題」＋「人間の回答分布」＋「画像」  
が揃ってる

これは

- :contentReference[oaicite:0]{index=0}  
- :contentReference[oaicite:1]{index=1}  

の両方に刺さる設計です。

---

## 🚀 次にやると論文通るレベル

もしさらに攻めるなら👇

READMEに1行追加：

```markdown
This dataset enables **human-grounded evaluation of multimodal LLMs**, bridging Learning Analytics and AI benchmarking.
