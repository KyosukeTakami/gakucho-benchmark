# gakucho-benchmark

A Human-Grounded Multimodal Benchmark from Japan’s National Assessment of Academic Ability (Grade 9)

## Overview

This repository provides a **curriculum-grounded multimodal benchmark** constructed from officially released items of Japan’s National Assessment of Academic Ability (Grade 9).

The dataset spans:
- Science
- Mathematics
- Japanese Language

It preserves:
- Figures and diagrams
- Mathematical expressions
- Vertical Japanese text
- Complex layouts (panels, annotations, speech balloons)

This enables **vision-language evaluation** beyond text-only reasoning.

---

## Motivation

Japanese K–12 exam materials are inherently multimodal and structured in ways that are:

- Pedagogically meaningful for students
- Challenging for current AI systems

Key challenges include:
- Vertical text and multi-column layouts
- Diagram–text alignment
- OCR noise in dense annotations
- Panel segmentation and spatial reasoning

This benchmark is designed to evaluate how well models can **reason over educational content as it is actually presented in classrooms**.

---

## Dataset Features

- 📊 Human performance statistics (aggregated student response rates)
- 🖼️ Original figures and layout-preserving assets
- 🧮 Mixed question types:
  - Multiple Choice (MC)
  - Open-ended (free response)
- 🇯🇵 Native Japanese educational context

A Multimodal Benchmark from Japan’s National Assessment of Academic Ability (Grade 9) This repository provides a curriculum-grounded multimodal benchmark built from officially released middle-school items in Science, Mathematics, and Japanese Language. The dataset preserves figures, math expressions, and vertical Japanese text, enabling vision-language evaluation rather than text-only reasoning.

Why this matters: Japanese K–12 exam materials mix vertical text, diagrams, charts, speech balloons, and irregular panel layouts, which are pedagogically helpful yet machine-unfriendly (OCR errors, panel segmentation, linking labels to figures). Our pipeline normalizes these into a structured, reproducible JSON schema

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

```

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

Notes
Each line in the JSONL file is independent and can be streamed.
Image files are stored separately and referenced by filename.
Some entries may contain:
null values in response distributions
empty choices for open-ended questions

