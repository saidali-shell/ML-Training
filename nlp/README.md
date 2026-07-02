# NLP Analysis of Volve Drilling Reports

This project takes **drilling reports** written by engineers and automatically finds common topics/themes using Natural Language Processing (NLP).

---

## What We Did (10 Steps)

| Step | What It Does | In Simple Words |
|------|-------------|-----------------|
| **1. Load Data** | Read the drilling report dataset | Opened the file and looked at what's inside |
| **2. Understand Data** | Count reports, wells, check for missing info | Found 1,759 reports from 23 wells, 23,447 comments total |
| **3. Extract Text** | Pull out the written comments from the dataset | Each report has an `activity` list with a `comments` field — we grabbed all those comments |
| **4. Clean Text** | Lowercase, remove punctuation/numbers, filter stopwords ("the", "and"), split into words, lemmatize ("running" → "run") | Made the text ready for analysis by removing noise and reducing words to their base form |
| **5. Explore Data** | Plot comment lengths, find most common words | Average comment = 11 words. Top words: drill, hole, mud, casing, cement |
| **6. Convert to Numbers** | TF-IDF: turns text into numbers computers can understand | Each comment becomes a row of 2,357 numbers representing word importance |
| **7. Find Topics** | LDA model: automatically discovers hidden themes | The model split all comments into 5 groups based on word patterns |
| **8. Visualize** | Word clouds, bar charts of top words per topic | Pictures showing what each topic is about |
| **9. Interpret** | Read real comments to name each topic | Gave each topic a meaningful label |
| **10. Conclude** | Summarize what the model found | NLP can automatically organize drilling reports by operation type |

---

## The 5 Topics Found

| Topic | Theme | What It's About | Key Words | % of Comments |
|-------|-------|-----------------|-----------|:------------:|
| 1 | **Circulation & Cementing** | Pumping mud, cementing casing, cleaning the hole | lpm, pipe, cement, mud, circulated, hole | 19.7% |
| 2 | **Pressure Testing & Well Control** | Testing equipment, checking pressure, BOP functions | tested, pressure, bop, riser, well, closed | 30.9% |
| 3 | **BHA & Equipment Handling** | Making up drilling tools, tripping in/out, toolbox talks | bha, rih, tool, handling, equipment | 17.2% |
| 4 | **Rig Floor Operations** | Installing/removing equipment on the rig floor | installed, removed, floor, rig, bushing, elevator | 18.3% |
| 5 | **Drilling Parameters** | Actual drilling: RPM, weight on bit, flow rate, ROP | rpm, hole, drilled, lpm, bar, flow | 14.0% |

---

## What You Need

- Python with these libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`, `wordcloud`
- Install missing ones: `pip install wordcloud`
- NLTK resources download automatically when you run the notebook

## How to Run

Open `nlp.ipynb` in Jupyter and run all cells (Cell → Run All).

---

## Summary

We took 23,447 drilling comments, cleaned them up, and used a topic model (LDA) to automatically group them into 5 operation categories: circulation, pressure testing, equipment handling, rig floor work, and drilling parameters. The model was trained without any labels — it figured out the patterns on its own. This shows how NLP can help organize thousands of drilling reports without manual reading.
