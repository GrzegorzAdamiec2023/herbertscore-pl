# HerBERTScore  

**HerBERTScore** is an evaluation metric for sentence similarity and text generation tasks, inspired by [BERTScore](https://arxiv.org/abs/1904.09675), but tailored for the **Polish language**.  
It leverages **HerBERT** (or any other transformer model) together with **Inverse Document Frequency (IDF)** statistics and a **baseline rescaling (b)** to provide more reliable similarity scores.  

> ✅ While the name suggests HerBERT, you can actually use **any Polish transformer** — or even a different language model, as long as you provide a matching dataset for IDF and baseline computation.  

---

## ✨ Features

- Compute **precision, recall, and F1 scores** between sentences.  
- Uses **IDF weighting** for more robust scoring.  
- Supports **baseline rescaling (b)** for precision and recall.  
- Save & load **IDF** and **baseline** to avoid recomputation.  
- Ready-to-use **NKJP (Polish National Corpus)** as the default dataset.  

---

## 📦 Installation

```bash
git clone https://github.com/yourname/herBERTScore.git
cd herBERTScore
pip install -r requirements.txt
```

---

## 🚀 Quickstart

```python
from herBERTscore.HerBERTScore import HerBERTScore

# Initialize with dataset (NKJP sample provided in repo)
hbs = HerBERTScore(file_path_to_texts="data/NKJP_test_texts.txt")

# Compute IDF (can take a while on a large dataset)
hbs.make_idf()

# Compute baseline (b) for precision and recall
# Note: reduce batch_size if you run out of GPU memory (default=100)
hbs.compute_b(batch_size=100)

# Compare sentences
s1 = ["Ala ma kota", "Programowanie w Pythonie jest przyjemne"]
s2 = ["Ala ma kocurka", "Python jest świetny do szybkiego prototypowania"]

x = hbs(s1, s2)
print(x["f1"])
```

Output:  

```
tensor([[0.91, 0.52],
        [0.48, 0.88]])
```

You can also check out the interactive demo in **`Example_usage.ipynb`**.  

---

## 💾 Saving & Loading State

Since computing **IDF** and **baseline** is expensive, you can save them once and reload later:  

```python
# Save
hbs.save_state("herBERTScoreState")

# Load
hbs.load_state("herBERTScoreState")
```

⚠️ IDF is tied to the tokenizer of the model used!  
Trying to load IDF from another model will raise an error.  

---

## 📚 Custom Dataset

If you want to use your own dataset instead of NKJP:  

- Provide a **plain text file** (`.txt`).  
- Each line should contain **one sentence or text fragment**.  
- Example format:  

```
To jest pierwszy przykład zdania.
Tutaj znajduje się drugie zdanie.
A to jest jeszcze inny fragment tekstu.
```

---

## 📖 Default Dataset: NKJP

The default dataset used in this repository is the **National Corpus of Polish (NKJP)**, available at:  
👉 [https://nkjp.pl/](https://nkjp.pl/)  

It covers the entirety of publicly available Polish texts in NKJP and serves as the basis for computing **IDF** and **baseline** values for HerBERTScore.  

---

## ⚡ Tips

- If you encounter **Out Of Memory (OOM)** on GPU, decrease `batch_size` in `compute_b()`.  
- Larger datasets for IDF yield more stable results, but also require more compute.  
- Always recompute or reload IDF and baseline when switching models.  

---

## 📝 License & Attribution

- Base dataset: [NKJP](https://nkjp.pl/)  
- Inspired by: [BERTScore](https://arxiv.org/abs/1904.09675)  
```
