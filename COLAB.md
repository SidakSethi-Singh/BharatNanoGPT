# BharatNanoGPT — Full 14M Run on Google Colab (T4 GPU)

> **Retrain v2 (9 Nov 2026):** the 8 Nov 6000-iter run overfitted (val 9.88)
> and its checkpoints were lost on reconnect. For the retrain, use
> **`notebooks/BharatNanoGPT_Colab_T4_Retrain1000_Drive.ipynb`** instead —
> 1000 iters, Drive save straight after training, fixed files + corpus
> embedded (no expiring share links). Same 3 steps: upload → T4 GPU →
> Run all, plus one Drive authorisation at the start. The old notebook
> below still works, but its share links expire 10 Nov and it does not
> save to Drive — use v2.

## Kya hai ye?
- Tiny model (~4M, CPU pe train hua) — samples fragmentary hain, pipeline ka proof tha.
- Full model (~14M params, 6 layers, 384 embd, vocab 8192, 6000 iters) — Colab free T4 GPU pe train hoga, bada corpus (2–5M tokens target) ke saath. Isi ka checkpoint INFRASHIFT demo me use hoga.

## Step-by-step — sirf ye karo

1. **Notebook kholo:** [Google Colab](https://colab.research.google.com/) → File → Upload notebook → `notebooks/BharatNanoGPT_Colab_T4.ipynb` select karo.
   - Ya: GitHub repo → `notebooks/BharatNanoGPT_Colab_T4.ipynb` → "Open in Colab" button.
2. **GPU select karo:** Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save.
3. **Run all dabao:** Runtime → Run all.
4. **Intezar karo (1–2 ghante):** Tab khula rakho, laptop sleep mat karo.
   - Pehla cell GPU check karega — `CUDA available: True` aur `Tesla T4` dikhna chahiye. Nahi dikha to step 2 dobara karo.
   - Corpus cell prebuilt corpus (120,280 tokens) download karta hai — Wikipedia se fetch nahi karta, kyunki Wikipedia Colab ko HTTP 429 (Too Many Requests) de deta hai. `Tokens: total` ~120,000 dikhna chahiye. Agar 20,000 se kam dikhe, to train mat karo.
   - Training cell sabse lamba chalega. Loss har 50 iter pe dikhega: shuru 7.6 → pehle 100 iter me ~5 tak girna chahiye.
   - Checkpoint har 500 iter pe `out/bharat/` me save hota hai.
5. **Evaluate cell khud chalega:** perplexity + 3 real samples (Hindi, Hinglish, English) print honge. Inhe README me dalna hai.
6. **Checkpoint download karo:** Last cell se `out/bharat/ckpt.pt` (~55 MB) aur `results/eval_samples.jsonl` download ho jayenge. Inhe apne laptop pe save karo.

**Note (2026-11-08 fix):** purane notebook me 2 bugs the — (1) corpus Wikipedia 429 se khaali reh jata tha (sirf 4,560 tokens), phir training `RuntimeError: random_ expects 'from' to be less than 'to'` se crash hoti thi kyunki val set (228 tokens) block_size (256) se chhota tha; (2) evaluate cell me `--max-tokens` likha tha, sahi flag `--max-new-tokens` hai. Dono fix ho gaye hain: `data/prepare.py` me 429 retry + `--from-text` option, `train.py` me clear error jab corpus chhota ho, aur notebook me prebuilt corpus flow.

## Demo me kaise use karna hai?
Download kiya hua `ckpt.pt` ko apne BharatNanoGPT folder me `out/bharat/ckpt.pt` ke roop me rakho, phir:
```bash
streamlit run app.py
```
App ko full checkpoint use karne ke liye config karna hai — agar app abhi tiny checkpoint load karta hai, to README / app.py me checkpoint path badal do.

## Agar kuch gadbad ho
- **GPU nahi mila:** Runtime → Change runtime type → T4 GPU, phir Restart and run all.
- **Session disconnect ho gaya:** Ghabrao mat — checkpoint save hai. Notebook dobara chalao aur training cell me `--init-from out/bharat/ckpt.pt` se resume karo.
- **Loss nahi gir raha:** Corpus check karo — `data/train.bin` kitna bada hai? Tiny run me 120k tokens the; full run me lakhon hone chahiye.
- **Colab free limit khatam:** Subah dobara try karo. Tab tak tiny checkpoint se demo chalao — wo pehle se verified hai.
- **Samples abhi bhi fragmentary hain:** To corpus aur bada karo (articles 400 → 800) ya iters badhao. README me honest result likhna — judges ko transparency pasand hai.

## Time estimate
- Setup + corpus: ~10–15 min
- Training (6000 iters, T4): ~1–2 ghante (estimate, GPU load pe depend karta hai)
- Evaluate + download: ~5 min
