"""BharatNanoGPT — Streamlit demo, built for a basic phone browser.

INFRASHIFT (Open Innovation) requirement: the demo must work end to end
on a basic phone browser, in Hindi or English. So this app is deliberately
mobile-first:
  - single column, no sidebar (on a phone the sidebar hides behind a
    hamburger menu and judges will not find it)
  - large tap targets: full-width buttons, min-height 56px, large text
  - Hindi + English sample prompts as visible buttons on the first screen
  - all controls in the main flow, inside expanders, not in a sidebar
  - no desktop-only layout, no multi-column comparison view

The model shown here was trained from scratch in this repo; it runs
fully offline after training. No LLM API is called at any point.
The checkpoint is PRE-TRAINED before the event (see README,
"Checkpoint strategy") — the demo loads it instantly.

Run:  streamlit run app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sample import generate_text  # noqa: E402

SAMPLE_PROMPTS = [
    ("🇮🇳 हिंदी: राम सुबह उठकर", "राम सुबह उठकर"),
    ("💬 Hinglish: bhai kal exam hai", "bhai kal exam hai aur mujhe"),
    ("🇬🇧 English: The capital of India is", "The capital of India is"),
    ("🇮🇳 हिंदी: भारत एक बड़ा देश है", "भारत एक बड़ा देश है"),
    ("💬 Hinglish: kal subah meeting hai", "mujhe kal subah 9 baje meeting hai"),
]

st.set_page_config(page_title="BharatNanoGPT", page_icon="🇮🇳",
                   initial_sidebar_state="collapsed")

# Mobile-first CSS: big text, big full-width buttons, no cramped columns.
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 18px; }
    .stButton > button {
        width: 100%; min-height: 56px; font-size: 18px;
        border-radius: 12px; white-space: normal; text-align: left;
        padding: 12px 16px;
    }
    .stTextInput input { font-size: 18px; min-height: 52px; }
    .stSelectbox, .stSlider { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🇮🇳 BharatNanoGPT")
st.markdown(
    "A tiny GPT **trained from scratch** on Hindi–English / Hinglish. "
    "No API, no internet needed — it runs on this phone. "
    "नीचे कोई भी prompt चुनें, फिर Generate दबाएँ।"
)

# Sample prompts visible on the first screen, as big tappable buttons.
st.subheader("Try one — एक prompt चुनें")
if "prompt" not in st.session_state:
    st.session_state.prompt = SAMPLE_PROMPTS[0][1]
for label, text in SAMPLE_PROMPTS:
    if st.button(label, key=f"sample-{text[:20]}"):
        st.session_state.prompt = text

prompt = st.text_input(
    "Your prompt / अपना prompt लिखें",
    value=st.session_state.prompt,
    key="prompt_input",
)
st.session_state.prompt = prompt

with st.expander("⚙️ Settings / सेटिंग्स"):
    checkpoint = st.selectbox(
        "Checkpoint",
        ["out/tiny/ckpt.pt", "out/bharat/ckpt.pt"],
        help="out/tiny = CPU smoke-test model (~4M params). "
             "out/bharat = full ~14M model (Colab).",
    )
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
    top_k = st.slider("Top-k", 1, 100, 40)
    max_tokens = st.slider("Max new tokens", 10, 200, 80)
# Defaults if the expander was never opened (Streamlit keeps widget state,
# but guard for the first render).
checkpoint = locals().get("checkpoint", "out/tiny/ckpt.pt")
temperature = locals().get("temperature", 0.8)
top_k = locals().get("top_k", 40)
max_tokens = locals().get("max_tokens", 80)

if st.button("✨ Generate / बनाएं", type="primary"):
    if not os.path.exists(checkpoint):
        st.error(f"Checkpoint not found: `{checkpoint}`. Train first: "
                 "`python train.py --config config/train_tiny.py`")
    elif not prompt.strip():
        st.warning("Please type or pick a prompt first. / पहले एक prompt चुनें।")
    else:
        with st.spinner("Local model generating... / मॉडल लिख रहा है..."):
            res = generate_text(checkpoint, prompt, max_tokens,
                                temperature, top_k)
        st.subheader("BharatNanoGPT — local, trained from scratch")
        st.write(res["text"])
        st.caption(f"checkpoint iter {res['checkpoint_iter']} · "
                   f"{res['latency_ms']} ms · {res['tokens_generated']} tokens")
        st.info(
            "Honest note: this is a tiny model trained on a small corpus. "
            "Expect short, sometimes repetitive completions — the point is "
            "that it was trained from scratch, and you can see the training "
            "data, the loss curve, and the tokenizer that produced it. / "
            "यह एक छोटा मॉडल है — कभी-कभी दोहराव हो सकता है, पर यह पूरी तरह "
            "हमारे अपने डेटा पर train हुआ है।"
        )
