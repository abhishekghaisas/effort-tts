"""Streamlit showcase: vocal-effort control for cloned voices, Chatterbox vs Qwen3-TTS steering.
Run locally:  streamlit run showcase/app.py      (after python showcase/build_showcase.py)
Optional password: put APP_PASSWORD = "..." in .streamlit/secrets.toml (local) or in the app's Secrets (hosted)."""

from pathlib import Path

import pandas as pd
import showcase_lib as L
import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="Vocal effort control for cloned voices", layout="wide")

# ----------------------------------------------------------------------------------------------------------------------------- access gate
try:
    _pw = st.secrets["APP_PASSWORD"]
except Exception:
    _pw = None
if _pw and not st.session_state.get("ok"):
    if st.text_input("Password", type="password") == _pw:
        st.session_state["ok"] = True
        st.rerun()
    st.stop()


@st.cache_data
def load():
    man = pd.read_csv(HERE / "manifest.csv")
    met = pd.read_csv(HERE / "metrics.csv")
    return man, met


@st.cache_data
def audio_bytes(rel, gain_db):
    sr, x = L.read_wav(HERE / rel)
    return L.to_wav_bytes(L.apply_gain_db(x, gain_db) if gain_db else x, sr)


import json

FILT = (
    json.load(open(HERE / "filter.json"))
    if (HERE / "filter.json").exists()
    else {"max_audio_s": 10.0, "max_wer": 0.2, "min_sim": 0.80, "min_wps": 1.6}
)
FILT_TEXT = f"ends on its own within {FILT['max_audio_s']:g} s, WER <= {FILT['max_wer']:g}, similarity >= {FILT['min_sim']:g}, at least {FILT['min_wps']:g} words/s and no duplicated words"

if not (HERE / "manifest.csv").exists():
    st.error("manifest.csv not found: run `python showcase/build_showcase.py` first.")
    st.stop()
man, met = load()
shown = man[man.shown]

# ----------------------------------------------------------------------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
max_gain = st.sidebar.radio(
    "Loudness",
    [0, 6, 9, 12],
    format_func=lambda g: (
        "Equal loudness (hear voice quality only)" if g == 0 else f"Ramp up to +{g} dB"
    ),
    help="All clips are stored at the same speech level. A ramp raises the level linearly from the reference setting up to k = 1.0 (Qwen) or exaggeration 1.2 "
    "(Chatterbox), with a soft limiter. Real loud speech is about +9 dB over regular speech.",
)
st.sidebar.caption(
    "Voices are synthetic clones of research participants (EARS dataset, CC BY-NC 4.0). Research and non-commercial use only. Chatterbox clips carry Resemble AI's watermark."
)
REF = {
    "qwen_dev": (0.0, 1.0),
    "qwen_anger": (0.0, 1.0),
    "chatterbox_dev": (0.5, 1.2),
    "chatterbox_ft_v0": (0.5, 1.2),
    "chatterbox_ft_v1": (0.5, 1.2),
}


def play(row):
    ref, high = REF[row.group]
    st.audio(
        audio_bytes(row.audio, L.ramp_gain(row.setting, ref, high, max_gain)), format="audio/wav"
    )


def caption(row):
    bits = [f"effort {row.effort:.1f}", f"similarity {row.sim:.2f}", f"WER {row.wer:.2f}"]
    if pd.notna(row.arousal):
        bits += [f"arousal {row.arousal:.2f}", f"valence {row.valence:.2f}"]
    st.caption(" · ".join(bits))
    if isinstance(row.flag, str) and row.flag:
        st.warning(row.flag, icon="⚠️")


tab_over, tab_jour, tab_cmp, tab_lad, tab_met, tab_neg, tab_about = st.tabs(
    ["Overview", "Journey", "Compare", "Ladder", "Metrics", "What did not work", "About and ethics"]
)

JOURNEY = [
    (
        "1.0",
        "Effort estimator",
        "A level-free classifier trained on real EARS reading styles scores vocal effort from spectral and voice-quality features.",
        "Separates loud from regular reading on unseen speakers (AUC about 0.93 to 0.99) and ranks real emotional speech by arousal. Not validated on synthetic speech; saturates near the level of real yelling.",
    ),
    (
        "1.0",
        "Chatterbox baseline",
        "5 speakers x 30 lines x 5 exaggeration settings, scored for effort, identity and speech problems.",
        "The exaggeration knob controls effort strongly (Spearman 0.77), but speaker similarity falls from 0.94 to 0.84 at the top on the full baseline (0.93 to 0.89 on the 3 speakers and 8 lines shown here), pacing changes, 54% of clips at the top setting have speech problems, and seeds differ by about 2.5 effort points.",
    ),
    (
        "1.0",
        "Human labels",
        "Blind listening labels on the top setting.",
        "Only 51% of clips judged fine; test-retest agreement of one rater was moderate (kappa 0.53).",
    ),
    (
        "1.0",
        "LoRA fine-tune v0",
        "Trained Chatterbox's own scalar input on 2 hours of EARS loud versus regular speech.",
        "Identity and intelligibility stayed good, but the effort range collapsed from 12.6 to 2.5 points (Spearman 0.21).",
    ),
    (
        "1.0",
        "LoRA fine-tune v1, extrapolation, token probe",
        "Stronger scalar training with loud clips repeated; scalars beyond the training range; a probe of the speech tokens.",
        "Spearman only 0.33; larger scalars did nothing (Spearman 0.03); loud reading is only partly visible in the speech tokens (AUC 0.75), consistent with a weak training signal. Fine-tuning was shelved.",
    ),
    (
        "2.0",
        "Steering direction",
        "Qwen3-TTS hidden-state difference between clones made from loud versus regular readings of 12 training speakers.",
        "The direction is consistent across speakers (cosine 0.80 to 0.91).",
    ),
    (
        "2.0",
        "Strength sweeps",
        "Strength k from 0.05 to 2.0 on one layer, plus layer choice and a multi-layer spread.",
        "A smooth effort control: at k = 1.0 the estimator effort rises by about 24 points with similarity 0.906 (unsteered 0.949) and WER near 0. Above 1.0 generations stop ending and the voice distorts; spreading over layers did not help.",
    ),
    (
        "2.0",
        "Level",
        "Measured the speech level of steered clips.",
        "Steering also raises level (+8 dB at k = 0.35, +25 dB at k = 1.0, near full scale), so every clip on this site is level-matched and loudness is a separate control.",
    ),
    (
        "2.0",
        "Stopping",
        "A hard length cap, then trimming to the end of the last word with ASR timestamps and the voice offset.",
        "The cap cut the last word. Trimming recovers 24 of 24 clips at k = 1.25 (similarity 0.894, WER median 0).",
    ),
    (
        "2.0",
        "Emotion",
        "Arousal, dominance and valence of steered clips against real EARS emotional speech.",
        "Arousal rises smoothly (k = 0.5 matches real loud speech) but valence drifts up (+0.13 at k = 1.0, +0.27 at 1.25) while real anger, fear and distress go down: the 'manic' impression.",
    ),
    (
        "2.0",
        "Anger direction",
        "The same method with real anger clips as the loud side.",
        "About half the valence drift at the same arousal (+0.03 to +0.07 at an arousal gain of 0.4) with similar identity; one listener heard no manic state and no distortion, and higher k as controlled, rational urgency.",
    ),
]

with tab_jour:
    st.header("Everything we tried")
    st.dataframe(
        pd.DataFrame(JOURNEY, columns=["Version", "Step", "What we did", "What happened"]),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("""
**Not done.** A held-out evaluation on untouched test speakers; a second listener; a verified effort measure for synthetic speech; cross-checks on more than three speakers and eight lines.
""")

with tab_over:
    st.title("Controlling vocal effort in a cloned voice")
    st.markdown("""
**Goal.** Make a zero-shot text-to-speech system speak a line with *more vocal effort* (up to shouting) in the cloned speaker's own voice, with one continuous knob.

**Version 1.0, Chatterbox.** Its built-in *exaggeration* knob controls effort strongly, but the cloned identity and the speech quality degrade as it goes up. Fine-tuning it (LoRA) to separate effort from identity did not work (see *What did not work*).

**Version 2.0, Qwen3-TTS with activation steering** (two directions: one from real loud versus regular readings, one from real anger clips).** A direction in the model's hidden states, learned from real loud versus regular readings of 12 training speakers, is added during generation with strength **k**. Identity stays much closer to the unsteered clone,
and the effort rises smoothly. The loud-reading direction is useful to roughly k = 1.0 to 1.25: **above about 1.0 it drifts towards excited or manic delivery** (measured as rising valence), and from 1.5 the endings and the voice degrade. The **anger direction** (built from real anger clips) drifts about half as much at the same arousal and, to one listener, sounds like controlled, rational urgency instead of mania; arousal stops rising around k = 1.0 and a few clips garble above that.

**How to read this site.** Only clips that pass one common filter are playable (they stop on their own, are intelligible, keep the speaker's identity and are not duplicated). The *Metrics* tab shows how many clips passed at every setting, so the curation is visible.
These are development lines and three development speakers, one sample per setting, one listener so far: a demonstration of what was tried, not a held-out evaluation.
""")
    st.dataframe(
        met[met.group.isin(["qwen_dev", "qwen_anger", "chatterbox_dev"])][
            ["model", "setting", "n_total", "n_shown", "pass_rate", "effort_median", "sim_median"]
        ].rename(columns={"setting": "k or exaggeration"}),
        hide_index=True,
    )

with tab_cmp:
    c0, c1 = st.columns(2)
    spk = c0.selectbox("Speaker", sorted(shown.speaker.unique()))
    lines = (
        shown[shown.speaker == spk][["line_id", "text"]].drop_duplicates().sort_values("line_id")
    )
    lid = c1.selectbox(
        "Line",
        lines.line_id,
        format_func=lambda i: f"{i}: {lines[lines.line_id == i].text.iloc[0]}",
    )
    qdir = "loud-reading direction"  # used when the anger clips are not in the build
    if (shown.group == "qwen_anger").any():
        qdir = st.radio(
            "Qwen steering direction",
            ["anger direction", "loud-reading direction"],
            horizontal=True,
            help="The loud-reading direction comes from real loud versus regular readings; the anger direction from real scripted anger clips (less drift towards excited delivery at the same arousal).",
        )
    qgrp = "qwen_anger" if qdir.startswith("anger") else "qwen_dev"
    left, right = st.columns(2)
    for col, grp, name, lab in (
        (left, qgrp, "Qwen3-TTS + steering (" + qdir + ")", "steering strength k"),
        (right, "chatterbox_dev", "Chatterbox (stock)", "exaggeration"),
    ):
        d = shown[
            (shown.group == grp) & (shown.speaker == spk) & (shown.line_id == lid)
        ].sort_values("setting")
        with col:
            st.subheader(name)
            if d.empty:
                st.info("No clip passed the filter for this speaker and line.")
                continue
            s = st.select_slider(
                lab.capitalize(),
                options=list(d.setting),
                value=d.setting.iloc[0],
                key=grp + "_slider",
            )
            r = d[d.setting == s].iloc[0]
            play(r)
            caption(r)

with tab_lad:
    names = {
        "Qwen3-TTS + steering": "qwen_dev",
        "Qwen3-TTS + steering (anger direction)": "qwen_anger",
        "Chatterbox (stock)": "chatterbox_dev",
    }
    names = {n: g for n, g in names.items() if (shown.group == g).any()}
    model = st.radio("Model", list(names), horizontal=True)
    grp = names[model]
    c0, c1 = st.columns(2)
    spk2 = c0.selectbox("Speaker ", sorted(shown.speaker.unique()), key="spk2")
    ids = shown[(shown.group == grp) & (shown.speaker == spk2)].line_id.unique()
    lid2 = c1.selectbox("Line ", sorted(ids), key="lid2") if len(ids) else None
    if lid2:
        for _, r in (
            shown[(shown.group == grp) & (shown.speaker == spk2) & (shown.line_id == lid2)]
            .sort_values("setting")
            .iterrows()
        ):
            a, b = st.columns([1, 3])
            a.markdown(f"**{'k' if grp.startswith('qwen') else 'exaggeration'} = {r.setting:g}**")
            with b:
                play(r)
                caption(r)
        st.caption("Settings missing from the ladder did not pass the common filter for this clip.")

with tab_met:
    q = met[met.group == "qwen_dev"].set_index("setting")
    c = met[met.group == "chatterbox_dev"].set_index("setting")
    cf = met[met.group == "chatterbox_full"].set_index("setting")
    st.subheader("Qwen3-TTS steering: effort, identity and emotion against k")
    if len(q):
        st.line_chart(
            q[["effort_median", "sim_median"]].rename(
                columns={
                    "effort_median": "effort score (estimator; saturates near 20)",
                    "sim_median": "speaker similarity",
                }
            )
        )
        if q.arousal_median.notna().any():
            st.line_chart(q[["arousal_median", "valence_median"]].dropna(how="all"))
            st.caption(
                "Arousal and valence from a dimensional emotion model trained on natural speech (research use). For real EARS speech of the same speakers: regular 0.31 / 0.46, loud 0.51 / 0.46, anger 0.70 / 0.24, "
                "fear 0.48 / 0.19, distress 0.62 / 0.16, ecstasy 0.71 / 0.63 (arousal / valence). Steered clips keep rising in arousal while valence climbs from k about 1.0, towards ecstasy and away from anger or fear."
            )
    qa = met[met.group == "qwen_anger"].set_index("setting")
    if len(qa) and len(q) and q.valence_median.notna().any() and qa.valence_median.notna().any():
        st.subheader("Two Qwen steering directions: valence and arousal against k")
        st.line_chart(
            pd.concat(
                {
                    "valence, loud-reading direction": q.valence_median,
                    "valence, anger direction": qa.valence_median,
                    "arousal, loud-reading direction": q.arousal_median,
                    "arousal, anger direction": qa.arousal_median,
                },
                axis=1,
            )
        )
        st.caption(
            "At similar arousal the anger direction drifts less towards positive valence (excited delivery) than the loud-reading direction; neither moves towards the low valence of real anger, fear or distress."
        )
    st.subheader("Chatterbox (stock): effort and identity against exaggeration")
    if len(c):
        st.line_chart(
            c[["effort_median", "sim_median"]].rename(
                columns={"effort_median": "effort score", "sim_median": "speaker similarity"}
            )
        )
    st.subheader("Share of clips that pass the presentability filter")
    parts = {"Qwen3-TTS (dev set)": q.pass_rate, "Chatterbox (dev set)": c.pass_rate}
    if len(qa):
        parts["Qwen3-TTS, anger direction (dev set)"] = qa.pass_rate
    if len(cf):
        parts["Chatterbox (full baseline, 5 speakers x 30 lines)"] = cf.pass_rate
    st.bar_chart(pd.concat(parts, axis=1))
    st.caption(
        "The full Chatterbox baseline has three seeds at exaggeration 1.6 (450 clips), one seed elsewhere (150 clips). A clip passes if it "
        + FILT_TEXT
        + "."
    )
    st.dataframe(met.drop(columns=["group"]), hide_index=True)

with tab_neg:
    st.markdown("""
**Effort-conditioned fine-tuning of Chatterbox (LoRA, two attempts)** used the model's own exaggeration input, trained on 2 hours of EARS loud and regular speech. The result was a negative one: identity and intelligibility stayed good, but the control lost most of its effect
(median effort at the top setting stayed near regular speech while stock Chatterbox reached loud speech), and scalars above the training range did nothing. The clips below are the same lines at low, medium and high settings.
""")
    ft = shown[shown.group.isin(["chatterbox_ft_v0", "chatterbox_ft_v1"])]
    if ft.empty:
        st.info("Build with --include-finetune to add the fine-tune clips.")
    for (grp, spk3, lid3), d in ft.groupby(["group", "speaker", "line_id"]):
        st.markdown(f"**{d.model.iloc[0]}** · {spk3} · {lid3}: {d.text.iloc[0]}")
        cols = st.columns(len(d))
        for col, (_, r) in zip(cols, d.sort_values("setting").iterrows()):
            with col:
                st.caption(f"scalar {r.setting:g}")
                play(r)

    if "failed_audio" in man.columns:
        fails = man[(~man.shown) & man.failed_audio.fillna("").astype(str).ne("")].sort_values(
            ["group", "reason", "setting"]
        )
        if len(fails):
            st.subheader("Examples of clips the filter removed")
            st.caption(
                "The filter is strict and Whisper is less reliable on shouted speech, so some removed clips may be better than their scores suggest."
            )
            for _, r in fails.iterrows():
                st.markdown(
                    f"**{r.model}** · {r.speaker} · {r.line_id} · setting {r.setting:g}: {r.text}"
                )
                st.caption(
                    f"removed because: {r.reason} · effort {r.effort:.1f} · similarity {r.sim:.2f} · WER {r.wer:.2f}"
                )
                st.audio(audio_bytes(r.failed_audio, 0), format="audio/wav")

with tab_about:
    st.markdown("""
**Voices and licence.** The cloned voices are synthetic imitations of participants of the EARS dataset (Meta, CC BY-NC 4.0). They are used here for non-commercial research. Please do not reuse, redistribute or use these clips to impersonate anyone.
No minors' voices are used.

**Disclosure.** Every clip is synthetic. Chatterbox outputs carry the Perth watermark of Resemble AI. I have not verified whether Qwen3-TTS outputs are watermarked.

**Models.** Chatterbox (Resemble AI, MIT). Qwen3-TTS (Alibaba Qwen team, reported Apache-2.0). Whisper small.en for transcripts, WavLM for speaker similarity, an arousal / dominance / valence model (audEERING, research use only).
The effort score comes from an estimator trained on real EARS reading styles; it is not validated on synthetic speech and saturates near the level of real yelling.

**Limitations.** Development lines and three development speakers, one seed per setting, one listener; the held-out evaluation on untouched test speakers was not run. Qwen steering raises speech level strongly (about +25 dB at k = 1), so all clips here are level-matched.
Steering is not emotion-neutral: it amplifies the arousal the line implies, and above k = 1.0 it tends to sound excited or manic, even on a line that calls for fear.

**Related work.** Akti and Waibel (arXiv 2610.07647) steer vocal effort in Qwen3-TTS for speech in noise; this project differs in goal (acting-level delivery of a cloned voice), evaluation (identity, emotion, curation visible) and its negative fine-tuning result.
""")
