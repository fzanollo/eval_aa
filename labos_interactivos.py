"""
Labos de Evaluación de Sistemas de AA, versión interactiva.

Correr con:   streamlit run labos_interactivos.py
Necesita:     pip install streamlit matplotlib   (numpy y pandas vienen con streamlit)

Es una simulación propia en numpy (no usa la librería expected_cost), pensada para
entender los conceptos moviendo perillas. Reproduce las ideas de los notebooks pero
NO los números exactos: los datos simulados no son los mismos.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Labos de Evaluación de Sistemas de AA", layout="wide")

BLUE, RED, GREEN, GREY, ORANGE = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f", "#ff7f0e"


# ----------------------------------------------------------------------------
# Utilidades matemáticas
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def lse(a):
    m = a.max(axis=1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))


@st.cache_data
def sim_binary(N, P1, d, seed):
    """Binario con LLRs *calibrados*: LLR | clase 1 ~ N(+d²/2, d), LLR | clase 0 ~ N(-d²/2, d)."""
    rng = np.random.default_rng(seed)
    t = (rng.random(N) < P1).astype(int)
    llr = np.where(t == 1, d**2 / 2, -d**2 / 2) + d * rng.standard_normal(N)
    return t, llr


def ec_at_thresholds(t, s, taus, c01, c10, P1c):
    """Costo esperado NORMALIZADO al decidir 1 cuando el score s > tau.
    c01: costo de decidir 1 siendo clase 0. c10: costo de decidir 0 siendo clase 1.
    P1c: prior de la clase 1 usada para el costo (puede ser distinta a la de los datos)."""
    P0c = 1 - P1c
    s0, s1 = np.sort(s[t == 0]), np.sort(s[t == 1])
    R01 = 1 - np.searchsorted(s0, taus, side="right") / len(s0)  # clase 0 decidida como 1
    R10 = np.searchsorted(s1, taus, side="right") / len(s1)      # clase 1 decidida como 0
    ec = P0c * c01 * R01 + P1c * c10 * R10
    return ec / min(P0c * c01, P1c * c10)  # normalizado por el sistema "naive"


def bayes_thr(c01, c10, P1):
    """Umbral de Bayes sobre el LLR: decidir 1 si LLR > log(c01·P0 / (c10·P1))."""
    return float(np.log(c01 * (1 - P1) / (c10 * P1)))


def fit_affine(z, t, with_shift, iters=60):
    """Regresión logística lineal 1-D: q1 = sigmoid(w·z + c). Newton con backtracking."""
    def obj(w, c):
        a = w * z + c
        return np.mean(np.logaddexp(0, a) - t * a)

    w, c = 1.0, 0.0
    for _ in range(iters):
        p = sigmoid(w * z + c)
        gw, gc = np.mean((p - t) * z), np.mean(p - t)
        W = p * (1 - p)
        if with_shift:
            H = np.array([[np.mean(W * z * z), np.mean(W * z)], [np.mean(W * z), np.mean(W)]]) + 1e-10 * np.eye(2)
            step = np.linalg.solve(H, [gw, gc])
        else:
            step = np.array([gw / (np.mean(W * z * z) + 1e-10), 0.0])
        f0, k = obj(w, c), 1.0
        while k > 1e-6 and obj(w - k * step[0], c - k * step[1]) > f0:
            k /= 2
        w, c = w - k * step[0], c - k * step[1]
        if np.abs(k * step).max() < 1e-9:
            break
    return w, c


def fig_small(w=5.2, h=3.3):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def show(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def init_state(defaults):
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def set_state(**kw):
    st.session_state.update(kw)


# ----------------------------------------------------------------------------
st.title("Evaluación de sistemas de AA: laboratorios interactivos")
st.caption(
    "Cada pestaña tiene perillas a la izquierda y gráficos a la derecha. "
    "Abajo de cada una hay experimentos guiados: hacelos en orden y fijate qué cambia."
)
tabs = st.tabs([
    "Labo 1 · Datos",
    "Labo 2 · Costo esperado y priors",
    "Labo 3 · Reglas de decisión",
    "Labo 3 · Préstamos",
    "Labo 4 · Calibración y PSRs",
])


# ============================================================================
# LABO 1: solo metadata (no hay features.csv en este proyecto)
# ============================================================================
with tabs[0]:
    st.subheader("¿Qué distribución de clases hay dentro de cada grupo?")
    st.write(
        "Acá no se entrena nada (no hay `features.csv`): solo se mira la metadata. "
        "Sirve para detectar **correlaciones espurias** y **dependencia del hablante** antes de decidir cómo dividir los datos."
    )
    EMO = ["Anger", "Happiness", "Sadness", "Neutral state"]
    base = Path(__file__).parent

    def load(name):
        for p in (Path(name), base / name):
            if p.exists():
                return pd.read_csv(p, index_col=0)
        return None

    ds = st.radio("Dataset", ["IEMOCAP", "RAVDESS"], horizontal=True)
    df = load(f"{ds}_metadata.csv")
    if df is None:
        st.info(f"Poné `{ds}_metadata.csv` en la misma carpeta que este archivo para ver esta pestaña.")
    else:
        df = df.copy()
        df["clase"] = np.array(EMO)[np.argmax(df[EMO].to_numpy(), axis=1)]
        if ds == "IEMOCAP":
            df["hablante"] = df.session_number.astype(str) + df.utterance_gender  # 10 actores (deducido de la metadata)
            df["tarea"] = df.performing_id.str.extract(r"(script|impro)")[0]
            df["género"] = df.utterance_gender
            df["sesión"] = df.session_number.astype(str)
            groups = ["hablante", "tarea", "género", "sesión"]
        else:
            df["hablante"] = df.speaker.astype(str).str.zfill(2)
            df["género"] = df.gender
            df["intensidad"] = df.emotional_string
            df["frase"] = df.statement.astype(str)
            groups = ["hablante", "género", "intensidad", "frase"]
        g = st.selectbox("Agrupar por", groups)
        ct = pd.crosstab(df[g], df["clase"], normalize="index")[EMO]
        overall = df["clase"].value_counts(normalize=True)[EMO]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = fig_small(6.5, 0.45 * len(ct) + 1.6)
            left = np.zeros(len(ct))
            for emo, col in zip(EMO, [RED, ORANGE, BLUE, GREY]):
                ax.barh(ct.index.astype(str), ct[emo], left=left, label=emo, color=col)
                left += ct[emo].to_numpy()
            ax.set_xlim(0, 1); ax.set_xlabel("proporción de cada clase dentro del grupo")
            ax.invert_yaxis(); ax.legend(ncol=4, bbox_to_anchor=(0, 1.02), loc="lower left", frameon=False, fontsize=8)
            show(fig)
        with c2:
            st.metric("Baseline mayoritario (accuracy)", f"{overall.max():.1%}", help="Lo que logra un sistema que siempre dice la clase más frecuente.")
            st.write("Proporción global de clases")
            st.dataframe((overall * 100).round(1).rename("%").to_frame())

    st.markdown("**Experimentos guiados**")
    st.markdown(
        "- **IEMOCAP, agrupar por *tarea***: *anger* pesa mucho más en *script* que en *impro*. "
        "Si un modelo aprende 'estilo de habla = script → enojo' y además dividís al azar, el test sale optimista.\n"
        "- **IEMOCAP, agrupar por *hablante***: cada actor tiene su propia mezcla de clases. "
        "Con split aleatorio los mismos 10 hablantes están en train y en test; por eso hay que dividir por sesión u hablante.\n"
        "- **RAVDESS, agrupar por *intensidad***: *neutral* solo existe con intensidad normal (96 muestras vs 192 del resto). Fijate qué barra falta."
    )


# ============================================================================
# LABO 2: costo esperado, umbral de Bayes, priors que matchean o no
# ============================================================================
with tabs[1]:
    init_state(dict(l2_d=2.8, l2_P1=0.10, l2_c10=2.0, l2_P1c=0.50, l2_same=False, l2_pi=0.50, l2_seed=0))
    st.subheader("Costo esperado en función del umbral de decisión")
    st.write(
        "Sistema binario cuyos scores son **LLRs calibrados**. Se decide *clase 1* si LLR > umbral. "
        "El costo depende de **cuánto cuesta cada error** y de **con qué priors lo evaluás**."
    )
    ctrl, out = st.columns([1, 2.2])
    with ctrl:
        st.markdown("**Los datos**")
        d = st.slider("Separación entre clases (d′)", 0.5, 5.0, key="l2_d", step=0.1,
                      help="Más alto = problema más fácil. 2.8 se parece al histograma de Simulacion.pdf.")
        P1 = st.slider("Prior de la clase 1 en los datos", 0.02, 0.98, key="l2_P1", step=0.01)
        seed = st.number_input("Semilla", 0, 999, key="l2_seed")
        st.markdown("**El costo**")
        b1, b2 = st.columns(2)
        b1.button("EC1: costos 0-1, priors 0.5", on_click=set_state,
                  kwargs=dict(l2_c10=1.0, l2_P1c=0.5, l2_same=False))
        b2.button("EC2: c10=2, priors de datos", on_click=set_state,
                  kwargs=dict(l2_c10=2.0, l2_same=True))
        c10 = st.select_slider("Costo de perder una clase 1 (c10). El de falsa alarma c01 = 1",
                               options=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0], key="l2_c10")
        same = st.checkbox("Usar en el costo las mismas priors que tienen los datos", key="l2_same")
        P1c = P1 if same else st.slider("Prior de la clase 1 con la que se calcula el costo", 0.02, 0.98, key="l2_P1c", step=0.01)
        st.markdown("**La posterior (para decidir con Bayes)**")
        pi1 = st.slider("Prior que usó el sistema para construir su posterior", 0.02, 0.98, key="l2_pi", step=0.01,
                        help="La posterior P(clase|x) ya trae unas priors 'incorporadas'. Si no son las del costo, las decisiones son subóptimas.")

    t, llr = sim_binary(20000, P1, d, int(seed))
    th_cost = bayes_thr(1.0, c10, P1c)
    th_post = bayes_thr(1.0, c10, pi1)
    lim = max(d**2 / 2 + 3 * d + 1, abs(th_cost) + 2, abs(th_post) + 2)
    taus = np.linspace(-lim, lim, 801)
    ec = ec_at_thresholds(t, llr, taus, 1.0, c10, P1c)
    th_best = taus[np.argmin(ec)]
    v = lambda th: float(ec_at_thresholds(t, llr, np.array([th]), 1.0, c10, P1c)[0])
    ec_cost, ec_post, ec_best = v(th_cost), v(th_post), float(ec.min())

    with out:
        m1, m2, m3 = st.columns(3)
        m1.metric("EC con umbral de Bayes (priors del costo)", f"{ec_cost:.3f}", help=f"umbral = {th_cost:.2f}")
        m2.metric("EC decidiendo con la posterior", f"{ec_post:.3f}", delta=f"{ec_post - ec_cost:+.3f}", delta_color="inverse",
                  help=f"umbral implícito = {th_post:.2f}")
        m3.metric("EC mínimo posible en estos datos", f"{ec_best:.3f}", help=f"umbral óptimo empírico = {th_best:.2f}")
        f1, f2 = st.columns(2)
        with f1:
            fig, ax = fig_small()
            ax.plot(taus, ec, color="k")
            ax.axhline(1, color=GREY, ls=":", lw=1); ax.text(taus[-1], 0.93, "sistema naive = 1", color=GREY, fontsize=8, ha="right")
            ax.axvline(th_cost, color=BLUE, ls="--", label="Bayes (priors del costo)")
            ax.axvline(th_post, color=RED, ls="--", label="umbral implícito en la posterior")
            ax.axvline(th_best, color=GREEN, ls="-", lw=1.5, label="mejor umbral empírico")
            ax.set_ylim(0, min(2.5, max(1.3, ec.max()))); ax.set_xlabel("umbral sobre el LLR"); ax.set_ylabel("EC normalizado")
            ax.legend(fontsize=7, frameon=False, loc="upper left")
            show(fig)
        with f2:
            fig, ax = fig_small()
            bins = np.linspace(-lim, lim, 70)
            ax.hist(llr[t == 0], bins, density=True, alpha=0.6, color=BLUE, label="clase 0")
            ax.hist(llr[t == 1], bins, density=True, alpha=0.6, color=RED, label="clase 1")
            ax.axvline(th_cost, color=BLUE, ls="--"); ax.axvline(th_post, color=RED, ls="--"); ax.axvline(th_best, color=GREEN)
            ax.set_xlabel("LLR"); ax.legend(fontsize=8, frameon=False)
            show(fig)
        if abs(th_cost - th_post) < 0.05:
            st.success("La posterior usa las mismas priors que el costo → los dos umbrales coinciden y decidís de forma óptima.")
        else:
            st.warning(
                f"La posterior usa priors distintas a las del costo → umbral {th_post:.2f} en vez de {th_cost:.2f}. "
                f"Costo extra por el desajuste: {ec_post - ec_cost:+.3f} (normalizado)."
            )

    st.markdown("**Experimentos guiados**")
    st.markdown(
        "1. Apretá **EC1**: el umbral de Bayes queda en 0 (costos iguales y priors 0.5). Después **EC2**: se corre a ≈1.5. "
        "*El score es el mismo; cambió lo que consideramos un buen desempeño.*\n"
        "2. Con EC2, mové la prior de la posterior hasta igualar la de los datos: la raya roja se tapa con la azul.\n"
        "3. Apretá **EC1** y poné la prior de la posterior en la de los datos (0.10): el costo sube bastante (en mi simulación ≈0.17 → 0.30; en el notebook, 0.19 → 0.34). La posterior con priors 0.1 te empuja a decidir 'clase 0' casi siempre, pero el costo pedía priors 0.5.\n"
        "4. La raya verde (óptimo empírico) casi coincide con la azul porque los LLRs están calibrados por construcción. "
        "En el Labo 4 vas a romper eso a propósito."
    )


# ============================================================================
# LABO 3a: Naive vs Argmax vs Bayes con distintas matrices de costo
# ============================================================================
@st.cache_data
def sim_multi(K, P0, sigma, N, seed):
    rng = np.random.default_rng(seed)
    pri = np.array([P0] + [(1 - P0) / (K - 1)] * (K - 1))
    t = rng.choice(K, size=N, p=pri)
    x = np.eye(K)[t] + sigma * rng.standard_normal((N, K))
    lp = x / sigma**2 + np.log(pri)   # log-likelihoods + log-prior
    lp = lp - lse(lp)                 # log-posteriors verdaderas (calibradas)
    return t, lp, pri


with tabs[2]:
    init_state(dict(l3_K=10, l3_P0=0.80, l3_sig=0.40, l3_imp=100, l3_a1=0.05, l3_a2=0.30))
    st.subheader("Tres formas de decidir con las mismas posteriors (calibradas)")
    st.write(
        "**Naive**: siempre la mejor decisión constante. **Argmax**: la clase más probable. "
        "**Bayes**: la decisión que minimiza el costo esperado. Las posteriors son las mismas, cambia solo la regla."
    )
    ctrl, out = st.columns([1, 2.4])
    with ctrl:
        K = st.slider("Cantidad de clases", 2, 10, key="l3_K")
        P0 = st.slider("Prior de la clase 0", 0.2, 0.95, key="l3_P0", step=0.01)
        sig = st.slider("Ruido de los datos (σ)", 0.2, 1.0, key="l3_sig", step=0.05, help="Más alto = problema más difícil.")
        imp = st.select_slider("Cuánto más grave es errarle a la última clase", [1, 3, 10, 30, 100, 300], key="l3_imp")
        a1 = st.slider("Costo de abstenerse (1)", 0.01, 0.9, key="l3_a1", step=0.01)
        a2 = st.slider("Costo de abstenerse (2)", 0.01, 0.9, key="l3_a2", step=0.01)

    t, lp, pri = sim_multi(K, P0, sig, 10000, 0)
    q = np.exp(lp)
    N = len(t)
    base_c = 1 - np.eye(K)
    imp_m = base_c.copy(); imp_m[-1, :] *= imp
    costs = {
        "0-1": base_c,
        "inversa a las priors\n(balanced)": base_c / pri[:, None] / K,
        f"última clase\n×{imp}": imp_m,
        f"abstención\nα={a1:.2f}": np.hstack([base_c, np.full((K, 1), a1)]),
        f"abstención\nα={a2:.2f}": np.hstack([base_c, np.full((K, 1), a2)]),
    }
    p_emp = np.bincount(t, minlength=K) / N
    rows, nec = {}, {"Argmax": [], "Bayes": []}
    for name, C in costs.items():
        Cb = (p_emp @ C).min()
        decs = {
            "Naive": np.full(N, int(np.argmin(pri @ C))),
            "Argmax": q.argmax(1),
            "Bayes": (q @ C).argmin(1),
        }
        for rule, dd in decs.items():
            e = C[t, dd].mean()
            cell = f"{e:.3f} / {e / Cb:.2f}"
            if C.shape[1] > K:
                cell += f"  ({(dd == K).mean() * 100:.0f}% abst.)"
            rows.setdefault(rule, {})[name.replace("\n", " ")] = cell
            if rule != "Naive":
                nec[rule].append(e / Cb)

    with out:
        fig, ax = fig_small(8.2, 3.6)
        x = np.arange(len(costs)); w = 0.36
        cap = 3.0
        for i, (rule, col) in enumerate([("Argmax", ORANGE), ("Bayes", BLUE)]):
            vals = np.array(nec[rule])
            ax.bar(x + (i - 0.5) * w, np.minimum(vals, cap), w, label=rule, color=col)
            for xi, vi in zip(x + (i - 0.5) * w, vals):
                ax.text(xi, min(vi, cap) + 0.04, f"{vi:.2f}" if vi <= cap else f"{vi:.0f}↑", ha="center", fontsize=8)
        ax.axhline(1, color="k", ls="--", lw=1); ax.text(len(costs) - 0.45, 1.03, "naive = 1", ha="right", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(list(costs.keys()), fontsize=8)
        ax.set_ylabel("costo esperado normalizado"); ax.set_ylim(0, cap + 0.3); ax.legend(frameon=False, ncol=2)
        show(fig)
        st.markdown("**Tabla**: `EC / EC normalizado` (y % de abstenciones cuando hay esa opción)")
        st.dataframe(pd.DataFrame(rows).T[list(v.replace("\n", " ") for v in costs.keys())])
        naive_txt = ", ".join(
            f"{n.replace(chr(10), ' ')}: {'abstenerse' if int(np.argmin(pri @ C)) == K else 'clase ' + str(int(np.argmin(pri @ C)))}"
            for n, C in costs.items()
        )
        st.caption(f"Mejor decisión constante (Naive) → {naive_txt}")

    st.markdown("**Experimentos guiados**")
    st.markdown(
        "1. Con costo **0-1** Argmax y Bayes dan exactamente lo mismo: argmax solo es óptimo para ese costo.\n"
        "2. Subí *cuánto más grave es la última clase*: Argmax empeora sin límite y Bayes se adapta (decide esa clase con menos confianza).\n"
        "3. En **abstención**, Argmax nunca se abstiene: si su error supera α, su costo normalizado pasa de 1, o sea **peor que ignorar el input**. Bayes abstiene cuando la confianza es baja.\n"
        "4. Bajá α hasta que el Naive sea abstenerse siempre (100% en la tabla): por eso hace falta normalizar.\n"
        "5. Subí σ (más difícil) o cambiá la prior de la clase 0 y mirá cómo se corre la mejor decisión constante."
    )


# ============================================================================
# LABO 3b: ejemplo de préstamos, de utilidades a costos
# ============================================================================
with tabs[3]:
    init_state(dict(ln_beta=0.5, ln_aS=0.5, ln_aH=0.9, ln_g=0.4, ln_x=0.4))
    st.subheader("Elegir costos = modelar el problema (ejemplo de préstamos)")
    st.write(
        "Hay **2 clases** (el cliente paga / no paga) pero **3 decisiones**: denegar (D), ofrecer tasa alta (AH) u ofrecer tasa estándar (AS). "
        "La lectura de AH/AS sale de los nombres de variables del notebook. Se parte de **utilidades** del negocio y se convierten en costos."
    )
    ctrl, out = st.columns([1, 2.2])
    with ctrl:
        beta = st.slider("β: % de buenos pagadores que aceptan la tasa alta", 0.0, 1.0, key="ln_beta", step=0.05)
        aS = st.slider("αS: interés total pagado con tasa estándar / capital", 0.1, 1.5, key="ln_aS", step=0.05)
        aH = st.slider("αH: interés total pagado con tasa alta / capital", 0.1, 1.5, key="ln_aH", step=0.05)
        g = st.slider("γ: fracción de la deuda que paga un moroso", 0.0, 1.0, key="ln_g", step=0.05)
        x0 = st.slider("Probabilidad de default de un cliente concreto", 0.0, 1.0, key="ln_x", step=0.01)

    U = np.array([[0, beta * aH, aS], [0, g * (1 + aH) - 1, g * (1 + aS) - 1]])
    C = -U
    C = C - C.min(axis=1, keepdims=True)  # estandarización: un cero en cada fila
    names = ["D (denegar)", "AH (tasa alta)", "AS (tasa estándar)"]
    xs = np.linspace(0, 1, 1001)
    cost_lines = np.outer(1 - xs, C[0]) + np.outer(xs, C[1])
    best = cost_lines.argmin(1)
    cuts = [0] + [i for i in range(1, len(xs)) if best[i] != best[i - 1]] + [len(xs) - 1]
    regions = [(xs[cuts[k]], xs[cuts[k + 1]], best[cuts[k]]) for k in range(len(cuts) - 1)]
    cols = [BLUE, RED, GREEN]

    with out:
        fig, ax = fig_small(6.6, 3.8)
        for j in range(3):
            ax.plot(xs, cost_lines[:, j], color=cols[j], label=names[j])
        for lo, hi, j in regions:
            ax.axvspan(lo, hi, color=cols[j], alpha=0.12)
        ax.axvline(x0, color="k", ls=":", lw=1.2)
        ax.set_xlabel("probabilidad de default"); ax.set_ylabel("costo esperado de cada decisión")
        ax.set_xlim(0, 1); ax.set_ylim(0, max(0.5, C.max() * 1.05)); ax.legend(frameon=False, fontsize=8)
        show(fig)
        st.markdown("**Regiones de decisión de Bayes**: " + " · ".join(
            f"{names[j]} si P(default) entre {lo:.2f} y {hi:.2f}" for lo, hi, j in regions))
        cx = C[0] * (1 - x0) + C[1] * x0
        st.info(f"Para P(default) = {x0:.2f}: costos D={cx[0]:.3f}, AH={cx[1]:.3f}, AS={cx[2]:.3f} → la decisión óptima es **{names[int(cx.argmin())]}**.")
        t1, t2 = st.columns(2)
        with t1:
            st.write("Utilidad (filas: clase real)")
            st.dataframe(pd.DataFrame(U, index=["Paga", "No paga"], columns=["D", "AH", "AS"]).round(3))
        with t2:
            st.write("Costo estandarizado = −U − mínimo de cada fila")
            st.dataframe(pd.DataFrame(C, index=["Paga", "No paga"], columns=["D", "AH", "AS"]).round(3))

    st.markdown("**Experimentos guiados**")
    st.markdown(
        "1. Valores por defecto: ofrecés tasa estándar si el riesgo es bajo, tasa alta en el medio y denegás si es muy alto. "
        "Como una decisión de Bayes es el mínimo de rectas, las regiones se cortan en umbrales sobre P(default).\n"
        "2. Bajá **β**: cada vez menos gente acepta la tasa alta y la región AH se achica hasta desaparecer.\n"
        "3. Subí **γ**: los morosos devuelven más, así que denegar deja de ser tan necesario y esa región se achica.\n"
        "4. Bajá **αH** hasta acercarlo (o pasarlo por debajo) de **αS**: la tasa alta deja de compensar el riesgo de que rechacen la oferta y la región AH desaparece."
    )


# ============================================================================
# LABO 4: calibración, PSRs
# ============================================================================
with tabs[4]:
    init_state(dict(l4_d=2.8, l4_P1=0.10, l4_a=1.0, l4_b=0.0, l4_pr="correctas", l4_s2=1.0, l4_alpha=4.0, l4_abs=0.10, l4_seed=0))
    st.subheader("Los mismos scores, distinta calibración")
    st.write(
        "Se sigue el diagrama de `Simulacion.pdf`: LLR verdadero → **escala/shift** (mc1) → se le suman las **priors** (correctas o incorrectas) → "
        "se multiplican los log-posteriors por un factor (**mc2**, sobreconfianza). Abajo se compara el sistema crudo con versiones calibradas."
    )
    ctrl, out = st.columns([1, 2.6])
    with ctrl:
        st.markdown("**Presets**")
        p1c, p2c = st.columns(2)
        p1c.button("Calibrado", on_click=set_state, kwargs=dict(l4_a=1.0, l4_b=0.0, l4_pr="correctas", l4_s2=1.0))
        p2c.button("Tipo mc1", on_click=set_state, kwargs=dict(l4_a=0.5, l4_b=-0.5, l4_pr="correctas", l4_s2=1.0))
        p3c, p4c = st.columns(2)
        p3c.button("Tipo mc2 (×10)", on_click=set_state, kwargs=dict(l4_a=1.0, l4_b=0.0, l4_pr="correctas", l4_s2=10.0))
        p4c.button("Priors incorrectas", on_click=set_state, kwargs=dict(l4_a=1.0, l4_b=0.0, l4_pr="invertidas", l4_s2=1.0))
        st.markdown("**Datos**")
        d4 = st.slider("Separación (d′)", 0.5, 5.0, key="l4_d", step=0.1)
        P1_4 = st.slider("Prior de la clase 1 en los datos", 0.02, 0.5, key="l4_P1", step=0.01)
        seed4 = st.number_input("Semilla ", 0, 999, key="l4_seed")
        st.markdown("**Cómo se rompe la calibración**")
        a4 = st.slider("Escala del LLR (a)", 0.2, 3.0, key="l4_a", step=0.1)
        b4 = st.slider("Shift del LLR (b)", -3.0, 3.0, key="l4_b", step=0.1)
        pr4 = st.radio("Priors que usa el sistema", ["correctas", "invertidas", "uniformes"], key="l4_pr", horizontal=True)
        s24 = st.slider("Factor sobre los log-posteriors (sobreconfianza)", 1.0, 20.0, key="l4_s2", step=0.5)
        st.markdown("**Costos a evaluar**")
        al4 = st.select_slider("Costo asimétrico: perder la clase 1 cuesta α veces más",
                               [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0], key="l4_alpha")
        ab4 = st.slider("Costo de abstenerse", 0.02, 0.6, key="l4_abs", step=0.01)

    t, llr = sim_binary(10000, P1_4, d4, int(seed4))
    Pe1 = t.mean()
    pi_used = {"correctas": P1_4, "invertidas": 1 - P1_4, "uniformes": 0.5}[pr4]
    z_raw = s24 * (a4 * llr + b4 + logit(pi_used))       # log-odds finales del sistema
    z_ideal = llr + logit(P1_4)                           # posterior verdadera
    w_s, _ = fit_affine(z_raw, t, with_shift=False)
    w_p, c_p = fit_affine(z_raw, t, with_shift=True)
    versions = {
        "Crudo": z_raw,
        "Calibrado: solo escala": w_s * z_raw,
        "Calibrado: escala + shift": w_p * z_raw + c_p,
        "Ideal (posterior verdadera)": z_ideal,
    }

    def nec_from_z(z, kind):
        q1 = sigmoid(z); q0 = 1 - q1
        if kind == "01":
            dec = (z > 0).astype(int)
            e = np.mean(dec != t); Cb = min(Pe1, 1 - Pe1); return e / Cb, 0.0
        if kind == "asym":
            dec = (z > -np.log(al4)).astype(int)     # decidir 1 si α·q1 > q0
            e = np.mean(np.where(t == 1, al4 * (dec == 0), 1.0 * (dec == 1)))
            Cb = min(Pe1 * al4, 1 - Pe1); return e / Cb, 0.0
        if kind == "abs":
            exp0, exp1 = q1, q0                      # costo esperado de decidir 0 / decidir 1
            dec = np.where(np.minimum(exp0, exp1) <= ab4, (exp1 < exp0).astype(int), 2)
            cost = np.where(dec == 2, ab4, (dec != t).astype(float))
            Cb = min(Pe1, 1 - Pe1, ab4); return cost.mean() / Cb, float((dec == 2).mean() * 100)
        if kind == "ce":
            lq1, lq0 = -np.logaddexp(0, -z), -np.logaddexp(0, z)
            ce = -np.mean(t * lq1 + (1 - t) * lq0)
            H = -(Pe1 * np.log(Pe1) + (1 - Pe1) * np.log(1 - Pe1)); return ce / H, 0.0
        if kind == "brier":
            return np.mean(2 * (q1 - t) ** 2) / (2 * Pe1 * (1 - Pe1)), 0.0

    labels = {"01": "Costo 0-1 (decisiones de Bayes)", "asym": f"Costo asimétrico α={al4:g}",
              "abs": f"Costo con abstención {ab4:.2f}", "ce": "Cross-entropy (PSR estricta)", "brier": "Brier (PSR estricta)"}
    table = {v: [] for v in versions}
    for kind in labels:
        for v, z in versions.items():
            val, pabs = nec_from_z(z, kind)
            table[v].append(f"{val:.3f}" + (f" ({pabs:.0f}% abst.)" if kind == "abs" else ""))
    df_m = pd.DataFrame(table, index=list(labels.values()))

    with out:
        st.markdown("**Todas las métricas están normalizadas por el sistema que solo usa las priors: 1 = igual que ignorar el input, >1 = peor.**")
        st.dataframe(df_m)
        f1, f2, f3 = st.columns(3)
        with f1:
            fig, ax = fig_small(4.4, 3.2)
            lo, hi = np.percentile(z_raw, [0.5, 99.5]); bins = np.linspace(lo, hi, 60)
            ax.hist(z_raw[t == 0], bins, density=True, alpha=0.6, color=BLUE, label="clase 0")
            ax.hist(z_raw[t == 1], bins, density=True, alpha=0.6, color=RED, label="clase 1")
            ax.axvline(0, color="k", ls="--", lw=1); ax.set_title("log-odds del sistema (crudo)", fontsize=9)
            ax.legend(fontsize=7, frameon=False); show(fig)
        with f2:
            fig, ax = fig_small(4.4, 3.2)
            ax.plot([0, 1], [0, 1], color="k", ls=":", lw=1)
            edges = np.linspace(0, 1, 11)
            for (name, z), col in zip(list(versions.items())[:3], [RED, ORANGE, GREEN]):
                qq = sigmoid(z); idx = np.digitize(qq, edges[1:-1])
                xs_, ys_, ns_ = [], [], []
                for k in range(10):
                    m = idx == k
                    if m.sum() >= 15:
                        xs_.append(qq[m].mean()); ys_.append(t[m].mean()); ns_.append(m.sum())
                ax.plot(xs_, ys_, "-o", color=col, ms=4, label=name, alpha=0.9)
            ax.set_xlabel("probabilidad que dice el sistema"); ax.set_ylabel("fracción real de clase 1")
            ax.set_title("calibración (más cerca de la diagonal = mejor)", fontsize=9); ax.legend(fontsize=6.5, frameon=False); show(fig)
        with f3:
            fig, ax = fig_small(4.4, 3.2)
            las = np.exp(np.arange(-4, 4.01, 0.5))
            qs = np.quantile(z_raw, np.linspace(0, 1, 401))
            curves = {"MAP (umbral 0.5)": [], "Bayes, crudo": [], "Bayes, calibrado": [], "Óptimo (umbral 'tramposo')": []}
            for a_ in las:
                Cb_ = min(Pe1 * a_, 1 - Pe1)
                def ecz(z, tau):
                    return float(ec_at_thresholds(t, z, np.array([tau]), 1.0, a_, Pe1)[0])
                curves["MAP (umbral 0.5)"].append(ecz(z_raw, 0.0))
                curves["Bayes, crudo"].append(ecz(z_raw, -np.log(a_)))
                zc = w_p * z_raw + c_p
                curves["Bayes, calibrado"].append(ecz(zc, -np.log(a_)))
                curves["Óptimo (umbral 'tramposo')"].append(float(ec_at_thresholds(t, z_raw, qs, 1.0, a_, Pe1).min()))
            for (name, vals), col in zip(curves.items(), [RED, ORANGE, GREEN, "k"]):
                ax.plot(np.log(las), np.minimum(vals, 3), color=col, label=name, lw=1.6 if name != "Óptimo (umbral 'tramposo')" else 1, ls="-" if name != "Óptimo (umbral 'tramposo')" else "--")
            ax.axhline(1, color=GREY, ls=":", lw=1); ax.axvline(np.log(al4), color="k", lw=0.6, alpha=0.4)
            ax.set_xlabel("log α (costo de perder la clase 1)"); ax.set_ylabel("EC normalizado (tope 3)")
            ax.set_title("descalibración según el punto de operación", fontsize=9); ax.legend(fontsize=6.5, frameon=False); show(fig)

        with st.expander("Log loss vs Brier vs costo 0-1 (por qué la cross-entropy castiga la sobreconfianza)"):
            qgrid = np.linspace(0.001, 0.999, 500)
            fig, ax = fig_small(6, 3.2)
            ax.plot(qgrid, -np.log(qgrid), color=BLUE, label="cross-entropy: −log q")
            ax.plot(qgrid, 2 * (1 - qgrid) ** 2, color=ORANGE, label="Brier: 2(1−q)²")
            ax.plot(qgrid, (qgrid < 0.5) * 2.0, color=GREY, ls="--", label="0-1 (×2, solo mira si q<0.5)")
            ax.set_ylim(0, 5); ax.set_xlabel("q = probabilidad que el sistema le dio a la clase correcta"); ax.set_ylabel("costo")
            ax.legend(frameon=False, fontsize=8); show(fig)
            st.caption("Cuando q→0 (sistema seguro de algo incorrecto) el log loss se va a infinito; el Brier queda acotado y el 0-1 ni distingue 0.4 de 0.01.")

    st.markdown("**Experimentos guiados**")
    st.markdown(
        "1. Con **Calibrado**, las cuatro columnas son prácticamente iguales (las diferencias chiquitas son ruido de muestreo): no hay nada que mejorar calibrando.\n"
        "2. **Tipo mc2 (×10)**: el costo 0-1 *no cambia* (multiplicar por un positivo no mueve el signo) pero cross-entropy explota. "
        "El costo 0-1 es una PSR no estricta: no ve la calibración. La cross-entropy sí. "
        "En el gráfico del medio, las probabilidades quedan pegadas a 0 y 1 lejos de la diagonal. Calibrar con **solo escala** lo arregla.\n"
        "3. **Priors incorrectas**: el costo 0-1 pasa de 1, o sea peor que ignorar el sistema. "
        "Calibrar con **solo escala no alcanza** (la escala no corrige un corrimiento), pero **escala + shift** sí (clase 6, p. 37).\n"
        "4. **Tipo mc1** con distintos α: en el gráfico de la derecha, la brecha entre *Bayes crudo* y *Bayes calibrado* depende del punto de operación; "
        "a veces es casi nula. Esa es la descalibración *en ese punto*.\n"
        "5. La curva 'Óptimo' elige el umbral mirando los datos de evaluación, por eso es optimista (clase 1, p. 26)."
    )
