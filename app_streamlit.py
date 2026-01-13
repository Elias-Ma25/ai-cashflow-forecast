import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from auth import require_login

require_login("KI-gestützter 13-Wochen-Cash-Forecast")

# ----------------------------------------------------
# 1. Setup (Seite konfigurieren)
# ----------------------------------------------------
st.set_page_config(
    page_title="KI-gestützter 13-Wochen-Cash-Forecast",
    page_icon="🤖",
    layout="wide"
)

PRIMARY_COLOR = "#0055A4"  # CFO-Blau
ACCENT_COLOR = "#0099E6"  # Blau-Ton für Forecast
NEG_COLOR = "#D9534F"  # Rot
POS_COLOR = "#5CB85C"  # Grün

st.title("💻KI-gestützter 13-Wochen-Cash-Forecast für CFO-Mandate")
st.markdown("""
Dieses Dashboard zeigt die synthetische Liquiditätsentwicklung und ermöglicht KI-unterstützte Analysen.
---
""")


# ----------------------------------------------------
# 2. Daten laden aus Arbeitspaket 1 und 2
# ----------------------------------------------------
@st.cache_data
def load_data():
    df_cash = pd.read_csv("cashflow_weekly.csv")
    df_cash["week_start"] = pd.to_datetime(df_cash["week_start"])

    df_forecast = pd.read_csv("forecast_13weeks.csv")
    df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])

    return df_cash, df_forecast


cash_df, forecast_df = load_data()

# Kalenderwochen vorbereiten
cash_df["kw"] = cash_df["week_start"].dt.isocalendar().week
forecast_df["kw"] = forecast_df["ds"].dt.isocalendar().week

# ----------------------------------------------------
# 3. Visualisierung Forecast + IST + Unsicherheitsband
# ----------------------------------------------------
st.subheader("🔹 Cashflow – Historisch & Forecast (13 Wochen)")

fig = px.line()

# Historische Werte
fig.add_scatter(
    x=cash_df["week_start"],
    y=cash_df["net_cash"],
    mode="lines",
    name="IST – Netto Cashflow"
)

# Forecast
fig.add_scatter(
    x=forecast_df["ds"],
    y=forecast_df["yhat"],
    mode="lines",
    name="Forecast – Netto Cashflow"
)

# Unsicherheitsband
fig.add_scatter(
    x=forecast_df["ds"],
    y=forecast_df["yhat_upper"],
    mode="lines",
    line=dict(width=0),
    showlegend=False
)

fig.add_scatter(
    x=forecast_df["ds"],
    y=forecast_df["yhat_lower"],
    fill="tonexty",
    mode="lines",
    line=dict(width=0),
    name="Unsicherheitsintervall"
)

fig.update_layout(
    height=500,
    xaxis_title="Datum",
    yaxis_title="Netto-Cashflow (€)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------
# 4. Szenarioanalyse für Arbeitspaket 4
# ----------------------------------------------
st.subheader("📐 Szenario- & Sensitivitätsanalyse (Was-wäre-wenn)")

st.markdown(
    "Passen Sie die Parameter an und analysieren Sie die Auswirkungen auf Cashflow und Liquidität."
)

col1, col2, col3 = st.columns(3)

umsatz_change = col1.slider("Umsatzänderung (%)", -20, 20, 0)
capex_change = col2.slider("CAPEX-/Kostenänderung (%)", -20, 20, 0)
dso_change = col3.slider("DSO-Veränderung (Tage)", -10, 10, 0)

if st.button("Szenario berechnen"):
    # Basisdaten kopieren
    scen_df = cash_df.copy()

    # 1) Umsatz wirkt auf Einzahlungen
    scen_df["cash_in_szen"] = scen_df["cash_in"] * (1 + umsatz_change / 100)

    # 2) CAPEX/Kosten wirkt auf Auszahlungen
    scen_df["cash_out_szen"] = scen_df["cash_out"] * (1 + capex_change / 100)

    # 3) DSO-Veränderung wirkt auf Einzahlungsvolumen (vereinfachtes Modell)
    scen_df["cash_in_szen"] = scen_df["cash_in_szen"] * (1 - dso_change * 0.01)
    # Neuer Netto-Cash
    scen_df["net_cash_szen"] = scen_df["cash_in_szen"] - scen_df["cash_out_szen"]

    # Kumulative Liquidität Szenario:
    base_start_liq = float(
        cash_df["cumulative_liquidity"].iloc[0] - cash_df["net_cash"].iloc[0]
    )
    scen_df["cum_liq_szen"] = base_start_liq + scen_df["net_cash_szen"].cumsum()

    # Szenarien speichern
    st.session_state["scen_df"] = scen_df
    st.session_state["scenario_active"] = True
    st.session_state["scenario_context"] = (
        f"Umsatz: {umsatz_change} %, "
        f"Kosten/CAPEX: {capex_change} %, "
        f"DSO: {dso_change} Tage."
    )

    # --- Plot 1: Netto-Cash Original vs. Szenario ---
    fig_scen_cash = px.line()
    fig_scen_cash.add_scatter(
        x=cash_df["week_start"],
        y=cash_df["net_cash"],
        mode="lines",
        name="IST – Netto-Cashflow",
    )
    fig_scen_cash.add_scatter(
        x=scen_df["week_start"],
        y=scen_df["net_cash_szen"],
        mode="lines",
        name="Szenario – Netto-Cashflow",
    )
    fig_scen_cash.update_layout(
        height=400,
        xaxis_title="Datum",
        yaxis_title="Netto-Cashflow (€)",
        template="plotly_white",
        title="Vergleich Netto-Cashflow: IST vs. Szenario",
    )
    st.plotly_chart(fig_scen_cash, use_container_width=True)

    # --- Plot 2: kumulative Liquidität Original vs. Szenario ---
    fig_scen_liq = px.line()
    fig_scen_liq.add_scatter(
        x=cash_df["week_start"],
        y=cash_df["cumulative_liquidity"],
        mode="lines",
        name="IST – kumulative Liquidität",
    )
    fig_scen_liq.add_scatter(
        x=scen_df["week_start"],
        y=scen_df["cum_liq_szen"],
        mode="lines",
        name="Szenario – kumulative Liquidität",
    )
    fig_scen_liq.update_layout(
        height=400,
        xaxis_title="Datum",
        yaxis_title="kumulative Liquidität (€)",
        template="plotly_white",
        title="Vergleich kumulative Liquidität: IST vs. Szenario",
    )
    st.plotly_chart(fig_scen_liq, use_container_width=True)

    # --- Sensitivitätsübersicht (vereinfachte KPI-Tabelle) ---
    base_final_liq = float(cash_df["cumulative_liquidity"].iloc[-1])
    scen_final_liq = float(scen_df["cum_liq_szen"].iloc[-1])
    diff_final = scen_final_liq - base_final_liq

    sens_data = pd.DataFrame([
        {
            "Parameter": "Umsatz",
            "Änderung": f"{umsatz_change} %",
            "Kommentar": "Höhere Einzahlungen bei positivem Wert",
        },
        {
            "Parameter": "CAPEX / Kosten",
            "Änderung": f"{capex_change} %",
            "Kommentar": "Höhere Auszahlungen bei positivem Wert",
        },
        {
            "Parameter": "DSO (Zahlungsziele)",
            "Änderung": f"{dso_change} Tage",
            "Kommentar": "Schnellere Zahlungseingänge bei negativer Zahl",
        },
        {
            "Parameter": "Kumulative Liquidität (Ende)",
            "Änderung": f"{diff_final:,.0f} €".replace(",", "."),
            "Kommentar": "Verbesserung/Verschlechterung gegenüber IST",
        },
    ])

    st.markdown("**📊 Sensitivitätsübersicht zum Szenario**")
    st.dataframe(sens_data, use_container_width=True)

    # Szenario-Kontext für LLM speichern (für Arbeitspaket 3+4)
    scen_context = (
        f"Szenarioparameter: Umsatzänderung = {umsatz_change} %, "
        f"CAPEX-/Kostenänderung = {capex_change} %, "
        f"DSO-Veränderung = {dso_change} Tage. "
        f"Veränderung der kumulativen Liquidität am Periodenende: {diff_final:,.0f} €."
    )
    st.session_state["scenario_context"] = scen_context

    st.info(
        "Szenario berechnet. Die Auswirkungen auf Netto-Cashflow und kumulative "
        "Liquidität sind in den Diagrammen und der Tabelle ersichtlich."
    )
else:
    # Falls noch kein Szenario berechnet wurde, Kontext leeren
    pass  # NICHTS machen – Szenario bleibt aktiv
   # st.session_state["scenario_context"] = None


# ----------------------------------------------------
# 5. LLM-Funktion (GPT-4o-mini Integration)
# ----------------------------------------------------
def extract_kw(question: str):
    """Extrahiert eine Kalenderwoche aus einer Frage."""
    match = re.search(r"\bkw\s*(\d{1,2})\b", question.lower())
    if match:
        return int(match.group(1))
    return None

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None

def ask_llm(question: str):
    """Automatische Analyse: IST oder Szenario-basiert."""

    scenario_active = st.session_state.get("scenario_active", False)

    df_used = (
        st.session_state["scen_df"].copy()
        if scenario_active else
        cash_df.copy()
    )

    mode = "Szenario" if scenario_active else "IST"

    scen_text = st.session_state.get("scenario_context", "Kein Szenario aktiv")

    # Formatting
    df_fmt = df_used.copy()
    for col in ["cash_in", "cash_out", "net_cash", "cumulative_liquidity"]:
        if col in df_fmt:
            df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:,.0f} €".replace(",", "."))

    df_preview = df_fmt.to_string(index=False)
    forecast_preview = forecast_df.to_string(index=False)

    prompt = f"""
    Du bist ein CFO-Experte für Liquiditätsplanung.
    Nutze die folgenden Daten, um die Frage des Users professionell zu beantworten.

    WICHTIG – Interpretation der Szenario-Parameter:
    - Umsatzänderung (%): Positive Werte = mehr Einzahlungen; negative = weniger Einzahlungen.
    - CAPEX/Kosten (%): Positive Werte = höhere Auszahlungen; negative Werte = Kostensenkung.
    - DSO-Veränderung (Tage): Positive Werte = spätere Zahlungseingänge (Cash-In reduziert sich);
        negative Werte = schneller Zahlungseingänge (Cash-In erhöht sich).

    1) MODUS: {"Szenario AKTIV" if scenario_active else "IST-Modus"}
       - Bei aktivem Szenario: Cashflow-Werte sind modifiziert
       - Forecast basiert IMMER auf IST-Daten, nicht auf Szenario

    2) VERWENDETE DATEN ({'Szenario' if scenario_active else 'IST'}):
    {df_preview}

    3) FORECAST (nächste 13 Wochen, immer IST-basiert):
    {forecast_preview}

    4) SZENARIO-KONTEXT (falls aktiv):
    {scen_text}

    FRAGE: "{question}"

    ANALYSE-REGELN:

    A) WENN FRAGE KALENDERWOCHE (KW) ENTHÄLT:
       - Führe spezifische Analyse dieser KW durch
       - Verwende historische Daten dieser KW
       - Vergleiche mit Vorwochen/Durchschnitt

    B) WENN SZENARIO AKTIV:
       - Beginne mit: "Im berechneten Szenario mit [Parameter]..."
       - Beschreibe EXPLIZIT Abweichungen gegenüber IST
       - Interpretiere Wirkung auf Netto-Cashflow UND kumulative Liquidität
       - Nenne größte Auswirkungen auf kritische Wochen

    C) WENN ALLGEMEINE/TRENDBEZOGENE FRAGE:
       - Analysiere kausal und konzeptionell
       - Identifiziere Haupttreiber (DSO, CAPEX, Umsatz)
       - Nutze historische Daten UND Forecast

    ANTWORT-ANFORDERUNGEN:
    - Professionelles Business-Deutsch
    - Kompakt (6-10 Sätze)
    - Strukturierte CFO-Analyse
    - Erklärung der wichtigsten Liquiditätstreiber
    - Klare Handlungsempfehlungen (CFO-Level)
    - Nutze immer: historische Daten + Forecast

    WICHTIGE EINSCHRÄNKUNGEN:
    - Nenne keine absoluten Euro-Werte, außer:
      * Sie sind explizit im Szenario-Kontext angegeben ODER
      * Der User fragt explizit danach (z.B. "Wie viel genau?")
    - Erfinde keine Zahlen oder Sachverhalte
    - Analysiere nur die Tabellendaten (keine Diagramme)
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700
    )

    return completion.choices[0].message.content

# ----------------------------------------------------
# 6. Chat-Interface mit echtem LLM (AP3+AP4)
# ----------------------------------------------------
st.subheader("🤖 KI-Chat – CFO-Analyse (IST + Szenario)")

user_input = st.text_input("Ihre Frage an die KI:")

analysis_box = st.container()

with analysis_box:
    # Validierung
    if st.button("Analyse starten"):
        if not user_input.strip():
            st.error("Bitte geben Sie eine Frage ein.")
            st.stop()

        if client is None:
            st.error("Kein API-Key gesetzt.")
            st.stop()

        kw = extract_kw(user_input)
        if kw and kw not in cash_df["kw"].values:
            st.warning("🔒 KI-Funktion ist deaktiviert (kein API-Key gesetzt).")
            st.stop()
	
        # Analyse durchführen
        with st.spinner("KI analysiert die Daten ..."):
            answer = ask_llm(user_input)
            if kw:
                answer = f"### Analyse für KW {kw}\n\n" + answer

        st.success(answer)

# ----------------------------------------------------
# 7. Footer
# ----------------------------------------------------
st.markdown("------------------------------------")
st.caption("Prototyp – KI-gestützte Liquiditätsplanung für CFO-Mandate © Elias")
