import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================
# Parameter für synthetische Daten
# ============================

START_DATE = "2025-05-10"   # Startdatum
N_WEEKS = 30                # 30 Wochen Cash-Forecast
SEED = 42                   # Reproduzierbarkeit

np.random.seed(SEED)

# ============================
# Hilfsfunktionen
# ============================

def generate_synthetic_transactions(start_date: str, n_weeks: int) -> pd.DataFrame:
    """
    Erzeugt synthetische Zahlungsströme (Ein- und Auszahlungen)
    mit Debitoren, Kreditoren, Zahlungszielen und wiederkehrenden Ausgaben.
    """
    start = datetime.fromisoformat(start_date)
    end = start + timedelta(weeks=n_weeks)

    dates = pd.date_range(start=start, end=end, freq="D")  # tägliche Basis

    records = []

    # Konfiguration: wie viele Zahlungen pro Tag?
    for current_date in dates:
        # Anzahl Debitoren-Einzahlungen (Kunden)
        n_in = np.random.poisson(lam=2)  # durchschnittlich 2 Eingänge pro Tag
        # Anzahl Kreditoren-Auszahlungen (Lieferanten, Miete, etc.)
        n_out = np.random.poisson(lam=3) # durchschnittlich 3 Ausgänge pro Tag

        # ---- Debitoren (Einzahlungen) ----
        for _ in range(n_in):
            amount = np.random.randint(2000, 15000)  # typische Kundenrechnung
            payment_terms = int(np.random.choice([14, 30, 45]))
            invoice_date = current_date - timedelta(days=payment_terms)

            records.append({
                "payment_date": current_date.date(),
                "direction": "IN",
                "counterparty_type": "Debitor",
                "category": "Kundenrechnung",
                "amount_eur": float(amount),
                "is_recurring": False,
                "payment_terms_days": payment_terms,
                "invoice_date": invoice_date.date()
            })

        # ---- Kreditoren (Auszahlungen) ----
        # 1) Wiederkehrende Ausgaben (Miete, Gehälter) an bestimmten Tagen
        # z.B. am Monatsanfang / Mitte
        if current_date.day in (1, 15):
            # Miete
            records.append({
                "payment_date": current_date.date(),
                "direction": "OUT",
                "counterparty_type": "Kreditor",
                "category": "Miete",
                "amount_eur": 8000.0,
                "is_recurring": True,
                "payment_terms_days": 0,
                "invoice_date": current_date.date()
            })
            # Gehälter
            records.append({
                "payment_date": current_date.date(),
                "direction": "OUT",
                "counterparty_type": "Kreditor",
                "category": "Gehalt",
                "amount_eur": 25000.0,
                "is_recurring": True,
                "payment_terms_days": 0,
                "invoice_date": current_date.date()
            })

        # 2) Variable Kreditoren (Lieferanten, sonstige)
        for _ in range(n_out):
            amount = np.random.randint(1000, 12000)
            category = np.random.choice(["Lieferant", "Sonstiges", "Marketing"])
            payment_terms = int(np.random.choice([14, 30]))
            invoice_date = current_date - timedelta(days=payment_terms)

            records.append({
                "payment_date": current_date.date(),
                "direction": "OUT",
                "counterparty_type": "Kreditor",
                "category": category,
                "amount_eur": float(amount),
                "is_recurring": False,
                "payment_terms_days": payment_terms,
                "invoice_date": invoice_date.date()
            })

    df = pd.DataFrame(records)

    # Sicherstellen, dass alles sauber sortiert ist
    df = df.sort_values("payment_date").reset_index(drop=True)
    return df


def aggregate_weekly_cashflow(transactions: pd.DataFrame,
                              initial_liquidity: float = 50000.0) -> pd.DataFrame:
    """
    Aggregiert Transaktionen auf Wochenebene.
    Ergebnis: wöchentlicher Cash-In, Cash-Out, Netto und kumulierte Liquidität.
    """
    # Richtung auf Vorzeichen mappen
    sign = transactions["direction"].map({"IN": 1, "OUT": -1})
    transactions["signed_amount"] = sign * transactions["amount_eur"]

    # Payment-Date als Datetime
    transactions["payment_date"] = pd.to_datetime(transactions["payment_date"])

    # Woche als Wochenbeginn (Montag) definieren
    transactions["week_start"] = transactions["payment_date"] - pd.to_timedelta(
        transactions["payment_date"].dt.weekday, unit="D"
    )

    weekly = transactions.groupby("week_start").agg(
        cash_in=("signed_amount", lambda x: x[x > 0].sum()),
        cash_out=("signed_amount", lambda x: -x[x < 0].sum())
    ).reset_index()

    weekly["net_cash"] = weekly["cash_in"] - weekly["cash_out"]

    # Kumulierte Liquidität
    weekly["cumulative_liquidity"] = initial_liquidity + weekly["net_cash"].cumsum()

    return weekly


# ============================
# Main: Daten erzeugen & speichern
# ============================

if __name__ == "__main__":
    print("Erzeuge synthetische Finanzdaten ...")

    transactions = generate_synthetic_transactions(START_DATE, N_WEEKS)
    print(f" {len(transactions)} Transaktionen erzeugt.")

    weekly_cashflow = aggregate_weekly_cashflow(transactions)
    print("Wöchentlicher Cashflow aggregiert.")

    # CSV speichern
    transactions.to_csv("transactions.csv", index=False)
    weekly_cashflow.to_csv("cashflow_weekly.csv", index=False)

    print("Dateien gespeichert:")
    print("   - transactions.csv")
    print("   - cashflow_weekly.csv")

    # Vorschau anzeigen
    print("\n Vorschau transactions.csv:")
    print(transactions.head())

    print("\n Vorschau cashflow_weekly.csv:")
    print(weekly_cashflow.head())
