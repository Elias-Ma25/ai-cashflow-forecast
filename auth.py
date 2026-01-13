import os
import streamlit as st


def require_login(app_name: str):
    """
    Erzwingt einen einfachen, sessionbasierten Login.
    Erwartet APP_PASSWORD als Umgebungsvariable.
    """
    APP_PASSWORD = os.getenv("APP_PASSWORD")

    # Falls kein Passwort gesetzt → kein Login notwendig
    if not APP_PASSWORD:
        return

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.set_page_config(page_title="Geschützte App", layout="centered")

        st.markdown(f"## 🔒 {app_name}")
        st.markdown("### Geschützte Demo-Anwendung")
        st.markdown("Bitte geben Sie das Passwort ein, um fortzufahren.")

        password = st.text_input("Passwort", type="password")
        login_clicked = st.button("Anmelden")

        if login_clicked:
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Falsches Passwort")

        st.stop()
