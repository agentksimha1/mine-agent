import streamlit as st
import requests
import asyncio

# ------------------------------
# CONFIG
# ------------------------------
API_URL = "http://localhost:8000"  # Change if deployed

st.set_page_config(
    page_title="Digital Mine Safety Officer",
    page_icon="⛏️",
    layout="wide"
)

st.title("⛏️ Digital Mine Safety Officer")
st.write("AI-powered mining safety assistant with RAG, DGMS updates & audit reporting.")

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
section = st.sidebar.radio(
    "Select Section",
    ["💬 Chat with AI", "📰 DGMS Updates", "📄 Audit Report PDF"]
)

# ============================================================
# SECTION 1 — CHAT WITH AI (RAG QUERY)
# ============================================================
if section == "💬 Chat with AI":
    st.header("💬 Ask Mining Safety Questions")
    query = st.text_input("Enter your question:", "")

    async def ask_query_async(query_text):
        try:
            response = await asyncio.to_thread(
                lambda: requests.post(f"{API_URL}/query", json={"query": query_text})
            )
            if response.status_code == 200:
                # Backend automatically handles L1/L2 caching
                return response.json().get("response", "")
            return "Server error."
        except Exception as e:
            return f"⚠️ Error: {e}"

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                response_text = asyncio.run(ask_query_async(query))
                st.success(response_text)
                st.info("⚡ Response served from cache if repeated query (backend L1/L2 cache)")

# ============================================================
# SECTION 2 — LIVE DGMS / MINING UPDATES
# ============================================================
elif section == "📰 DGMS Updates":
    st.header("📰 Recent Mining Updates & Risk Classification")

    async def fetch_updates_async():
        try:
            response = await asyncio.to_thread(lambda: requests.get(f"{API_URL}/updates"))
            if response.status_code == 200:
                return response.json().get("updates", [])
            return []
        except Exception as e:
            st.error(f"⚠️ Error fetching updates: {e}")
            return []

    if st.button("Fetch Latest Updates"):
        with st.spinner("Fetching latest mining safety updates..."):
            updates = asyncio.run(fetch_updates_async())
            if updates:
                for item in updates:
                    with st.expander(f"📌 {item['title']}"):
                        st.markdown(f"**Published:** {item['published']}")
                        st.markdown(f"🔗 [Open Article]({item['link']})")
                        st.markdown(f"### 🛑 Safety Analysis\n{item['danger_analysis']}")
                st.info("⚡ Safety analyses served from cache if already queried (backend L1/L2).")
            else:
                st.warning("No updates found.")

# ============================================================
# SECTION 3 — AUDIT REPORT PDF GENERATOR
# ============================================================
elif section == "📄 Audit Report PDF":
    st.header("📄 Generate Mining Safety Audit Report (PDF)")

    state = st.text_input("State:", "All States")
    year = st.text_input("Year:", "All Years")
    hazard_type = st.text_input("Hazard Type:", "All Hazards")

    async def generate_pdf_async(state, year, hazard_type):
        try:
            response = await asyncio.to_thread(
                lambda: requests.post(
                    f"{API_URL}/audit_report_pdf",
                    json={"state": state, "year": year, "hazard_type": hazard_type},
                )
            )
            return response
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            return None

    if st.button("Generate PDF Report"):
        with st.spinner("Generating audit report..."):
            response = asyncio.run(generate_pdf_async(state, year, hazard_type))
            if response and response.status_code == 200:
                st.success("PDF generated successfully!")
                st.download_button(
                    label="📥 Download PDF",
                    data=response.content,
                    file_name=f"Audit_Report_{state}_{year}.pdf",
                    mime="application/pdf",
                )
                st.info("⚡ Report served from cache if identical request was made before.")
            else:
                st.error("Failed to generate PDF report.")

