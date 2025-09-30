import os
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Agno imports
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.csv_toolkit import CsvTools
from agno.tools.duckdb import DuckDbTools

# -------------------------
# Load .env
# -------------------------
load_dotenv()
AGNO_API_KEY = os.getenv("AGNO_API_KEY")
FLIGHTS_CSV = os.getenv("FLIGHTS_CSV", "flights.csv")
HOTELS_CSV = os.getenv("HOTELS_CSV", "hotels.csv")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "travel.duckdb")

# -------------------------
# Page config + CSS
# -------------------------
st.set_page_config(page_title="Travel Agent (CsvTools + DuckDB + Google Search)", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .header {background:linear-gradient(90deg,#4f46e5,#06b6d4);padding:18px;border-radius:12px;color:white}
    .card {border-radius:10px;padding:12px;border:1px solid #eee;box-shadow: 0 4px 12px rgba(15,23,42,0.04);background:#fff}
    .small {font-size:12px;color:#6b7280}
    .badge {display:inline-block;padding:6px 8px;border-radius:8px;font-size:12px;margin-right:6px;background:#f1f5f9}
    .price {font-weight:700;font-size:18px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header"><h1 style="margin:0">✈️ Travel Booking Assistant</h1></div>', unsafe_allow_html=True)
st.write("")

if not AGNO_API_KEY:
    st.sidebar.warning("AGNO_API_KEY not found in .env; agent features limited for language model responses.")

# -------------------------
# Caching & resources
# -------------------------
@st.cache_data(ttl=3600)
def load_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def get_booking_db(db_path: str = "bookings_csv_tools.db"):
    class BookingDB:
        def __init__(self, db_path):
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self._init()
        def _init(self):
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bookings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    origin TEXT, destination TEXT, travel_date TEXT,
                    flight_json TEXT, hotel_json TEXT, itinerary TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """); self.conn.commit()
        def save(self, origin, destination, travel_date, flight_obj, hotel_obj, itinerary) -> Dict[str, Any]:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO bookings (origin,destination,travel_date,flight_json,hotel_json,itinerary) VALUES (?,?,?,?,?,?)",
                (origin, destination, travel_date, json.dumps(flight_obj), json.dumps(hotel_obj), itinerary)
            )
            self.conn.commit()
            booking_id = cur.lastrowid
            return {"booking_id": booking_id, "confirmation_code": f"TRV-{booking_id:06d}"}
        def list_recent(self, limit=10):
            return pd.read_sql_query(f"SELECT id,origin,destination,travel_date,created_at FROM bookings ORDER BY created_at DESC LIMIT {limit}", self.conn)
    return BookingDB(db_path)

@st.cache_resource
def get_csv_toolkit(flights_path: str, hotels_path: str):
    return CsvTools(csvs=[Path(flights_path), Path(hotels_path)])

@st.cache_resource
def get_duckdb_tool(db_path: str = DUCKDB_PATH):
    return DuckDbTools(db_path=db_path)

@st.cache_resource
def get_agents_with_tools(api_key: Optional[str],
                          _csv_toolkit_obj,
                          _duckdb_tool,
                          _google_search_tool,
                          temperature: float = 0.2):
    """
    Returns (search_agent, itinerary_agent, booking_agent, news_agent)
    Note: leading underscores tell Streamlit not to hash these unhashable tool objects.
    """
    if not api_key:
        class StubAgent:
            def __init__(self, instructions: str): self.instructions = instructions
            def run(self, prompt: str): return type("R", (), {"content": f"[LOCAL STUB] {self.instructions} | PROMPT: {prompt}"})
        stub = StubAgent("stub")
        return stub, stub, stub, stub

    model = Gemini(id="gemini-2.0-flash-001", api_key=api_key, temperature=temperature)

    base_instructions = [
        "Prefer deterministic tool functions (CsvTools/DuckDbTools) over guessing.",
        "When running SQL, prefer DuckDbTools.run_query and inspect queries first."
    ]   

    # collect tools
    tools = []
    if _csv_toolkit_obj: tools.append(_csv_toolkit_obj)
    if _duckdb_tool: tools.append(_duckdb_tool)

    search_agent = Agent(model=model, tools=tools, instructions=base_instructions + ["You are SearchAgent. Use tools to find flights & hotels."], markdown=False)
    itinerary_agent = Agent(model=model, tools=tools, instructions=["You are ItineraryAgent. Create concise day-by-day itineraries."], markdown=False)
    booking_agent = Agent(model=model, tools=tools, instructions=["You are BookingAgent. Return short confirmation notes."], markdown=False)
   
    return search_agent, itinerary_agent, booking_agent

# -------------------------
# Init resources
# -------------------------
booking_db = get_booking_db()
emailer = type("E", (), {"send": staticmethod(lambda to, s, b: {"to": to, "status": "sent"})})()

# CSV paths check
flights_path = Path(FLIGHTS_CSV)
hotels_path = Path(HOTELS_CSV)
if not flights_path.exists() or not hotels_path.exists():
    st.error(f"CSV files not found: {flights_path.resolve()}, {hotels_path.resolve()}")
    st.stop()

# load CSVs
flights_df_all = load_csv_cached(str(flights_path))
hotels_df_all = load_csv_cached(str(hotels_path))

# get tools (cached)
csv_toolkit = get_csv_toolkit(str(flights_path), str(hotels_path))
duckdb_tool = get_duckdb_tool(DUCKDB_PATH)

# instantiate agents (cached)
search_agent, itinerary_agent, booking_agent = get_agents_with_tools(AGNO_API_KEY, csv_toolkit, duckdb_tool, None)

# -------------------------
# UI inputs (compact) — no defaults
# -------------------------
origin_prefill = st.session_state.get("last_search", {}).get("origin", "")
destination_prefill = st.session_state.get("last_search", {}).get("destination", "")

with st.form("search_form", clear_on_submit=False):
    c1, c2, c3 = st.columns([2.6, 2.6, 1.6])
    with c1:
        origin = st.text_input("Origin", value=origin_prefill, placeholder="e.g. Delhi")
        destination = st.text_input("Destination", value=destination_prefill, placeholder="e.g. Paris")
    with c2:
        travel_dt = st.date_input("Travel Date", value=date(2025,10,5))
        nights = st.number_input("Nights", 1, 14, 3)
    with c3:
        budget = st.number_input("Hotel budget / night (USD)", 50, 700, 150)
        user_email = st.text_input("Email (optional)", "")
    submitted = st.form_submit_button("🔎 Search")

# Session state initialization
if "top_flights" not in st.session_state: st.session_state.top_flights = []
if "top_hotels" not in st.session_state: st.session_state.top_hotels = []
if "itinerary_text" not in st.session_state: st.session_state.itinerary_text = ""
if "last_search" not in st.session_state: st.session_state.last_search = {}
if "selected_flight_id" not in st.session_state: st.session_state.selected_flight_id = None
if "selected_hotel_id" not in st.session_state: st.session_state.selected_hotel_id = None

# -------------------------
# Card render helpers
# -------------------------
def _set_selected_flight(fid):
    st.session_state.selected_flight_id = str(fid)

def render_flight_card(f: Dict[str, Any], key_prefix=""):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    cols = st.columns([3,2,2,1])
    with cols[0]:
        st.write(f"**{f['airline']} {f['flight_number']}**")
        st.write(f"<div class='small'>{f['origin']} → {f['destination']} · {f.get('duration','')} · {f.get('num_stops',0)} stop(s)</div>", unsafe_allow_html=True)
        st.write(f"<div class='small'>Depart: {f['depart_datetime']}  •  Arrive: {f['arrive_datetime']}</div>", unsafe_allow_html=True)
    with cols[1]:
        st.metric(label="Price (USD)", value=f"${f['price_usd']}", delta=None)
        st.write(f"<div class='small'>Cabin: {f.get('cabin','Economy')}</div>", unsafe_allow_html=True)
    with cols[2]:
        badges = ""
        if str(f.get("refundable","False")).lower() in ("true","1","yes"): badges += "<span class='badge'>Refundable</span>"
        if str(f.get("wifi","False")).lower() in ("true","1","yes"): badges += "<span class='badge'>Wi-Fi</span>"
        badges += f"<span class='badge'>{f.get('bag_allowance_kg',20)}kg baggage</span>"
        st.markdown(badges, unsafe_allow_html=True)
    with cols[3]:
        sel_key = f"flight_select_btn_{key_prefix}_{f['flight_id']}"
        st.button("Select", key=sel_key, on_click=_set_selected_flight, args=(f["flight_id"],))
        if str(st.session_state.get("selected_flight_id")) == str(f["flight_id"]):
            st.markdown("<div style='color:green;font-weight:600;margin-top:6px'>✓ Selected</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def _set_selected_hotel(hid):
    st.session_state.selected_hotel_id = str(hid)

def render_hotel_card(h: Dict[str, Any], key_prefix=""):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    cols = st.columns([3,2,2])
    with cols[0]:
        st.write(f"**{h['name']}**  — <span class='small'>{h.get('city','')}</span>", unsafe_allow_html=True)
        st.write(f"<div class='small'>{h.get('address','')}</div>", unsafe_allow_html=True)
        st.write(f"<div class='small'>Amenities: {h.get('amenities','—')}</div>", unsafe_allow_html=True)
    with cols[1]:
        st.metric(label="Per night", value=f"${h['price_per_night']}")
        st.write(f"<div class='small'>Rating: {h['rating']} ★ · {h.get('stars','')}⭐</div>", unsafe_allow_html=True)
    with cols[2]:
        nights = st.session_state.last_search.get('nights', 3)
        st.write(f"<div class='small'>Total for {nights} nights: <strong>${int(h['price_per_night']*nights)}</strong></div>", unsafe_allow_html=True)
        sel_key = f"hotel_select_btn_{key_prefix}_{h['hotel_id']}"
        st.button("Select", key=sel_key, on_click=_set_selected_hotel, args=(h["hotel_id"],))
        if str(st.session_state.get("selected_hotel_id")) == str(h["hotel_id"]):
            st.markdown("<div style='color:green;font-weight:600;margin-top:6px'>✓ Selected</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Search & caching logic
# -------------------------
def do_search_and_cache(o, d, dt_str, bud, nights_val):
    fl = flights_df_all[
        (flights_df_all.origin.str.lower() == o.strip().lower()) &
        (flights_df_all.destination.str.lower() == d.strip().lower()) &
        (flights_df_all.depart_datetime.str.startswith(dt_str))
    ].sort_values("price_usd").head(12)
    ho = hotels_df_all[
        (hotels_df_all.city.str.lower() == d.strip().lower()) &
        (hotels_df_all.price_per_night <= bud) &
        (hotels_df_all.availability_rooms > 0)
    ].sort_values(["price_per_night","rating"]).head(12)
    st.session_state.top_flights = fl.head(8).to_dict(orient="records")
    st.session_state.top_hotels = ho.head(8).to_dict(orient="records")
    st.session_state.itinerary_text = ""
    st.session_state.last_search = {"origin": o, "destination": d, "travel_date": dt_str, "budget": bud, "nights": nights_val}

# Validate and run search
if submitted:
    if not origin.strip() or not destination.strip():
        st.warning("Please enter both an Origin and a Destination before searching.")
    else:
        dstr = travel_dt.strftime("%Y-%m-%d")
        do_search_and_cache(origin, destination, dstr, budget, nights)
        

# -------------------------
# Show results panels
# -------------------------
st.markdown("### Search results")
if st.session_state.top_flights or st.session_state.top_hotels:
    cols = st.columns([1,1])
    with cols[0]:
        st.markdown("#### Flights")
        if st.session_state.top_flights:
            for f in st.session_state.top_flights:
                render_flight_card(f, key_prefix="cached")
        else:
            st.info("No flights found for the selected date.")
    with cols[1]:
        st.markdown("#### Hotels")
        if st.session_state.top_hotels:
            for h in st.session_state.top_hotels:
                render_hotel_card(h, key_prefix="cached")
        else:
            st.info("No hotels within budget or no availability.")

# -------------------------
# Itinerary generation
# -------------------------
st.markdown("---")
st.markdown("### Itinerary")
if not st.session_state.itinerary_text:
    if AGNO_API_KEY and (st.session_state.top_flights or st.session_state.top_hotels):
        try:
            prompt = (
                f"Create a {st.session_state.last_search.get('nights',3)}-day itinerary for {st.session_state.last_search.get('destination','')}. "
                f"Flights: {json.dumps(st.session_state.top_flights)} Hotels: {json.dumps(st.session_state.top_hotels)} "
                "Return a day-by-day plan with 1-2 activities and a dining suggestion."
            )
            it_resp = itinerary_agent.run(prompt)
            st.session_state.itinerary_text = it_resp.content
        except Exception:
            st.session_state.itinerary_text = "Itinerary could not be generated (agent error)."
    else:
        st.session_state.itinerary_text = "No itinerary yet. Search and then generate."

st.info(st.session_state.itinerary_text)

# -------------------------
# Booking selection + confirm
# -------------------------
st.markdown("---")
st.markdown("### Selected")
sel_f = st.session_state.get("selected_flight_id")
sel_h = st.session_state.get("selected_hotel_id")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Flight:**")
    if sel_f:
        fobj = next((f for f in st.session_state.top_flights if str(f["flight_id"]) == str(sel_f)), None)
        if fobj:
            st.write(f"{fobj['airline']} {fobj['flight_number']} — ${fobj['price_usd']}")
        else:
            st.write(f"Flight {sel_f} (not in cached list)")
    else:
        st.write("No flight selected.")
with col2:
    st.markdown("**Hotel:**")
    if sel_h:
        hobj = next((h for h in st.session_state.top_hotels if str(h["hotel_id"]) == str(sel_h)), None)
        if hobj:
            st.write(f"{hobj['name']} — ${hobj['price_per_night']}/night")
        else:
            st.write(f"Hotel {sel_h} (not in cached list)")
    else:
        st.write("No hotel selected.")

if st.button("✅ Confirm & Save Booking"):
    if not sel_f or not sel_h:
        st.error("Please select both a flight and a hotel before confirming.")
    else:
        flight_obj = next((f for f in st.session_state.top_flights if str(f["flight_id"]) == str(sel_f)), None)
        hotel_obj = next((h for h in st.session_state.top_hotels if str(h["hotel_id"]) == str(sel_h)), None)
        if not flight_obj or not hotel_obj:
            st.error("Selected items not found in cached results. Re-run search or re-select.")
        else:
            receipt = booking_db.save(
                st.session_state.last_search.get("origin", origin),
                st.session_state.last_search.get("destination", destination),
                st.session_state.last_search.get("travel_date", travel_dt.strftime("%Y-%m-%d")),
                flight_obj, hotel_obj, st.session_state.itinerary_text
            )
            st.success(f"Booking saved — Confirmation: **{receipt['confirmation_code']}**")
            if user_email:
                sent = emailer.send(user_email, f"Booking {receipt['confirmation_code']}", f"Details: {receipt}")
                st.info(f"(Mock) email sent to {user_email} — {sent['status']}")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**Confirmation:** `{receipt['confirmation_code']}`")
            st.markdown(f"- From: **{st.session_state.last_search.get('origin')}**")
            st.markdown(f"- To: **{st.session_state.last_search.get('destination')}**")
            st.markdown(f"- Date: **{st.session_state.last_search.get('travel_date')}**")
            st.markdown(f"- Flight: {flight_obj['airline']} {flight_obj['flight_number']} — ${flight_obj['price_usd']}")
            st.markdown(f"- Hotel: {hotel_obj['name']} — ${hotel_obj['price_per_night']}/night")
            st.markdown('</div>', unsafe_allow_html=True)

# Sidebar recent bookings
st.sidebar.header("Recent bookings")
try:
    recent = booking_db.list_recent(8)
    if recent.empty:
        st.sidebar.write("No bookings yet.")
    else:
        st.sidebar.dataframe(recent)
except Exception:
    st.sidebar.write("Booking DB unavailable.")
