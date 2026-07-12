"""
NTPC IT Helpdesk Chatbot + Maintenance Alert System
====================================================
Internship Project | NTPC IT Division
Built with: Python, Streamlit, Google Gemini API, Pandas

Features:
  1. AI-powered IT troubleshooting chatbot (Gemini 2.5 Flash)
  2. Maintenance alert popups (overdue + due soon)
  3. Warranty alert popups (expired + expiring soon)
  4. Vendor card after every AI solution
  5. [NEW] Charts & Graphs Dashboard (plotly)
  6. [NEW] PDF Report Generator (fpdf2)
  7. [NEW] Ticket Management System (CSV-backed)
  8. [NEW] Maintenance History Log (CSV-backed)
"""

import os
from datetime import datetime, date

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from fpdf import FPDF

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NTPC IT Helpdesk",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.user-bubble {
    background: #1f6feb22; border: 1px solid #1f6feb55;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px; margin: 8px 0 8px 60px;
    font-size: 0.95rem; color: #79c0ff;
}
.bot-bubble {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px; margin: 8px 60px 8px 0;
    font-size: 0.95rem; color: #e6edf3;
}
.vendor-card {
    background: #0d2235; border: 1px solid #1f6feb;
    border-left: 4px solid #1f6feb; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0;
}
.vendor-card .vendor-title { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#79c0ff; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }
.vendor-card .vendor-name  { font-size:1rem; font-weight:600; color:#e6edf3; }
.vendor-card .vendor-contact { font-size:0.85rem; color:#8b949e; margin-top:4px; }
.alert-critical { background:#3d1010; border-left:4px solid #f85149; border-radius:6px; padding:12px 16px; margin:6px 0; }
.alert-high     { background:#2d1f00; border-left:4px solid #e3a414; border-radius:6px; padding:12px 16px; margin:6px 0; }
.alert-medium   { background:#0d2235; border-left:4px solid #388bfd; border-radius:6px; padding:12px 16px; margin:6px 0; }
.alert-ok       { background:#0d2b1e; border-left:4px solid #3fb950; border-radius:6px; padding:10px 14px; margin:4px 0; }
.warranty-expired  { background:#3d1010; border-left:4px solid #f85149; border-radius:6px; padding:12px 16px; margin:6px 0; }
.warranty-expiring { background:#2d1f00; border-left:4px solid #e3a414; border-radius:6px; padding:12px 16px; margin:6px 0; }
.ticket-open     { background:#0d2235; border-left:4px solid #388bfd; border-radius:6px; padding:12px 16px; margin:6px 0; }
.ticket-progress { background:#2d1f00; border-left:4px solid #e3a414; border-radius:6px; padding:12px 16px; margin:6px 0; }
.ticket-closed   { background:#0d2b1e; border-left:4px solid #3fb950; border-radius:6px; padding:10px 14px; margin:4px 0; }
.history-card    { background:#161b22; border-left:4px solid #3fb950; border-radius:6px; padding:10px 14px; margin:4px 0; }
.metric-box { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; text-align:center; }
.metric-num   { font-size:2rem; font-family:'IBM Plex Mono',monospace; font-weight:600; }
.metric-label { font-size:0.8rem; color:#8b949e; margin-top:4px; }
.stTextInput > div > div > input { background:#161b22 !important; border:1px solid #30363d !important; color:#e6edf3 !important; border-radius:6px !important; }
.stSelectbox > div > div { background:#161b22 !important; border:1px solid #30363d !important; color:#e6edf3 !important; }
.stButton > button { background:#238636 !important; color:white !important; border:none !important; border-radius:6px !important; font-family:'IBM Plex Mono',monospace !important; font-weight:600 !important; padding:8px 20px !important; }
.stButton > button:hover { background:#2ea043 !important; }
.popup-overlay  { background:#1a0000; border:2px solid #f85149; border-radius:10px; padding:20px; margin-bottom:20px; }
.warranty-popup { background:#1a1000; border:2px solid #e3a414; border-radius:10px; padding:20px; margin-bottom:20px; }
.ntpc-badge { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#f0c040; background:#1a1500; border:1px solid #f0c040; border-radius:4px; padding:2px 8px; display:inline-block; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GEMINI API SETUP
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
if not GEMINI_API_KEY:
    st.error("❌ **GEMINI_API_KEY not set!**\n\n`$env:GEMINI_API_KEY='your-key-here'`\n\nGet a free key at: https://aistudio.google.com/app/apikey")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────
# FILE PATHS FOR PERSISTENT DATA
# ─────────────────────────────────────────────
TICKETS_FILE = "ntpc_tickets.csv"
HISTORY_FILE = "ntpc_maintenance_history.csv"

def init_tickets():
    if not os.path.exists(TICKETS_FILE):
        pd.DataFrame(columns=[
            "ticket_id","raised_by","asset_id","device_type","issue",
            "priority","status","assigned_to","raised_date","closed_date","notes"
        ]).to_csv(TICKETS_FILE, index=False)

def init_history():
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=[
            "log_id","asset_id","device_type","maintenance_date",
            "done_by","type","cost","notes"
        ]).to_csv(HISTORY_FILE, index=False)

def load_tickets():
    init_tickets()
    return pd.read_csv(TICKETS_FILE)

def save_tickets(df):
    df.to_csv(TICKETS_FILE, index=False)

def load_history():
    init_history()
    return pd.read_csv(HISTORY_FILE)

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def parse_vendor(vendor_str):
    if not vendor_str or str(vendor_str).strip() in ["", "nan"]:
        return "Not Available", ""
    v = str(vendor_str).strip()
    parts = v.split(",", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return v, ""


# ─────────────────────────────────────────────
# LOAD & PROCESS DATASET
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    file_path = "NTPC_IT-PROJECT.csv"
    if not os.path.exists(file_path):
        st.error("❌ Data file **'NTPC_IT-PROJECT.csv'** not found. Place it in the same folder as ntpc_helpdesk.py.")
        st.stop()

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "Maintanence Period":     "maintenance_period",
        "Maintanence Date(Last)": "last_maintenance",
        "Crticality":             "criticality",
        "Asset_ID":               "asset_id",
        "Device_Type":            "device_type",
        "Asset_Make":             "asset_make",
        "Model_Details":          "model",
        "Purchase Date":          "purchase_date",
        "WARR END DT":            "warranty_end",
        "Vendor Details":         "vendor_details",
    }, inplace=True)

    def period_to_days(p):
        p = str(p).strip().lower()
        if "3 month"  in p: return 90
        if "6 month"  in p: return 180
        if "12 month" in p: return 365
        return 180

    df["period_days"]     = df["maintenance_period"].apply(period_to_days)
    today                 = pd.Timestamp(date.today())
    df["last_maintenance"]= pd.to_datetime(df["last_maintenance"], dayfirst=True, errors="coerce")
    df["purchase_date"]   = pd.to_datetime(df["purchase_date"],    dayfirst=True, errors="coerce")
    df["warranty_end"]    = pd.to_datetime(df["warranty_end"],     dayfirst=True, errors="coerce")
    df["next_maintenance"]= df["last_maintenance"] + pd.to_timedelta(df["period_days"], unit="D")
    df["days_remaining"]  = (df["next_maintenance"] - today).dt.days
    df["warranty_days_left"] = (df["warranty_end"] - today).dt.days

    def warranty_status(d):
        if pd.isna(d):  return "⚪ NO DATA"
        if d < 0:       return "🔴 EXPIRED"
        if d <= 90:     return "🟠 EXPIRING SOON"
        if d <= 365:    return "🟡 VALID (<1yr)"
        return          "🟢 VALID"

    df["warranty_status"] = df["warranty_days_left"].apply(warranty_status)
    crit_map = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    df["crit_weight"] = df["criticality"].map(crit_map).fillna(2)

    def urgency(row):
        d = row["days_remaining"]
        if pd.isna(d): return 0
        if d <= 0:     return 999 * row["crit_weight"]
        return round(row["crit_weight"] * (1 / (d + 1)) * 1000, 2)

    df["urgency_score"] = df.apply(urgency, axis=1)

    def maint_status(d):
        if pd.isna(d): return "UNKNOWN"
        if d < 0:      return "🔴 OVERDUE"
        if d <= 7:     return "🔴 DUE THIS WEEK"
        if d <= 30:    return "🟠 DUE THIS MONTH"
        if d <= 60:    return "🟡 UPCOMING"
        return         "🟢 OK"

    df["status"] = df["days_remaining"].apply(maint_status)
    df["vendor_details"] = df["vendor_details"].fillna("")
    df["asset_age_years"] = ((today - df["purchase_date"]).dt.days / 365).round(1)
    return df.sort_values("urgency_score", ascending=False)


df = load_data()

COMMON_ERRORS = [
    "System not turning on / no power",
    "Computer running very slow",
    "Blue screen / system crash (BSOD)",
    "Internet not working / no network",
    "Printer not printing / offline",
    "Printer paper jam",
    "Projector not displaying / no signal",
    "Camera / CCTV feed not showing",
    "Software not opening / crashing",
    "System overheating / fan noise",
    "Mouse / keyboard not working",
    "Monitor no display / blank screen",
    "Virus / malware suspected",
    "Server not responding",
    "Photocopier error / not working",
    "Other (type below)",
]


# ─────────────────────────────────────────────
# VENDOR CARD
# ─────────────────────────────────────────────
def show_vendor_card(asset_row):
    vendor_name, vendor_contact = parse_vendor(asset_row.get("vendor_details", ""))
    asset_id    = asset_row.get("asset_id", "N/A")
    device      = asset_row.get("device_type", "N/A")
    warr_status = asset_row.get("warranty_status", "⚪ NO DATA")
    warr_days   = asset_row.get("warranty_days_left", None)

    if pd.notna(warr_days) if not isinstance(warr_days, str) else False:
        warr_str = (f"<span style='color:#f85149'>Expired {abs(int(warr_days))} days ago</span>"
                    if warr_days < 0 else
                    f"<span style='color:#3fb950'>Valid for {int(warr_days)} more days</span>")
    else:
        warr_str = "<span style='color:#8b949e'>No warranty data</span>"

    st.markdown(f"""
<div class="vendor-card">
  <div class="vendor-title">📞 Vendor & Warranty — {asset_id} ({device})</div>
  <div class="vendor-name">🏢 {vendor_name}</div>
  <div class="vendor-contact">📱 {vendor_contact if vendor_contact else 'Contact not available'}</div>
  <div class="vendor-contact" style="margin-top:6px">🛡️ Warranty: {warr_status} &nbsp;|&nbsp; {warr_str}</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AI SOLUTION GENERATOR
# ─────────────────────────────────────────────
def get_ai_solution(error_desc, device_info):
    device_context = ""
    if device_info:
        warr_days  = device_info.get("warranty_days_left", None)
        warr_note  = ""
        try:
            if pd.notna(warr_days):
                warr_note = (f"WARRANTY EXPIRED {abs(int(warr_days))} days ago — escalate to vendor!" if warr_days < 0
                             else f"WARRANTY EXPIRING in {int(warr_days)} days — contact vendor soon." if warr_days <= 90
                             else f"Warranty valid for {int(warr_days)} more days.")
        except Exception:
            pass
        vendor_name, vendor_contact = parse_vendor(device_info.get("vendor_details", ""))
        device_context = f"""
Device from NTPC Asset Register:
- Asset ID: {device_info.get('asset_id','N/A')} | Type: {device_info.get('device_type','N/A')}
- Make: {device_info.get('asset_make','N/A')} | Model: {device_info.get('model','N/A')}
- Purchase Date: {device_info.get('purchase_date','N/A')} | Criticality: {device_info.get('criticality','N/A')}
- Last Maintenance: {device_info.get('last_maintenance','N/A')} | Next Due: {device_info.get('next_maintenance','N/A')}
- Days Until Due: {device_info.get('days_remaining','N/A')} days
- Warranty: {device_info.get('warranty_status','N/A')} — {warr_note}
- Vendor: {vendor_name} | {vendor_contact}
"""
    else:
        device_context = "Device not found in NTPC asset register. Providing general advice."

    prompt = f"""You are an expert IT helpdesk assistant for NTPC (National Thermal Power Corporation).
{device_context}
Reported Problem: {error_desc}
Today: {date.today().strftime('%d %B %Y')}

Respond with these exact sections:
## 🔍 Problem Diagnosis
## 🛠️ Step-by-Step Fix
## 🔧 Maintenance Required?
## 🧹 Cleaning Required?
## ⚠️ Preventive Measures
## 📞 Escalation

Be specific, practical, and beginner-friendly."""

    try:
        model    = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"""## ⚠️ AI Service Unavailable
**For '{error_desc}':**
1. Restart the device first
2. Check all cable connections
3. Check event logs
4. Contact vendor if issue persists
**Error:** {str(e)}"""


# ─────────────────────────────────────────────
# ALERT POPUPS
# ─────────────────────────────────────────────
def show_maintenance_alerts():
    urgent    = df[df["days_remaining"] <= 7].copy()
    if urgent.empty: return
    overdue   = urgent[urgent["days_remaining"] < 0]
    this_week = urgent[urgent["days_remaining"] >= 0]
    st.markdown('<div class="popup-overlay">', unsafe_allow_html=True)
    st.markdown("### 🚨 URGENT MAINTENANCE ALERTS")
    for _, row in overdue.iterrows():
        vn, vc = parse_vendor(row["vendor_details"])
        st.markdown(f'<div class="alert-critical"><strong>🔴 OVERDUE: {row["asset_id"]}</strong> — {row["device_type"]} ({row["asset_make"]})<br><small>Overdue by <strong>{abs(int(row["days_remaining"]))} days</strong> | Criticality: <strong>{row["criticality"]}</strong> | 📞 {vn}{(" — "+vc) if vc else ""}</small></div>', unsafe_allow_html=True)
    for _, row in this_week.iterrows():
        vn, vc = parse_vendor(row["vendor_details"])
        st.markdown(f'<div class="alert-high"><strong>🟠 DUE SOON: {row["asset_id"]}</strong> — {row["device_type"]} ({row["asset_make"]})<br><small>Due in <strong>{int(row["days_remaining"])} days</strong> | Criticality: <strong>{row["criticality"]}</strong> | 📞 {vn}</small></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_warranty_alerts():
    expired  = df[df["warranty_days_left"] < 0].copy()
    expiring = df[(df["warranty_days_left"] >= 0) & (df["warranty_days_left"] <= 90)].copy()
    if expired.empty and expiring.empty: return
    st.markdown('<div class="warranty-popup">', unsafe_allow_html=True)
    st.markdown("### 🛡️ WARRANTY ALERTS")
    for _, row in expired.iterrows():
        vn, vc = parse_vendor(row["vendor_details"])
        exp_date = row["warranty_end"].strftime('%d %b %Y') if pd.notna(row["warranty_end"]) else "Unknown"
        st.markdown(f'<div class="warranty-expired"><strong>🔴 EXPIRED: {row["asset_id"]}</strong> — {row["device_type"]} ({row["asset_make"]})<br><small>Expired: <strong>{exp_date}</strong> | 📞 {vn}{(" — "+vc) if vc else ""}</small></div>', unsafe_allow_html=True)
    for _, row in expiring.iterrows():
        vn, vc = parse_vendor(row["vendor_details"])
        exp_date = row["warranty_end"].strftime('%d %b %Y') if pd.notna(row["warranty_end"]) else "Unknown"
        st.markdown(f'<div class="warranty-expiring"><strong>🟠 EXPIRING: {row["asset_id"]}</strong> — {row["device_type"]} ({row["asset_make"]})<br><small>Expires: <strong>{exp_date}</strong> (in {int(row["warranty_days_left"])} days) | 📞 {vn}</small></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# [NEW] PDF REPORT GENERATOR
# ─────────────────────────────────────────────
def sanitize(text, max_len=None):
    """Remove non-latin1 characters so fpdf Helvetica doesn't crash."""
    text = str(text)
    # Replace common unicode punctuation with ASCII equivalents
    text = (text
            .replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2018", "'")   # left single quote
            .replace("\u2019", "'")   # right single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u20b9", "Rs.") # rupee sign
            .replace("\u00e9", "e")   # e with accent
            .replace("\u00e0", "a"))  # a with accent
    # Strip anything still outside latin-1
    text = text.encode("latin-1", errors="ignore").decode("latin-1")
    if max_len:
        text = text[:max_len]
    return text

def generate_pdf_report():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header bar
    pdf.set_fill_color(0, 48, 135)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 7)
    pdf.cell(0, 10, "NTPC IT Division - Maintenance & Warranty Report")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 20)
    pdf.cell(0, 6, sanitize(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}  |  Total Assets: {len(df)}"))
    pdf.ln(18)

    # Summary row
    overdue_c = len(df[df["days_remaining"] < 0])
    warr_exp  = len(df[df["warranty_days_left"] < 0])
    warr_soon = len(df[(df["warranty_days_left"] >= 0) & (df["warranty_days_left"] <= 90)])
    due_week  = len(df[(df["days_remaining"] >= 0) & (df["days_remaining"] <= 7)])

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, "SUMMARY")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_fill_color(245, 245, 245)
    pdf.cell(45, 8, f"Maint. Overdue: {overdue_c}",  border=1, fill=True)
    pdf.cell(45, 8, f"Due This Week: {due_week}",     border=1, fill=True)
    pdf.cell(50, 8, f"Warranty Expired: {warr_exp}", border=1, fill=True)
    pdf.cell(50, 8, f"Expiring Soon: {warr_soon}",   border=1, fill=True)
    pdf.ln(12)

    # Overdue maintenance table
    overdue_df = df[df["days_remaining"] < 0].copy()
    if not overdue_df.empty:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(220, 50, 50)
        pdf.cell(0, 8, f"OVERDUE MAINTENANCE ({len(overdue_df)} assets)")
        pdf.ln(8)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(0, 48, 135)
        pdf.set_text_color(255, 255, 255)
        for col, w in [("Asset ID",25),("Device",35),("Brand",30),("Overdue",20),("Criticality",25),("Vendor",55)]:
            pdf.cell(w, 7, col, border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(0, 0, 0)
        for i, (_, row) in enumerate(overdue_df.iterrows()):
            fill = i % 2 == 0
            pdf.set_fill_color(255, 235, 235) if fill else pdf.set_fill_color(255, 255, 255)
            vn, _ = parse_vendor(row["vendor_details"])
            pdf.cell(25, 6, sanitize(row["asset_id"],   12), border=1, fill=fill)
            pdf.cell(35, 6, sanitize(row["device_type"],18), border=1, fill=fill)
            pdf.cell(30, 6, sanitize(row["asset_make"], 15), border=1, fill=fill)
            pdf.cell(20, 6, f"{abs(int(row['days_remaining']))}d ago", border=1, fill=fill)
            pdf.cell(25, 6, sanitize(row["criticality"],12), border=1, fill=fill)
            pdf.cell(55, 6, sanitize(vn, 28),                border=1, fill=fill)
            pdf.ln()
        pdf.ln(6)

    # Warranty expired table
    warr_exp_df = df[df["warranty_days_left"] < 0].copy()
    if not warr_exp_df.empty:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(220, 50, 50)
        pdf.cell(0, 8, f"EXPIRED WARRANTIES ({len(warr_exp_df)} assets)")
        pdf.ln(8)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(0, 48, 135)
        pdf.set_text_color(255, 255, 255)
        for col, w in [("Asset ID",25),("Device",35),("Brand",30),("Expired On",30),("Vendor",70)]:
            pdf.cell(w, 7, col, border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(0, 0, 0)
        for i, (_, row) in enumerate(warr_exp_df.iterrows()):
            fill = i % 2 == 0
            pdf.set_fill_color(255, 235, 235) if fill else pdf.set_fill_color(255, 255, 255)
            vn, vc = parse_vendor(row["vendor_details"])
            exp_date = row["warranty_end"].strftime('%d %b %Y') if pd.notna(row["warranty_end"]) else "Unknown"
            vendor_str = sanitize(f"{vn}{(' - '+vc[:18]) if vc else ''}", 38)
            pdf.cell(25, 6, sanitize(row["asset_id"],   12), border=1, fill=fill)
            pdf.cell(35, 6, sanitize(row["device_type"],18), border=1, fill=fill)
            pdf.cell(30, 6, sanitize(row["asset_make"], 15), border=1, fill=fill)
            pdf.cell(30, 6, sanitize(exp_date),              border=1, fill=fill)
            pdf.cell(70, 6, vendor_str,                      border=1, fill=fill)
            pdf.ln()
        pdf.ln(6)

    # Footer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "NTPC IT Division | Confidential | Generated by NTPC IT Helpdesk System", align="C")

    return bytes(pdf.output())


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ntpc-badge">⚡ NTPC IT DIVISION</div>', unsafe_allow_html=True)
    st.markdown("## IT Helpdesk System")
    st.markdown("---")

    page = st.radio("Navigate to:", [
        "💬 IT Helpdesk Chatbot",
        "📊 Charts Dashboard",
        "🔔 Maintenance Dashboard",
        "🎫 Ticket Management",
        "📋 Maintenance History",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**🔍 Quick Device Lookup**")
    search_id = st.text_input("Enter Asset ID", placeholder="PC_101")
    if search_id:
        match = df[df["asset_id"].str.upper() == search_id.strip().upper()]
        if not match.empty:
            r = match.iloc[0]
            st.success(f"**{r['asset_id']}** found!")
            st.caption(f"{r['device_type']} | {r['asset_make']} {r['model']}")
            st.caption(f"Maintenance: {r['status']}")
            st.caption(f"Warranty: {r['warranty_status']}")
            days = int(r['days_remaining']) if pd.notna(r['days_remaining']) else 'N/A'
            if isinstance(days, (int, float)) and days < 0:
                st.error(f"Overdue by {abs(int(days))} days!")
            elif isinstance(days, (int, float)) and days <= 30:
                st.warning(f"Maintenance due in {int(days)} days")
            wdays = r['warranty_days_left']
            if pd.notna(wdays):
                if wdays < 0:   st.error(f"Warranty expired {abs(int(wdays))} days ago!")
                elif wdays <= 90: st.warning(f"Warranty expiring in {int(wdays)} days!")
            vn, vc = parse_vendor(r['vendor_details'])
            st.caption(f"📞 {vn}")
            if vc: st.caption(f"   {vc}")
        else:
            st.warning("Asset ID not found")

    st.markdown("---")
    st.caption(f"Dataset: {len(df)} assets loaded")
    st.caption(f"Last refresh: {datetime.now().strftime('%d %b %Y %H:%M')}")


# ═══════════════════════════════════════════════════════
# PAGE 1 — IT HELPDESK CHATBOT
# ═══════════════════════════════════════════════════════
if "💬 IT Helpdesk Chatbot" in page:
    st.markdown("# ⚡ NTPC IT Helpdesk")
    st.markdown("*AI-powered troubleshooting assistant — NTPC IT Division*")
    show_maintenance_alerts()
    show_warranty_alerts()
    st.markdown("---")

    for k, v in [("step",1),("error_desc",""),("selected_asset",None),("chat_history",[]),("solution_shown",False)]:
        if k not in st.session_state: st.session_state[k] = v

    for msg in st.session_state.chat_history:
        cls = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{cls}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    if st.session_state.step == 1:
        st.markdown('<div class="bot-bubble">🤖 <strong>Welcome to NTPC IT Helpdesk!</strong><br>What problem are you facing?</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2,1])
        with col1:
            error_choice = st.selectbox("Select issue:", ["-- Select an issue --"] + COMMON_ERRORS, label_visibility="collapsed")
        with col2:
            if st.button("Next →", key="btn_s1"):
                if error_choice not in ["-- Select an issue --", "Other (type below)"]:
                    st.session_state.error_desc = error_choice
                    st.session_state.chat_history += [{"role":"user","content":f"My problem: {error_choice}"},{"role":"assistant","content":"Got it! Which device is affected?"}]
                    st.session_state.step = 2; st.rerun()
                else: st.warning("Please select an issue")
        custom = st.text_input("Or describe your issue:", placeholder="e.g. Screen flickers when opening Excel")
        if custom and st.button("Submit", key="btn_custom"):
            st.session_state.error_desc = custom
            st.session_state.chat_history += [{"role":"user","content":f"My problem: {custom}"},{"role":"assistant","content":"Got it! Which device is affected?"}]
            st.session_state.step = 2; st.rerun()

    elif st.session_state.step == 2:
        st.markdown("**Step 2: Identify your device**")
        col1, col2 = st.columns(2)
        with col1:
            dt_filter = st.selectbox("Device Type:", ["All"] + sorted(df["device_type"].unique().tolist()))
        fdf = df if dt_filter == "All" else df[df["device_type"] == dt_filter]
        opts = [f"{r['asset_id']} — {r['device_type']} ({r['asset_make']} {r['model']})" for _, r in fdf.iterrows()]
        with col2:
            sel = st.selectbox("Select device:", ["-- Select --"] + opts)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Get Solution ✓", key="btn_s2"):
                if sel != "-- Select --":
                    aid = sel.split(" — ")[0]
                    row = df[df["asset_id"] == aid].iloc[0]
                    st.session_state.selected_asset = row.to_dict()
                    st.session_state.chat_history += [{"role":"user","content":f"Device: {aid} — {row['device_type']}"},{"role":"assistant","content":"Found device ✅. Generating solution..."}]
                    st.session_state.step = 3; st.rerun()
                else: st.warning("Please select a device")
        with c2:
            if st.button("Not in list", key="btn_noasset"):
                st.session_state.selected_asset = {}
                st.session_state.chat_history += [{"role":"user","content":"Device not in register"},{"role":"assistant","content":"No problem! Giving general advice."}]
                st.session_state.step = 3; st.rerun()

    elif st.session_state.step == 3 and not st.session_state.solution_shown:
        with st.spinner("🤖 Generating solution..."):
            sol = get_ai_solution(st.session_state.error_desc, st.session_state.selected_asset)
        asset = st.session_state.selected_asset
        if asset:
            days = asset.get("days_remaining")
            try:
                if pd.notna(days) and float(days) <= 30:
                    st.warning(f"⚠️ {asset.get('asset_id')} is due for maintenance in **{int(days)} days**.")
            except Exception: pass
        st.markdown('<div class="bot-bubble">', unsafe_allow_html=True)
        st.markdown(sol)
        st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.chat_history.append({"role":"assistant","content":sol})
        if asset: show_vendor_card(asset)
        st.session_state.solution_shown = True
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 New Issue", key="btn_rst"):
                for k in ["step","error_desc","selected_asset","chat_history","solution_shown"]: del st.session_state[k]
                st.rerun()
        with c2:
            if st.button("📋 Follow-up", key="btn_fu"): st.session_state.step = 4; st.rerun()

    elif st.session_state.step == 3 and st.session_state.solution_shown:
        if st.session_state.selected_asset: show_vendor_card(st.session_state.selected_asset)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 New Issue", key="btn_rst2"):
                for k in ["step","error_desc","selected_asset","chat_history","solution_shown"]: del st.session_state[k]
                st.rerun()
        with c2:
            if st.button("📋 Follow-up", key="btn_fu2"): st.session_state.step = 4; st.rerun()

    elif st.session_state.step == 4:
        if st.session_state.selected_asset: show_vendor_card(st.session_state.selected_asset)
        fu = st.text_input("Ask a follow-up:", placeholder="e.g. How do I update the BIOS?")
        if fu and st.button("Send", key="btn_send"):
            with st.spinner("Thinking..."):
                resp = get_ai_solution(f"Original: {st.session_state.error_desc}\nFollow-up: {fu}", st.session_state.selected_asset)
            st.session_state.chat_history += [{"role":"user","content":fu},{"role":"assistant","content":resp}]
            st.rerun()
        if st.button("🔄 New Issue", key="btn_new"):
            for k in ["step","error_desc","selected_asset","chat_history","solution_shown"]: del st.session_state[k]
            st.rerun()


# ═══════════════════════════════════════════════════════
# PAGE 2 — [NEW] CHARTS DASHBOARD
# ═══════════════════════════════════════════════════════
elif "📊 Charts Dashboard" in page:
    st.markdown("# 📊 Charts Dashboard")
    st.markdown("*Visual analytics for NTPC IT asset health*")
    st.markdown("---")

    # PDF Download button
    try:
        pdf_bytes = generate_pdf_report()
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"NTPC_IT_Report_{date.today().strftime('%d%m%Y')}.pdf",
            mime="application/pdf",
            help="Download a formatted PDF report of all overdue and expired items"
        )
    except Exception as e:
        st.warning(f"⚠️ PDF generation failed: {str(e)[:120]}")

    st.markdown("---")

    # Row 1 — Pie charts
    col1, col2, col3 = st.columns(3)

    with col1:
        crit_counts = df["criticality"].value_counts().reset_index()
        crit_counts.columns = ["Criticality", "Count"]
        fig = px.pie(crit_counts, names="Criticality", values="Count",
                     title="Assets by Criticality",
                     color_discrete_sequence=["#f85149","#e3a414","#388bfd","#3fb950"],
                     hole=0.4)
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                          font_color="#e6edf3", title_font_color="#e6edf3", legend_font_color="#e6edf3")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        color_map = {"🔴 OVERDUE":"#f85149","🔴 DUE THIS WEEK":"#ff6b6b",
                     "🟠 DUE THIS MONTH":"#e3a414","🟡 UPCOMING":"#f0c040","🟢 OK":"#3fb950","UNKNOWN":"#8b949e"}
        fig2 = px.pie(status_counts, names="Status", values="Count",
                      title="Maintenance Status Distribution",
                      color="Status", color_discrete_map=color_map, hole=0.4)
        fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                           font_color="#e6edf3", title_font_color="#e6edf3", legend_font_color="#e6edf3")
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        warr_counts = df["warranty_status"].value_counts().reset_index()
        warr_counts.columns = ["Status","Count"]
        fig3 = px.pie(warr_counts, names="Status", values="Count",
                      title="Warranty Status Distribution",
                      color_discrete_sequence=["#f85149","#e3a414","#f0c040","#3fb950","#8b949e"],
                      hole=0.4)
        fig3.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                           font_color="#e6edf3", title_font_color="#e6edf3", legend_font_color="#e6edf3")
        st.plotly_chart(fig3, use_container_width=True)

    # Row 2 — Bar charts
    col4, col5 = st.columns(2)

    with col4:
        type_counts = df.groupby("device_type").size().reset_index(name="Count")
        fig4 = px.bar(type_counts, x="device_type", y="Count",
                      title="Asset Count by Device Type",
                      color="Count", color_continuous_scale="Blues")
        fig4.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                           font_color="#e6edf3", title_font_color="#e6edf3",
                           xaxis=dict(tickangle=-30, color="#8b949e"),
                           yaxis=dict(color="#8b949e"))
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        overdue_by_type = df[df["days_remaining"] < 0].groupby("device_type").size().reset_index(name="Overdue")
        if not overdue_by_type.empty:
            fig5 = px.bar(overdue_by_type, x="device_type", y="Overdue",
                          title="Overdue Maintenance by Device Type",
                          color="Overdue", color_continuous_scale="Reds")
            fig5.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                               font_color="#e6edf3", title_font_color="#e6edf3",
                               xaxis=dict(tickangle=-30, color="#8b949e"),
                               yaxis=dict(color="#8b949e"))
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.success("✅ No overdue maintenance items!")

    # Row 3 — Timeline & Age
    col6, col7 = st.columns(2)

    with col6:
        next_30 = df[(df["days_remaining"] >= 0) & (df["days_remaining"] <= 30)].copy()
        next_30 = next_30.sort_values("days_remaining")
        if not next_30.empty:
            fig6 = px.bar(next_30, x="asset_id", y="days_remaining",
                          title="Upcoming Maintenance (Next 30 Days)",
                          color="criticality",
                          color_discrete_map={"Critical":"#f85149","High":"#e3a414","Medium":"#388bfd","Low":"#3fb950"})
            fig6.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                               font_color="#e6edf3", title_font_color="#e6edf3",
                               xaxis=dict(tickangle=-45, color="#8b949e"),
                               yaxis=dict(color="#8b949e", title="Days Remaining"))
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No maintenance due in next 30 days.")

    with col7:
        age_df = df.dropna(subset=["asset_age_years"]).copy()
        fig7 = px.histogram(age_df, x="asset_age_years", nbins=10,
                            title="Asset Age Distribution (Years)",
                            color_discrete_sequence=["#388bfd"])
        fig7.add_vline(x=5, line_dash="dash", line_color="#f85149",
                       annotation_text="5yr EOL threshold", annotation_font_color="#f85149")
        fig7.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                           font_color="#e6edf3", title_font_color="#e6edf3",
                           xaxis=dict(color="#8b949e", title="Age (Years)"),
                           yaxis=dict(color="#8b949e", title="Count"))
        st.plotly_chart(fig7, use_container_width=True)

    # Row 4 — Urgency top 10
    top10 = df.nlargest(10, "urgency_score")[["asset_id","device_type","criticality","urgency_score","status"]]
    fig8 = px.bar(top10, x="asset_id", y="urgency_score",
                  title="Top 10 Assets by Urgency Score",
                  color="criticality",
                  color_discrete_map={"Critical":"#f85149","High":"#e3a414","Medium":"#388bfd","Low":"#3fb950"})
    fig8.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                       font_color="#e6edf3", title_font_color="#e6edf3",
                       xaxis=dict(color="#8b949e"), yaxis=dict(color="#8b949e", title="Urgency Score"))
    st.plotly_chart(fig8, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 3 — MAINTENANCE DASHBOARD
# ═══════════════════════════════════════════════════════
elif "🔔 Maintenance Dashboard" in page:
    st.markdown("# 🔔 Maintenance Dashboard")
    st.markdown("*Real-time asset maintenance and warranty status*")
    show_maintenance_alerts()
    show_warranty_alerts()
    st.markdown("---")

    oc = len(df[df["days_remaining"] < 0])
    wc = len(df[(df["days_remaining"] >= 0) & (df["days_remaining"] <= 7)])
    mc = len(df[(df["days_remaining"] > 7)  & (df["days_remaining"] <= 30)])
    gc = len(df[df["days_remaining"] > 30])
    we = len(df[df["warranty_days_left"] < 0])
    ws = len(df[(df["warranty_days_left"] >= 0) & (df["warranty_days_left"] <= 90)])

    cols = st.columns(6)
    for col, num, label, color in zip(cols,
        [oc,wc,mc,gc,we,ws],
        ["MAINT. OVERDUE","DUE THIS WEEK","DUE THIS MONTH","MAINT. OK","WARR. EXPIRED","WARR. EXPIRING"],
        ["#f85149","#e3a414","#388bfd","#3fb950","#f85149","#e3a414"]):
        col.markdown(f'<div class="metric-box"><div class="metric-num" style="color:{color}">{num}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    cf1,cf2,cf3,cf4 = st.columns(4)
    with cf1: fs = st.selectbox("Maintenance:", ["All","🔴 OVERDUE","🔴 DUE THIS WEEK","🟠 DUE THIS MONTH","🟡 UPCOMING","🟢 OK"])
    with cf2: ft = st.selectbox("Device Type:", ["All"] + sorted(df["device_type"].unique().tolist()))
    with cf3: fc = st.selectbox("Criticality:", ["All","Critical","High","Medium","Low"])
    with cf4: fw = st.selectbox("Warranty:", ["All","🔴 EXPIRED","🟠 EXPIRING SOON","🟡 VALID (<1yr)","🟢 VALID","⚪ NO DATA"])

    filt = df.copy()
    if fs != "All": filt = filt[filt["status"] == fs]
    if ft != "All": filt = filt[filt["device_type"] == ft]
    if fc != "All": filt = filt[filt["criticality"] == fc]
    if fw != "All": filt = filt[filt["warranty_status"] == fw]

    st.markdown(f"**Showing {len(filt)} assets**")
    for _, row in filt.iterrows():
        days     = row["days_remaining"]
        days_str = f"{int(days)} days" if pd.notna(days) else "Unknown"
        cc       = ("alert-critical" if pd.notna(days) and days < 0 else
                    "alert-high"     if pd.notna(days) and days <= 7 else
                    "alert-medium"   if pd.notna(days) and days <= 30 else "alert-ok")
        nd  = row["next_maintenance"].strftime("%d %b %Y") if pd.notna(row["next_maintenance"]) else "N/A"
        wd  = row["warranty_end"].strftime("%d %b %Y")     if pd.notna(row["warranty_end"])     else "N/A"
        vn, vc = parse_vendor(row["vendor_details"])
        st.markdown(f'<div class="{cc}"><strong>{row["status"]} | {row["asset_id"]}</strong> — {row["device_type"]} <span style="color:#8b949e;font-size:0.85rem">{row["asset_make"]} {row["model"]}</span><br><small>Next maintenance: <strong>{nd}</strong> | Days: <strong>{days_str}</strong> | Criticality: <strong>{row["criticality"]}</strong> | Urgency: <strong>{row["urgency_score"]:.1f}</strong></small><br><small>🛡️ Warranty: <strong>{row["warranty_status"]}</strong> (expires: {wd}) | 📞 <strong>{vn}</strong>{(" — "+vc) if vc else ""}</small></div>', unsafe_allow_html=True)

    with st.expander("📊 View Full Table"):
        st.dataframe(filt[["asset_id","device_type","asset_make","model","criticality","status","days_remaining","warranty_status","warranty_end","vendor_details","urgency_score"]].rename(columns={"asset_id":"ID","device_type":"Type","asset_make":"Brand","model":"Model","criticality":"Criticality","status":"Maint. Status","days_remaining":"Days Left","warranty_status":"Warranty","warranty_end":"Warr. End","vendor_details":"Vendor","urgency_score":"Urgency"}), use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════
# PAGE 4 — [NEW] TICKET MANAGEMENT
# ═══════════════════════════════════════════════════════
elif "🎫 Ticket Management" in page:
    st.markdown("# 🎫 Ticket Management System")
    st.markdown("*Raise, track, and close IT helpdesk tickets*")
    st.markdown("---")

    tickets = load_tickets()

    # Summary
    open_t     = len(tickets[tickets["status"] == "Open"])         if not tickets.empty else 0
    prog_t     = len(tickets[tickets["status"] == "In Progress"])  if not tickets.empty else 0
    closed_t   = len(tickets[tickets["status"] == "Closed"])       if not tickets.empty else 0
    total_t    = len(tickets)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><div class="metric-num" style="color:#388bfd">{open_t}</div><div class="metric-label">OPEN</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="metric-num" style="color:#e3a414">{prog_t}</div><div class="metric-label">IN PROGRESS</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="metric-num" style="color:#3fb950">{closed_t}</div><div class="metric-label">CLOSED</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-box"><div class="metric-num" style="color:#e6edf3">{total_t}</div><div class="metric-label">TOTAL</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Raise new ticket
    with st.expander("➕ Raise New Ticket", expanded=True):
        tc1, tc2 = st.columns(2)
        with tc1:
            raised_by   = st.text_input("Your Name *", placeholder="e.g. Ramesh Kumar")
            asset_sel   = st.selectbox("Affected Asset *", ["-- Select --"] + df["asset_id"].tolist())
            issue_desc  = st.text_area("Issue Description *", placeholder="Describe the problem in detail...", height=100)
        with tc2:
            priority    = st.selectbox("Priority *", ["Low","Medium","High","Critical"])
            assigned_to = st.text_input("Assign To", placeholder="e.g. IT Team / Suresh Singh")
            notes       = st.text_input("Additional Notes", placeholder="Any extra info...")

        if st.button("🎫 Raise Ticket", key="btn_raise"):
            if raised_by and asset_sel != "-- Select --" and issue_desc:
                asset_row   = df[df["asset_id"] == asset_sel].iloc[0]
                ticket_id   = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                new_row     = {
                    "ticket_id":   ticket_id,
                    "raised_by":   raised_by,
                    "asset_id":    asset_sel,
                    "device_type": asset_row["device_type"],
                    "issue":       issue_desc,
                    "priority":    priority,
                    "status":      "Open",
                    "assigned_to": assigned_to,
                    "raised_date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "closed_date": "",
                    "notes":       notes
                }
                tickets = pd.concat([tickets, pd.DataFrame([new_row])], ignore_index=True)
                save_tickets(tickets)
                st.success(f"✅ Ticket **{ticket_id}** raised successfully!")
                st.rerun()
            else:
                st.warning("Please fill in Name, Asset, and Issue Description")

    st.markdown("---")

    # View & Update Tickets
    st.markdown("### 📋 All Tickets")
    tf1, tf2 = st.columns(2)
    with tf1: t_filter = st.selectbox("Filter by Status:", ["All","Open","In Progress","Closed"])
    with tf2: t_priority = st.selectbox("Filter by Priority:", ["All","Critical","High","Medium","Low"])

    disp = tickets.copy() if not tickets.empty else tickets
    if not disp.empty:
        if t_filter   != "All": disp = disp[disp["status"]   == t_filter]
        if t_priority != "All": disp = disp[disp["priority"] == t_priority]

        for i, (_, row) in enumerate(disp.iterrows()):
            cc = ("ticket-closed"   if row["status"] == "Closed" else
                  "ticket-progress" if row["status"] == "In Progress" else "ticket-open")
            st.markdown(f"""
<div class="{cc}">
<strong>{row["ticket_id"]}</strong> &nbsp;|&nbsp; 
<span style="color:#79c0ff">{row["asset_id"]} — {row["device_type"]}</span> &nbsp;|&nbsp;
Priority: <strong>{row["priority"]}</strong> &nbsp;|&nbsp; Status: <strong>{row["status"]}</strong><br>
<small>👤 Raised by: <strong>{row["raised_by"]}</strong> on {row["raised_date"]} 
{(" | 🔧 Assigned to: <strong>"+str(row["assigned_to"])+"</strong>") if str(row.get("assigned_to","")) not in ["","nan"] else ""}
{(" | ✅ Closed: "+str(row["closed_date"])) if str(row.get("closed_date","")) not in ["","nan"] else ""}</small><br>
<small>📝 {row["issue"][:120]}{"..." if len(str(row["issue"])) > 120 else ""}</small>
</div>""", unsafe_allow_html=True)

            # Update status inline
            with st.expander(f"Update {row['ticket_id']}"):
                u1, u2 = st.columns(2)
                with u1:
                    new_status = st.selectbox("New Status:", ["Open","In Progress","Closed"], key=f"st_{i}")
                with u2:
                    update_note = st.text_input("Update Note:", key=f"un_{i}")
                if st.button("Update", key=f"upd_{i}"):
                    idx = tickets[tickets["ticket_id"] == row["ticket_id"]].index[0]
                    tickets.at[idx, "status"] = new_status
                    if update_note: tickets.at[idx, "notes"] = update_note
                    if new_status == "Closed":
                        tickets.at[idx, "closed_date"] = datetime.now().strftime("%d-%m-%Y %H:%M")
                    save_tickets(tickets)
                    st.success("✅ Ticket updated!")
                    st.rerun()
    else:
        st.info("No tickets yet. Raise your first ticket above!")


# ═══════════════════════════════════════════════════════
# PAGE 5 — [NEW] MAINTENANCE HISTORY LOG
# ═══════════════════════════════════════════════════════
elif "📋 Maintenance History" in page:
    st.markdown("# 📋 Maintenance History Log")
    st.markdown("*Log completed maintenance tasks and track service history*")
    st.markdown("---")

    history = load_history()

    # Summary
    total_logs = len(history)
    this_month = 0
    total_cost = 0
    if not history.empty:
        history["maintenance_date"] = pd.to_datetime(history["maintenance_date"], dayfirst=True, errors="coerce")
        today_ts = pd.Timestamp(date.today())
        this_month = len(history[history["maintenance_date"].dt.month == today_ts.month])
        try:
            total_cost = pd.to_numeric(history["cost"], errors="coerce").sum()
        except Exception: pass

    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="metric-box"><div class="metric-num" style="color:#3fb950">{total_logs}</div><div class="metric-label">TOTAL LOGS</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="metric-num" style="color:#388bfd">{this_month}</div><div class="metric-label">THIS MONTH</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="metric-num" style="color:#e3a414">₹{int(total_cost):,}</div><div class="metric-label">TOTAL COST</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Log new maintenance
    with st.expander("➕ Log Completed Maintenance", expanded=True):
        lc1, lc2 = st.columns(2)
        with lc1:
            log_asset    = st.selectbox("Asset *", ["-- Select --"] + df["asset_id"].tolist(), key="log_asset")
            maint_date   = st.date_input("Maintenance Date *", value=date.today(), key="log_date")
            done_by      = st.text_input("Done By *", placeholder="e.g. Vendor / Internal IT", key="log_by")
        with lc2:
            maint_type   = st.selectbox("Type *", ["Preventive Maintenance","Corrective Repair","Cleaning","Software Update","Hardware Upgrade","Inspection","Other"], key="log_type")
            cost         = st.number_input("Cost (₹)", min_value=0, value=0, key="log_cost")
            log_notes    = st.text_area("Notes", placeholder="What was done?", height=80, key="log_notes")

        if st.button("📋 Log Maintenance", key="btn_log"):
            if log_asset != "-- Select --" and done_by:
                asset_row = df[df["asset_id"] == log_asset].iloc[0]
                log_id    = f"LOG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                new_log   = {
                    "log_id":            log_id,
                    "asset_id":          log_asset,
                    "device_type":       asset_row["device_type"],
                    "maintenance_date":  maint_date.strftime("%d-%m-%Y"),
                    "done_by":           done_by,
                    "type":              maint_type,
                    "cost":              cost,
                    "notes":             log_notes
                }
                history = pd.concat([history, pd.DataFrame([new_log])], ignore_index=True)
                save_history(history)
                st.success(f"✅ Maintenance logged as **{log_id}**!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("Please fill in Asset and Done By fields")

    st.markdown("---")

    # View history
    st.markdown("### 📜 Maintenance History")
    hf1, hf2 = st.columns(2)
    with hf1: h_asset  = st.selectbox("Filter by Asset:",  ["All"] + df["asset_id"].tolist(), key="hf_asset")
    with hf2: h_type   = st.selectbox("Filter by Type:",   ["All","Preventive Maintenance","Corrective Repair","Cleaning","Software Update","Hardware Upgrade","Inspection","Other"], key="hf_type")

    if not history.empty:
        disp_h = history.copy()
        if h_asset != "All": disp_h = disp_h[disp_h["asset_id"] == h_asset]
        if h_type  != "All": disp_h = disp_h[disp_h["type"]     == h_type]
        disp_h = disp_h.sort_values("maintenance_date", ascending=False)

        for _, row in disp_h.iterrows():
            cost_str = f"₹{int(row['cost']):,}" if pd.notna(row['cost']) and str(row['cost']) not in ['0','0.0','nan'] else "No cost recorded"
            st.markdown(f"""
<div class="history-card">
<strong>✅ {row["log_id"]}</strong> &nbsp;|&nbsp;
<span style="color:#79c0ff">{row["asset_id"]} — {row["device_type"]}</span> &nbsp;|&nbsp;
<strong>{row["type"]}</strong><br>
<small>📅 Date: <strong>{row["maintenance_date"]}</strong> &nbsp;|&nbsp;
🔧 By: <strong>{row["done_by"]}</strong> &nbsp;|&nbsp;
💰 Cost: <strong>{cost_str}</strong></small>
{('<br><small>📝 '+str(row["notes"])+'</small>') if str(row.get("notes","")) not in ["","nan"] else ""}
</div>""", unsafe_allow_html=True)

        # Download history as CSV
        csv_data = disp_h.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download History CSV", csv_data,
                           f"NTPC_Maintenance_History_{date.today()}.csv", "text/csv")
    else:
        st.info("No maintenance logs yet. Log your first maintenance task above!")
