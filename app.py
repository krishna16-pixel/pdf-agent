import streamlit as st
import json, time, re, io
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak, KeepTogether, Preformatted
)
from reportlab.lib import colors

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI PDF Report Generator",
    page_icon="📄",
    layout="centered"
)

st.markdown("""
    <style>
        .main { background-color: #f8f9fb; }
        .stTextArea textarea { font-size: 15px; }
        .stButton > button {
            background-color: #1f4788;
            color: white;
            font-size: 16px;
            padding: 0.6em 2em;
            border-radius: 8px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #003d82;
            color: white;
        }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown("<h1 style='text-align:center; color:#1f4788;'>📄 AI PDF Report Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Enter any topic and get a professional multi-page PDF report instantly.</p>", unsafe_allow_html=True)
st.divider()

# =========================
# SIDEBAR — API KEY
# =========================

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("---")
    st.markdown("**Supported Section Types:**")
    st.markdown("""
    - 📝 Text paragraphs
    - 🔵 Bullet points
    - ✔✘ Advantages vs Disadvantages
    - 📊 Comparison tables
    - 🖥️ ASCII diagrams
    """)
    st.markdown("---")
    st.caption("Powered by Groq LLaMA 3.1 + ReportLab")

# =========================
# GROQ CLIENT
# =========================

@st.cache_resource
def get_client(key):
    return ChatGroq(api_key=key, model="llama-3.1-8b-instant")

# =========================
# JSON CLEANER
# =========================

def clean_json_text(text):
    text = text.replace("```json", "").replace("```", "")
    text = text.replace("\u201c", '\\"').replace("\u201d", '\\"')
    text = text.replace("\u2018", "\\'").replace("\u2019", "\\'")

    def fix_newlines(s):
        result = []
        inside = False
        for i, ch in enumerate(s):
            if ch == '"' and (i == 0 or s[i-1] != '\\'):
                inside = not inside
                result.append(ch)
            elif ch == '\n' and inside:
                result.append('\\n')
            elif ch == '\r' and inside:
                pass
            elif ch == '\t' and inside:
                result.append('  ')
            else:
                result.append(ch)
        return ''.join(result)

    text = fix_newlines(text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return text[start:end]


def extract_json(text):
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        return json.loads(cleaned)

# =========================
# PROMPT
# =========================

def build_prompt(user_input):
    return f"""
You are a professional academic report writer.

TOPIC: {user_input}

CRITICAL JSON RULES:
1. Use ONLY straight double quotes. NEVER smart/curly quotes.
2. Escape inner quotes as \\"
3. No real newlines inside strings — use \\n instead.
4. No trailing commas. No comments.
5. Output ONLY raw JSON — no explanation, no markdown.

CONTENT RULES:
- Pick ONLY sections relevant to the topic.
- "Advantages vs Disadvantages" ONLY if topic benefits from comparison.
- "How It Works" for technical/process topics.
- "History" for historical topics.
- "Types" if topic has variants.
- "Applications" for tech/science topics.
- "Comparison Table" if comparing multiple things.
- "ASCII Diagram" for structural/process/flow topics.
- Always include Introduction and Conclusion.
- Every section MUST be 150-200+ words minimum.

JSON FORMAT:
{{
  "title": "Full Report Title",
  "content": [
    {{
      "type": "text",
      "heading": "Introduction",
      "body": "long detailed paragraph..."
    }},
    {{
      "type": "bullets",
      "heading": "Key Concepts",
      "items": ["Point 1 — detailed...", "Point 2 — detailed..."]
    }},
    {{
      "type": "table_adv_dis",
      "heading": "Advantages vs Disadvantages",
      "advantages": ["Advantage 1...", "Advantage 2..."],
      "disadvantages": ["Disadvantage 1...", "Disadvantage 2..."]
    }},
    {{
      "type": "comparison_table",
      "heading": "Comparison",
      "columns": ["Feature", "Option A", "Option B"],
      "rows": [["Row label", "Value A", "Value B"]]
    }},
    {{
      "type": "ascii_diagram",
      "heading": "Diagram: How It Works",
      "caption": "Brief description",
      "diagram": "+--------+     +--------+\\n| Step 1 | --> | Step 2 |\\n+--------+     +--------+"
    }},
    {{
      "type": "text",
      "heading": "Conclusion",
      "body": "long conclusion paragraph..."
    }}
  ]
}}

TOPIC: {user_input}
"""

# =========================
# STYLES
# =========================

def get_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=styles["Heading1"],
            fontSize=26, alignment=TA_CENTER,
            textColor=colors.HexColor("#1f4788"), spaceAfter=10),
        "heading": ParagraphStyle("heading", parent=styles["Heading2"],
            fontSize=15, textColor=colors.HexColor("#003d82"),
            spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("body", parent=styles["Normal"],
            fontSize=11, leading=17),
        "bullet": ParagraphStyle("bullet", parent=styles["Normal"],
            fontSize=11, leading=17, leftIndent=14, spaceAfter=4),
        "ascii": ParagraphStyle("ascii", parent=styles["Normal"],
            fontName="Courier", fontSize=9, leading=13, leftIndent=10,
            backColor=colors.HexColor("#f4f6fb"),
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=6, spaceBefore=6),
        "caption": ParagraphStyle("caption", parent=styles["Normal"],
            fontSize=9, alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"), spaceAfter=8),
    }

# =========================
# PDF BUILDER → returns bytes
# =========================

def create_pdf_bytes(title, content_list):
    buffer = io.BytesIO()       # ← write to memory, not disk
    S = get_styles()

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.7*inch, bottomMargin=0.7*inch,
        leftMargin=0.8*inch, rightMargin=0.8*inch
    )

    elements = []

    # Title page
    elements.append(Spacer(1, 2.8*inch))
    elements.append(Paragraph(title, S["title"]))
    elements.append(Spacer(1, 0.3*inch))
    line = Table([[""]], colWidths=[5*inch], rowHeights=[3])
    line.setStyle(TableStyle([
        ("LINEABOVE",   (0,0), (-1,0), 2, colors.HexColor("#1f4788")),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
    ]))
    elements.append(line)
    elements.append(PageBreak())

    for section in content_list:
        stype   = section.get("type", "text")
        heading = section.get("heading", "")

        parts = [Paragraph(heading, S["heading"]), Spacer(1, 0.04*inch)]

        if stype == "text":
            body = str(section.get("body", "")).replace("\\n", "<br/>").replace("\n", "<br/>")
            parts.append(Paragraph(body, S["body"]))

        elif stype == "bullets":
            for item in section.get("items", []):
                t = str(item).replace("\\n", " ").replace("\n", " ")
                parts.append(Paragraph("• " + t, S["bullet"]))

        elif stype == "table_adv_dis":
            adv = section.get("advantages", [])
            dis = section.get("disadvantages", [])
            max_len = max(len(adv), len(dis), 1)
            td = [[Paragraph("<b>✔ Advantages</b>", S["body"]),
                   Paragraph("<b>✘ Disadvantages</b>", S["body"])]]
            for i in range(max_len):
                a = str(adv[i]) if i < len(adv) else ""
                d = str(dis[i]) if i < len(dis) else ""
                td.append([Paragraph(a, S["body"]), Paragraph(d, S["body"])])
            cw = (doc.width - 0.1*inch) / 2
            tbl = Table(td, colWidths=[cw, cw], repeatRows=1)
            tbl.setStyle(TableStyle([
                ("GRID",         (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND",   (0,0),(-1,0),  colors.HexColor("#1f4788")),
                ("TEXTCOLOR",    (0,0),(-1,0),  colors.white),
                ("VALIGN",       (0,0),(-1,-1), "TOP"),
                ("FONTSIZE",     (0,0),(-1,-1), 10),
                ("LEFTPADDING",  (0,0),(-1,-1), 8),
                ("RIGHTPADDING", (0,0),(-1,-1), 8),
                ("TOPPADDING",   (0,0),(-1,-1), 6),
                ("BOTTOMPADDING",(0,0),(-1,-1), 6),
                *[("BACKGROUND",(0,r),(-1,r), colors.HexColor("#f0f4ff"))
                  for r in range(2, max_len+1, 2)]
            ]))
            parts.append(tbl)

        elif stype == "comparison_table":
            cols = section.get("columns", [])
            rows = section.get("rows", [])
            if not cols: continue
            cw = doc.width / len(cols)
            td = [[Paragraph(f"<b>{c}</b>", S["body"]) for c in cols]]
            for row in rows:
                td.append([Paragraph(str(cell), S["body"]) for cell in row])
            tbl = Table(td, colWidths=[cw]*len(cols), repeatRows=1)
            tbl.setStyle(TableStyle([
                ("GRID",         (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND",   (0,0),(-1,0),  colors.HexColor("#003d82")),
                ("TEXTCOLOR",    (0,0),(-1,0),  colors.white),
                ("VALIGN",       (0,0),(-1,-1), "TOP"),
                ("FONTSIZE",     (0,0),(-1,-1), 10),
                ("LEFTPADDING",  (0,0),(-1,-1), 8),
                ("RIGHTPADDING", (0,0),(-1,-1), 8),
                ("TOPPADDING",   (0,0),(-1,-1), 6),
                ("BOTTOMPADDING",(0,0),(-1,-1), 6),
                *[("BACKGROUND",(0,r),(-1,r), colors.HexColor("#eef2ff"))
                  for r in range(2, len(rows)+1, 2)]
            ]))
            parts.append(tbl)

        elif stype == "ascii_diagram":
            diagram = section.get("diagram", "").replace("\\n", "\n")
            caption = section.get("caption", "")
            box = Table([[Preformatted(diagram, S["ascii"])]], colWidths=[doc.width])
            box.setStyle(TableStyle([
                ("BOX",          (0,0),(-1,-1), 1,   colors.HexColor("#1f4788")),
                ("BACKGROUND",   (0,0),(-1,-1),      colors.HexColor("#f4f6fb")),
                ("LEFTPADDING",  (0,0),(-1,-1), 10),
                ("RIGHTPADDING", (0,0),(-1,-1), 10),
                ("TOPPADDING",   (0,0),(-1,-1), 8),
                ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ]))
            parts.append(box)
            if caption:
                parts.append(Paragraph(f"Fig: {caption}", S["caption"]))

        elements.append(KeepTogether(parts[:2]))
        for part in parts[2:]:
            elements.append(part)
        elements.append(Spacer(1, 0.2*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()   # return bytes

# =========================
# MAIN UI
# =========================

topic = st.text_area(
    "📝 Enter your report topic:",
    placeholder="e.g. Artificial Intelligence, Solar Energy vs Coal, TCP/IP Networking...",
    height=120
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate = st.button("🚀 Generate PDF Report", use_container_width=True)

if generate:
    if not api_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar.")
    elif not topic.strip():
        st.warning("⚠️ Please enter a topic.")
    else:
        with st.spinner("⏳ Generating content with AI..."):
            try:
                client   = get_client(api_key)
                response = client.invoke(build_prompt(topic))
                text     = response.content
                data     = extract_json(text)
                title    = data["title"]
                content  = data["content"]

                st.success(f"✅ Parsed **{len(content)} sections** — building PDF...")

                with st.spinner("📄 Building PDF..."):
                    pdf_bytes = create_pdf_bytes(title, content)

                st.balloons()

                # Section preview
                with st.expander("📋 Report Sections Preview"):
                    for i, sec in enumerate(content, 1):
                        st.markdown(f"**{i}. {sec.get('heading','')}** — `{sec.get('type','')}`")

                # Download button
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{title[:40].replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")
                with st.expander("🔍 Debug Info"):
                    st.code(str(e))