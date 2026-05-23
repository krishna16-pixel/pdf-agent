import streamlit as st
import json, re, io
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

st.markdown(
    "<h1 style='text-align:center; color:#1f4788;'>📄 AI PDF Report Generator</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#555;'>Enter any topic and get a professional multi-page PDF report instantly.</p>",
    unsafe_allow_html=True
)
st.divider()

# =========================
# API KEY FROM SECRETS
# =========================

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("⚠️ GROQ_API_KEY not found in Streamlit secrets. Please add it in Settings → Secrets.")
    st.stop()

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("### 📄 AI PDF Report Generator")
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
    st.markdown("**How to use:**")
    st.markdown("""
1. Type any topic below
2. Click Generate
3. Download your PDF
    """)
    st.markdown("---")
    st.caption("Powered by Groq LLaMA 3.1 + ReportLab")

# =========================
# GROQ CLIENT
# =========================

@st.cache_resource
def get_client(key):
    return ChatGroq(
        api_key=key,
        model="llama-3.1-8b-instant",
        temperature=0
    )

# =========================
# JSON CLEANER
# =========================

def clean_and_parse(text):
    # Strip markdown fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Fix smart/curly quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Extract JSON object only
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    text = text[start:end]

    # Fix real newlines inside JSON strings
    def fix_newlines(s):
        result = []
        inside = False
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '\\' and i + 1 < len(s):
                result.append(ch)
                result.append(s[i + 1])
                i += 2
                continue
            if ch == '"':
                inside = not inside
            if inside and ch == '\n':
                result.append('\\n')
            elif inside and ch == '\r':
                pass
            else:
                result.append(ch)
            i += 1
        return ''.join(result)

    text = fix_newlines(text)

    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}\n\nRAW (first 300 chars):\n{text[:300]}")

# =========================
# PLAN PROMPT
# =========================

def build_plan_prompt(user_input):
    return f"""
You are a JSON API. Output ONLY valid JSON. No markdown. No explanation. No prose.

TOPIC: {user_input}

Decide the best sections for a professional PDF report on this topic.

SECTION SELECTION RULES:
- Always include Introduction and Conclusion (type: text)
- Technical/process topics: include How It Works (type: text), ascii_diagram
- Topics with pros/cons: include Advantages vs Disadvantages (type: table_adv_dis)
- Topics with variants: include Types or Classification (type: bullets)
- Comparison topics: include Comparison Table (type: comparison_table)
- Science/tech topics: include Applications (type: bullets)
- Historical topics: include History (type: text)

OUTPUT THIS EXACT FORMAT — VALID JSON ONLY:
{{
  "title": "Full Report Title Here",
  "sections": [
    {{"heading": "Introduction", "type": "text"}},
    {{"heading": "How It Works", "type": "text"}},
    {{"heading": "Key Concepts", "type": "bullets"}},
    {{"heading": "Advantages vs Disadvantages", "type": "table_adv_dis"}},
    {{"heading": "Diagram: Process Flow", "type": "ascii_diagram"}},
    {{"heading": "Applications", "type": "bullets"}},
    {{"heading": "Conclusion", "type": "text"}}
  ]
}}

TOPIC: {user_input}
"""

# =========================
# SECTION PROMPT
# =========================

def build_section_prompt(topic, title, heading, stype):

    formats = {
        "text": (
            '{{\n'
            '  "type": "text",\n'
            '  "heading": "' + heading + '",\n'
            '  "body": "Write 200 to 250 words here as one paragraph. No real newlines inside the string."\n'
            '}}'
        ),
        "bullets": (
            '{{\n'
            '  "type": "bullets",\n'
            '  "heading": "' + heading + '",\n'
            '  "items": [\n'
            '    "Point 1 — detailed explanation in 2 to 3 sentences.",\n'
            '    "Point 2 — detailed explanation in 2 to 3 sentences.",\n'
            '    "Point 3 — detailed explanation in 2 to 3 sentences.",\n'
            '    "Point 4 — detailed explanation in 2 to 3 sentences."\n'
            '  ]\n'
            '}}'
        ),
        "table_adv_dis": (
            '{{\n'
            '  "type": "table_adv_dis",\n'
            '  "heading": "' + heading + '",\n'
            '  "advantages": [\n'
            '    "Advantage 1 — detailed explanation.",\n'
            '    "Advantage 2 — detailed explanation.",\n'
            '    "Advantage 3 — detailed explanation."\n'
            '  ],\n'
            '  "disadvantages": [\n'
            '    "Disadvantage 1 — detailed explanation.",\n'
            '    "Disadvantage 2 — detailed explanation.",\n'
            '    "Disadvantage 3 — detailed explanation."\n'
            '  ]\n'
            '}}'
        ),
        "ascii_diagram": (
            '{{\n'
            '  "type": "ascii_diagram",\n'
            '  "heading": "' + heading + '",\n'
            '  "caption": "One line description of what the diagram shows",\n'
            '  "diagram": "+----------+     +----------+     +----------+\\n|  Step 1  | --> |  Step 2  | --> |  Step 3  |\\n+----------+     +----------+     +----------+"\n'
            '}}'
        ),
        "comparison_table": (
            '{{\n'
            '  "type": "comparison_table",\n'
            '  "heading": "' + heading + '",\n'
            '  "columns": ["Feature", "Option A", "Option B"],\n'
            '  "rows": [\n'
            '    ["Feature 1", "Value A", "Value B"],\n'
            '    ["Feature 2", "Value A", "Value B"],\n'
            '    ["Feature 3", "Value A", "Value B"]\n'
            '  ]\n'
            '}}'
        )
    }

    fmt = formats.get(stype, formats["text"])

    return f"""
You are a JSON API. Output ONLY valid JSON. No markdown. No explanation. No extra text.

STRICT RULES:
1. Use straight double quotes only — never smart or curly quotes
2. Escape any quote inside a string value as \\"
3. No real newlines inside string values — use \\n only
4. No trailing commas
5. Output ONLY the JSON object — nothing before or after it

REPORT TOPIC: {topic}
REPORT TITLE: {title}
SECTION TO WRITE: {heading}

Write detailed, professional, informative content for this section.

OUTPUT FORMAT:
{fmt}
"""

# =========================
# STYLES
# =========================

def get_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=styles["Heading1"],
            fontSize=26, alignment=TA_CENTER,
            textColor=colors.HexColor("#1f4788"), spaceAfter=10
        ),
        "heading": ParagraphStyle(
            "heading", parent=styles["Heading2"],
            fontSize=15, textColor=colors.HexColor("#003d82"),
            spaceBefore=12, spaceAfter=6
        ),
        "body": ParagraphStyle(
            "body", parent=styles["Normal"],
            fontSize=11, leading=17
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=styles["Normal"],
            fontSize=11, leading=17, leftIndent=14, spaceAfter=4
        ),
        "ascii": ParagraphStyle(
            "ascii", parent=styles["Normal"],
            fontName="Courier", fontSize=9, leading=13,
            leftIndent=10,
            backColor=colors.HexColor("#f4f6fb"),
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=6, spaceBefore=6
        ),
        "caption": ParagraphStyle(
            "caption", parent=styles["Normal"],
            fontSize=9, alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"), spaceAfter=8
        ),
    }

# =========================
# PDF BUILDER
# =========================

def create_pdf_bytes(title, content_list):
    buffer = io.BytesIO()
    S      = get_styles()

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch
    )

    elements = []

    # ── Title page ──────────────────────────────────────────
    elements.append(Spacer(1, 2.8 * inch))
    elements.append(Paragraph(title, S["title"]))
    elements.append(Spacer(1, 0.3 * inch))

    line = Table([[""]], colWidths=[5 * inch], rowHeights=[3])
    line.setStyle(TableStyle([
        ("LINEABOVE",   (0, 0), (-1, 0), 2, colors.HexColor("#1f4788")),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(line)
    elements.append(PageBreak())

    # ── Sections ────────────────────────────────────────────
    for section in content_list:

        stype   = section.get("type", "text")
        heading = section.get("heading", "")

        parts = [
            Paragraph(heading, S["heading"]),
            Spacer(1, 0.04 * inch)
        ]

        # ── Plain text ──────────────────────────────────────
        if stype == "text":
            body = str(section.get("body", ""))
            body = body.replace("\\n", "<br/>").replace("\n", "<br/>")
            parts.append(Paragraph(body, S["body"]))

        # ── Bullet list ─────────────────────────────────────
        elif stype == "bullets":
            for item in section.get("items", []):
                t = str(item).replace("\\n", " ").replace("\n", " ")
                parts.append(Paragraph("• " + t, S["bullet"]))

        # ── Advantages vs Disadvantages ─────────────────────
        elif stype == "table_adv_dis":
            adv     = section.get("advantages", [])
            dis     = section.get("disadvantages", [])
            max_len = max(len(adv), len(dis), 1)

            td = [[
                Paragraph("<b>✔ Advantages</b>", S["body"]),
                Paragraph("<b>✘ Disadvantages</b>", S["body"])
            ]]
            for i in range(max_len):
                a = str(adv[i]) if i < len(adv) else ""
                d = str(dis[i]) if i < len(dis) else ""
                td.append([
                    Paragraph(a, S["body"]),
                    Paragraph(d, S["body"])
                ])

            cw  = (doc.width - 0.1 * inch) / 2
            tbl = Table(td, colWidths=[cw, cw], repeatRows=1)
            tbl.setStyle(TableStyle([
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND",    (0, 0), (-1,  0), colors.HexColor("#1f4788")),
                ("TEXTCOLOR",     (0, 0), (-1,  0), colors.white),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE",      (0, 0), (-1, -1), 10),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                *[("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f0f4ff"))
                  for r in range(2, max_len + 1, 2)]
            ]))
            parts.append(tbl)

        # ── Comparison table ────────────────────────────────
        elif stype == "comparison_table":
            cols = section.get("columns", [])
            rows = section.get("rows", [])
            if not cols:
                continue

            cw  = doc.width / len(cols)
            td  = [[Paragraph(f"<b>{c}</b>", S["body"]) for c in cols]]
            for row in rows:
                td.append([Paragraph(str(cell), S["body"]) for cell in row])

            tbl = Table(td, colWidths=[cw] * len(cols), repeatRows=1)
            tbl.setStyle(TableStyle([
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND",    (0, 0), (-1,  0), colors.HexColor("#003d82")),
                ("TEXTCOLOR",     (0, 0), (-1,  0), colors.white),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE",      (0, 0), (-1, -1), 10),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                *[("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef2ff"))
                  for r in range(2, len(rows) + 1, 2)]
            ]))
            parts.append(tbl)

        # ── ASCII diagram ────────────────────────────────────
        elif stype == "ascii_diagram":
            diagram = section.get("diagram", "").replace("\\n", "\n")
            caption = section.get("caption", "")

            box = Table(
                [[Preformatted(diagram, S["ascii"])]],
                colWidths=[doc.width]
            )
            box.setStyle(TableStyle([
                ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#1f4788")),
                ("BACKGROUND",    (0, 0), (-1, -1),    colors.HexColor("#f4f6fb")),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                ("TOPPADDING",    (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            parts.append(box)

            if caption:
                parts.append(Paragraph(f"Fig: {caption}", S["caption"]))

        # Keep heading with first content, rest flows freely
        elements.append(KeepTogether(parts[:2]))
        for part in parts[2:]:
            elements.append(part)
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()

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
    if not topic.strip():
        st.warning("⚠️ Please enter a topic.")
    else:
        try:
            client = get_client(api_key)

            # ── Step 1: Plan ─────────────────────────────────────
            with st.spinner("🧠 Planning report structure..."):
                plan_resp    = client.invoke(build_plan_prompt(topic))
                plan         = clean_and_parse(plan_resp.content)
                report_title = plan["title"]
                sections     = plan["sections"]

            st.info(f"📋 **{report_title}** — {len(sections)} sections planned")

            # ── Step 2: Generate each section ────────────────────
            content_list = []
            progress     = st.progress(0)
            status       = st.empty()

            for i, sec in enumerate(sections):
                heading = sec["heading"]
                stype   = sec["type"]

                status.markdown(
                    f"✍️ Writing **{heading}**... ({i + 1}/{len(sections)})"
                )

                try:
                    prompt   = build_section_prompt(topic, report_title, heading, stype)
                    response = client.invoke(prompt)
                    parsed   = clean_and_parse(response.content)
                    content_list.append(parsed)

                except Exception as sec_err:
                    st.warning(f"⚠️ Skipped '{heading}': {sec_err}")
                    content_list.append({
                        "type":    "text",
                        "heading": heading,
                        "body":    "Content for this section could not be generated."
                    })

                progress.progress((i + 1) / len(sections))

            # ── Step 3: Build PDF ─────────────────────────────────
            status.markdown("📄 Building PDF...")
            pdf_bytes = create_pdf_bytes(report_title, content_list)

            progress.progress(1.0)
            status.empty()
            st.balloons()
            st.success(f"✅ PDF ready — {len(content_list)} sections")

            # Section preview
            with st.expander("📋 Report Sections Preview"):
                for i, sec in enumerate(content_list, 1):
                    icon = {
                        "text":             "📝",
                        "bullets":          "🔵",
                        "table_adv_dis":    "✔✘",
                        "comparison_table": "📊",
                        "ascii_diagram":    "🖥️"
                    }.get(sec.get("type", ""), "📄")
                    st.markdown(
                        f"{icon} **{i}. {sec.get('heading', '')}** "
                        f"— `{sec.get('type', '')}`"
                    )

            # Download button
            safe_title = report_title[:40].replace(' ', '_')
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"{safe_title}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            with st.expander("🔍 Debug Info"):
                st.code(str(e))
