"""Builds docs/MediScan_Frontend.pdf — routes, design system and structure of apps/web.

Routes are parsed from apps/web/src/app/routes.ts so this document cannot drift from the app.
Run:  uv run --with reportlab python docs/src/build_frontend_pdf.py
"""

import os
import re
from collections import OrderedDict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "MediScan_Frontend.pdf")
ROUTES_TS = os.path.join(ROOT, "apps", "web", "src", "app", "routes.ts")
SHOTS = os.environ.get("SHOTS_DIR", "")  # optional directory with landing.png, result.png, ...
BASE_URL = "http://localhost:5173"

# ---------- parse routes.ts ----------
src = open(ROUTES_TS, encoding="utf-8").read()
pattern = re.compile(
    r'\{\s*path:\s*"([^"]+)",\s*name:\s*"([^"]+)",\s*group:\s*"([^"]+)",\s*description:\s*'
    r'"([^"]+)",(?:\s*example:\s*"([^"]+)",?)?\s*\}',
    re.S,
)
routes = [
    {"path": m[0], "name": m[1], "group": m[2], "description": m[3], "example": m[4] or ""}
    for m in pattern.findall(src)
]
assert routes, "no routes parsed — check the regex against routes.ts"
groups: "OrderedDict[str, list]" = OrderedDict()
for g in ["Public", "Auth", "Patient", "Doctor", "Admin", "Developer"]:
    groups[g] = [r for r in routes if r["group"] == g]

# ---------- styles (same family as the baseline document) ----------
ss = getSampleStyleSheet()
NAVY = colors.HexColor("#0B3D5C")
GREEN = colors.HexColor("#047857")
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13.5, spaceAfter=5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10.5, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
MONO = ParagraphStyle("mono", parent=CELL, fontName="Courier", fontSize=7.6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], keepWithNext=1, fontName="Helvetica-Bold", fontSize=17, leading=21, spaceBefore=6, spaceAfter=8, textColor=GREEN)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], keepWithNext=1, fontName="Helvetica-Bold", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5, textColor=GREEN)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], keepWithNext=1, fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#065F46"))
CODE = ParagraphStyle("code", parent=ss["Code"], fontName="Courier", fontSize=7.4, leading=9.2, backColor=colors.HexColor("#F1F8F5"), borderPadding=5, leftIndent=4, spaceAfter=8)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=26, leading=32, textColor=GREEN, alignment=0, spaceAfter=6)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=12, leading=16, textColor=colors.HexColor("#444444"))
CALLOUT = ParagraphStyle("callout", parent=BODY, backColor=colors.HexColor("#ECFDF5"), borderPadding=6, borderColor=colors.HexColor("#6EE7B7"), borderWidth=0.6, leftIndent=2, spaceBefore=4, spaceAfter=10)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))

story = []
P = lambda t, s=BODY: story.append(Paragraph(t, s))  # noqa: E731
h1 = lambda t: story.append(Paragraph(t, H1))  # noqa: E731
h2 = lambda t: story.append(Paragraph(t, H2))  # noqa: E731
h3 = lambda t: story.append(Paragraph(t, H3))  # noqa: E731
code = lambda t: story.append(Preformatted(t.strip("\n"), CODE))  # noqa: E731
sp = lambda n=6: story.append(Spacer(1, n))  # noqa: E731
callout = lambda t: story.append(Paragraph(t, CALLOUT))  # noqa: E731


def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=10) for i in items], bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8))
    sp(3)


def table(header, rows, widths, mono_cols=()):
    data = [[Paragraph(f'<font color="white">{h}</font>', CELLB) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), MONO if i in mono_cols else CELL) for i, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F2F8F5")))
    t.setStyle(TableStyle(st))
    story.append(t)
    sp(8)


def shot(name, caption, width=170 * mm):
    path = os.path.join(SHOTS, f"{name}.png") if SHOTS else ""
    if not path or not os.path.exists(path):
        return
    img = Image(path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    story.append(img)
    P(caption, SMALL)
    sp(6)


# ====================================================================
# COVER
# ====================================================================
sp(110)
P("MediScan", TITLE)
P("Frontend — Pages, Routes & Design System", ParagraphStyle("t2", parent=TITLE, fontSize=18, leading=22))
sp(10)
P("Reference for <b>apps/web</b>: every page and how to reach it, the 'Clinical Glass' design system, the "
  "application structure, what is simulated during the building phase, and how the UI will be wired to the API.", SUB)
sp(30)
table(["Field", "Value"], [
    ["Version", "1.0 (building phase)"],
    ["Date", "18 September 2026"],
    ["App", "apps/web — React 19 + TypeScript + Vite 8 + Tailwind 4"],
    ["Run locally", "cd apps/web && npm install && npm run dev  ->  " + BASE_URL],
    ["Routes", f"{len(routes)} pages, all reachable without authentication (guards are intentionally off)"],
    ["Source of truth for routes", "apps/web/src/app/routes.ts (this document is generated from it)"],
    ["Companion document", "MediScan_Engineering_Baseline.pdf (architecture, findings, roadmap)"],
], [42 * mm, 128 * mm])
sp(16)
callout("<b>Building phase.</b> No login is enforced and every screen runs on mock data shaped like the MediScan domain "
        "(Observation, CriticalFlag, RiskAssessment, ...). Forms and buttons navigate through the intended flow "
        "but do not call a server. Section 6 lists exactly what is simulated.")
story.append(PageBreak())

# ====================================================================
# 1. QUICK START
# ====================================================================
h1("1. Quick start")
code(f"""
cd apps/web
npm install
npm run dev            # {BASE_URL}
npm run typecheck      # tsc -b --noEmit
npm run lint           # eslint
npm run build          # production bundle in dist/
""")
P("Open <b>" + BASE_URL + "/pages</b> for a clickable index of every route with copy-to-clipboard URLs.")
P("Prerequisites: Node 22+. No backend, database or Docker is required to run the UI in this phase.")

# ====================================================================
# 2. ROUTES
# ====================================================================
h1("2. All pages and how to reach them")
P(f"{len(routes)} routes grouped by portal. Paths with <font face='Courier'>:id</font> include a working example URL.")
for g, rs in groups.items():
    if not rs:
        continue
    h2(f"2.{list(groups).index(g) + 1} {g}")
    rows = []
    for r in rs:
        url = BASE_URL + (r["example"] or r["path"])
        rows.append([r["name"], r["path"], url, r["description"]])
    table(["Page", "Route", "Open", "What is on it"], rows, [30 * mm, 30 * mm, 48 * mm, 62 * mm], mono_cols=(1, 2))

h2("2.7 The main flow, end to end")
code(f"""
{BASE_URL}/                        landing
  -> /reports/upload               drop a file or "Use sample" -> simulated upload + extraction
  -> /reports/r1051/review         verify values, accept the Creatinine suggestion (12 -> 1.2), edit, confirm
  -> /reports/r1042/result         critical flags, risk cards, explanation, values, doctor comment, specialists
  -> /doctors -> /doctors/d3       pick a day and a slot
  -> /book-appointment/d3          order summary -> Khalti (sandbox) -> "Simulate successful payment"
  -> /payment/success              confirmation
Doctor side:  /doctor/dashboard -> /doctor/patients/p2 (critical platelet flag) -> comment on the report
Admin side:   /admin/dashboard -> /admin/doctors (approve / reject a licence)
""")
story.append(PageBreak())

# ====================================================================
# 3. SCREENSHOTS (optional)
# ====================================================================
if SHOTS and os.path.isdir(SHOTS):
    h1("3. Screenshots")
    P("Captured at 1440px with `npm run shoot` (puppeteer, software WebGL).", SMALL)
    shot("landing", "Landing — hero with the animated glass DNA helix (three.js) and floating result preview.")
    shot("result", "Report result — safety status, six risk cards incl. NOT_ASSESSABLE, explanation with citations, doctor comment.")
    shot("admin", "Admin overview — platform stats, money, licences awaiting review.")
    shot("login", "Sign in — split layout with the floating glass orb.")
    story.append(PageBreak())

# ====================================================================
# 4. DESIGN SYSTEM
# ====================================================================
h1("4. Design system — 'Clinical Glass'")
P("White paper, emerald light, frosted surfaces. Only green and white are used for brand colour; red and amber appear "
  "solely for clinical severity. Everything is a token in <font face='Courier'>src/index.css</font> under "
  "<font face='Courier'>@theme</font>.")
h2("4.1 Tokens")
table(["Token", "Value", "Use"], [
    ["--color-brand-50 ... 950", "#ECFDF5 ... #022C22 (emerald scale)", "buttons, badges, gradients, sidebar active state"],
    ["--color-ink / ink-soft / ink-muted", "#0B1F1A / #3F5A52 / #7B918A", "text hierarchy"],
    ["--color-paper", "#F7FBF9", "page background (with three soft radial green gradients)"],
    ["--color-critical / warn", "#DC2626 / #D97706", "critical flags, out-of-range values only"],
    ["--font-display", "Sora 400-800", "headings, big numbers"],
    ["--font-sans", "Manrope 400-700", "body, UI"],
    ["--shadow-glass", "inset highlight + 20px emerald drop", "all glass surfaces"],
    ["--shadow-glow", "1px emerald ring + 40px emerald blur", "hover / selected states"],
    ["--radius-xl2 / xl3", "1.25rem / 1.75rem", "cards"],
], [45 * mm, 60 * mm, 65 * mm])
h2("4.2 Surface classes")
table(["Class", "Recipe", "Where"], [
    [".glass", "bg-white/55 + backdrop-blur-xl + white/70 border + shadow-glass", "cards, tables, list rows"],
    [".glass-strong", "bg-white/75 + backdrop-blur-2xl", "navbar when scrolled, sidebar, top bar, forms"],
    [".glass-dark", "bg-brand-950/70 + white/10 border, white text", "footer, 'How it works' section"],
    [".glass-pill", "rounded-full, bg-white/60, blur-lg", "tabs, chips, secondary buttons"],
    [".grid-dots", "radial dot grid masked to the centre", "hero / section backdrops"],
    [".text-gradient", "emerald -> teal gradient clipped to text", "one highlighted phrase per heading"],
], [32 * mm, 80 * mm, 58 * mm])
h2("4.3 Components (src/components/ui)")
table(["Component", "Variants / props", "Notes"], [
    ["Button", "primary, secondary, ghost, danger, glass · sm/md/lg · icon, trailing · to (renders Link)", "pill shape, gradient primary, press scale"],
    ["GlassCard", "strong, dark, hover, padded (framer-motion div)", "base surface for everything"],
    ["Badge", "brand, neutral, critical, warn, info · dot", "uppercase 11px"],
    ["Input / Textarea", "label, hint, icon", "glass field, focus ring"],
    ["Stat", "label, value, delta, icon", "dashboard KPI tile with soft glow"],
    ["SectionHeading / Eyebrow / PageHead", "eyebrow, title, lead, align, dark · crumbs, actions", "marketing vs portal headings"],
    ["Table / Td", "head[]", "glass wrapper, zebra-free, hover rows"],
    ["Tabs", "tabs[{id,label,count}] · layoutId pill", "animated selected pill"],
    ["Avatar, Progress, EmptyState, LinkCard", "-", "small primitives"],
    ["RiskCard, CriticalBanner, JobProgress, StatusBadge", "(pages/patient/shared.tsx)", "domain components used by patient and doctor portals"],
], [45 * mm, 70 * mm, 55 * mm])
h2("4.4 Motion")
bullets([
    "<b>Lenis</b> smooth scrolling on the document (lerp 0.09); the chat widget opts out with <font face='Courier'>data-lenis-prevent</font>.",
    "<b>ScrollProgress</b>: 3px gradient bar at the top of every layout.",
    "<b>Reveal</b>: fade + rise + un-blur when 1% of the element enters the viewport (once). Stagger with <font face='Courier'>delay</font>.",
    "<b>PageTransition</b>: route-level fade/slide; portals animate the <font face='Courier'>&lt;main&gt;</font> on path change.",
    "<b>Counter</b> (count-up numbers), <b>Marquee</b> (conditions ribbon), spring-animated tab and sidebar pills.",
    "Hover: cards lift 4px and gain the emerald glow; buttons scale 0.98 on press.",
])
h2("4.5 Three.js scenes (src/components/three)")
bullets([
    "<b>HeroScene</b> — a double helix of 68 frosted-glass and solid emerald beads with rungs, orbiting glass orbs, "
    "emerald sparkles, city environment lighting. Rotates slowly; tilts toward the pointer; parallaxes and fades on scroll. "
    "Lazy-loaded chunk (~950 kB) so it only ships on the landing page.",
    "<b>OrbScene</b> — a single floating glass icosahedron with sparkles, used on auth pages and marketing heroes.",
    "Both use <font face='Courier'>meshPhysicalMaterial</font> with transmission (real refraction) tinted with brand greens; "
    "DPR capped at 1.6 for performance. Desktop only (hidden below <font face='Courier'>lg</font>).",
])
story.append(PageBreak())

# ====================================================================
# 5. STRUCTURE
# ====================================================================
h1("5. Application structure")
code("""
apps/web/
|-- index.html                 fonts (Sora, Manrope), favicon /logo-mark.svg
|-- public/logo-mark.svg       MediScan mark (scan-frame corners + pulse line) - swap when a brand asset exists
`-- src/
    |-- main.tsx               React root
    |-- index.css              @theme tokens, glass utilities, keyframes
    |-- app/
    |   |-- App.tsx            router: PublicLayout / AppLayout(role) / bare auth pages; lazy chunks
    |   `-- routes.ts          ROUTES[] - single source of truth (router index page, this PDF)
    |-- components/
    |   |-- brand/Logo.tsx     LogoMark, Logo (wordmark)
    |   |-- ui/index.tsx       component kit (section 4.3)
    |   |-- motion/index.tsx   Reveal, PageTransition, ScrollProgress, SmoothScroll, Counter, Marquee
    |   |-- three/HeroScene.tsx HeroScene, OrbScene
    |   |-- layout/            PublicLayout (navbar+footer), AppLayout (sidebar per role)
    |   `-- chat/ChatWidget.tsx floating assistant (scripted replies in this phase)
    |-- pages/
    |   |-- public/            Landing, Marketing (Services, About, Contact, Privacy, Doctors, DoctorProfilePublic)
    |   |-- auth/Auth.tsx      Login, Signup, VerifyOtp, AdminLogin
    |   |-- patient/           Dashboard, Reports (list, upload, review, result), Account (appointments, profile, book, success), shared
    |   |-- doctor/Doctor.tsx  dashboard, appointments, patients, patient detail, availability, profile, verify
    |   |-- admin/Admin.tsx    dashboard, doctors, patients, create admin
    |   `-- dev/Dev.tsx        PagesIndex, StyleGuide, NotFound
    |-- mocks/data.ts          domain-shaped fixtures (see section 6)
    `-- lib/                   cn(), formatters
""")
h2("5.1 CUPID in the frontend")
table(["Property", "How it shows up here"], [
    ["Composable", "Pages compose the kit; features do not import each other. Layouts own chrome, pages own content."],
    ["Unix", "web renders and collects input. It never computes clinical meaning: thresholds, ranges and NOT_ASSESSABLE come from data."],
    ["Predictable", "Every state has a rendering (assessed / not assessable / awaiting review / failed). routes.ts is the one list of pages."],
    ["Idiomatic", "TypeScript strict, React Router 7, TanStack Query provider (ready for the generated client), Tailwind 4 tokens, ESLint + Prettier."],
    ["Domain-based", "Types and component names mirror docs/architecture.md: Observation, CriticalFlag, RiskAssessment, ReportJob, Consultation."],
], [30 * mm, 140 * mm])

# ====================================================================
# 6. SIMULATED VS REAL
# ====================================================================
h1("6. What is simulated in this phase")
table(["Area", "Today (mock)", "Phase 2/3 wiring"], [
    ["Authentication", "Any submit -> OTP page -> dashboard. 'Skip to dashboard' link. No guards.", "identity API; RouteGuards per role; HttpOnly cookie session"],
    ["Upload & extraction", "Timers: 1.2 s upload, 2.2 s extraction, then /reports/r1051/review", "POST /reports -> ReportJob; poll job status; real Observations with bbox"],
    ["Review", "Local state edits; accept/reject suggestions; 'Confirm & analyse' waits 2.6 s then opens r1042", "PUT observations; worker runs rules -> assess -> explain"],
    ["Result", "Fixtures r1042 (clean) and r1039 (critical platelets)", "GET report result; PDF export"],
    ["Chat widget", "Two scripted messages, canned reply after 0.9 s", "SSE stream from api -> llm_gateway with citations"],
    ["Doctors & booking", "6 fixture doctors; slot picker; Khalti panel is a styled placeholder", "consultation + billing APIs; Khalti ePayment redirect + server verification"],
    ["Doctor portal", "Fixture patients/appointments; comment saved to local state", "policy-checked patient access; audit events"],
    ["Admin portal", "Approve/reject updates local state", "administration API; licence review decisions logged"],
    ["Charts", "Static trend and per-day arrays", "Observation time series per analyte"],
], [30 * mm, 70 * mm, 70 * mm])
h2("6.1 Swapping mocks for the API")
bullets([
    "Generate the client: <font face='Courier'>openapi-typescript packages/contracts/openapi/api.yaml -o src/api/schema.ts</font> (make contracts).",
    "Replace imports from <font face='Courier'>@/mocks/data</font> with TanStack Query hooks in <font face='Courier'>src/api/</font>; keep the types — they were written to match the contracts.",
    "Add <font face='Courier'>RouteGuards</font> around the three portal layouts once identity is live; the 'Building phase · no auth' badge is removed then.",
    "Replace timers in Upload/Review with job polling on <font face='Courier'>ReportJob.status</font> using the existing JobProgress component.",
])
story.append(PageBreak())

# ====================================================================
# 7. CHECKLISTS
# ====================================================================
h1("7. Frontend checklists")
h2("7.1 Adding a page")
bullets([
    "Add the route to <font face='Courier'>src/app/routes.ts</font> first (name, group, description, example).",
    "Create the component under <font face='Courier'>src/pages/&lt;group&gt;/</font>; use PageHead (portal) or SectionHeading (public).",
    "Register it in <font face='Courier'>App.tsx</font> under the right layout; lazy import.",
    "Use only kit components and tokens; no raw hex colours, no new fonts.",
    "Every server state must render: loading, empty, error, and domain states (e.g. NOT_ASSESSABLE).",
    "Run <font face='Courier'>npm run typecheck && npm run lint</font>; regenerate this PDF with <font face='Courier'>make doc-frontend</font>.",
])
h2("7.2 Accessibility & performance")
bullets([
    "All icon-only buttons have <font face='Courier'>aria-label</font>; focus rings via <font face='Courier'>.ring-focus</font>.",
    "3D scenes are <font face='Courier'>aria-hidden</font> and desktop-only; text never depends on them.",
    "Three.js and Recharts are separate lazy chunks; initial route JS is ~160 kB gzip.",
    "Respect <font face='Courier'>prefers-reduced-motion</font> before launch (Phase 5 item).",
])
sp(20)
P("<i>Generated from apps/web/src/app/routes.ts. End of document.</i>", SMALL)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(20 * mm, 12 * mm, "MediScan - Frontend v1.0 - 18 Sep 2026 - Internal")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm, title="MediScan Frontend", author="MediScan team")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", os.path.abspath(OUT), "-", len(routes), "routes")
