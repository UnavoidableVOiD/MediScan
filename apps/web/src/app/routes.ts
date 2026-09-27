/**
 * Single source of truth for every page. Used by the router, the /pages index
 * and the docs generator (docs/src/build_frontend_pdf.py reads this file).
 */
export type RouteGroup = "Public" | "Auth" | "Patient" | "Doctor" | "Admin" | "Developer";

export interface RouteDef {
  path: string;
  name: string;
  group: RouteGroup;
  description: string;
  example?: string; // concrete URL when the path has params
}

export const ROUTES: RouteDef[] = [
  // ---------------------------------------------------------------- public
  {
    path: "/",
    name: "Landing",
    group: "Public",
    description: "Hero with 3D helix, how it works, portals, trust, CTA.",
  },
  {
    path: "/services",
    name: "Services",
    group: "Public",
    description: "The six capabilities as cards + call to action.",
  },
  {
    path: "/doctors",
    name: "Find doctors",
    group: "Public",
    description: "Verified doctors with specialization filter.",
  },
  {
    path: "/doctors/:id",
    name: "Doctor profile & booking",
    group: "Public",
    description: "Bio, stats, licence note, day/slot picker.",
    example: "/doctors/d1",
  },
  { path: "/about", name: "About", group: "Public", description: "Mission, principles, team." },
  {
    path: "/contact",
    name: "Contact",
    group: "Public",
    description: "Contact details and a message form.",
  },
  {
    path: "/privacy",
    name: "Privacy",
    group: "Public",
    description: "Draft privacy policy in plain language.",
  },

  // ------------------------------------------------------------------ auth
  {
    path: "/login",
    name: "Sign in",
    group: "Auth",
    description: "Email/password + Google; continues to OTP.",
  },
  {
    path: "/signup",
    name: "Create account",
    group: "Auth",
    description: "Patient or doctor role selection.",
  },
  { path: "/verify-otp", name: "Verify OTP", group: "Auth", description: "Six-digit code entry." },
  { path: "/admin/login", name: "Admin sign in", group: "Auth", description: "Staff-only login." },

  // --------------------------------------------------------------- patient
  {
    path: "/dashboard",
    name: "Patient dashboard",
    group: "Patient",
    description: "Critical banner, stats, trend chart, upcoming consultation, recent reports.",
  },
  {
    path: "/reports",
    name: "My reports",
    group: "Patient",
    description: "All uploads with pipeline status.",
  },
  {
    path: "/reports/upload",
    name: "Upload report",
    group: "Patient",
    description: "Drag-and-drop → simulated upload/extraction → review.",
  },
  {
    path: "/reports/:id/review",
    name: "Review extracted values",
    group: "Patient",
    description:
      "Document preview with highlights, per-value confidence, accept/reject suggestions, edit, confirm.",
    example: "/reports/r1051/review",
  },
  {
    path: "/reports/:id/result",
    name: "Report result",
    group: "Patient",
    description:
      "Critical flags, risk cards (assessed / not assessable), explanation (patient / clinician), values table, doctor comment, suggested specialists.",
    example: "/reports/r1042/result",
  },
  {
    path: "/appointments",
    name: "Appointments",
    group: "Patient",
    description: "Upcoming and past consultations, pay / cancel.",
  },
  {
    path: "/book-appointment/:id",
    name: "Book & pay",
    group: "Patient",
    description: "Order summary, Khalti (sandbox) checkout.",
    example: "/book-appointment/d3?slot=18:00",
  },
  {
    path: "/payment/success",
    name: "Payment success",
    group: "Patient",
    description: "Confirmation with transaction details.",
  },
  {
    path: "/profile",
    name: "Patient profile",
    group: "Patient",
    description: "Personal details, data export, access log, delete account.",
  },

  // ---------------------------------------------------------------- doctor
  {
    path: "/doctor/dashboard",
    name: "Doctor dashboard",
    group: "Doctor",
    description: "Today's consultations, revenue, weekly chart.",
  },
  {
    path: "/doctor/appointments",
    name: "Doctor appointments",
    group: "Doctor",
    description: "Upcoming / completed / cancelled with linked reports.",
  },
  {
    path: "/doctor/patients",
    name: "Doctor patients",
    group: "Doctor",
    description: "Linked and booked patients with risk badges.",
  },
  {
    path: "/doctor/patients/:id",
    name: "Patient detail",
    group: "Doctor",
    description:
      "Critical banner, clinician summary, risk cards, values, history, comment + private notes.",
    example: "/doctor/patients/p2",
  },
  {
    path: "/doctor/availability",
    name: "Availability",
    group: "Doctor",
    description: "Add/remove bookable windows.",
  },
  {
    path: "/doctor/profile",
    name: "Doctor profile",
    group: "Doctor",
    description: "Public profile editor.",
  },
  {
    path: "/doctor/verify",
    name: "Verification",
    group: "Doctor",
    description: "Licence status timeline and documents.",
  },

  // ----------------------------------------------------------------- admin
  {
    path: "/admin/dashboard",
    name: "Admin dashboard",
    group: "Admin",
    description: "Platform stats, reports/day chart, money, licences awaiting review.",
  },
  {
    path: "/admin/doctors",
    name: "Admin · doctors",
    group: "Admin",
    description: "Pending / verified / rejected with approve & reject panel.",
  },
  {
    path: "/admin/patients",
    name: "Admin · patients",
    group: "Admin",
    description: "Searchable patient accounts.",
  },
  {
    path: "/admin/create-admin",
    name: "Create admin",
    group: "Admin",
    description: "Sub-admin creation with permission summary.",
  },

  // ------------------------------------------------------------- developer
  {
    path: "/pages",
    name: "Pages index",
    group: "Developer",
    description: "Every route with a live link (this list).",
  },
  {
    path: "/styleguide",
    name: "Style guide",
    group: "Developer",
    description: "Tokens, type, buttons, badges, cards.",
  },
];

export const resolve = (r: RouteDef) => r.example ?? r.path;
