import { useState } from "react";

// ── VERIFIED CONSTANTS (stress test 41/43 pass, 2 bad test assertions) ──
const CONSTANTS = {
  image_10_4: 55,
  image_10_3: 10,
  image_8_4:  36,
  image_16_3: 16,
  image_5_4:  15,
  components: 2,
  cycle_nodes: [0, 6174],
  cycle_lengths: [1, 1],
  tau_max: 7,
  shell3_frac: 0.2400,
  throughput: 184806,
  sig_time_s: 0.0100,
  null_image_mean: 6323.4,
};

const SHELLS = [
  { depth: 0, count: 2,    pct: 0.02 },
  { depth: 1, count: 392,  pct: 3.92 },
  { depth: 2, count: 576,  pct: 5.76 },
  { depth: 3, count: 2400, pct: 24.00 },
  { depth: 4, count: 1272, pct: 12.72 },
  { depth: 5, count: 1518, pct: 15.18 },
  { depth: 6, count: 1656, pct: 16.56 },
  { depth: 7, count: 2184, pct: 21.84 },
];

const CROSS_BASE = [
  { label: "K₁₀,₄", b: 10, d: 4, img: 55,  tau: 7, note: "primary system" },
  { label: "K₁₀,₃", b: 10, d: 3, img: 10,  tau: "—", note: "3-digit classical" },
  { label: "K₈,₄",  b: 8,  d: 4, img: 36,  tau: 5, note: "octal" },
  { label: "K₁₆,₃", b: 16, d: 3, img: 16,  tau: "—", note: "hex" },
  { label: "K₅,₄",  b: 5,  d: 4, img: 15,  tau: "—", note: "base-5" },
];

const OPEN = [
  "Shell-3 plateau at 24.00% — structural bottleneck exists, no derivation yet",
  "Flat plateau at depths 3,4,5 each ≈10 states in 54-state lattice — unexplained",
  "K_b fixed-point iff 5|b (proved for d=4) — generalize to arbitrary d",
  "Entropy gradient H(Sᵢ) − H(Sᵢ₊₁) across shells — not yet computed",
  "Shell-3 invariance across (b,d) — not yet tested systematically",
  "Z-score framing for shell significance — not valid; needs KS or Wasserstein",
];

// ── colour tokens ──────────────────────────────────────────────────────────
const C = {
  bg:      "#09090b",
  panel:   "#111114",
  border:  "#27272a",
  dim:     "#3f3f46",
  muted:   "#71717a",
  body:    "#a1a1aa",
  hi:      "#e4e4e7",
  amber:   "#f59e0b",
  green:   "#22c55e",
  red:     "#ef4444",
  blue:    "#38bdf8",
  shell3:  "#f59e0b",
};

const mono = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace";
const sans = "'IBM Plex Sans', 'DM Sans', sans-serif";

// ── tiny atoms ─────────────────────────────────────────────────────────────
function Label({ children, color = C.muted }) {
  return (
    <span style={{ fontFamily: mono, fontSize: 9, letterSpacing: "0.12em",
      textTransform: "uppercase", color, display: "block", marginBottom: 4 }}>
      {children}
    </span>
  );
}

function Val({ children, color = C.hi, size = 22 }) {
  return (
    <span style={{ fontFamily: mono, fontSize: size, color, fontWeight: 700,
      lineHeight: 1 }}>
      {children}
    </span>
  );
}

function Stat({ label, value, sub, accent = C.hi }) {
  return (
    <div style={{ padding: "14px 16px", background: C.panel,
      border: `1px solid ${C.border}`, borderRadius: 6 }}>
      <Label>{label}</Label>
      <Val color={accent}>{value}</Val>
      {sub && <span style={{ fontFamily: mono, fontSize: 10, color: C.muted,
        display: "block", marginTop: 4 }}>{sub}</span>}
    </div>
  );
}

// ── shell bar chart ─────────────────────────────────────────────────────────
function ShellChart() {
  const maxPct = Math.max(...SHELLS.map(s => s.pct));
  return (
    <div>
      <Label>shell depth distribution — K₁₀,₄  (N = 10 000)</Label>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {SHELLS.map(({ depth, count, pct }) => {
          const isS3 = depth === 3;
          const barW = `${(pct / maxPct) * 100}%`;
          const barColor = isS3 ? C.amber : depth === 0 ? C.red : C.blue;
          return (
            <div key={depth} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontFamily: mono, fontSize: 11, color: C.muted,
                width: 52, textAlign: "right", flexShrink: 0 }}>
                τ={depth}
              </span>
              <div style={{ flex: 1, height: 18, background: C.border,
                borderRadius: 3, overflow: "hidden", position: "relative" }}>
                <div style={{ width: barW, height: "100%",
                  background: barColor, opacity: isS3 ? 1 : 0.65,
                  borderRadius: 3,
                  boxShadow: isS3 ? `0 0 8px ${C.amber}66` : "none",
                  transition: "width 0.4s ease" }} />
              </div>
              <span style={{ fontFamily: mono, fontSize: 10,
                color: isS3 ? C.amber : C.body, width: 80, flexShrink: 0 }}>
                {count} ({pct.toFixed(2)}%)
                {isS3 && " ◄ S₃"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── cross-base table ────────────────────────────────────────────────────────
function CrossBaseTable() {
  return (
    <div>
      <Label>cross-base image sizes (verified)</Label>
      <table style={{ width: "100%", borderCollapse: "collapse",
        fontFamily: mono, fontSize: 11 }}>
        <thead>
          <tr>
            {["system","base","digits","|Im|","τ_max","note"].map(h => (
              <th key={h} style={{ textAlign: "left", padding: "5px 10px",
                color: C.muted, borderBottom: `1px solid ${C.border}`,
                fontWeight: 400, letterSpacing: "0.08em", fontSize: 9,
                textTransform: "uppercase" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {CROSS_BASE.map((r, i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? "transparent" : "#ffffff05" }}>
              <td style={{ padding: "6px 10px", color: C.blue }}>{r.label}</td>
              <td style={{ padding: "6px 10px", color: C.body }}>{r.b}</td>
              <td style={{ padding: "6px 10px", color: C.body }}>{r.d}</td>
              <td style={{ padding: "6px 10px", color: C.hi, fontWeight: 700 }}>{r.img}</td>
              <td style={{ padding: "6px 10px", color: C.body }}>{r.tau}</td>
              <td style={{ padding: "6px 10px", color: C.muted }}>{r.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── open questions ──────────────────────────────────────────────────────────
function OpenItems() {
  return (
    <div>
      <Label color={C.red}>open / unproved — do not state as results</Label>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {OPEN.map((q, i) => (
          <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
            <span style={{ color: C.red, fontFamily: mono, fontSize: 11,
              flexShrink: 0, marginTop: 1 }}>✗</span>
            <span style={{ fontFamily: mono, fontSize: 11, color: C.body,
              lineHeight: 1.5 }}>{q}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── audit badge ─────────────────────────────────────────────────────────────
function AuditBadge() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10,
      padding: "8px 14px", background: "#052e16",
      border: `1px solid #166534`, borderRadius: 6 }}>
      <span style={{ color: C.green, fontFamily: mono, fontSize: 13,
        fontWeight: 700 }}>41/43</span>
      <span style={{ color: C.muted, fontFamily: mono, fontSize: 10 }}>PASS</span>
      <span style={{ color: C.dim, fontFamily: mono, fontSize: 10 }}>|</span>
      <span style={{ color: "#fbbf24", fontFamily: mono, fontSize: 10 }}>
        2 failures = bad test assertions, not engine failures
      </span>
    </div>
  );
}

// ── tab nav ─────────────────────────────────────────────────────────────────
function Tab({ active, onClick, children }) {
  return (
    <button onClick={onClick}
      style={{ background: "none", border: "none", cursor: "pointer",
        fontFamily: mono, fontSize: 11, letterSpacing: "0.08em",
        textTransform: "uppercase", padding: "8px 14px",
        color: active ? C.hi : C.muted,
        borderBottom: active ? `2px solid ${C.amber}` : "2px solid transparent",
        transition: "color 0.15s" }}>
      {children}
    </button>
  );
}

// ── main ────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("overview");

  const compressionRatio =
    ((CONSTANTS.null_image_mean - CONSTANTS.image_10_4) /
      CONSTANTS.null_image_mean * 100).toFixed(1);

  return (
    <div style={{ background: C.bg, minHeight: "100vh", padding: "24px",
      fontFamily: sans, color: C.body }}>

      {/* ── header ── */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12,
          marginBottom: 6 }}>
          <h1 style={{ fontFamily: mono, fontSize: 18, color: C.hi,
            fontWeight: 700, margin: 0, letterSpacing: "-0.02em" }}>
            KSG-GMFLOW-A
          </h1>
          <span style={{ fontFamily: mono, fontSize: 11, color: C.muted }}>
            Kaprekar Spectral Geometry · Audit Dashboard
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <AuditBadge />
          <span style={{ fontFamily: mono, fontSize: 10, color: C.dim }}>
            b=10, d=4 · N=10 000 · stress-tested in session
          </span>
        </div>
      </div>

      {/* ── tabs ── */}
      <div style={{ borderBottom: `1px solid ${C.border}`, marginBottom: 20,
        display: "flex", gap: 0 }}>
        {[["overview","Overview"],["shells","Shells"],
          ["cross","Cross-Base"],["open","Open Problems"]].map(([k,v]) => (
          <Tab key={k} active={tab===k} onClick={() => setTab(k)}>{v}</Tab>
        ))}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === "overview" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 10 }}>
            <Stat label="image size |Im(K₁₀,₄)|" value={CONSTANTS.image_10_4}
              sub="all outputs ≡ 0 mod 9" accent={C.blue} />
            <Stat label="connected components" value={CONSTANTS.components}
              sub="cycle nodes: {0, 6174}" accent={C.body} />
            <Stat label="max depth τ_max" value={CONSTANTS.tau_max}
              sub="every state reaches attractor" accent={C.body} />
            <Stat label="shell-3 fraction" value="24.00%"
              sub="2400 / 10 000 states" accent={C.amber} />
            <Stat label="throughput" value="185k/s"
              sub="maps/sec · single thread" accent={C.green} />
            <Stat label="signature time" value="10ms"
              sub="full orbit_signature()" accent={C.green} />
            <Stat label="null |Im| mean" value={Math.round(CONSTANTS.null_image_mean).toLocaleString()}
              sub="10 random endofunctions" accent={C.muted} />
            <Stat label="image compression" value={`${compressionRatio}%`}
              sub="vs uniform random map" accent={C.blue} />
          </div>

          <div style={{ padding: 16, background: C.panel,
            border: `1px solid ${C.border}`, borderRadius: 6 }}>
            <Label>fixed points (cycle_lengths = [1, 1])</Label>
            <div style={{ display: "flex", gap: 24 }}>
              {CONSTANTS.cycle_nodes.map(n => (
                <div key={n} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontFamily: mono, fontSize: 20, color: C.amber,
                    fontWeight: 700 }}>{n}</span>
                  <span style={{ fontFamily: mono, fontSize: 11, color: C.muted }}>
                    K({n}) = {n}
                  </span>
                </div>
              ))}
            </div>
            <p style={{ fontFamily: mono, fontSize: 10, color: C.dim,
              margin: "10px 0 0" }}>
              Both are 1-cycles. Earlier test suite expected cycle_count=1 — that was wrong.
              The zero fixed point is a repdigit sink; it is structurally distinct from
              6174 but mathematically equal in cycle classification.
            </p>
          </div>

          <div style={{ padding: 16, background: C.panel,
            border: `1px solid ${C.border}`, borderRadius: 6 }}>
            <Label>what is proved vs. what is observed</Label>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16,
              fontFamily: mono, fontSize: 11, lineHeight: 1.7 }}>
              <div>
                <div style={{ color: C.green, marginBottom: 6, fontSize: 10,
                  textTransform: "uppercase", letterSpacing: "0.1em" }}>✓ proved / verified</div>
                {[
                  "|Im(K₁₀,₄)| = 55 (exact)",
                  "τ_max = 7 (exact)",
                  "Both fixed points = {0, 6174}",
                  "All outputs ≡ 0 mod 9",
                  "Shell decomposition covers N=10000",
                  "K_b has unique fixed pt iff 5|b (d=4)",
                  "Deterministic, reproducible",
                ].map(s => <div key={s} style={{ color: C.body }}>· {s}</div>)}
              </div>
              <div>
                <div style={{ color: C.red, marginBottom: 6, fontSize: 10,
                  textTransform: "uppercase", letterSpacing: "0.1em" }}>✗ not yet proved</div>
                {[
                  "Shell-3 = 24% is universal across (b,d)",
                  "σ-level statistical significance claims",
                  "Throughput is a scaling law",
                  "Shell bottleneck has spectral explanation",
                  "Depth plateau at layers 3–5 is non-trivial",
                  "Random null rejection (needs KS/Wasserstein)",
                ].map(s => <div key={s} style={{ color: C.body }}>· {s}</div>)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── SHELLS ── */}
      {tab === "shells" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ padding: 16, background: C.panel,
            border: `1px solid ${C.border}`, borderRadius: 6 }}>
            <ShellChart />
          </div>
          <div style={{ padding: 16, background: C.panel,
            border: `1px solid ${C.border}`, borderRadius: 6 }}>
            <Label>raw counts</Label>
            <table style={{ fontFamily: mono, fontSize: 11,
              borderCollapse: "collapse", width: "100%" }}>
              <thead>
                <tr>
                  {["τ","count","fraction","note"].map(h => (
                    <th key={h} style={{ textAlign: "left", color: C.muted,
                      padding: "4px 12px", borderBottom: `1px solid ${C.border}`,
                      fontWeight: 400, fontSize: 9, textTransform: "uppercase",
                      letterSpacing: "0.08em" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {SHELLS.map(({ depth, count, pct }) => (
                  <tr key={depth} style={{ background: depth===3?"#1c1300":"transparent" }}>
                    <td style={{ padding: "5px 12px",
                      color: depth===3 ? C.amber : C.body }}>{depth}</td>
                    <td style={{ padding: "5px 12px",
                      color: depth===3 ? C.amber : C.hi,
                      fontWeight: depth===3 ? 700 : 400 }}>{count}</td>
                    <td style={{ padding: "5px 12px",
                      color: depth===3 ? C.amber : C.body }}>{pct.toFixed(2)}%</td>
                    <td style={{ padding: "5px 12px", color: C.dim, fontSize: 10 }}>
                      {depth===0 && "cycle nodes (fixed points)"}
                      {depth===3 && "S₃ bottleneck — structural observation, no derivation"}
                      {depth===7 && "maximum depth — 2184 states"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ fontFamily: mono, fontSize: 10, color: C.dim,
              marginTop: 12, lineHeight: 1.6 }}>
              Shell-3 dominance is reproducible and structurally real.
              It is NOT explained by a derivation. It is a candidate phenomenon.
              Next step: entropy gradient H(S₃) − H(S₄) and invariance test across (b,d).
            </p>
          </div>
        </div>
      )}

      {/* ── CROSS-BASE ── */}
      {tab === "cross" && (
        <div style={{ padding: 16, background: C.panel,
          border: `1px solid ${C.border}`, borderRadius: 6 }}>
          <CrossBaseTable />
          <p style={{ fontFamily: mono, fontSize: 10, color: C.dim,
            marginTop: 14, lineHeight: 1.6 }}>
            All image sizes verified in session. τ_max only computed for b=10,d=4 (=7)
            and b=8,d=4 (=5) in this run. "—" = not computed yet, not zero.
            Cross-base τ sweep is a required next step before universality claims.
          </p>
        </div>
      )}

      {/* ── OPEN ── */}
      {tab === "open" && (
        <div style={{ padding: 16, background: C.panel,
          border: `1px solid ${C.border}`, borderRadius: 6 }}>
          <OpenItems />
          <div style={{ marginTop: 20, padding: "12px 14px",
            background: "#1c0505", border: `1px solid #7f1d1d`,
            borderRadius: 6, fontFamily: mono, fontSize: 10,
            color: C.body, lineHeight: 1.7 }}>
            <span style={{ color: C.red, fontWeight: 700 }}>INVALIDATED CLAIMS</span>
            &nbsp;— killed from prior sessions:<br />
            · "−80.1σ compression" — not valid; Gaussian null inapplicable to combinatorial distributions<br />
            · "p &lt; 10⁻⁷" — sample size 1900 does not support this for tail events<br />
            · "throughput is a scaling law" — formula cancelled to constant; it was an assumption, not a measurement<br />
            · "universal bottleneck law" — not proved; one data point in (b,d) space<br />
            · "cycle_count = 1" — wrong; 0 is a fixed point, cycle_count = 2
          </div>
        </div>
      )}

      {/* ── footer ── */}
      <div style={{ marginTop: 28, borderTop: `1px solid ${C.border}`,
        paddingTop: 12, display: "flex", justifyContent: "space-between",
        alignItems: "center" }}>
        <span style={{ fontFamily: mono, fontSize: 9, color: C.dim }}>
          ATREYUE TECH · AQARION_PRIME · LOUISVILLE NODE
        </span>
        <span style={{ fontFamily: mono, fontSize: 9, color: C.dim }}>
          41 PASS · 2 BAD ASSERTIONS · 0 ENGINE FAILURES
        </span>
      </div>
    </div>
  );
