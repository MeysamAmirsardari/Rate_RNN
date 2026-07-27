// Three technical slides: the pipeline, the learning rule, the constants.
//   node make_algorithm_slides.js

const PptxGenJS = require("pptxgenjs");

const INK = "2B2B2B";
const GRAY = "7A7A7A";
const FAINT = "F2F2F2";
const BLUE = "3D6FA8";
const GREEN = "2F7D5D";
const F = "Arial";
const MONO = "Courier New";

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";
pres.title = "Pipeline and learning rule";

function head(s, title, lead) {
  s.background = { color: "FFFFFF" };
  s.addText(title, { x: 0.6, y: 0.34, w: 12.2, h: 0.55, fontFace: F,
    fontSize: 25, bold: true, color: INK, margin: 0 });
  if (lead) s.addText(lead, { x: 0.6, y: 0.94, w: 12.2, h: 0.36, fontFace: F,
    fontSize: 13, color: GRAY, margin: 0 });
}

function code(s, text, x, y, w, h, size) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: FAINT },
    line: { color: "E2E2E2", width: 1 } });
  s.addText(text, { x: x + 0.28, y: y + 0.16, w: w - 0.5, h: h - 0.32,
    fontFace: MONO, fontSize: size || 14, color: INK, lineSpacing: 21,
    margin: 0, valign: "top" });
}

// =====================================================================
// 1.  Pipeline
// =====================================================================
let s = pres.addSlide();
head(s, "Pipeline", "One time step, dt = 1 ms. N channels, K units.");

const stage = [
  ["stimulus", "s(t)", "(N,)", "tones, one channel per tone"],
  ["layer 1", "E(t) = model0(s)", "(N,)", "excitatory rate, unchanged, read only"],
  ["slow conductance", "r += dt(-r + E)/tau_rise\ns += dt(-s + r)/tau_decay", "(N,)",
   "tau_rise 40 ms, tau_decay 150 ms"],
  ["coincidence map", "D = E s^T,   diag(D) = 0", "(N,N)", "D[i,j] = E_i * s_j"],
  ["unit response", "y_k = relu( <M_k , D> )", "(K,)", "M_k >= 0, shape (N,N)"],
];

let y0 = 1.55;
stage.forEach((row, i) => {
  const h = i === 2 ? 1.02 : 0.78;
  s.addText(row[0], { x: 0.6, y: y0, w: 2.25, h, fontFace: F, fontSize: 13,
    bold: true, color: BLUE, valign: "middle", margin: 0 });
  code(s, row[1], 2.95, y0, 5.5, h, 13.5);
  s.addText(row[2], { x: 8.6, y: y0, w: 0.95, h, fontFace: MONO, fontSize: 12,
    color: GRAY, valign: "middle", margin: 0 });
  s.addText(row[3], { x: 9.6, y: y0, w: 3.2, h, fontFace: F, fontSize: 11.5,
    color: GRAY, valign: "middle", margin: 0 });
  if (i < stage.length - 1) {
    s.addText("v", { x: 5.4, y: y0 + h - 0.04, w: 0.6, h: 0.3, fontFace: F,
      fontSize: 12, color: GRAY, align: "center", margin: 0 });
  }
  y0 += h + 0.26;
});

s.addText("Feedforward only. Layer 1 is never modified and receives nothing back.",
  { x: 0.6, y: 6.72, w: 12.2, h: 0.4, fontFace: F, fontSize: 13, bold: true,
    color: INK, margin: 0 });
s.addNotes("D is the only thing layer 2 ever sees.");

// =====================================================================
// 2.  Learning rule
// =====================================================================
s = pres.addSlide();
head(s, "Learning rule",
  "Four local rules, applied every time step after D is formed. No labels, no error signal, no feedback.");

code(s,
  "if ||D||  >  gate * running_peak(||D||):        plasticity gate\n" +
  "\n" +
  "     Dh    =  D / ||D||\n" +
  "\n" +
  "     c_k   =  <M_k , Dh> / ||M_k||               1.  match      cosine, direction only\n" +
  "\n" +
  "     w     =  argmax_k  c_k                      2.  compete    winner take all\n" +
  "\n" +
  "     M_w  +=  eta * ( Dh - M_w ),  clip >= 0     3.  learn      instar, winner only\n" +
  "\n" +
  "\n" +
  "M_k  *=  ( 1 - lambda )      for every k         4.  forget     decay, every unit,\n" +
  "                                                                every step, gated or not",
  0.6, 1.62, 12.2, 3.55, 13.5);

s.addText("Commitment", { x: 0.6, y: 5.42, w: 2.4, h: 0.4, fontFace: F,
  fontSize: 14, bold: true, color: GREEN, margin: 0 });
code(s, "unit k is committed  <=>  ||M_k||  >  commit_frac * max_j ||M_j||",
  2.95, 5.32, 9.85, 0.62, 13);

s.addText([
  { text: "Symmetry is broken only by the random initial masks. ", options: { breakLine: false } },
  { text: "The number of committed units is an outcome, not a setting.", options: { bold: true } },
], { x: 0.6, y: 6.25, w: 12.2, h: 0.4, fontFace: F, fontSize: 13, color: INK,
  margin: 0 });
s.addText("Optional vigilance (ART): if c_w < rho, recruit argmin_k ||M_k|| instead of letting w learn. Set to 0 here, so unused.",
  { x: 0.6, y: 6.72, w: 12.2, h: 0.4, fontFace: F, fontSize: 12, color: GRAY,
    margin: 0 });
s.addNotes("Rules 3 and 4 together are the survival competition.");

// =====================================================================
// 3.  Constants and name
// =====================================================================
s = pres.addSlide();
head(s, "Constants, and what the algorithm is called", null);

s.addTable([
  [{ text: "symbol", options: { bold: true } },
   { text: "value", options: { bold: true } },
   { text: "meaning", options: { bold: true } }],
  ["dt", "1 ms", "integration step"],
  ["tau_rise", "40 ms", "rise of the slow conductance"],
  ["tau_decay", "150 ms", "decay; must span one chunk"],
  ["K", "8", "units available, deliberately more than needed"],
  ["eta", "5e-3 per step", "instar learning rate"],
  ["lambda", "1e-4 per step", "synaptic decay"],
  ["gate", "0.15", "fraction of running peak ||D|| needed to learn"],
  ["M init", "U(0, 0.05)", "random, non negative, diagonal zero"],
  ["commit_frac", "0.20", "commitment threshold on ||M||"],
  ["rho", "0", "vigilance, disabled"],
], { x: 0.6, y: 1.45, w: 6.55, colW: [1.6, 1.85, 3.1], rowH: 0.35,
  fontFace: F, fontSize: 11.5, color: INK,
  border: { pt: 0.5, color: "DDDDDD" }, valign: "middle" });

s.addText("Name", { x: 7.5, y: 1.45, w: 5.3, h: 0.4, fontFace: F, fontSize: 15,
  bold: true, color: GREEN, margin: 0 });
s.addText([
  { text: "Competitive Hebbian learning", options: { bold: true, breakLine: true } },
  { text: "instar rule (Grossberg 1976) + winner take all + synaptic decay.", options: { breakLine: true } },
], { x: 7.5, y: 1.95, w: 5.3, h: 1.0, fontFace: F, fontSize: 12.5, color: INK,
  lineSpacing: 19, margin: 0, valign: "top" });

s.addText("Equivalent to", { x: 7.5, y: 3.0, w: 5.3, h: 0.35, fontFace: F,
  fontSize: 13, bold: true, color: GREEN, margin: 0 });
s.addText([
  { text: "online spherical k-means on D (cosine assignment, centroid update)", options: { bullet: true, breakLine: true } },
  { text: "vector quantisation; a self organising map with zero neighbourhood", options: { bullet: true, breakLine: true } },
  { text: "online non negative dictionary learning with a 1-sparse code", options: { bullet: true } },
], { x: 7.5, y: 3.42, w: 5.3, h: 1.5, fontFace: F, fontSize: 12, color: INK,
  lineSpacing: 18, margin: 0, valign: "top" });

s.addText("Not", { x: 7.5, y: 4.95, w: 5.3, h: 0.35, fontFace: F, fontSize: 13,
  bold: true, color: "C1553B", margin: 0 });
s.addText("backpropagation, predictive coding, Oja's rule, or batch NMF.",
  { x: 7.5, y: 5.35, w: 5.3, h: 0.4, fontFace: F, fontSize: 12, color: INK,
    margin: 0 });

s.addShape(pres.ShapeType.rect, { x: 0.6, y: 6.05, w: 12.2, h: 1.0,
  fill: { color: FAINT }, line: { color: "E2E2E2", width: 1 } });
s.addText([
  { text: "The rule is standard and forty years old. ", options: { breakLine: false } },
  { text: "The non standard part is what it is applied to: ", options: { bold: true, breakLine: false } },
  { text: "an instantaneous coincidence map rather than a rate vector.", options: { breakLine: true } },
  { text: "Competitive learning on a rate vector learns which channels co-occur. On a coincidence map it learns which channel follows which.", options: {} },
], { x: 0.95, y: 6.2, w: 11.6, h: 0.75, fontFace: F, fontSize: 13, color: INK,
  lineSpacing: 20, margin: 0, valign: "middle" });
s.addNotes("Methods sentence: unsupervised competitive Hebbian learning (instar rule with winner take all and synaptic decay, equivalent to online spherical k-means) applied to the directional coincidence map.");

pres.writeFile({ fileName: "algorithm_3slides.pptx" })
  .then(() => console.log("wrote algorithm_3slides.pptx"));
