// Builds the 25 slide walkthrough of the two layer algorithm.
// Deliberately plain: white background, Arial, a couple of muted colours,
// one idea per slide, small hand built diagrams.
//
//   node make_slides.js

const PptxGenJS = require("pptxgenjs");

const INK = "2B2B2B";
const GRAY = "7A7A7A";
const FAINT = "EDEDED";
const BLUE = "3D6FA8";
const RED = "C1553B";
const GREEN = "2F7D5D";
const PURPLE = "6B5B95";
const F = "Arial";
const MONO = "Courier New";

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";        // 13.33 x 7.5 in.  Set before any slide.
pres.author = "Rate RNN project";
pres.title = "A two layer algorithm for learning ordered chunks";

let N = 0;

// ---- small helpers -------------------------------------------------
function newSlide(title, lead) {
  N += 1;
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText(title, {
    x: 0.65, y: 0.42, w: 12.0, h: 0.62, fontFace: F, fontSize: 26,
    bold: true, color: INK, margin: 0,
  });
  if (lead) {
    s.addText(lead, {
      x: 0.65, y: 1.12, w: 12.0, h: 0.44, fontFace: F, fontSize: 14,
      color: GRAY, margin: 0,
    });
  }
  s.addText(String(N), {
    x: 12.5, y: 6.92, w: 0.5, h: 0.3, fontFace: F, fontSize: 10,
    color: GRAY, align: "right", margin: 0,
  });
  return s;
}

function bullets(slide, items, opt) {
  const o = Object.assign({ x: 0.65, y: 1.85, w: 6.1, fontSize: 15 }, opt || {});
  // Height must be derived, not fixed: a fixed 3.9 in box placed low on the
  // slide runs off the bottom edge.
  const h = o.h || Math.max(0.6, Math.min(3.9, 7.05 - o.y));
  slide.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: true, breakLine: i !== items.length - 1 },
    })),
    {
      x: o.x, y: o.y, w: o.w, h, fontFace: F, fontSize: o.fontSize,
      color: INK, lineSpacing: 22, paraSpaceAfter: 10, margin: 0, valign: "top",
    }
  );
}

function note(slide, text) { slide.addNotes(text); }

// a labelled token box
function token(slide, x, y, label, color, w, h) {
  slide.addShape(pres.ShapeType.rect, {
    x, y, w: w || 0.85, h: h || 0.72, fill: { color },
    line: { color, width: 1 },
  });
  slide.addText(label, {
    x, y, w: w || 0.85, h: h || 0.72, fontFace: F, fontSize: 18, bold: true,
    color: "FFFFFF", align: "center", valign: "middle", margin: 0,
  });
}

// a 2x2 coincidence map with one cell filled
function miniMap(slide, x, y, fillRow, fillCol, caption, capColor) {
  const c = 0.62;
  const labels = ["A", "B"];
  for (let r = 0; r < 2; r++) {
    for (let k = 0; k < 2; k++) {
      const on = r === fillRow && k === fillCol;
      slide.addShape(pres.ShapeType.rect, {
        x: x + k * c, y: y + r * c, w: c, h: c,
        fill: { color: on ? PURPLE : "FFFFFF" },
        line: { color: on ? PURPLE : "C9C9C9", width: 1 },
      });
    }
  }
  for (let r = 0; r < 2; r++) {
    slide.addText(labels[r], {
      x: x - 0.42, y: y + r * c, w: 0.36, h: c, fontFace: F, fontSize: 12,
      color: GRAY, align: "right", valign: "middle", margin: 0,
    });
    slide.addText(labels[r], {
      x: x + r * c, y: y + 2 * c + 0.03, w: c, h: 0.3, fontFace: F,
      fontSize: 12, color: GRAY, align: "center", margin: 0,
    });
  }
  slide.addText("firing now", {
    x: x - 1.75, y: y + c - 0.15, w: 1.3, h: 0.3, fontFace: F, fontSize: 10,
    color: GRAY, align: "right", margin: 0,
  });
  slide.addText("fired recently", {
    x: x - 0.1, y: y + 2 * c + 0.34, w: 1.5, h: 0.3, fontFace: F,
    fontSize: 10, color: GRAY, align: "center", margin: 0,
  });
  if (caption) {
    slide.addText(caption, {
      x: x - 0.5, y: y - 0.46, w: 2.4, h: 0.36, fontFace: F, fontSize: 13,
      bold: true, color: capColor || INK, align: "center", margin: 0,
    });
  }
}

function formula(slide, text, x, y, w, size) {
  slide.addShape(pres.ShapeType.rect, {
    x, y, w, h: 0.86, fill: { color: FAINT }, line: { color: FAINT, width: 1 },
  });
  slide.addText(text, {
    x, y, w, h: 0.86, fontFace: MONO, fontSize: size || 17, color: INK,
    align: "center", valign: "middle", margin: 0,
  });
}

function callout(slide, text, x, y, w, color) {
  slide.addText(text, {
    x, y, w, h: 0.9, fontFace: F, fontSize: 15, bold: true,
    color: color || GREEN, valign: "middle", margin: 0,
  });
}

// =====================================================================
// 1
let s = pres.addSlide();
s.background = { color: "FFFFFF" };
s.addText("Learning ordered chunks from a cortical model", {
  x: 0.9, y: 2.3, w: 11.5, h: 0.9, fontFace: F, fontSize: 34, bold: true,
  color: INK, margin: 0,
});
s.addText("A two layer algorithm, start to finish", {
  x: 0.9, y: 3.25, w: 11.5, h: 0.5, fontFace: F, fontSize: 18, color: GRAY,
  margin: 0,
});
s.addText("What it computes, why each piece is there, what it does and does not do",
  { x: 0.9, y: 3.95, w: 11.5, h: 0.5, fontFace: F, fontSize: 14, color: GRAY,
    margin: 0 });
N = 1;
note(s, "The goal is that by the end you can rebuild this from memory.");

// 2
s = newSlide("The problem", "Two chunks made of exactly the same two tones");
token(s, 0.8, 2.3, "A", RED); token(s, 1.75, 2.3, "B", BLUE);
s.addText("chunk AB", { x: 0.8, y: 3.1, w: 1.8, h: 0.3, fontFace: F,
  fontSize: 13, color: GRAY, align: "center", margin: 0 });
token(s, 0.8, 3.7, "B", BLUE); token(s, 1.75, 3.7, "A", RED);
s.addText("chunk BA", { x: 0.8, y: 4.5, w: 1.8, h: 0.3, fontFace: F,
  fontSize: 13, color: GRAY, align: "center", margin: 0 });
bullets(s, [
  "Each chunk contains one A tone and one B tone. Nothing else differs.",
  "Average anything over a chunk and the two are identical.",
  "So no code based on how much each channel fired can ever separate them.",
  "The only difference is the order.",
], { x: 4.2, y: 2.2, w: 8.3 });
callout(s, "Everything that follows exists to read that order.", 4.2, 5.5, 8.3, INK);
note(s, "This is the whole motivation. A rate code is provably at chance here.");

// 3
s = newSlide("Two layers", "Feedforward only. Layer 1 is never modified.");
s.addShape(pres.ShapeType.rect, { x: 1.2, y: 2.4, w: 3.6, h: 1.5,
  fill: { color: "FFFFFF" }, line: { color: INK, width: 1.5 } });
s.addText("Layer 1\nmodel0, a cortical A1 model", { x: 1.2, y: 2.4, w: 3.6,
  h: 1.5, fontFace: F, fontSize: 14, color: INK, align: "center",
  valign: "middle", margin: 0 });
s.addText("→", { x: 4.95, y: 2.75, w: 0.8, h: 0.8, fontFace: F, fontSize: 30,
  color: GRAY, align: "center", margin: 0 });
s.addShape(pres.ShapeType.rect, { x: 5.9, y: 2.4, w: 3.6, h: 1.5,
  fill: { color: "FFFFFF" }, line: { color: PURPLE, width: 1.5 } });
s.addText("Layer 2\nchunk selective units", { x: 5.9, y: 2.4, w: 3.6, h: 1.5,
  fontFace: F, fontSize: 14, color: PURPLE, align: "center",
  valign: "middle", margin: 0 });
s.addText("excitatory rate E", { x: 4.6, y: 3.95, w: 1.6, h: 0.3, fontFace: F,
  fontSize: 11, color: GRAY, align: "center", margin: 0 });
bullets(s, [
  "Layer 1 turns tones into per channel excitatory rates E.",
  "Layer 2 reads only E. It sends nothing back.",
  "Layer 2 adds exactly one new state variable, and four learning rules.",
], { x: 1.2, y: 4.7, w: 11.0 });
note(s, "Keeping layer 1 untouched means any result is attributable to layer 2.");

// 4
s = newSlide("Layer 1, in one slide", "Used as given. No changes were made to it.");
bullets(s, [
  "One excitatory and one inhibitory unit per tonotopic channel.",
  "Inhibition is slow and tone selective, in the style of SST interneurons.",
  "Thalamic input passes through short term depression, which gives adaptation.",
  "Recurrent excitatory to excitatory weights learn by a Hebbian rule.",
  "Layer 2 uses only the excitatory rate E. Everything else stays inside layer 1.",
]);
callout(s, "Treat it as a box that converts tones into per channel activity.",
  0.65, 5.6, 11.5, INK);
note(s, "Details of layer 1 do not matter for the algorithm being presented.");

// 5
s = newSlide("Where the order information actually lives",
  "Not in who fired, but in who fired before whom");
bullets(s, [
  "Who fired: A and B, in both chunks. No information.",
  "Who fired first: this is the entire signal.",
  "To see it, a neuron must combine two things at the same instant:",
], { y: 2.0 });
s.addShape(pres.ShapeType.rect, { x: 1.4, y: 4.1, w: 4.3, h: 1.0,
  fill: { color: "FFFFFF" }, line: { color: BLUE, width: 1.5 } });
s.addText("what is firing NOW", { x: 1.4, y: 4.1, w: 4.3, h: 1.0, fontFace: F,
  fontSize: 15, bold: true, color: BLUE, align: "center", valign: "middle",
  margin: 0 });
s.addText("×", { x: 5.85, y: 4.35, w: 0.6, h: 0.5, fontFace: F, fontSize: 24,
  color: GRAY, align: "center", margin: 0 });
s.addShape(pres.ShapeType.rect, { x: 6.6, y: 4.1, w: 4.3, h: 1.0,
  fill: { color: "FFFFFF" }, line: { color: RED, width: 1.5 } });
s.addText("what fired RECENTLY", { x: 6.6, y: 4.1, w: 4.3, h: 1.0, fontFace: F,
  fontSize: 15, bold: true, color: RED, align: "center", valign: "middle",
  margin: 0 });
note(s, "The product of a fast signal and a slow one is the core operation.");

// 6
s = newSlide("The rule that makes it work",
  "The two things multiplied must have different timescales");
formula(s, "D[i,j]  =  integral of  f_i(t) * g_j(t)  dt", 1.6, 2.1, 10.1);
bullets(s, [
  "If f and g are the same signal, then D[i,j] equals D[j,i] for every pair.",
  "The matrix is symmetric, so the order information is exactly zero.",
  "Not small. Zero, analytically, for any weights you could ever choose.",
], { x: 0.65, y: 3.3, w: 11.8 });
callout(s, "Order selectivity comes entirely from the difference between the two timescales.",
  0.65, 5.5, 11.8, GREEN);
note(s, "This is a theorem, not a tuning issue. It also tells you the fix.");

// 7
s = newSlide("The same theorem you already know",
  "This is how motion detection works");
bullets(s, [
  "A Reichardt detector multiplies a delayed signal from one place by an undelayed signal from another.",
  "Make the two delays equal and direction selectivity disappears completely.",
  "Temporal order in a sound is the same computation as motion in an image.",
  "So the design question is only: what plays the role of the delay?",
]);
note(s, "Useful for an audience that knows vision better than audition.");

// 8
s = newSlide("The one new variable: a slow conductance",
  "This is layer 2's memory of what just happened");
formula(s, "E  ->  [ rise 40 ms ]  ->  [ decay 150 ms ]  ->  s", 1.6, 2.1, 10.1);
bullets(s, [
  "s is a two stage low pass filter of the layer 1 rate E.",
  "E is the fast factor, s is the slow one. That is the required timescale gap.",
  "The decay must span a chunk, so that the first token is still present when the last arrives.",
  "Biologically this is an NMDA like conductance or a dendritic plateau.",
], { x: 0.65, y: 3.3, w: 11.8 });
note(s, "One variable per channel. That is the entire addition to the model.");

// 9
s = newSlide("Why the conductance must rise slowly",
  "Otherwise a channel coincides with itself");
bullets(s, [
  "With an instant rise, a channel's own trace is already large while it fires.",
  "Units then learn 'A is on', which layer 1 already tells you, instead of 'B follows A'.",
  "A slow rise means a channel's own trace is still small during its own tone.",
], { x: 0.65, y: 2.0, w: 11.8 });
s.addShape(pres.ShapeType.rect, { x: 1.6, y: 4.15, w: 4.4, h: 1.2,
  fill: { color: "FFFFFF" }, line: { color: GRAY, width: 1.25 } });
s.addText("instant rise\ncross term / self term  =  1.2", { x: 1.6, y: 4.15,
  w: 4.4, h: 1.2, fontFace: F, fontSize: 14, color: GRAY, align: "center",
  valign: "middle", margin: 0 });
s.addShape(pres.ShapeType.rect, { x: 7.0, y: 4.15, w: 4.4, h: 1.2,
  fill: { color: "FFFFFF" }, line: { color: GREEN, width: 1.5 } });
s.addText("40 ms rise\ncross term / self term  =  4.5", { x: 7.0, y: 4.15,
  w: 4.4, h: 1.2, fontFace: F, fontSize: 14, bold: true, color: GREEN,
  align: "center", valign: "middle", margin: 0 });
callout(s, "Measured before building anything else. It decided the design.",
  0.65, 5.7, 11.8, INK);
note(s, "The numbers come from a direct measurement on the real layer 1 output.");

// 10
s = newSlide("The coincidence map", "One matrix per moment in time");
formula(s, "D(t)[i,j]  =  E_i(t)  *  s_j(t)", 1.6, 2.05, 10.1);
bullets(s, [
  "Row i: channel i is firing now.",
  "Column j: channel j fired recently.",
  "So entry [i,j] reads as 'i is firing now, and j fired just before it'.",
  "This single matrix is the entire input to layer 2.",
], { x: 0.65, y: 3.25, w: 11.8 });
note(s, "Everything downstream is just pattern matching against this matrix.");

// 11
s = newSlide("What the map looks like", "The same two chunks, at the moment they complete");
miniMap(s, 3.3, 2.9, 1, 0, "chunk AB", RED);
miniMap(s, 8.6, 2.9, 0, 1, "chunk BA", BLUE);
callout(s, "Two chunks that no rate code can separate produce two different matrices.",
  0.65, 5.8, 11.8, GREEN);
note(s, "AB lights up B after A. BA lights up A after B. Clean and readable.");

// 12
s = newSlide("A layer 2 unit is one mask",
  "Same shape as the coincidence map, and never negative");
bullets(s, [
  "Each unit holds one matrix M, of the same size as D.",
  "Non negative, in keeping with Dale's principle.",
  "The mask is literally a picture of the chunk the unit is looking for.",
  "You can read a unit's meaning straight off its weights, with no decoder.",
]);
miniMap(s, 8.9, 2.6, 1, 0, "a unit that means\n'B after A'", PURPLE);
note(s, "Interpretability is a property of the design, not an afterthought.");

// 13
s = newSlide("How a unit responds", "Overlap between its mask and the current map");
formula(s, "y  =  relu( sum over i,j of  M[i,j] * D[i,j] )", 1.6, 2.05, 10.1);
bullets(s, [
  "A matched filter: how much does what is happening look like what I encode?",
  "Because D is a product, the unit only responds when something is firing now AND something was firing before.",
  "Biologically: a dendritic subunit receiving a fast input from one channel and a slow input from another.",
  "Single pyramidal dendrites are known to discriminate input sequence order this way.",
], { x: 0.65, y: 3.25, w: 11.8 });
note(s, "Branco, Clark and Hausser 2010 is the direct experimental precedent.");

// 14
s = newSlide("One wiring rule: pair different channels",
  "A subunit combines two distinct inputs, never a channel with itself");
bullets(s, [
  "The first token of a chunk has no predecessor, so at that moment the map is purely diagonal.",
  "If self pairs are allowed, units happily learn that, and you get redundant 'this tone is on' detectors.",
  "Measured: with self pairs, four units commit, two useful and two redundant, and the readout falls to chance.",
  "Excluding them makes this a layer about transitions, which is the point.",
]);
callout(s, "First tokens teach identity. Second tokens teach order. We want the second.",
  0.65, 5.7, 11.8, INK);
note(s, "This was found by a failed run, not assumed in advance.");

// 15
s = newSlide("Learning, rule 1 of 4: match",
  "Every unit says how well the current map fits it");
formula(s, "c_k  =  < M_k , D >  /  ( |M_k| * |D| )", 1.6, 2.15, 10.1);
bullets(s, [
  "A cosine, so it compares direction only and ignores magnitude.",
  "That lets a unit with weak weights still recognise its own pattern.",
  "Which is what allows a faded unit to be recruited again later.",
], { x: 0.65, y: 3.4, w: 11.8 });
note(s, "Match on direction, respond on magnitude. The two are kept separate.");

// 16
s = newSlide("Rule 2: compete", "One winner takes the moment");
formula(s, "winner  =  the unit with the largest c_k", 1.6, 2.15, 10.1);
bullets(s, [
  "This is strong lateral inhibition between layer 2 units.",
  "Only the winner is allowed to learn.",
  "Winner take all is what stops two units from encoding the same chunk: the loser gets nothing.",
], { x: 0.65, y: 3.4, w: 11.8 });
note(s, "Competition is what makes the units divide the work between them.");

// 17
s = newSlide("Rule 3: learn", "The winner moves toward what just happened");
formula(s, "M_winner  +=  eta * ( D_normalised  -  M_winner )", 1.6, 2.15, 10.1);
bullets(s, [
  "Ordinary Hebbian potentiation toward the active pattern, then clipped at zero.",
  "Repeated exposure sharpens the mask onto one recurring transition.",
  "Learning only happens when there is enough drive, which is a plasticity threshold.",
], { x: 0.65, y: 3.4, w: 11.8 });
note(s, "Instar learning in Grossberg's sense.");

// 18
s = newSlide("Rule 4: forget", "Every unit decays, winner or not");
formula(s, "M_k  =  M_k  *  ( 1 - lambda )      for every unit k", 1.6, 2.15, 10.1);
bullets(s, [
  "Synaptic pruning. Use it or lose it.",
  "A unit that keeps winning gains more than it loses, and grows.",
  "A unit that never wins only decays, and its mask goes to zero.",
], { x: 0.65, y: 3.4, w: 11.8 });
note(s, "This single line is what makes the population size self limiting.");

// 19
s = newSlide("How the number of chunks emerges",
  "The count is a result, not a setting");
s.addShape(pres.ShapeType.rect, { x: 1.1, y: 2.4, w: 4.7, h: 1.5,
  fill: { color: "FFFFFF" }, line: { color: GREEN, width: 1.5 } });
s.addText("wins often\ngain beats decay\ngrows and sharpens", { x: 1.1, y: 2.4,
  w: 4.7, h: 1.5, fontFace: F, fontSize: 14, color: GREEN, align: "center",
  valign: "middle", margin: 0 });
s.addShape(pres.ShapeType.rect, { x: 7.0, y: 2.4, w: 4.7, h: 1.5,
  fill: { color: "FFFFFF" }, line: { color: GRAY, width: 1.25 } });
s.addText("never wins\nonly decay\nfades to zero and goes silent", { x: 7.0,
  y: 2.4, w: 4.7, h: 1.5, fontFace: F, fontSize: 14, color: GRAY,
  align: "center", valign: "middle", margin: 0 });
bullets(s, [
  "Start with more units than you think you need.",
  "The stream decides how many survive, and the survivors are genuinely silent, not merely suppressed.",
  "A faded unit keeps its direction, so a new chunk appearing later can still recruit it.",
], { x: 0.65, y: 4.4, w: 11.8 });
note(s, "Stability and plasticity together, without a separate mechanism.");

// 20
s = newSlide("The whole algorithm", "Everything above, on one page");
s.addShape(pres.ShapeType.rect, { x: 0.9, y: 1.85, w: 11.5, h: 4.2,
  fill: { color: FAINT }, line: { color: "DDDDDD", width: 1 } });
s.addText(
  "for each time step:\n" +
  "    s     <- slow two stage filter of the layer 1 rate E\n" +
  "    D     <- outer product of E and s        (zero on the diagonal)\n" +
  "    y_k   <- relu( < M_k , D > )             for every unit k\n\n" +
  "    if the drive is above the plasticity threshold:\n" +
  "        c_k    <- cosine match between M_k and D\n" +
  "        winner <- argmax over k of c_k\n" +
  "        M_win  <- M_win + eta * ( D_normalised - M_win ),  clipped at 0\n\n" +
  "    M_k <- M_k * ( 1 - lambda )              for every unit k",
  { x: 1.25, y: 2.1, w: 10.8, h: 3.7, fontFace: MONO, fontSize: 13.5,
    color: INK, lineSpacing: 20, margin: 0, valign: "top" });
note(s, "Four local rules. No backpropagation, no batch step, no global signal.");

// ---- worked example ------------------------------------------------
// Every number in this section is taken from the running model, not invented.

s = newSlide("A worked example",
  "Two units, two chunks. Every number from here on is a real one.");
bullets(s, [
  "Two layer 2 units. With two channels each mask has just two usable entries.",
  "Entry one means: B is firing now and A fired recently.",
  "Entry two means the reverse: A is firing now and B fired recently.",
  "They start with small random weights and know nothing.",
], { x: 0.65, y: 1.95, w: 7.0 });
s.addShape(pres.ShapeType.rect, { x: 8.1, y: 2.3, w: 4.4, h: 1.15,
  fill: { color: "FFFFFF" }, line: { color: PURPLE, width: 1.5 } });
s.addText("unit 1   [ 0.030 , 0.020 ]", { x: 8.1, y: 2.3, w: 4.4, h: 1.15,
  fontFace: MONO, fontSize: 15, color: PURPLE, align: "center",
  valign: "middle", margin: 0 });
s.addShape(pres.ShapeType.rect, { x: 8.1, y: 3.75, w: 4.4, h: 1.15,
  fill: { color: "FFFFFF" }, line: { color: PURPLE, width: 1.5 } });
s.addText("unit 2   [ 0.018 , 0.035 ]", { x: 8.1, y: 3.75, w: 4.4, h: 1.15,
  fontFace: MONO, fontSize: 15, color: PURPLE, align: "center",
  valign: "middle", margin: 0 });
s.addText("[ B after A , A after B ]", { x: 8.1, y: 5.0, w: 4.4, h: 0.35,
  fontFace: F, fontSize: 11, color: GRAY, align: "center", margin: 0 });
note(s, "Two channels keeps the arithmetic small enough to follow by hand.");

s = newSlide("Step 1: find the moment that carries the information",
  "An AB chunk. Tone A runs 0 to 50 ms, tone B runs 80 to 130 ms.");
formula(s,
  "at 116 ms after chunk onset:\n" +
  "    E = [ A 0.00 , B 12.06 ]        s = [ A 2.43 , B 0.60 ]",
  1.1, 2.15, 11.1, 15);
bullets(s, [
  "B is firing now: its rate is 12.06 and A has already fallen silent.",
  "But A is still strongly present in the slow conductance, at 2.43.",
  "B's own trace is only 0.60, because the conductance rises slowly.",
], { x: 0.65, y: 3.5, w: 11.8 });
callout(s, "B is firing now, and A is still in memory. That is the whole event.",
  0.65, 5.6, 11.8, GREEN);
note(s, "This moment is found automatically, not selected by hand.");

s = newSlide("Step 2: multiply the two", "This is the coincidence map");
formula(s,
  "D[ B after A ]  =  E_B  x  s_A  =  12.06  x  2.43  =  29.33\n" +
  "D[ A after B ]  =  E_A  x  s_B  =   0.00  x  0.60  =   0.00",
  1.1, 2.1, 11.1, 15);
bullets(s, [
  "One entry is large. The other is exactly zero, because A is not firing.",
  "Normalised, the map is [ 1.00 , 0.00 ]: pure 'B after A'.",
  "A rate code, looking at the same chunk, sees one A tone and one B tone and stops there.",
], { x: 0.65, y: 3.45, w: 11.8 });
note(s, "The product is what converts an order into a number.");

s = newSlide("Step 3: every unit measures the fit, the best one wins",
  "Cosine between each mask and the map");
formula(s,
  "unit 1:   0.030 / sqrt( 0.030^2 + 0.020^2 )   =   0.83\n" +
  "unit 2:   0.018 / sqrt( 0.018^2 + 0.035^2 )   =   0.46",
  1.1, 2.1, 11.1, 15);
bullets(s, [
  "Unit 1 happens to lean slightly toward 'B after A', so it fits better.",
  "The cosine ignores overall size, so a weak unit can still recognise its own pattern.",
  "Unit 1 wins this moment. Only unit 1 is allowed to learn from it.",
], { x: 0.65, y: 3.45, w: 11.8 });
callout(s, "The starting difference was tiny and random. Competition amplifies it.",
  0.65, 5.7, 11.8, INK);
note(s, "Symmetry breaking comes entirely from the random initial weights.");

s = newSlide("Step 4: the winner moves toward what it just saw",
  "Ordinary Hebbian learning, one step");
formula(s,
  "M1  <-  M1  +  0.14 x ( [1.00, 0.00]  -  M1 )\n" +
  "    =  [0.030, 0.020]  +  0.14 x [ 0.970 , -0.020 ]   =   [ 0.166 , 0.017 ]",
  1.1, 2.1, 11.1, 14);
bullets(s, [
  "The first entry grows toward 1. The second shrinks toward 0.",
  "The 0.14 is the learning rate accumulated over the roughly 30 ms the drive stays above threshold.",
  "Nothing else in the network changes on this step.",
], { x: 0.65, y: 3.45, w: 11.8 });
note(s, "Move toward the input, clip at zero. That is the whole update.");

s = newSlide("Step 5: then every mask decays a little",
  "Winner and loser alike");
formula(s,
  "unit 1:  [0.166, 0.017]  x 0.939  ->  [ 0.155 , 0.016 ]    won, net gain\n" +
  "unit 2:  [0.018, 0.035]  x 0.939  ->  [ 0.017 , 0.033 ]    lost, net loss",
  1.1, 2.1, 11.1, 14);
bullets(s, [
  "Decay is applied to every unit at every time step, whether it won or not.",
  "Unit 1 gained far more from learning than it lost to decay.",
  "Unit 2 only lost. Repeat that enough times and its mask reaches zero.",
], { x: 0.65, y: 3.45, w: 11.8 });
callout(s, "Winning beats the decay. Not winning does not. That is the whole pruning mechanism.",
  0.65, 5.7, 11.8, GREEN);
note(s, "This is where the population size comes from.");

s = newSlide("Step 6: the same five steps, eight chunks in a row",
  "Alternating AB and BA. Nothing is labelled and nothing is supervised.");
s.addTable(
  [
    [{ text: "chunk", options: { bold: true } },
     { text: "winner", options: { bold: true } },
     { text: "unit 1  [B after A, A after B]", options: { bold: true } },
     { text: "unit 2  [B after A, A after B]", options: { bold: true } }],
    ["start", "", "[0.030, 0.020]", "[0.018, 0.035]"],
    ["1   AB", "unit 1", "[0.155, 0.016]", "[0.017, 0.033]"],
    ["2   BA", "unit 2", "[0.146, 0.015]", "[0.014, 0.158]"],
    ["3   AB", "unit 1", "[0.249, 0.012]", "[0.013, 0.148]"],
    ["4   BA", "unit 2", "[0.234, 0.012]", "[0.010, 0.251]"],
    ["5   AB", "unit 1", "[0.320, 0.009]", "[0.010, 0.235]"],
    ["6   BA", "unit 2", "[0.300, 0.009]", "[0.008, 0.321]"],
    ["7   AB", "unit 1", "[0.374, 0.007]", "[0.007, 0.302]"],
    ["8   BA", "unit 2", "[0.351, 0.007]", "[0.006, 0.375]"],
  ],
  { x: 1.5, y: 1.95, w: 10.3, colW: [1.5, 1.6, 3.6, 3.6], rowH: 0.33,
    fontFace: MONO, fontSize: 12, color: INK, border: { pt: 0.5, color: "DDDDDD" },
    align: "center", valign: "middle" }
);
callout(s, "Each unit's own entry climbs while its other entry is squeezed to nothing.",
  1.5, 5.95, 10.3, INK);
note(s, "Read down either column: one number grows, the other decays away.");

s = newSlide("Step 7: where it settles",
  "After about sixty chunks the two units have divided the work");
formula(s,
  "unit 1  =  [ 0.51 , 0.00 ]              unit 2  =  [ 0.00 , 0.54 ]",
  1.1, 2.0, 11.1, 15);
s.addTable(
  [
    [{ text: "", options: { bold: true } },
     { text: "AB chunk", options: { bold: true } },
     { text: "BA chunk", options: { bold: true } }],
    ["unit 1", "15.0", "0.0"],
    ["unit 2", "0.0", "15.2"],
  ],
  { x: 3.6, y: 3.15, w: 6.1, colW: [2.0, 2.05, 2.05], rowH: 0.38,
    fontFace: MONO, fontSize: 13, color: INK,
    border: { pt: 0.5, color: "DDDDDD" }, align: "center", valign: "middle" }
);
bullets(s, [
  "Each unit ends up owning exactly one chunk, and is silent for the other.",
  "Nobody assigned them. The only asymmetry at the start was random noise.",
], { x: 0.65, y: 5.05, w: 11.8 });
note(s, "Compare with the real run: selectivity plus 0.91 and minus 0.90.");

s = newSlide("The same arithmetic for the other chunk",
  "Nothing special was done for AB. BA works out symmetrically.");
formula(s,
  "BA chunk, 117 ms after onset:\n" +
  "    E = [ A 11.65 , B 0.00 ]        s = [ A 0.65 , B 2.40 ]",
  1.1, 2.0, 11.1, 15);
formula(s,
  "D[ A after B ]  =  11.65  x  2.40  =  28.01          D[ B after A ]  =  0.00",
  1.1, 3.25, 11.1, 14);
bullets(s, [
  "Now A is firing and B is the one still held in the slow conductance.",
  "The map lights up the opposite entry, so unit 2 wins and unit 1 learns nothing.",
  "Same rule, same constants, opposite outcome. That is all the mechanism there is.",
], { x: 0.65, y: 4.4, w: 11.8 });
note(s, "Worth showing so nobody suspects the AB case was special.");

// 21
s = newSlide("Result 1: AB against BA, fifty fifty",
  "Eight units available, no labels given");
bullets(s, [
  "Exactly two units commit. The other six fade to zero and stay silent.",
  "Their masks read 'B after A' and 'A after B'. You can see it in the weights.",
  "100 percent correct on a fresh stream with the weights frozen.",
  "Five independent seeds: two units and 100 percent every time.",
]);
miniMap(s, 9.1, 2.40, 1, 0, "unit 3", PURPLE);
miniMap(s, 9.1, 5.05, 0, 1, "unit 0", PURPLE);
note(s, "Selectivity index plus 0.91 and minus 0.90.");

// 22
s = newSlide("Result 2: the controls", "What the result depends on");
bullets(s, [
  "Make the slow conductance fast: performance falls to chance, exactly as the theorem predicts.",
  "Make it much slower than a chunk: it bridges the silence and spare units start duplicating.",
  "On the raw stimulus, where every rate cue is provably zero, the same units still score 100 percent.",
  "Through layer 1 a weak rate cue does exist, because adaptation makes the first tone louder than the second. Worth saying out loud.",
]);
callout(s, "The middle timescale range is the only one that recovers both the order and the right number of units.",
  0.65, 5.7, 11.8, INK);
note(s, "The raw stimulus control is the airtight version of the claim.");

// 23
s = newSlide("Result 3: when one order becomes rare",
  "Ninety percent AB against ten percent BA");
bullets(s, [
  "Between 80 and 90 percent the vocabulary collapses from two units to one.",
  "Only the frequent chunk keeps a detector.",
  "The rare chunk is still perfectly detectable, because the detector simply fails to fire for it.",
  "But nothing names it. It is encoded as prediction error, not as an object.",
]);
callout(s, "Testable prediction: at strong oddball ratios expect mismatch without deviant specific decoding.",
  0.65, 5.7, 11.8, GREEN);
note(s, "This is the panel that connects the model to the recordings.");

// 24
s = newSlide("Result 4: the Saffran test",
  "Four words hidden in one continuous stream, boundaries marked only by statistics");
bullets(s, [
  "It prefers the within word transitions, 75 percent against 40 percent chance. Scrambled control gives zero.",
  "It passes the classic word against part word test, area under the ROC 0.86.",
  "But no unit spans a whole word, and no unit passes the test on its own.",
  "It passes by counting: a word contains two frequent transitions, a part word only one.",
]);
callout(s, "It reproduces the behaviour by a weaker mechanism than the behaviour is usually taken to show.",
  0.65, 5.7, 11.8, RED);
note(s, "Say this plainly rather than reporting the ROC and stopping.");

// 25
s = newSlide("What it does, and what it does not",
  "The honest summary");
s.addText("Does", { x: 0.9, y: 1.95, w: 5.2, h: 0.4, fontFace: F, fontSize: 17,
  bold: true, color: GREEN, margin: 0 });
bullets(s, [
  "Reads temporal order that no rate code can see",
  "Discovers how many chunks exist, rather than being told",
  "Produces units you can read directly off the weights",
  "Uses four local rules and one new variable",
], { x: 0.9, y: 2.45, w: 5.2, fontSize: 13.5 });
s.addText("Does not", { x: 6.9, y: 1.95, w: 5.4, h: 0.4, fontFace: F,
  fontSize: 17, bold: true, color: RED, margin: 0 });
bullets(s, [
  "Represent a whole word as a single object",
  "Find chunk boundaries without a silence or a statistic",
  "Benefit from layer 1: the raw stimulus works just as well",
  "Behave as a hierarchy, on the evidence so far",
], { x: 6.9, y: 2.45, w: 5.4, fontSize: 13.5 });
callout(s, "The last point matters most: this is two parallel mechanisms, not a hierarchy.",
  0.9, 5.9, 11.4, INK);
note(s, "Ending on the limitation is deliberate.");

pres.writeFile({ fileName: "two_layer_algorithm.pptx" })
  .then(() => console.log(`wrote two_layer_algorithm.pptx with ${N} slides`));
