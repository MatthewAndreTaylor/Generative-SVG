let lines = [];
let currentLine = [];
let isDrawing = false;
let tolerance = 2.0;
const width = 384;
const height = 384;

let showKeyPoints = false;
let lastTime = new Date();
let tokenizer;
var currentMode = "generation";

function applyModeUI() {
  const container = document.getElementById("canvas-container");
  const genRadio = document.getElementById("mode-generation");
  const shouldHide = genRadio && genRadio.checked;
  if (container) {
    if (shouldHide) {
      container.classList.add("is-hidden");
    } else {
      container.classList.remove("is-hidden");
    }
  }
  currentMode = shouldHide ? "generation" : "completion";
}

function getClassLabel() {
  const select = document.getElementById("example-class");
  const opt = select.options[select.selectedIndex];
  const idxAttr = opt.getAttribute("data-index");
  const parsed = parseInt(idxAttr, 10);
  return parsed;
}

function getSelectedModel() {
  const sel = document.getElementById("model-select");
  if (sel && sel.value) return sel.value;
  if (window.SELECTED_MODEL) return window.SELECTED_MODEL;
  return "small";
}

function setup() {
  let cnv = createCanvas(width, height);
  cnv.parent("canvas-container");
  background(255);
  stroke(0);
  strokeWeight(2);
  noFill();
  // Hook up mode radios
  const genRadio = document.getElementById("mode-generation");
  const compRadio = document.getElementById("mode-completion");
  if (genRadio) genRadio.addEventListener("change", applyModeUI);
  if (compRadio) compRadio.addEventListener("change", applyModeUI);
  // Apply initial mode state
  applyModeUI();
  document.getElementById("clear").onclick = () => {
    lines = [];
    background(255);
    const container = document.getElementById("svg-output");
    container.innerHTML = "(generate something to see output)";
    container.classList.remove("svg-grid");
    delete container.dataset.initialized;
  };
}

function draw() {
  background(255);
  for (let line of lines) {
    drawSmoothLine(line);
    drawKeyPoints(line);
  }
  if (isDrawing) {
    drawSmoothLine(currentLine);
    drawKeyPoints(currentLine);
  }
}


function mousePressed() {
  if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
    isDrawing = true;
    currentLine = [createVector(mouseX, mouseY)];
  }
}
function mouseDragged() {
  var ms = new Date();
  if (ms - lastTime < 25) {
    return;
  }
  lastTime = ms;
  if (isDrawing) {
    let prev = currentLine[currentLine.length - 1];
    let curr = createVector(mouseX, mouseY);
    if (p5.Vector.dist(prev, curr) > 2) {
      currentLine.push(curr);
    }
  }
}
function mouseReleased() {
  if (isDrawing) {
    isDrawing = false;
    const simplified = rdpSimplify(currentLine, tolerance);
    lines.push(simplified);
  }
}

function rdpSimplify(points, epsilon) {
  if (points.length < 3) return points;
  let dmax = 0;
  let index = 0;
  const start = points[0];
  const end = points[points.length - 1];
  for (let i = 1; i < points.length - 1; i++) {
    const d = perpendicularDistance(points[i], start, end);
    if (d > dmax) {
      index = i;
      dmax = d;
    }
  }
  if (dmax > epsilon) {
    const left = rdpSimplify(points.slice(0, index + 1), epsilon);
    const right = rdpSimplify(points.slice(index), epsilon);
    return left.slice(0, -1).concat(right);
  } else {
    return [start, end];
  }
}
function perpendicularDistance(p, start, end) {
  const num = Math.abs(
    (end.y - start.y) * p.x - (end.x - start.x) * p.y + end.x * start.y - end.y * start.x
  );
  const den = dist(start.x, start.y, end.x, end.y);
  return den === 0 ? 0 : num / den;
}
function drawSmoothLine(points) {
  if (points.length < 2) return;
  stroke(0);
  strokeWeight(2);
  noFill();
  beginShape();
  for (let p of points) vertex(p.x, p.y);
  endShape();
}
function drawKeyPoints(points) {
  if (!showKeyPoints) return;
  noStroke();
  fill(255, 0, 0);
  for (let p of points) circle(p.x, p.y, 6);
}

async function modelExample(tokens) {
  const eosId = tokenizer.vocab.get("END");
  const classLabel = getClassLabel();
  const start_tokens =
    currentMode === "completion" ? tokens.slice(0, -1) : [tokenizer.vocab.get("START")];

  const body = {
    start_tokens: start_tokens,
    eos_id: eosId,
    class_label: classLabel,
    model: getSelectedModel()
  };

  const response = await fetch("/sample", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    console.error("Error from model:", await response.text());
    return tokens;
  }

  const data = await response.json();
  return data.tokens;
}

async function updateSVG() {
  const genBtn = document.getElementById("generate");
  if (genBtn) {
    genBtn.disabled = true;
    genBtn.classList.add("is-loading");
    genBtn.setAttribute("aria-busy", "true");
  }

  try {
    const strokes = lines.map((l) => l.map((p) => [p.x, p.y]));
    const tokens = tokenizer.encode(strokes);
    const modelTokens = await modelExample(tokens);
    const svg = tokenizer.decode(modelTokens);
    const container = document.getElementById("svg-output");
    // Initialize grid, first time (remove placeholder text)
    if (!container.dataset.initialized) {
      container.innerHTML = "";
      container.classList.add("svg-grid");
      container.dataset.initialized = "true";
    }
    // Wrapper grid cell for the new SVG
    const cell = document.createElement("div");
    cell.className = "svg-item";
    cell.innerHTML = svg;

    // Add a download button overlay
    const dlBtn = document.createElement("button");
    dlBtn.type = "button";
    dlBtn.className = "svg-download-btn";
    dlBtn.title = "Download SVG";
    dlBtn.setAttribute("aria-label", "Download SVG");
    dlBtn.innerHTML = "⤓";

    dlBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const filename = `sketch_${new Date().toISOString().replace(/[:.]/g, "-")}.svg`;
      const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    cell.appendChild(dlBtn);
    container.appendChild(cell);
  } catch (err) {
    console.error("Generate failed:", err);
    alert("Error during generation. Inference server may be unavailable.");
  } finally {
    if (genBtn) {
      genBtn.disabled = false;
      genBtn.classList.remove("is-loading");
      genBtn.removeAttribute("aria-busy");
    }
  }
}

class DeltaPenPositionTokenizer {
  constructor(bins, width, height) {
    this.bins = bins;
    this.width = width;
    this.height = height;
    this.vocab = new Map();
    this.invVocab = new Map();
    let idx = 0;
    for (let x = -bins; x <= bins; x++) {
      for (let y = -bins; y <= bins; y++) {
        this.vocab.set(`${x},${y}`, idx);
        this.invVocab.set(idx, [x, y]);
        idx++;
      }
    }
    for (let penToken of ["MOVE", "PAD", "START", "END"]) {
      this.vocab.set(penToken, idx);
      this.invVocab.set(idx, penToken);
      idx++;
    }
  }
  quantizeLine(points) {
    const bins = this.bins;
    const scaleX = (bins - 1) / this.width;
    const scaleY = (bins - 1) / this.height;
    return points.map(([x, y]) => {
      let qx = x * scaleX;
      let qy = y * scaleY;
      qx = Math.max(0, Math.min(Math.round(qx), bins - 1));
      qy = Math.max(0, Math.min(Math.round(qy), bins - 1));
      return [qx, qy];
    });
  }
  encode(strokes) {
    const tokens = [this.vocab.get("START")];
    let prev = [0, 0];
    for (let stroke of strokes) {
      if (!stroke.length) continue;
      const qStroke = this.quantizeLine(stroke);
      tokens.push(this.vocab.get("MOVE"));
      let [x0, y0] = qStroke[0];
      let dx = x0 - prev[0];
      let dy = y0 - prev[1];
      prev = [x0, y0];
      const firstKey = `${dx},${dy}`;
      tokens.push(this.vocab.get(firstKey) ?? this.vocab.get("PAD"));
      for (let i = 1; i < qStroke.length; i++) {
        const [x, y] = qStroke[i];
        dx = x - prev[0];
        dy = y - prev[1];
        prev = [x, y];
        const key = `${dx},${dy}`;
        tokens.push(this.vocab.get(key) ?? this.vocab.get("PAD"));
      }
    }
    tokens.push(this.vocab.get("END"));
    return tokens;
  }
  decode(tokens, strokeWidth = 0.3) {
    const svgParts = [
      `<svg viewBox="0 0 ${this.bins} ${this.bins}" xmlns="http://www.w3.org/2000/svg"><g stroke-width="${strokeWidth}">`
    ];
    let strokes = [];
    let currentStroke = [];
    let x = 0;
    let y = 0;

    for (let token of tokens) {
      const item = this.invVocab.get(token);

      if (item === "START" || item === "PAD") continue;
      if (item === "END") break;

      if (item === "MOVE") {
        if (currentStroke.length > 0) {
          strokes.push(currentStroke);
          currentStroke = [];
        }
        continue;
      }

      const [dx, dy] = item;
      x += dx;
      y += dy;
      currentStroke.push(createVector(x, y));
    }

    if (currentStroke.length > 0) strokes.push(currentStroke);

    for (let stroke of strokes) {
      const sampled = resampleStroke(stroke, 20); // like num_samples=20
      const beziers = fitCurve(sampled, 1.0); // like maxError
      const pathD = bezierArrayToSVG(beziers);
      svgParts.push(`<path d="${pathD}" stroke="black" fill="none" />`);
    }

    svgParts.push("</g></svg>");
    return svgParts.join("");
  }
}

function resampleStroke(points, numSamples = 20) {
  if (points.length <= numSamples) return points.slice();

  const u = [0];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    total += p5.Vector.dist(points[i], points[i - 1]);
    u.push(total);
  }
  u.forEach((_, i) => {
    u[i] /= total;
  });

  const resampled = [];
  for (let i = 0; i < numSamples; i++) {
    const t = i / (numSamples - 1);
    let k = 0;
    while (k < u.length - 2 && u[k + 1] < t) k++;
    const tSeg = (t - u[k]) / (u[k + 1] - u[k] || 1);
    const p = p5.Vector.lerp(points[k], points[k + 1], tSeg);
    resampled.push(p);
  }
  return resampled;
}

// Greedy fit of cubic Bezier curves to a set of points
function dot(a, b) {
  return a.x * b.x + a.y * b.y;
}

function sub(a, b) {
  return createVector(a.x - b.x, a.y - b.y);
}

function normalize(v) {
  const m = Math.sqrt(v.x * v.x + v.y * v.y);
  return m === 0 ? createVector(0, 0) : createVector(v.x / m, v.y / m);
}

const Bezier = {
  q(ctrl, t) {
    const mt = 1 - t;
    return createVector(
      mt ** 3 * ctrl[0].x +
        3 * mt ** 2 * t * ctrl[1].x +
        3 * mt * t ** 2 * ctrl[2].x +
        t ** 3 * ctrl[3].x,

      mt ** 3 * ctrl[0].y +
        3 * mt ** 2 * t * ctrl[1].y +
        3 * mt * t ** 2 * ctrl[2].y +
        t ** 3 * ctrl[3].y
    );
  },

  qprime(ctrl, t) {
    const mt = 1 - t;
    return createVector(
      3 * mt ** 2 * (ctrl[1].x - ctrl[0].x) +
        6 * mt * t * (ctrl[2].x - ctrl[1].x) +
        3 * t ** 2 * (ctrl[3].x - ctrl[2].x),

      3 * mt ** 2 * (ctrl[1].y - ctrl[0].y) +
        6 * mt * t * (ctrl[2].y - ctrl[1].y) +
        3 * t ** 2 * (ctrl[3].y - ctrl[2].y)
    );
  },

  qprimeprime(ctrl, t) {
    const mt = 1 - t;
    return createVector(
      6 * mt * (ctrl[2].x - 2 * ctrl[1].x + ctrl[0].x) +
        6 * t * (ctrl[3].x - 2 * ctrl[2].x + ctrl[1].x),

      6 * mt * (ctrl[2].y - 2 * ctrl[1].y + ctrl[0].y) +
        6 * t * (ctrl[3].y - 2 * ctrl[2].y + ctrl[1].y)
    );
  }
};

function chordLengthParameterize(points) {
  const u = [0];
  let total = 0;

  for (let i = 1; i < points.length; i++) {
    total += p5.Vector.dist(points[i], points[i - 1]);
    u.push(total);
  }

  for (let i = 1; i < u.length; i++) u[i] /= total;
  return u;
}

function newtonRaphsonRootFind(bez, point, u) {
  const q = Bezier.q(bez, u);
  const q1 = Bezier.qprime(bez, u);
  const q2 = Bezier.qprimeprime(bez, u);

  const diff = sub(q, point);
  const numerator = dot(diff, q1);
  const denominator = dot(q1, q1) + dot(diff, q2);

  if (Math.abs(denominator) < 1e-9) return u;
  return u - numerator / denominator;
}

function reparameterize(bez, points, u) {
  return u.map((ui, i) => newtonRaphsonRootFind(bez, points[i], ui));
}

function computeMaxError(points, bez, u) {
  let maxDist = 0;
  let splitPoint = Math.floor(points.length / 2);

  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const q = Bezier.q(bez, u[i]);
    const d = p5.Vector.dist(p, q);
    if (d > maxDist) {
      maxDist = d;
      splitPoint = i;
    }
  }
  return [maxDist, splitPoint];
}

function generateBezier(points, u, leftTangent, rightTangent) {
  const bez = [points[0], null, null, points[points.length - 1]];
  let C = [
    [0, 0],
    [0, 0]
  ];
  let X = [0, 0];

  let A = [];
  for (let ui of u) {
    const A1 = p5.Vector.mult(leftTangent, 3 * (1 - ui) ** 2 * ui);
    const A2 = p5.Vector.mult(rightTangent, 3 * (1 - ui) * ui ** 2);
    A.push([A1, A2]);
  }

  for (let i = 0; i < points.length; i++) {
    const [A1, A2] = A[i];
    C[0][0] += dot(A1, A1);
    C[0][1] += dot(A1, A2);
    C[1][0] += dot(A1, A2);
    C[1][1] += dot(A2, A2);

    const tmp = sub(
      points[i],
      Bezier.q([points[0], points[0], points.at(-1), points.at(-1)], u[i])
    );
    X[0] += dot(A1, tmp);
    X[1] += dot(A2, tmp);
  }

  const detC0C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1];
  let alphaL, alphaR;

  if (Math.abs(detC0C1) > 1e-12) {
    alphaL = (X[1] * C[0][0] - X[0] * C[1][0]) / detC0C1;
    alphaR = (X[0] * C[1][1] - X[1] * C[0][1]) / detC0C1;
  } else {
    alphaL = alphaR = 0;
  }

  const segLength = p5.Vector.dist(points[0], points.at(-1));
  const eps = 1e-6 * segLength;

  if (alphaL < eps || alphaR < eps) {
    bez[1] = p5.Vector.add(points[0], p5.Vector.mult(leftTangent, segLength / 3));
    bez[2] = p5.Vector.add(points.at(-1), p5.Vector.mult(rightTangent, segLength / 3));
  } else {
    bez[1] = p5.Vector.add(points[0], p5.Vector.mult(leftTangent, alphaL));
    bez[2] = p5.Vector.add(points.at(-1), p5.Vector.mult(rightTangent, alphaR));
  }

  return bez;
}

function fitCubic(points, leftTangent, rightTangent, error) {
  if (points.length === 2) {
    const dist = p5.Vector.dist(points[0], points[1]) / 3;
    return [
      [
        points[0],
        p5.Vector.add(points[0], p5.Vector.mult(leftTangent, dist)),
        p5.Vector.add(points[1], p5.Vector.mult(rightTangent, dist)),
        points[1]
      ]
    ];
  }

  let u = chordLengthParameterize(points);
  let bez = generateBezier(points, u, leftTangent, rightTangent);
  let [maxError, splitPoint] = computeMaxError(points, bez, u);

  if (maxError < error) return [bez];

  if (maxError < error * error) {
    for (let i = 0; i < 20; i++) {
      const uPrime = reparameterize(bez, points, u);
      bez = generateBezier(points, uPrime, leftTangent, rightTangent);

      const [m, sp] = computeMaxError(points, bez, uPrime);
      if (m < error) return [bez];

      u = uPrime;
      maxError = m;
      splitPoint = sp;
    }
  }

  const centerTangent = normalize(sub(points[splitPoint - 1], points[splitPoint + 1]));

  return [
    ...fitCubic(points.slice(0, splitPoint + 1), leftTangent, centerTangent, error),
    ...fitCubic(points.slice(splitPoint), p5.Vector.mult(centerTangent, -1), rightTangent, error)
  ];
}

function fitCurve(points, error) {
  if (!points || points.length < 2) return [];
  const leftTangent = normalize(sub(points[1], points[0]));
  const rightTangent = normalize(sub(points[points.length - 2], points[points.length - 1]));
  return fitCubic(points, leftTangent, rightTangent, error);
}

function bezierArrayToSVG(beziers) {
  if (!beziers || beziers.length === 0) return "";
  const cmds = [];
  const first = beziers[0][0];
  cmds.push(`M ${first.x} ${first.y}`);

  // Then add a cubic command for each Bézier
  for (const bez of beziers) {
    const [p0, p1, p2, p3] = bez;
    // p0 is already the current point; SVG only needs p1, p2, p3
    cmds.push(`C ${p1.x} ${p1.y} ${p2.x} ${p2.y} ${p3.x} ${p3.y}`);
  }

  return cmds.join(" ");
}
