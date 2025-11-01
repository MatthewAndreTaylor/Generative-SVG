let lines = [];
let currentLine = [];
let isDrawing = false;
let tolerance = 2.0;
let bins = 64;
let showKeyPoints = false;
let lastTime = new Date();
let tokenizer;

function setup() {
  const width = 512;
  const height = 512;

  let cnv = createCanvas(width, height);
  cnv.parent("canvas-container");
  background(255);
  stroke(0);
  strokeWeight(2);
  noFill();

  // Initialize tokenizer here
  tokenizer = new DeltaPenPositionTokenizer(bins, width, height);

  document.getElementById("clear").onclick = () => {
    lines = [];
    background(255);
    document.getElementById("svg-output").innerHTML =
      "(draw something to see SVG)";
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
    let simplified = rdpSimplify(currentLine, tolerance);
    lines.push(simplified);
    updateSVG();
  }
}

function rdpSimplify(points, epsilon) {
  if (points.length < 3) return points;

  let dmax = 0,
    index = 0;
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
    (end.y - start.y) * p.x -
      (end.x - start.x) * p.y +
      end.x * start.y -
      end.y * start.x
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

function updateSVG() {
  const strokes = lines.map((l) => l.map((p) => [p.x, p.y]));
  const tokens = tokenizer.encode(strokes);
  const svg = tokenizer.decode(tokens);
  document.getElementById("svg-output").innerHTML = svg;
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
    const width = this.width;
    const height = this.height;

    const scaleX = (bins - 1) / width;
    const scaleY = (bins - 1) / height;

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

  decode(tokens, strokeWidth = 0.4) {
    const svgParts = [
      `<svg viewBox="0 0 ${this.bins} ${this.bins}" xmlns="http://www.w3.org/2000/svg"><g stroke-width="${strokeWidth}">`,
    ];
    let pathCmds = [];
    let x = 0;
    let y = 0;

    for (let token of tokens) {
      const item = this.invVocab.get(token);
      if (item === "START" || item === "PAD") continue;
      if (item === "END") break;

      if (item === "MOVE") {
        if (pathCmds.length) {
          svgParts.push(
            `<path d="${pathCmds.join(" ")}" stroke="black" fill="none"/>`
          );
          pathCmds = [];
        }
        continue;
      }

      const [dx, dy] = item;
      x += dx;
      y += dy;

      if (!pathCmds.length) pathCmds.push(`M ${x} ${y}`);
      else pathCmds.push(`L ${x} ${y}`);
    }

    if (pathCmds.length)
      svgParts.push(
        `<path d="${pathCmds.join(" ")}" stroke="black" fill="none"/>`
      );

    svgParts.push("</g></svg>");
    return svgParts.join("\n");
  }
}
