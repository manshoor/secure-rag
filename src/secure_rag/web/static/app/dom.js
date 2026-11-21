// DOM helpers. Server data only ever reaches the page as text nodes: no HTML strings anywhere.

const SVG_NS = "http://www.w3.org/2000/svg"; // a namespace name, never fetched

// Stroke icons from the boards, as element lists.
const ICONS = {
  shield: [["path", { d: "M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6z" }], ["polyline", { points: "9 12 11 14 15 10" }]],
  check: [["polyline", { points: "20 6 9 17 4 12" }]],
  clock: [["circle", { cx: "12", cy: "12", r: "9" }], ["polyline", { points: "12 7 12 12 15 14" }]],
  warn: [["path", { d: "M12 3l9.5 17h-19z" }], ["line", { x1: "12", y1: "10", x2: "12", y2: "14" }], ["line", { x1: "12", y1: "17", x2: "12.01", y2: "17" }]],
  cross: [["circle", { cx: "12", cy: "12", r: "9" }], ["line", { x1: "15", y1: "9", x2: "9", y2: "15" }], ["line", { x1: "9", y1: "9", x2: "15", y2: "15" }]],
  stop: [["circle", { cx: "12", cy: "12", r: "9" }], ["line", { x1: "8", y1: "12", x2: "16", y2: "12" }]],
  clip: [["path", { d: "M21 12.5l-8.5 8.5a6 6 0 01-8.5-8.5l9-9a4 4 0 015.7 5.7l-9 9a2 2 0 01-2.8-2.8l8.3-8.3" }]],
  info: [["circle", { cx: "12", cy: "12", r: "9" }], ["line", { x1: "12", y1: "11", x2: "12", y2: "16" }], ["line", { x1: "12", y1: "8", x2: "12.01", y2: "8" }]],
};

/** An icon element; decorative, so screen readers skip it (the words next to it carry the meaning). */
export function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "icon");
  for (const [tag, attrs] of ICONS[name] || []) {
    const part = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
      part.setAttribute(key, value);
    }
    svg.append(part);
  }
  return svg;
}

/**
 * Build an element. props: class, text, on{Event} handlers, and plain attributes.
 * Children: nodes or strings (added as text); null and false are skipped; arrays are flattened.
 *   h("button", { class: "btn-ok", onClick: go }, "Approve")
 */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) {
      continue;
    }
    if (key === "class") {
      node.className = value;
      continue;
    }
    if (key === "text") {
      node.textContent = String(value);
      continue;
    }
    if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
      continue;
    }
    node.setAttribute(key, value === true ? "" : String(value));
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) {
      continue;
    }
    if (Array.isArray(child)) {
      append(node, child);
      continue;
    }
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/** Replace every child of a node. */
export function fill(node, ...children) {
  node.replaceChildren();
  append(node, children);
  return node;
}

/** A status word with its icon, so colour is never the only signal. */
export function status(kind, words) {
  const icons = { ok: "check", wait: "clock", warn: "warn", bad: "cross", muted: "stop" };
  return h("span", { class: `status status-${kind}` }, icon(icons[kind] || "info"), words);
}

const AED = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/** "33015.00" -> "33,015.00"; money stays a string from the server, this only adds separators. */
export function money(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  const [whole, cents = "00"] = String(value).split(".");
  const sign = whole.startsWith("-") ? "-" : "";
  return `${sign}${AED.format(Number(whole.replace("-", ""))).split(".")[0]}.${cents.padEnd(2, "0").slice(0, 2)}`;
}

export function plural(count, word, many = `${word}s`) {
  return `${count} ${count === 1 ? word : many}`;
}
