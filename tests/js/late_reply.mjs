// Drives the chat page's app.js with a fake DOM and a fake server, to replay late replies.
//   node tests/js/late_reply.mjs <scenario>     exit 0: pass, 1: fail (message on stderr)
// One scenario per process: app.js is a module and starts once.

const APP = new URL("../../src/secure_rag/static/app/app.js", import.meta.url).href;
const scenario = process.argv[2];

// ---- a fake DOM: just what app.js uses ----------------------------------------------------------

class Node {}
class Text extends Node {
  constructor(text) { super(); this.text = text; this.parent = null; }
  get textContent() { return this.text; }
}
class El extends Node {
  constructor(tag) {
    super();
    Object.assign(this, { tagName: tag, children: [], attrs: {}, listeners: {}, parent: null, className: "", hidden: false, disabled: false, value: "", files: [] });
    const self = this;
    this.classList = {
      add: (c) => { if (!self.cls().includes(c)) self.className = `${self.className} ${c}`.trim(); },
      remove: (c) => { self.className = self.cls().filter((x) => x !== c).join(" "); },
      toggle: (c) => { if (self.cls().includes(c)) { self.classList.remove(c); return false; } self.classList.add(c); return true; },
      contains: (c) => self.cls().includes(c),
    };
  }
  cls() { return this.className.split(/\s+/).filter(Boolean); }
  setAttribute(k, v) { this.attrs[k] = v; if (k === "disabled") this.disabled = true; if (k === "id") this.id = v; if (k === "value") this.value = v; }
  getAttribute(k) { return this.attrs[k]; }
  addEventListener(n, f) { (this.listeners[n] ||= []).push(f); }
  dispatch(n) { for (const f of this.listeners[n] || []) f({ preventDefault() {}, currentTarget: this, target: this }); }
  click() { if (!this.disabled) this.dispatch("click"); }
  append(...nodes) { for (const n of nodes) { const c = n instanceof Node ? n : new Text(String(n)); c.parent?.removeChild?.(c); c.parent = this; this.children.push(c); } }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); c.parent = null; }
  remove() { this.parent?.removeChild(this); }
  replaceChildren(...nodes) { for (const c of this.children) c.parent = null; this.children = []; this.append(...nodes); }
  set textContent(t) { this.replaceChildren(new Text(t)); }
  get textContent() { return this.children.map((c) => c.textContent).join(""); }
  all() { return this.children.flatMap((c) => (c instanceof El ? [c, ...c.all()] : [])); }
  match(sel) { return sel.split(",").map((s) => s.trim()).some((s) => (s.startsWith(".") ? this.cls().includes(s.slice(1)) : this.tagName === s)); }
  querySelector(sel) { return this.all().find((e) => e.match(sel)) || null; }
  querySelectorAll(sel) { return this.all().filter((e) => e.match(sel)); }
  closest(sel) { let e = this; while (e) { if (e.match(sel)) return e; e = e.parent; } return null; }
  scrollIntoView() {}
  focus() {}
}
const byId = {};
const store = () => ({ m: {}, getItem(k) { return this.m[k] ?? null; }, setItem(k, v) { this.m[k] = v; } });
Object.assign(globalThis, {
  Node,
  document: {
    createElement: (t) => new El(t), createElementNS: (_, t) => new El(t), createTextNode: (t) => new Text(t),
    getElementById: (id) => (byId[id] ||= new El("div")), documentElement: new El("html"), addEventListener() {}, visibilityState: "visible",
  },
  location: { href: "http://x/app?as=omar#case=R1", hash: "#case=R1" },
  history: { replaceState() {} },
  sessionStorage: store(),
  localStorage: store(),
  matchMedia: () => ({ matches: false }),
});

// ---- a fake server -----------------------------------------------------------------------------

const base = {
  run_id: "R1", entry: "boq", mode: "agent", requester: "khalid", lines: [], rfq: [], review: [], packages: [], quotes: [], comparison: [],
  agent_steps: [], events: [], orders: [], cost_usd: 0, tokens: 0, policy: {}, summaries: {}, approver_roles: ["finance"], vendor_names: {}, decide_problem: "",
};
const waitingQuotes = { ...base, status: "awaiting_quotes", pending_gate: null, can_decide: false, total: "0.00", budget_after: null, steps: [{ name: "wait_quotes", status: "waiting", detail: { round: 1 } }] };
const gate = (run_id, name, total) => ({ ...base, run_id, status: `awaiting_${name}`, pending_gate: name, can_decide: true, total, budget_after: "1.00", steps: [{ name: `gate_${name}`, status: "waiting", detail: { roles: [name] } }] });
const losingBid = { ...waitingQuotes, status: "awaiting_quote_decision", steps: [], quotes: [{ vendor: "Losing Bidder LLC", package: "p", total: "9999.00", round: 1, problems: [], flags: [], counted: true }] };
const approverDone = { ...gate("R1", "finance", "100.00"), status: "done", pending_gate: null, steps: [], can_decide: false,
  lines: [{ item_no: "1", description: "Glass", vendor: "V", quantity: "1", unit: "m2", code: "quote", unit_price: "7777.00", total: "7777.00" }], orders: [{ vendor: "V", lines: 1, total: "7777.00" }] };

const routes = {}; // path -> (user, opts) => response promise
const calls = [];
const json = (body, status = 200) => ({ ok: status < 400, status, statusText: "", json: async () => body });
globalThis.fetch = (path, opts = {}) => {
  const user = opts.headers?.["X-User"];
  calls.push(`${opts.method || "GET"} ${path} as ${user}`);
  if (path === "/ui/users") return Promise.resolve(json([{ user_id: "omar", name: "Omar P" }, { user_id: "khalid", name: "Khalid S" }]));
  if (path === "/me") return Promise.resolve(json({ user_id: user, name: user, roles: user === "omar" ? ["procurement", "finance"] : ["site_engineer"], llm: "offline", demo: true }));
  const route = Object.keys(routes).find((prefix) => path.startsWith(prefix));
  return route ? routes[route](user, opts) : Promise.resolve(json({ detail: "nope" }, 404));
};

// ---- helpers --------------------------------------------------------------------------------------

const tick = () => new Promise((resolve) => setTimeout(resolve, 30));
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const log = () => byId.log;
const button = (label) => log().querySelectorAll("button").find((b) => b.textContent.startsWith(label));
function check(ok, message) {
  if (!ok) {
    console.error(`FAIL ${scenario}: ${message}`);
    process.exit(1);
  }
}
async function switchTo(userId) {
  byId.persona.value = userId;
  byId.persona.dispatch("change");
  await tick(); await tick();
}

// ---- scenarios --------------------------------------------------------------------------------------

const scenarios = {
  // Omar's late demo reply must not land on Khalid's screen (it holds losing bids).
  async deliver_after_switch() {
    let release;
    routes["/workflows/R1/events"] = () => new Promise((resolve) => { release = () => resolve(json(losingBid)); });
    routes["/workflows/R1"] = () => Promise.resolve(json(waitingQuotes));
    routes["/workflows?scope=mine"] = (user) => Promise.resolve(json(user === "khalid" ? [{ run_id: "R1", entry: "boq", status: "awaiting_quotes" }] : []));
    routes["/workflows?scope"] = () => Promise.resolve(json([]));
    await import(APP); await tick(); await tick();
    button("Simulate vendor replies").click(); await tick();
    await switchTo("khalid");
    byId["mine-list"].querySelector("button").click(); await tick(); await tick();
    release(); await tick(); await tick();
    check(!log().textContent.includes("9,999.00"), "Khalid sees Omar's late reply with a losing bid");
  },

  // An approver's late decision reply must not land on the requester's screen.
  async decide_after_switch() {
    let release;
    routes["/workflows/R1/decision"] = () => new Promise((resolve) => { release = () => resolve(json(approverDone)); });
    routes["/workflows/R1"] = (user) => Promise.resolve(json(user === "omar" ? gate("R1", "finance", "100.00") : { ...gate("R1", "finance", "100.00"), can_decide: false, decide_problem: "requester" }));
    routes["/workflows?scope=mine"] = (user) => Promise.resolve(json(user === "khalid" ? [{ run_id: "R1", entry: "boq", status: "awaiting_finance" }] : []));
    routes["/workflows?scope"] = () => Promise.resolve(json([]));
    await import(APP); await tick(); await tick();
    button("Approve AED").click(); await tick();
    await switchTo("khalid");
    byId["mine-list"].querySelector("button").click(); await tick(); await tick();
    release(); await tick(); await tick();
    check(!log().textContent.includes("7,777.00"), "Khalid sees the approver's view with a quoted line price");
  },

  // A late reply for another case must not replace the open case's card; a click decides the case it shows.
  async wrong_case() {
    let release;
    routes["/workflows/R1/events"] = () => new Promise((resolve) => { release = () => resolve(json(gate("R1", "procurement", "111.00"))); });
    routes["/workflows/R1/decision"] = () => new Promise(() => {});
    routes["/workflows/R2/decision"] = () => new Promise(() => {});
    routes["/workflows/R1"] = () => Promise.resolve(json(waitingQuotes));
    routes["/workflows/R2"] = () => Promise.resolve(json(gate("R2", "procurement", "99999.00")));
    routes["/workflows?scope=waiting"] = () => Promise.resolve(json([{ run_id: "R2", entry: "boq", status: "awaiting_procurement" }]));
    routes["/workflows?scope"] = () => Promise.resolve(json([]));
    await import(APP); await tick(); await tick();
    button("Simulate vendor replies").click(); await tick();
    byId["waiting-list"].querySelector("button").click(); await tick(); await tick();
    release(); await tick(); await tick();
    check(log().querySelector(".card-wait").textContent.includes("99,999.00"), "the open case's card was replaced by another case");
    button("Approve").click(); await tick();
    const posts = calls.filter((c) => c.startsWith("POST") && c.includes("/decision"));
    check(posts.length === 1 && posts[0].startsWith("POST /workflows/R2/"), `decision went to ${posts}`);
  },

  // One failed poll must not hide the case once polls succeed again.
  async failed_poll() {
    let failNext = 0;
    routes["/workflows/R1"] = () => {
      if (failNext > 0) { failNext -= 1; return Promise.resolve(json({ detail: "boom" }, 500)); }
      return Promise.resolve(json(gate("R1", "finance", "100.00")));
    };
    routes["/workflows?scope"] = () => Promise.resolve(json([]));
    await import(APP); await tick(); await tick();
    check(!!button("Approve AED"), "no approve button at start");
    failNext = 1;
    await wait(1700);
    await wait(3400);
    check(!!button("Approve AED"), "the approve button stays gone after good polls");
  },
};

if (!scenarios[scenario]) {
  console.error(`unknown scenario ${scenario}`);
  process.exit(2);
}
await scenarios[scenario]();
process.exit(0);
