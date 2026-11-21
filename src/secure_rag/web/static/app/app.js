// Secure RAG chat page. Plain modules, no build step.
//
//   persona ─► /me, lists ─► chat log ◄─ POST /chat (answers, notices, case links)
//                               └── open case ◄─ GET /workflows/{id}, polled while visible
//   Decisions only from card buttons (POST /workflows/{id}/decision); typed text never decides.

import * as api from "./api.js";
import { caseCards, statusWords } from "./case.js";
import { fill, h, icon, status } from "./dom.js";
import { renderPanel } from "./panel.js";

const CASE_POLL_MS = 1500;
const LIST_POLL_MS = 5000;
const DEFAULT_PERSONA = "khalid";
const UPLOAD_MODE = "agent";
const DEFAULT_UPLOAD_TEXT = "Please price this BOQ.";
const TERMINAL = new Set(["done", "stopped", "rejected"]);
const GATE_CARDS = new Set(["rfq", "exception", "approval"]);
const OPENS_CASE = new Set(["case_status", "explain_case", "decide_request"]);
const PERSONA_KEY = "secure-rag-persona";
const LARGE_KEY = "secure-rag-large";

// Starting suggestions per role: what each person would try first.
const SUGGESTIONS = [
  ["site_engineer", ["Price a BOQ", "My cases"]],
  ["procurement", ["What is waiting on me?", "My cases"]],
  ["finance", ["What is waiting on me?", "What is the approval limit for purchases?"]],
  ["executive", ["What is waiting on me?"]],
  ["hr", ["How many leave days can I carry over?", "When must I submit a leave request?"]],
  ["employee", ["How many leave days can I carry over?"]],
];

const $ = (id) => document.getElementById(id);
const nodes = {
  log: $("log"), suggestions: $("suggestions"), composer: $("composer"), message: $("message"), file: $("file"),
  attach: $("attach"), attached: $("attached"), persona: $("persona"), badge: $("mode-badge"), demoTag: $("demo-tag"),
  waitingList: $("waiting-list"), waitingCount: $("waiting-count"), mineList: $("mine-list"), menu: $("menu"), menuCount: $("menu-count"),
  rail: $("rail"), panel: $("panel"), panelClose: $("panel-close"), banner: $("offline-banner"), large: $("large"),
  panelNodes: { title: $("panel-title"), tabs: $("tabs"), body: $("tab-panel") },
};

const state = {
  me: null,
  runId: null,
  view: null,
  caseNode: null,
  seen: new Set(),
  tab: "overview",
  file: null,
  busy: false,
  timer: null,
  listTimer: null,
  generation: 0, // bumps on persona switch: late replies for the old persona are dropped
};

// ---- storage (per-viewer conveniences only; the page works without it) ------------------------------

function load(storage, key) {
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function save(storage, key, value) {
  try {
    storage.setItem(key, value);
  } catch {
    // private window or blocked storage: nothing to keep
  }
}

// ---- errors and the offline banner ---------------------------------------------------------------

function reachable(ok) {
  nodes.banner.hidden = ok;
  if (!ok) {
    fill(nodes.banner, icon("warn"), h("span", {}, h("b", {}, "Can't reach the server. "), "Checking again every few seconds. Nothing was lost."));
  }
}

function alertCard(kind, title, text, action) {
  return h("div", { class: `alert alert-${kind}`, role: "alert" }, icon(kind === "bad" ? "cross" : "warn"),
    h("div", { class: "alert-body" }, h("span", { class: "alert-title" }, title), text ? h("span", {}, text) : null, action || null));
}

function say(...children) {
  const node = h("div", {}, ...children);
  nodes.log.append(node);
  node.scrollIntoView({ block: "end", behavior: "smooth" });
  return node;
}

function errorCard(error, fallbackTitle) {
  if (error.status === api.UNREACHABLE) {
    reachable(false);
    return alertCard("bad", "Can't reach the server", "Your message was not sent. Try again in a few seconds.");
  }
  if (error.status === 429 || error.status >= 500) {
    return alertCard("warn", "The server is busy", "Nothing was changed. Try again in a few seconds.");
  }
  return alertCard("bad", fallbackTitle, error.detail);
}

// ---- persona -------------------------------------------------------------------------------------

async function startPersona(userId) {
  state.generation += 1;
  api.actAs(userId);
  save(sessionStorage, PERSONA_KEY, userId);
  const url = new URL(location.href);
  url.searchParams.set("as", userId);
  history.replaceState(null, "", url);

  // A new person sees nothing the last one saw.
  const generation = state.generation;
  state.me = null;
  closeCase();
  nodes.log.replaceChildren();
  nodes.suggestions.replaceChildren();
  nodes.waitingList.replaceChildren();
  nodes.mineList.replaceChildren();
  for (const badge of [nodes.waitingCount, nodes.menuCount]) {
    badge.hidden = true;
  }
  clearFile();

  let me;
  try {
    me = await api.me();
    reachable(true);
  } catch (error) {
    if (generation === state.generation) {
      say(errorCard(error, "Could not sign in as this person"));
    }
    return;
  }
  if (generation !== state.generation) {
    return; // a later switch won
  }
  state.me = me;
  const llm = state.me.llm === "claude" ? "Claude" : "Offline";
  fill(nodes.badge, llm === "Claude" ? "Claude · paid model calls" : "Offline · model cost $0.00");
  nodes.demoTag.hidden = !state.me.demo;
  nodes.attach.hidden = !state.me.roles.includes("site_engineer");
  nodes.message.placeholder = state.me.roles.includes("site_engineer") ? "Ask a question, or attach a BOQ to price" : "Ask a question or about a case";
  showSuggestions();
  say(h("p", { class: "reply" }, `Hello ${state.me.name.split(" ")[0]}. Ask about company documents or your cases.`));
  await refreshLists();
}

function showSuggestions() {
  const roles = new Set(state.me?.roles || []);
  const found = SUGGESTIONS.find(([role]) => roles.has(role));
  fill(nodes.suggestions, (found ? found[1] : []).map((text) => h("button", { type: "button", class: "chip", onClick: () => send(text) }, text)));
}

// ---- lists ---------------------------------------------------------------------------------------

function listRow(row) {
  const [kind, word] = statusWords(row.status);
  return h("li", {}, h("button", {
    type: "button", class: "rail-item", "aria-current": String(row.run_id === state.runId),
    onClick: () => { nodes.rail.classList.remove("open"); openCase(row.run_id); },
  }, h("span", { class: "rail-item-title" }, `${row.entry === "boq" ? "BOQ" : "Request"} case ${row.run_id}`), status(kind, word)));
}

async function refreshLists() {
  if (!state.me) {
    return; // not signed in as anyone yet
  }
  const generation = state.generation;
  let waiting;
  let mine;
  try {
    [waiting, mine] = await Promise.all([api.runs("waiting"), api.runs("mine")]);
    reachable(true);
  } catch (error) {
    if (error.status === api.UNREACHABLE) {
      reachable(false);
    }
    return;
  }
  if (generation !== state.generation) {
    return;
  }
  fill(nodes.waitingList, waiting.length ? waiting.map(listRow) : h("li", { class: "empty" }, "Nothing right now."));
  fill(nodes.mineList, mine.length ? mine.map(listRow) : h("li", { class: "empty" }, "No cases yet."));
  for (const badge of [nodes.waitingCount, nodes.menuCount]) {
    badge.hidden = waiting.length === 0;
    fill(badge, String(waiting.length));
  }
  nodes.menu.setAttribute("aria-label", `Cases, ${waiting.length} waiting`);
}

// ---- the open case -------------------------------------------------------------------------------

function closeCase() {
  clearTimeout(state.timer);
  state.runId = null;
  state.view = null;
  state.caseNode = null;
  state.drawn = "";
  state.seen = new Set();
  nodes.panel.hidden = true;
}

async function openCase(runId, view = null) {
  if (runId !== state.runId) {
    clearTimeout(state.timer);
    state.runId = runId;
    state.seen = new Set();
    state.tab = "overview";
    state.caseNode?.remove();
    state.caseNode = null;
    state.view = null;
    state.drawn = "";
  }

  // The live case always sits at the end of the chat.
  if (!state.caseNode) {
    state.caseNode = h("div", { class: "case-live", "aria-label": "Open case" });
  }
  nodes.log.append(state.caseNode);
  if (view) {
    draw(view);
  }
  await refreshCase();

  // Show what needs the person: the decision card if one waits, else the case itself.
  const decision = state.caseNode?.querySelector(".card-wait");
  (decision || state.caseNode)?.scrollIntoView({ block: decision ? "center" : "start", behavior: "smooth" });
  refreshLists();
}

async function refreshCase() {
  clearTimeout(state.timer);
  const runId = state.runId;
  const generation = state.generation;
  if (!runId) {
    return;
  }
  try {
    const view = await api.run(runId);
    reachable(true);
    if (runId !== state.runId || generation !== state.generation) {
      return;
    }
    draw(view);
  } catch (error) {
    if (runId !== state.runId || generation !== state.generation) {
      return;
    }
    if (error.status === 404) {
      // The same words whether the case exists or not.
      fill(state.caseNode, alertCard("bad", "Not found or not visible to you", "The same reply whether the case exists or not.",
        h("button", { type: "button", class: "btn-plain", onClick: () => send("My cases") }, "Back to my cases")));
      nodes.panel.hidden = true;
      state.runId = null;
      return;
    }
    if (error.status !== api.UNREACHABLE) {
      fill(state.caseNode, errorCard(error, "Could not load the case"));
      state.drawn = ""; // the next good poll draws the case again
    }
  }
  schedule();
}

function schedule() {
  clearTimeout(state.timer);
  if (!state.runId || (state.view && TERMINAL.has(state.view.status))) {
    return;
  }
  state.timer = setTimeout(() => {
    if (document.visibilityState === "visible") {
      refreshCase();
    } else {
      schedule();
    }
  }, CASE_POLL_MS);
}

function draw(view) {
  // Redraw only on change: the chat log is a live region, and a redraw is announced.
  const json = JSON.stringify(view);
  if (json === state.drawn) {
    return;
  }
  state.drawn = json;
  const before = state.view;
  state.view = view;
  const ctx = { me: state.me, decide, deliver, showPanel };
  fill(state.caseNode, caseCards(view, state.seen, ctx));
  if (!nodes.panel.hidden || !before) {
    drawPanel();
  }
  if (before && before.status !== view.status) {
    refreshLists();
  }
}

function drawPanel() {
  if (!state.view) {
    return;
  }
  nodes.panel.hidden = false;
  state.tab = renderPanel(nodes.panelNodes, state.view, state.tab, (tab) => {
    state.tab = tab;
    drawPanel();
  });
}

function showPanel() {
  drawPanel();
  nodes.panelClose.focus();
}

// ---- decisions and demo events -------------------------------------------------------------------

function lockButtons(button) {
  const card = button.closest("section, .demo-control");
  const buttons = card ? [...card.querySelectorAll("button")] : [button];
  for (const item of buttons) {
    item.disabled = true;
  }
  return () => {
    for (const item of buttons) {
      item.disabled = false;
    }
  };
}

// A reply counts only for the person and case it was asked for: a late one is dropped.
function stillCurrent(runId, generation) {
  return runId === state.runId && generation === state.generation;
}

async function decide(runId, gate, choice, button) {
  const unlock = lockButtons(button);
  const generation = state.generation;
  try {
    const view = await api.decide(runId, choice, gate);
    if (stillCurrent(runId, generation)) {
      draw(view);
      schedule();
      refreshLists();
    }
  } catch (error) {
    if (!stillCurrent(runId, generation)) {
      return;
    }
    unlock();
    if (error.status === 409) {
      say(alertCard("warn", "This case moved on", "Someone decided this step already. Nothing was changed by your click.",
        h("button", { type: "button", class: "btn-plain", onClick: () => openCase(runId) }, "Show the latest")));
      return;
    }
    if (error.status === 404) {
      say(alertCard("bad", "Not found or not visible to you", "The same reply whether the case exists or not."));
      closeCase();
      refreshLists();
      return;
    }
    say(errorCard(error, "The decision was not saved"));
  }
}

async function deliver(runId, event, sample, button) {
  const unlock = lockButtons(button);
  const generation = state.generation;
  try {
    const view = await api.deliver(runId, event, sample);
    if (stillCurrent(runId, generation)) {
      draw(view);
      schedule();
      refreshLists();
    }
  } catch (error) {
    if (!stillCurrent(runId, generation)) {
      return;
    }
    unlock();
    say(errorCard(error, "The demo event was refused"));
  }
}

// ---- chat ----------------------------------------------------------------------------------------

function chatCard(card) {
  switch (card.kind) {
    case "answer":
      return h("section", { class: "card", "aria-label": "Answer" },
        h("div", { class: "answer" },
          h("p", {}, card.text),
          card.citations?.length ? h("div", { class: "sources" }, h("span", { class: "eyebrow" }, "Sources"),
            card.citations.map((cite) => h("span", { class: "source-chip" }, h("b", {}, cite.source_id), ` ${cite.title} › ${cite.section}`))) : null,
          h("span", { class: "checkline" }, icon("check"), "Only documents you may read · every sentence cited")));
    case "refusal":
      return h("section", { class: "card", "aria-label": "Answer" },
        h("div", { class: "answer" }, h("span", { class: "status status-muted" }, icon("stop"), card.text),
          h("span", { class: "muted small" }, "Answers come only from documents your roles allow.")));
    case "notice":
      return h("div", { class: "notice" }, icon("info"), h("div", { class: "alert-body" }, h("span", {}, card.text),
        card.samples?.length ? h("div", { class: "sources" }, card.samples.map((name) =>
          h("button", { type: "button", class: "chip", onClick: () => startSample(name) }, `Use sample ${name}`))) : null));
    case "case":
    case "rfq":
    case "exception":
    case "approval": {
      const [kind, word] = statusWords(card.status || (GATE_CARDS.has(card.kind) ? `awaiting_${card.gate === "quotes" ? "quote_decision" : card.gate}` : ""));
      return h("div", { class: "notice" }, status(kind, word),
        h("span", {}, `Case ${card.run_id} · started by ${card.requester}`),
        h("button", { type: "button", class: "btn-plain", onClick: () => openCase(card.run_id) }, "Open"));
    }
    default:
      return null; // timeline, quotes and orders come from the live case, not from chat cards
  }
}

async function send(text) {
  const message = text.trim();
  if (state.busy || (!message && !state.file)) {
    return;
  }
  if (state.file) {
    await startCase(state.file, message || DEFAULT_UPLOAD_TEXT);
    return;
  }
  state.busy = true;
  const generation = state.generation;
  say(h("div", { class: "bubble-me" }, message));
  nodes.message.value = "";
  try {
    const reply = await api.chat(message, state.runId);
    reachable(true);
    if (generation !== state.generation) {
      return;
    }
    say(h("p", { class: "reply" }, reply.text), reply.cards.map(chatCard));

    // Cases the reply points at open live below it.
    const linked = reply.cards.filter((card) => card.run_id);
    const gates = reply.cards.filter((card) => GATE_CARDS.has(card.kind));
    if (OPENS_CASE.has(reply.intent) && linked.length) {
      await openCase(linked[0].run_id);
    } else if (reply.intent === "waiting_on_me" && gates.length === 1) {
      await openCase(gates[0].run_id);
    }
  } catch (error) {
    if (generation === state.generation) {
      nodes.message.value = message; // kept, so nothing is lost
      say(errorCard(error, "The message was refused"));
    }
  } finally {
    state.busy = false;
  }
}

async function startCase(file, message) {
  state.busy = true;
  const generation = state.generation;
  say(h("div", { class: "bubble-me" }, message, h("div", { class: "bubble-file" }, icon("clip"), file.name)));
  nodes.message.value = "";
  clearFile();
  try {
    const view = await api.uploadBoq(file, UPLOAD_MODE);
    reachable(true);
    if (generation !== state.generation) {
      return;
    }
    say(h("p", { class: "reply" }, `Case ${view.run_id} started. Each step shows below as it happens.`));
    await openCase(view.run_id, view);
  } catch (error) {
    if (generation !== state.generation) {
      return;
    }
    if ([413, 415, 422].includes(error.status)) {
      say(alertCard("bad", `${file.name} was not read`, error.detail,
        h("button", { type: "button", class: "btn-plain", onClick: () => nodes.file.click() }, "Choose another file")));
      return;
    }
    say(errorCard(error, `${file.name} was not read`));
  } finally {
    state.busy = false;
  }
}

async function startSample(name) {
  if (state.busy) {
    return; // one case per click
  }
  state.busy = true;
  const generation = state.generation;
  let file;
  try {
    file = await api.sampleFile(name);
  } catch (error) {
    if (generation === state.generation) {
      say(errorCard(error, "That sample is not available"));
    }
    return;
  } finally {
    state.busy = false;
  }
  if (generation === state.generation) {
    await startCase(file, DEFAULT_UPLOAD_TEXT);
  }
}

function clearFile() {
  state.file = null;
  nodes.file.value = "";
  nodes.attached.hidden = true;
  nodes.attached.replaceChildren();
}

function chooseFile(file) {
  state.file = file;
  fill(nodes.attached, icon("clip"), h("span", {}, file.name),
    h("button", { type: "button", class: "btn-ghost", onClick: clearFile }, "Remove"));
  nodes.attached.hidden = false;
  nodes.message.focus();
}

// ---- wiring --------------------------------------------------------------------------------------

function wire() {
  fill($("logo"), icon("shield"));
  fill(nodes.attach, icon("clip"));
  nodes.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    send(nodes.message.value);
  });
  nodes.attach.addEventListener("click", () => nodes.file.click());
  nodes.file.addEventListener("change", () => {
    if (nodes.file.files[0]) {
      chooseFile(nodes.file.files[0]);
    }
  });
  nodes.persona.addEventListener("change", () => startPersona(nodes.persona.value));
  $("new-chat").addEventListener("click", () => {
    closeCase();
    nodes.log.replaceChildren();
    nodes.rail.classList.remove("open");
    showSuggestions();
    nodes.message.focus();
  });
  nodes.menu.addEventListener("click", () => {
    const open = nodes.rail.classList.toggle("open");
    nodes.menu.setAttribute("aria-expanded", String(open));
  });
  nodes.panelClose.addEventListener("click", () => {
    nodes.panel.hidden = true;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      nodes.rail.classList.remove("open");
      if (matchMedia("(max-width: 1100px)").matches) {
        nodes.panel.hidden = true;
      }
    }
  });
  nodes.large.addEventListener("click", () => {
    const on = document.documentElement.classList.toggle("large");
    nodes.large.setAttribute("aria-pressed", String(on));
    save(localStorage, LARGE_KEY, on ? "on" : "off");
  });
  if (load(localStorage, LARGE_KEY) === "on") {
    document.documentElement.classList.add("large");
    nodes.large.setAttribute("aria-pressed", "true");
  }
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      refreshLists();
    }
  });
  state.listTimer = setInterval(() => {
    if (document.visibilityState === "visible" && state.me) {
      refreshLists();
    }
  }, LIST_POLL_MS);
}

async function boot() {
  wire();
  let people;
  try {
    people = await api.people();
  } catch (error) {
    say(error.status === 401
      ? alertCard("bad", "Sign-in is off", "This page needs RAG_AUTH=header (local use only).")
      : errorCard(error, "Could not load the people list"));
    return;
  }
  fill(nodes.persona, people.map((person) => h("option", { value: person.user_id }, person.name)));
  const asked = new URL(location.href).searchParams.get("as") || load(sessionStorage, PERSONA_KEY) || DEFAULT_PERSONA;
  const userId = people.some((person) => person.user_id === asked) ? asked : people[0].user_id;
  nodes.persona.value = userId;
  await startPersona(userId);

  const linked = new URLSearchParams(location.hash.slice(1)).get("case");
  if (linked && /^[A-Za-z0-9_-]{1,40}$/.test(linked)) {
    await openCase(linked);
  }
}

boot();
