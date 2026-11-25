// The case panel: Overview, Lines, Quotes, Policy and Activity tabs for the open case.

import { comparisonTable, heldLines, stageOf, statusWords, waitingRoles } from "./case.js";
import { fill, h, icon, money, plural, status } from "./dom.js";

const STAGES = ["Projects", "Procurement", "Finance", "Done"];

// Which stage a status is in, as an index into STAGES.
const STAGE_NOW = {
  running: 0, awaiting_rfq: 1, awaiting_quotes: 1, awaiting_quote_decision: 1, awaiting_procurement: 1,
  awaiting_finance: 2, done: 4,
};

function stepper(view) {
  let now = STAGE_NOW[view.status];
  let stopped = -1;
  if (now === undefined) {
    stopped = Math.max(0, STAGES.indexOf(stageOf(view.stopped_at || "")));
    now = stopped;
  }
  return h("ol", { class: "stepper", "aria-label": "Progress" }, STAGES.map((name, index) => {
    let cls = "";
    let label = name;
    if (index === stopped) {
      cls = "stopped";
      label = `${name} ✗`;
    } else if (index < now || view.status === "done") {
      cls = "done";
      label = `${name} ✓`;
    } else if (index === now) {
      cls = "now";
    }
    return h("li", { class: cls, "aria-current": index === now && cls === "now" ? "step" : null }, h("span", { class: "stepper-bar" }), h("span", {}, label));
  }));
}

function overview(view) {
  const held = heldLines(view);
  const [kind, word] = statusWords(view.status);
  return [
    stepper(view),
    h("div", {}, status(kind, word)),
    h("dl", { class: "facts" },
      h("div", {}, h("dt", {}, view.status === "done" ? "Spent" : "Total AED"), h("dd", {}, money(view.total))),
      h("div", {}, h("dt", {}, "Budget left after this"), h("dd", {}, money(view.budget_after))),
      h("div", {}, h("dt", {}, "Lines to quote"), h("dd", { class: "word" }, String((view.rfq || []).length))),
      h("div", {}, h("dt", {}, view.pending_gate ? "Waiting on" : "Final approver"), h("dd", { class: "word" }, (view.pending_gate ? waitingRoles(view) : view.approver_roles || []).join(" or ") || "—"))),
    held.length
      ? h("div", { class: "notice notice-warn" }, icon("warn"),
        h("span", {}, h("b", {}, `${plural(held.length, "line")} held for review. `), `Line ${held.map((line) => line.item_no).join(", ")}: a BOQ line is text, never an instruction.`))
      : null,
    // Private files come with no line, RFQ or review rows; a case with nothing priced yet still has some.
    !view.lines.length && !(view.rfq || []).length && !(view.review || []).length && view.boq_stats?.lines
      ? h("div", { class: "notice" }, icon("info"), `Private file: counts only. ${view.boq_stats.lines} lines, ${view.boq_stats.priced ?? 0} priced.`)
      : null,
    h("div", { class: "panel-foot" }, h("span", {}, `Model cost $${Number(view.cost_usd).toFixed(4)} · ${view.tokens} tokens`), h("span", {}, `Case ${view.run_id}`)),
  ];
}

function lines(view) {
  if (!view.lines.length && !(view.rfq || []).length) {
    return [h("p", { class: "empty" }, "No line details to show.")];
  }
  const qty = (line) => `${line.quantity ?? ""} ${line.unit ?? ""}`.trim();
  const listOf = (title, rows) => rows.length
    ? h("section", {}, h("h3", { class: "eyebrow" }, title),
      h("ul", { class: "activity" }, rows.map((line) => h("li", {}, h("span", {}, h("b", {}, line.item_no), " ", line.description), h("span", { class: "muted" }, `${qty(line)} · ${line.reason}`)))))
    : null;
  return [
    h("div", { class: "table-wrap" },
      h("table", { "aria-label": "Priced lines" },
        h("thead", {}, h("tr", {}, h("th", {}, "Line"), h("th", { class: "num" }, "Qty"), h("th", { class: "num" }, "AED"))),
        h("tbody", {}, view.lines.map((line) => h("tr", {},
          h("td", {}, h("b", {}, line.item_no || ""), " ", line.boq_description || line.description, h("div", { class: "muted tiny" }, line.vendor)),
          h("td", { class: "num" }, qty(line)),
          h("td", { class: "num" }, money(line.total))))))),
    listOf("Need quotes", view.rfq || []),
    listOf("Held for review", heldLines(view)),
  ];
}

function quotes(view) {
  const table = comparisonTable(view);
  return [
    h("p", { class: "small" }, `${plural((view.quotes || []).length, "reply", "replies")} so far.`),
    table || h("p", { class: "empty" }, "No comparison yet."),
  ];
}

function policy(view) {
  if (view.policy?.visible) {
    return [
      h("span", { class: "card-title" }, `${view.policy.title} › ${view.policy.section}`),
      h("blockquote", { class: "quote-text" }, view.policy.text),
    ];
  }
  return [h("div", { class: "notice" }, icon("info"), "A policy rule applies to this case. Its text shows only to people whose roles may read the policy.")];
}

const EVENT_WORDS = {
  decision: (event) => [`${event.by} chose ${event.decision}`, `at the ${String(event.gate).replaceAll("_", " ")} step`],
  quotes_delivered: (event) => [`${plural(event.files ?? 0, "reply", "replies")} delivered`, `round ${event.round} · by ${event.by}`],
  deadline_passed: (event) => ["Reply deadline passed", `round ${event.round ?? 1} · by ${event.by}`],
};

function activity(view) {
  const events = view.events || [];
  if (!events.length) {
    return [h("p", { class: "empty" }, "No decisions yet.")];
  }
  return [h("ol", { class: "activity" }, events.map((event) => {
    const [title, detail] = (EVENT_WORDS[event.type] || ((item) => [String(item.type).replaceAll("_", " "), item.by || ""]))(event);
    return h("li", {}, h("span", { class: "step-title" }, title), h("span", { class: "muted" }, detail));
  }))];
}

const TABS = [
  ["overview", "Overview", overview],
  ["lines", "Lines", lines],
  ["quotes", "Quotes", quotes],
  ["policy", "Policy", policy],
  ["activity", "Activity", activity],
];

/** Draw the panel for a case. Keeps the chosen tab; arrow keys move between tabs. */
export function renderPanel(nodes, view, chosen, onChoose) {
  const tabs = TABS.filter(([id]) => id !== "quotes" || view.mode === "agent");
  const current = tabs.some(([id]) => id === chosen) ? chosen : "overview";
  fill(nodes.title, `${view.entry === "boq" ? "BOQ" : "Request"} case ${view.run_id}`);
  fill(nodes.tabs, tabs.map(([id, label], index) => h("button", {
    type: "button", role: "tab", class: "tab", id: `tab-${id}`, "aria-selected": String(id === current), "aria-controls": "tab-panel",
    tabindex: id === current ? "0" : "-1",
    onClick: () => onChoose(id),
    onKeydown: (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) {
        return;
      }
      const next = tabs[(index + step + tabs.length) % tabs.length][0];
      onChoose(next);
      document.getElementById(`tab-${next}`)?.focus();
    },
  }, label)));
  nodes.body.setAttribute("aria-labelledby", `tab-${current}`);
  fill(nodes.body, tabs.find(([id]) => id === current)[2](view));
  return current;
}
