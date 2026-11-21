// One case in the chat, drawn from GET /workflows/{id} as the viewer may see it.
// The server decides who sees what; this file only words it. It never computes money.
//
//   case card (status + timeline) ─ quotes card ─ demo control ─ gate card ─ purchase orders ─ notice

import { h, icon, money, plural, status } from "./dom.js";

// ---- words ---------------------------------------------------------------------------------------

const STATUS_WORDS = {
  running: ["muted", "Running"],
  awaiting_rfq: ["wait", "Waiting: quote requests"],
  awaiting_quotes: ["wait", "Waiting: vendor replies"],
  awaiting_quote_decision: ["wait", "Waiting: quote decision"],
  awaiting_procurement: ["wait", "Waiting: procurement"],
  awaiting_finance: ["wait", "Waiting: finance"],
  done: ["ok", "Done"],
  stopped: ["bad", "Stopped"],
  rejected: ["bad", "Rejected"],
};

export function statusWords(code) {
  return STATUS_WORDS[code] || ["muted", String(code).replaceAll("_", " ")];
}

const STEP_STATUS = { done: ["ok", "Done"], waiting: ["wait", "Waiting"], stopped: ["bad", "Stopped"], rejected: ["bad", "Rejected"] };
const OUTCOME = { done: ["ok", "Done"], refused: ["bad", "Refused"], error: ["bad", "Error"], stopped: ["muted", "Stopped"] };

const GATE_NAMES = { rfq: "quote requests", quotes: "quote decision", procurement: "procurement", finance: "finance" };

const PROBLEM_WORDS = {
  wrong_sender: "sent from the wrong address",
  vendor_not_approved: "vendor not approved",
  not_requested: "not asked to quote",
  currency_not_aed: "not in AED",
  expired: "expired",
  unknown_line: "quotes a line we did not ask for",
  unreadable_number: "a number could not be read",
  quantity_differs: "quantity differs",
  amount_differs: "amount does not match quantity × rate",
  missing_line: "a line is missing",
  unit_differs: "unit differs",
  duplicate_line: "a line appears twice",
  unreadable_file: "file could not be read",
};
const REFUSED_PROBLEMS = new Set(["vendor_not_approved", "not_requested", "wrong_sender"]);

const PACKAGE_STATUS = {
  open: ["wait", "Waiting"], won: ["ok", "Winner picked"], short: ["warn", "Too few valid quotes"],
  dropped: ["muted", "Dropped"], tender: ["warn", "Open tender"], no_vendors: ["warn", "No approved vendor"], not_sent: ["muted", "Not sent"],
};

const DECIDED = { approve: "Approved", reject: "Rejected", extend: "Extended once", drop: "Dropped the short lines" };
const words = (value) => String(value).replaceAll("_", " ");
const list = (items) => items.join(" or ");

// Stage of a workflow step, for the badge.
function stageOf(name) {
  if (name.startsWith("boq_")) {
    return "Projects";
  }
  if (name === "finance" || name === "gate_finance") {
    return "Finance";
  }
  if (name === "execute") {
    return "Done";
  }
  if (["gate_rfq", "wait_quotes", "gate_quotes", "hold", "gate_procurement"].includes(name)) {
    return "Procurement";
  }
  return "Workflow";
}

// What one workflow step did, in words: [title, detail].
function stepWords(step) {
  const d = step.detail || {};
  const decided = d.approver ? `${DECIDED[d.decision] || words(d.decision)} by ${d.approver}` : "";
  switch (step.name) {
    case "boq_parse":
      return ["Read the BOQ", `${plural(d.lines ?? 0, "line")} in ${plural(d.sections ?? 0, "section")}.${d.flags ? ` ${d.flags} flagged.` : ""}`];
    case "boq_match":
      return ["Match to the catalogue", `${d.matched ?? 0} matched, ${d.low_score ?? 0} weak, ${d.unmatched ?? 0} with no match, ${d.held ?? 0} held.`];
    case "boq_price":
      return ["Price from the catalogue", `${plural(d.priced ?? 0, "line")} priced, AED ${money(d.total_aed)}. ${plural(d.rfq ?? 0, "line")} need quotes.`];
    case "gate_rfq":
      return step.status === "waiting" ? ["Approve quote requests", `Waiting on ${list(d.roles || ["procurement"])}.`] : ["Quote requests", `${decided}${d.requests ? `, ${plural(d.requests, "request")}` : ""}.`];
    case "wait_quotes":
      return step.status === "waiting"
        ? [`Vendor replies, round ${d.round ?? 1}`, "Waiting for replies or the deadline."]
        : [`Vendor replies, round ${d.round ?? 1}`, `${d.received ?? 0} received: ${d.valid ?? 0} valid, ${d.invalid ?? 0} invalid, ${d.flagged ?? 0} flagged.`];
    case "gate_quotes":
      return step.status === "waiting" ? ["Too few valid quotes", `Waiting on ${list(d.roles || ["procurement"])}.`] : ["Quote decision", `${decided}.`];
    case "hold":
      return ["Hold the budget", `AED ${money(d.total_aed)} held for ${plural(d.lines ?? 0, "line")}.`];
    case "gate_procurement":
      return step.status === "waiting" ? ["Procurement approval", `Waiting on ${list(d.roles || ["procurement"])}.`] : ["Procurement approval", `${decided}.`];
    case "finance":
      return ["Check budget and approver", `Approver: ${list(d.approver_roles || [])}.`];
    case "gate_finance":
      return step.status === "waiting" ? ["Finance approval", `Waiting on ${list(d.roles || ["finance"])}.`] : ["Finance approval", `${decided}.`];
    case "execute":
      return ["Write purchase orders", `${plural(d.written ?? 0, "order")} written, AED ${money(d.total_aed)}.`];
    default:
      return [words(step.name), Object.entries(d).map(([key, value]) => `${words(key)}: ${Array.isArray(value) ? value.join(", ") : value}`).join(" · ")];
  }
}

const TOOL_TITLES = {
  policy_rule: "Check the quotation rule",
  pick_vendors: "Pick approved vendors",
  draft_rfqs: "Draft quote requests",
  request_rfq_approval: "Ask procurement to approve sending",
  collect_quotes: "Collect vendor replies",
  compare_quotes: "Compare quotes",
  chase_vendor: "Remind a vendor",
  ask_procurement: "Ask procurement to decide",
  finalize_pricing: "Add the winning quotes",
  finish: "Finish",
};

function resultWords(result) {
  return Object.entries(result || {})
    .filter(([, value]) => typeof value !== "object")
    .map(([key, value]) => `${words(key)}: ${value === true ? "yes" : value === false ? "no" : value}`)
    .join(" · ");
}

// ---- timeline --------------------------------------------------------------------------------------

// Agent steps sit between workflow steps: each pause step follows the agent step that asked for it.
const PAUSE_TOOL = { gate_rfq: "request_rfq_approval", wait_quotes: "collect_quotes", gate_quotes: "ask_procurement", hold: "finalize_pricing" };

function timelineRows(view) {
  const rows = [];
  const agent = view.agent_steps || [];
  let next = 0;
  const flushTo = (tool) => {
    while (next < agent.length) {
      const step = agent[next];
      next += 1;
      rows.push({ key: `agent-${step.n}`, agent: step });
      if (tool && step.tool === tool) {
        return;
      }
    }
  };

  view.steps.forEach((step, index) => {
    if (!step.name.startsWith("boq_")) {
      flushTo(PAUSE_TOOL[step.name]);
    }
    rows.push({ key: `step-${index}-${step.name}`, step });
    // Lines held for review show right after matching: they are never priced.
    if (step.name === "boq_match") {
      for (const line of heldLines(view)) {
        rows.push({ key: `held-${line.item_no}`, held: line });
      }
    }
  });
  flushTo(null);
  return rows;
}

/** BOQ lines held because their text looks like an instruction. */
export function heldLines(view) {
  return (view.review || []).filter((line) => String(line.reason).startsWith("possible injection"));
}

function timelineRow(row, isNew) {
  const cls = `step${isNew ? " step-new" : ""}`;
  if (row.agent) {
    const step = row.agent;
    const [kind, word] = OUTCOME[step.status] || ["muted", words(step.status)];
    return h("li", { class: cls },
      h("span", { class: "stage stage-procurement" }, "Agent"),
      h("div", { class: "step-main" },
        h("span", { class: "step-title" }, TOOL_TITLES[step.tool] || words(step.tool), step.fallback ? " " : null, step.fallback ? h("span", { class: "tag" }, "fallback planner") : null),
        step.why ? h("span", { class: "muted small step-why" }, step.why) : null,
        resultWords(step.result) ? h("span", { class: "muted small" }, resultWords(step.result)) : null),
      status(kind, word));
  }
  if (row.held) {
    return h("li", { class: cls },
      h("span", { class: "stage" }, "Projects"),
      h("div", { class: "step-main" },
        h("span", { class: "step-title" }, `Hold line ${row.held.item_no}`),
        h("span", { class: "muted small" }, "The description reads like an instruction. Held for a person to review, not priced.")),
      status("warn", "Flagged"));
  }
  const step = row.step;
  const [title, detail] = stepWords(step);
  const stage = stageOf(step.name);
  const [kind, word] = STEP_STATUS[step.status] || ["muted", words(step.status)];
  return h("li", { class: cls },
    h("span", { class: `stage stage-${stage.toLowerCase()}` }, stage),
    h("div", { class: "step-main" }, h("span", { class: "step-title" }, title), detail ? h("span", { class: "muted small" }, detail) : null),
    status(kind, word));
}

// ---- package labels ---------------------------------------------------------------------------------

function packageLabel(view, packageId) {
  const pkg = (view.packages || []).find((item) => item.id === packageId);
  if (!pkg) {
    return words(packageId);
  }
  // A line waits on the RFQ list until its quote wins; then it is a priced line.
  const text = Object.fromEntries([
    ...(view.lines || []).map((line) => [line.item_no, line.boq_description || line.description]),
    ...(view.rfq || []).map((line) => [line.item_no, line.description]),
  ]);
  return pkg.items
    .map((item) => {
      const qty = `${pkg.quantities?.[item] ?? ""} ${pkg.units?.[item] ?? ""}`.trim();
      return text[item] ? `${item} ${text[item]} · ${qty}` : `${item} · ${qty}`;
    })
    .join("; ");
}

function waitingRoles(view) {
  const step = [...view.steps].reverse().find((item) => item.status === "waiting");
  return step?.detail?.roles || view.approver_roles || [];
}

// ---- cards --------------------------------------------------------------------------------------

function caseCard(view, seen, ctx) {
  const [kind, word] = statusWords(view.status);
  const rows = timelineRows(view);
  return h("section", { class: "card", "aria-label": "Case" },
    h("div", { class: "card-head" },
      h("div", { class: "card-head-text" },
        h("span", { class: "eyebrow" }, `Case · ${view.mode === "agent" ? "agent mode" : "classic"}`),
        h("span", { class: "card-title" }, `${view.entry === "boq" ? "BOQ" : "Request"} case ${view.run_id}`),
        h("span", { class: "muted small" }, `Started by ${view.requester}`)),
      h("span", { class: `pill pill-${kind}` }, status(kind, word)),
      h("button", { type: "button", class: "btn-ghost", onClick: ctx.showPanel }, "Case details")),
    h("ol", { class: "timeline", "aria-label": "Steps" }, rows.map((row) => timelineRow(row, !seen.has(row.key)))));
}

function quoteCheck(quote) {
  if (quote.counted === false) {
    return status("muted", "Replaced by a later quote");
  }
  const problems = quote.problems || [];
  if (problems.length) {
    const refused = problems.some((problem) => REFUSED_PROBLEMS.has(problem));
    return status("bad", `${refused ? "Refused" : "Invalid"} · ${problems.map((problem) => PROBLEM_WORDS[problem] || words(problem)).join(", ")}`);
  }
  if ((quote.flags || []).length) {
    return status("warn", `Valid · ${quote.flags.map(words).join(", ")} flagged`);
  }
  return status("ok", "Valid");
}

function quotesCard(view) {
  const byPackage = new Map();
  for (const quote of view.quotes) {
    if (!byPackage.has(quote.package)) {
      byPackage.set(quote.package, []);
    }
    byPackage.get(quote.package).push(quote);
  }
  const hidden = view.quotes.some((quote) => quote.total === undefined);
  const rows = [];
  for (const [packageId, quotes] of byPackage) {
    rows.push(h("tr", { class: "group-row" }, h("td", { colspan: "3" }, packageLabel(view, packageId))));
    for (const quote of quotes) {
      rows.push(h("tr", {},
        h("td", {}, quote.vendor, quote.round > 1 ? h("span", { class: "muted small" }, ` · round ${quote.round}`) : null),
        h("td", { class: "num" }, money(quote.total)),
        h("td", {}, quoteCheck(quote))));
    }
  }
  return h("section", { class: "card", "aria-label": "Quotes received" },
    h("div", { class: "card-head" },
      h("span", { class: "card-title" }, `${plural(view.quotes.length, "reply", "replies")} checked`),
      h("span", { class: "muted small" }, "Prices are read from the quote files. No model sets a price.")),
    h("div", { class: "table-wrap" },
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Vendor"), h("th", { class: "num" }, "Total AED"), h("th", {}, "Check"))),
        h("tbody", {}, rows))),
    hidden ? h("div", { class: "card-note muted" }, "Quote prices show to procurement, finance and executives only.") : null,
    comparisonTable(view));
}

export function comparisonTable(view) {
  const rows = (view.comparison || []).filter((row) => row.status === "won" || row.status === "short");
  if (!rows.length) {
    return null;
  }
  return h("div", { class: "table-wrap" },
    h("table", { "aria-label": "Lowest valid quote per line" },
      h("thead", {}, h("tr", {}, h("th", {}, "Line"), h("th", {}, "Lowest valid quote"), h("th", { class: "num" }, "AED"), h("th", {}, "Valid quotes"))),
      h("tbody", {}, rows.map((row) => {
        const [kind, word] = PACKAGE_STATUS[row.status] || ["muted", words(row.status)];
        return h("tr", {},
          h("td", {}, packageLabel(view, row.package)),
          h("td", {}, row.vendor || status(kind, word)),
          h("td", { class: "num" }, money(row.total)),
          h("td", {}, `${row.valid_quotes} of ${row.needed}`));
      }))));
}

function demoControl(view, ctx) {
  const round = [...view.steps].reverse().find((step) => step.name === "wait_quotes")?.detail?.round ?? 1;
  return h("div", { class: "demo-control", role: "group", "aria-label": "Demo control" },
    h("span", { class: "eyebrow" }, "Demo control"),
    h("button", { type: "button", class: "btn-plain", onClick: (event) => ctx.deliver(view.run_id, "quotes_delivered", round > 1 ? "round2" : "round1", event.currentTarget) }, "Simulate vendor replies"),
    h("button", { type: "button", class: "btn-plain", onClick: (event) => ctx.deliver(view.run_id, "deadline_passed", null, event.currentTarget) }, "Deadline passed"),
    h("span", { class: "muted small" }, "Stands in for email and a scheduler."));
}

const BUTTONS = {
  rfq: { approve: ["btn-ok", "Approve sending"], reject: ["btn-bad", "Reject"] },
  quotes: { extend: ["btn-wait", "Extend once"], drop: ["btn-plain", "Drop the short lines"] },
};

function gateButtons(view, ctx, choices) {
  const gate = view.pending_gate;
  const reasonId = `why-${view.run_id}`;
  const buttons = choices.map((choice) => {
    const [cls, label] = BUTTONS[gate]?.[choice] || (choice === "approve" ? ["btn-ok", gate === "finance" ? `Approve AED ${money(view.total)}` : "Approve"] : ["btn-bad", words(choice)[0].toUpperCase() + words(choice).slice(1)]);
    return h("button", {
      type: "button", class: cls, disabled: !view.can_decide, "aria-describedby": view.can_decide ? null : reasonId,
      onClick: (event) => ctx.decide(view.run_id, gate, choice, event.currentTarget),
    }, label);
  });
  const note = view.can_decide
    ? h("span", { class: "muted small" }, `You may decide: ${list(waitingRoles(view))}, not the requester.`)
    : h("span", { class: "why-not", id: reasonId }, icon("info"), view.decide_problem || "You can't decide this step.");
  return h("div", { class: "card-actions" }, buttons, note);
}

function policyNote(view) {
  const rule = (view.agent_steps || []).find((step) => step.tool === "policy_rule")?.result?.rule;
  const cited = view.policy?.visible ? `${view.policy.title} › ${view.policy.section}` : "";
  if (!rule && !cited) {
    return null;
  }
  return h("span", {}, rule ? h("b", {}, `Rule: ${rule}. `) : null, cited ? h("span", { class: "muted" }, cited) : null);
}

function gateCard(view, ctx) {
  const gate = view.pending_gate;
  const roles = waitingRoles(view);
  const eyebrow = view.can_decide ? `Your decision · ${GATE_NAMES[gate] || gate}` : `Waiting on ${list(roles)} · ${GATE_NAMES[gate] || gate}`;
  const head = (title) => h("div", { class: "card-head" },
    h("div", { class: "card-head-text" },
      h("span", { class: "eyebrow eyebrow-wait" }, eyebrow),
      h("span", { class: "card-title" }, title),
      h("span", { class: "muted small" }, `${view.entry === "boq" ? "BOQ" : "Request"} case ${view.run_id} · started by ${view.requester}`)));

  if (gate === "rfq") {
    const rfq = Object.fromEntries((view.rfq || []).map((line) => [line.item_no, line]));
    const count = (view.packages || []).reduce((sum, pkg) => sum + pkg.vendors.length, 0);
    return h("section", { class: "card card-wait", "aria-label": "Approval" },
      head(`Send ${plural(count, "request")} for quotation`),
      h("div", { class: "table-wrap" },
        h("table", {},
          h("thead", {}, h("tr", {}, h("th", {}, "Line"), h("th", { class: "num" }, "Qty"), h("th", {}, "Approved vendors asked"))),
          h("tbody", {}, (view.packages || []).flatMap((pkg) => pkg.items.map((item) => h("tr", {},
            h("td", {}, h("b", {}, item), " ", rfq[item]?.description || ""),
            h("td", { class: "num" }, `${pkg.quantities?.[item] ?? ""} ${pkg.units?.[item] ?? ""}`),
            h("td", {}, h("ul", { class: "vendor-list" }, pkg.vendors.map((id) => h("li", {}, view.vendor_names?.[id] || id)))))))))),
      h("div", { class: "card-note" },
        policyNote(view),
        h("span", { class: "muted" }, "Vendors come from the vendor master; the agent cannot add one. Requests go to the outbox, once per vendor.")),
      gateButtons(view, ctx, ["approve", "reject"]));
  }

  if (gate === "quotes") {
    const short = (view.comparison || []).filter((row) => row.status === "short");
    return h("section", { class: "card card-wait", "aria-label": "Decision needed" },
      head(short.length === 1 ? `${short[0].valid_quotes} of ${short[0].needed} valid quotes. The policy needs ${short[0].needed}.` : `${plural(short.length, "line")} with too few valid quotes`),
      h("div", { class: "card-body" },
        short.map((row) => h("span", {}, h("b", {}, packageLabel(view, row.package)), `: ${row.valid_quotes} of ${row.needed} valid.`)),
        h("span", { class: "muted small" }, "The agent will not go on with fewer. Extend once to ask these vendors again, or take the short lines out of this order.")),
      gateButtons(view, ctx, ["extend", "drop"]));
  }

  const note = view.summaries?.[gate];
  return h("section", { class: "card card-wait", "aria-label": "Approval" },
    head(`Approve AED ${money(view.total)}`),
    h("div", { class: "card-body" },
      h("dl", { class: "facts" },
        h("div", {}, h("dt", {}, "Total AED"), h("dd", {}, money(view.total))),
        h("div", {}, h("dt", {}, "Budget left after hold"), h("dd", {}, money(view.budget_after))),
        h("div", {}, h("dt", {}, "Final approver"), h("dd", { class: "word" }, list(view.approver_roles || []) || "—")),
        h("div", {}, h("dt", {}, "Lines to quote"), h("dd", { class: "word" }, String((view.rfq || []).length)))),
      comparisonTable(view),
      heldLines(view).length ? h("div", { class: "notice notice-warn" }, icon("warn"), `${plural(heldLines(view).length, "line")} held and not in this total: the text reads like an instruction.`) : null,
      note ? h("div", { class: "small" }, note.split("\n").map((part) => h("p", { class: "reply" }, part))) : null,
      policyNote(view)),
    gateButtons(view, ctx, ["approve", "reject"]));
}

function ordersCard(view) {
  if (!view.orders.length) {
    // Classic runs write one PO for the run; private agent runs show no order details.
    const text = view.mode === "agent" ? "Purchase orders are written. Details are hidden for private files." : `One purchase order for AED ${money(view.total)} is written to the outbox.`;
    return h("div", { class: "alert alert-ok", role: "status" }, icon("check"),
      h("div", { class: "alert-body" }, h("span", { class: "alert-title" }, "Done"), h("span", {}, text)));
  }
  return h("section", { class: "card", "aria-label": "Purchase orders" },
    h("div", { class: "card-head" },
      h("div", { class: "card-head-text" },
        h("span", { class: "eyebrow eyebrow-ok" }, "Done"),
        h("span", { class: "card-title" }, `${plural(view.orders.length, "purchase order")}, one per vendor`),
        h("span", { class: "muted small" }, "Written to the outbox once each. Running the case again writes nothing new."))),
    h("div", { class: "table-wrap" },
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Vendor"), h("th", { class: "num" }, "Lines"), h("th", { class: "num" }, "AED"))),
        h("tbody", {},
          view.orders.map((order) => h("tr", {}, h("td", {}, order.vendor), h("td", { class: "num" }, String(order.lines)), h("td", { class: "num" }, money(order.total)))),
          h("tr", { class: "total-row" }, h("td", { colspan: "2" }, "Total"), h("td", { class: "num" }, money(view.total)))))),
    heldLines(view).length ? h("div", { class: "card-note" }, h("span", { class: "status status-warn" }, icon("warn"), `${plural(heldLines(view).length, "line")} stays held. A person must review it.`)) : null);
}

/**
 * Every card for one case, in order.
 * ctx: { me, decide(runId, gate, choice, button), deliver(runId, event, sample, button), showPanel() }
 * Buttons send the run they were drawn for, never "whatever is open now".
 * seen: keys of timeline rows already shown, so only new rows animate in.
 */
export function caseCards(view, seen, ctx) {
  const cards = [caseCard(view, seen, ctx)];
  if ((view.quotes || []).length) {
    cards.push(quotesCard(view));
  }
  if (ctx.me?.demo && view.status === "awaiting_quotes") {
    cards.push(demoControl(view, ctx));
  }
  if (view.pending_gate) {
    cards.push(gateCard(view, ctx));
  }
  if (view.status === "done") {
    cards.push(ordersCard(view));
  }
  if (view.status === "stopped" || view.status === "rejected") {
    cards.push(h("div", { class: "alert alert-bad", role: "status" }, icon("cross"),
      h("div", { class: "alert-body" }, h("span", { class: "alert-title" }, view.status === "stopped" ? "Stopped" : "Rejected"), view.reason ? h("span", {}, view.reason) : null)));
  }
  for (const row of timelineRows(view)) {
    seen.add(row.key);
  }
  return cards;
}

export { packageLabel, waitingRoles, stageOf };
