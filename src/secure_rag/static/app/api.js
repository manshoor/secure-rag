// The page's only way to the server. Every call names the person acting (local header sign-in).

export const UNREACHABLE = 0; // no HTTP status: the server could not be reached

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

let actingAs = "";

export function actAs(userId) {
  actingAs = userId;
}

async function call(method, path, { json, form } = {}) {
  const headers = { "X-User": actingAs };
  let body;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  }
  if (form !== undefined) {
    body = form;
  }

  let response;
  try {
    response = await fetch(path, { method, headers, body, credentials: "same-origin" });
  } catch {
    throw new ApiError(UNREACHABLE, "Can't reach the server.");
  }
  if (response.ok) {
    return response.json();
  }
  throw new ApiError(response.status, await errorText(response));
}

// FastAPI errors: {"detail": "text"} or {"detail": [{"msg": ...}, ...]} for a refused body.
async function errorText(response) {
  try {
    const { detail } = await response.json();
    if (Array.isArray(detail)) {
      return detail.map((item) => item.msg).join("; ");
    }
    return String(detail || response.statusText);
  } catch {
    return response.statusText;
  }
}

export const me = () => call("GET", "/me");
export const people = () => call("GET", "/ui/users");
export const chat = (message, runId) => call("POST", "/chat", { json: runId ? { message, run_id: runId } : { message } });
export const run = (runId) => call("GET", `/workflows/${encodeURIComponent(runId)}`);
export const runs = (scope) => call("GET", `/workflows?scope=${encodeURIComponent(scope)}`);
export const decide = (runId, decision, gate) => call("POST", `/workflows/${encodeURIComponent(runId)}/decision`, { json: { decision, gate } });
export const deliver = (runId, event, sample) =>
  call("POST", `/workflows/${encodeURIComponent(runId)}/events`, { json: sample ? { event, sample } : { event } });

export function uploadBoq(file, mode) {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  return call("POST", "/workflows/boq", { form });
}

/** One of the invented sample BOQs, as a File the upload can send. */
export async function sampleFile(name) {
  let response;
  try {
    response = await fetch(`/ui/samples/${encodeURIComponent(name)}`, { credentials: "same-origin" });
  } catch {
    throw new ApiError(UNREACHABLE, "Can't reach the server.");
  }
  if (!response.ok) {
    throw new ApiError(response.status, "That sample is not available.");
  }
  return new File([await response.blob()], name);
}
