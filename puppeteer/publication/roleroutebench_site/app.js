const state = { data: null, packet: null };
const byId = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const percent = (part, total) => total ? `${Math.round(part * 100 / total)}%` : "—";

async function loadData() {
  const response = await fetch("benchmark.json");
  if (!response.ok) throw new Error(`data request failed (${response.status})`);
  return response.json();
}

function renderResults(runs) {
  byId("results-body").innerHTML = runs.map((run) => `
    <tr>
      <td><strong>${esc(run.domain_label)}</strong></td>
      <td>${run.packet_count}</td>
      <td>${run.trajectory_passed}/${run.packet_count} <span class="badge">${percent(run.trajectory_passed, run.packet_count)}</span></td>
      <td>${run.independent_end_task_passed}/${run.packet_count} <span class="badge">${percent(run.independent_end_task_passed, run.packet_count)}</span></td>
      <td>${run.primary_routing_decisions}</td>
      <td>${run.suppressed_by_budget}</td>
    </tr>`).join("");
}

function unique(values) { return [...new Set(values.filter(Boolean))].sort(); }
function fillSelect(select, values, label) {
  select.innerHTML = `<option value="">${label}</option>` + values.map((value) => `<option value="${esc(value)}">${esc(value)}</option>`).join("");
}

function availablePackets() {
  const domain = byId("domain-filter").value;
  return state.data.packets.filter((packet) => !domain || packet.domain === domain);
}

function updatePacketOptions(keepSelection = true) {
  const select = byId("packet-filter");
  const previous = keepSelection ? select.value : "";
  const packets = availablePackets();
  select.innerHTML = packets.map((packet) => `<option value="${esc(packet.packet_id)}">${esc(packet.packet_id)}</option>`).join("");
  if (packets.some((packet) => packet.packet_id === previous)) select.value = previous;
  renderPacket();
}

function renderPacket() {
  const id = byId("packet-filter").value;
  state.packet = state.data.packets.find((packet) => packet.packet_id === id) || availablePackets()[0];
  if (!state.packet) return;
  byId("packet-filter").value = state.packet.packet_id;
  const p = state.packet;
  byId("packet-summary").innerHTML = `
    <div class="packet-summary-head">
      <div><h3>${esc(p.packet_id)}</h3><div class="badges"><span class="badge">${esc(p.domain_label)}</span><span class="badge">${esc(p.executor_model)}</span><span class="badge">${p.primary_routing_decisions} primary decisions</span></div></div>
      <div class="badges"><span class="status ${p.trajectory_passed ? "success" : "failed"}">trajectory ${p.trajectory_passed ? "pass" : "fail"}</span><span class="status ${p.independent_end_task_passed ? "success" : "failed"}">trusted ${p.independent_end_task_passed ? "pass" : "fail"}</span></div>
    </div><p>${esc(p.task)}</p>`;
  renderSteps();
}

function renderSteps() {
  const role = byId("role-filter").value;
  const status = byId("status-filter").value;
  const rows = state.packet.steps.filter((step) => (!role || step.role === role) && (!status || step.event_status === status));
  byId("trajectory").innerHTML = rows.length ? rows.map((step) => `
    <article class="trajectory-step">
      <div class="step-number">#${String(step.index).padStart(2,"0")}</div>
      <div class="step-role"><strong>${esc(step.role)}</strong><small>budget before: ${esc(step.budget_before)} · confidence: ${step.confidence == null ? "—" : Number(step.confidence).toFixed(2)}</small><span class="event-pill">${esc(step.event_status)} · ${esc(step.semantic_delta)}</span></div>
      <div class="state-change"><small>state before → after</small><code>${esc(step.state_before || "not executed")}</code><code>→ ${esc(step.state_after || "not executed")}</code></div>
      <div class="choice-list">top choices<br>${step.top_choices.map((choice) => `${esc(choice.role)} ${(choice.probability * 100).toFixed(0)}%`).join(" · ") || "not recorded"}</div>
    </article>`).join("") : `<p class="error">No step matches the active role and status filters.</p>`;
}

async function init() {
  try {
    state.data = await loadData();
    renderResults(state.data.runs);
    const domains = state.data.runs.map((run) => run.domain);
    byId("domain-filter").innerHTML += domains.map((domain) => `<option value="${esc(domain)}">${esc(state.data.runs.find((run) => run.domain === domain).domain_label)}</option>`).join("");
    fillSelect(byId("role-filter"), unique(state.data.packets.flatMap((packet) => packet.steps.map((step) => step.role))), "All roles");
    fillSelect(byId("status-filter"), unique(state.data.packets.flatMap((packet) => packet.steps.map((step) => step.event_status))), "All statuses");
    updatePacketOptions(false);
    byId("domain-filter").addEventListener("change", () => updatePacketOptions(false));
    byId("packet-filter").addEventListener("change", renderPacket);
    byId("role-filter").addEventListener("change", renderSteps);
    byId("status-filter").addEventListener("change", renderSteps);
  } catch (error) {
    byId("results-body").innerHTML = `<tr><td colspan="6" class="error">Could not load explorer data: ${esc(error.message)}</td></tr>`;
    byId("trajectory").innerHTML = `<p class="error">Explorer data is unavailable.</p>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
