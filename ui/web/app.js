(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const LS_ACTIVE = "movieidss.activeTab";
  const LS_API = "movieidss.apiUrl";
  const SS_OUT_PREFIX = "movieidss.out.";
  const SS_ROWS_PREFIX = "movieidss.rows.";

  const apiInput = $("#apiUrl");
  const healthBtn = $("#btnHealth");
  const healthStatus = $("#healthStatus");

  // init: restore API + active tab + cached outputs
  apiInput.value = localStorage.getItem(LS_API) || apiInput.value || "http://127.0.0.1:8000";
  apiInput.addEventListener("input", () => localStorage.setItem(LS_API, apiInput.value.trim()));

  const initTab = localStorage.getItem(LS_ACTIVE) || "forecasting";
  $$(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === initTab));
  $$(".panel").forEach(p => p.classList.toggle("active", p.id === initTab));

  // restore cached results per task
  ["forecasting","predictive","classification","clustering","nlp"].forEach(task => {
    try {
      const out = sessionStorage.getItem(SS_OUT_PREFIX + task);
      const rows = sessionStorage.getItem(SS_ROWS_PREFIX + task);
      if (out) setOut(task, JSON.parse(out));
      if (rows) renderTable(task, JSON.parse(rows));
    } catch {}
  });

  // tabs
  $$(".tab").forEach(btn =>
    btn.addEventListener("click", () => {
      $$(".tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const id = btn.dataset.tab;
      $$(".panel").forEach(p => p.classList.remove("active"));
      document.getElementById(id).classList.add("active");
      localStorage.setItem(LS_ACTIVE, id);
    })
  );

  // health badge
  function setBadge(text, ok) {
    healthStatus.textContent = text;
    healthStatus.style.background = ok
      ? "linear-gradient(180deg,#50fa7b,#8be9fd)"
      : "#ff5555";
    healthStatus.style.color = ok ? "#0b0e11" : "#f8f8f2";
  }

async function apiFetch(path, opts){
  const base = apiInput.value.trim();
  const url = base.replace(/\/$/, "") + path;

  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    throw new Error("NETWORK: " + (e?.message || e));
  }

  const text = await res.text();
  let json;
  try { json = text ? JSON.parse(text) : {}; }
  catch { throw new Error(`BAD_JSON ${res.status}: ${text.slice(0,200)}`); }

  if (!res.ok) {
    // прокинем detail из FastAPI, если есть
    const detail = json?.detail ? ` | detail: ${JSON.stringify(json.detail)}` : "";
    throw new Error(`${res.status} ${res.statusText}${detail}`);
  }
  return json;
}

  healthBtn.addEventListener("click", async () => {
    try {
      await apiFetch("/health");
      setBadge("ok", true);
    } catch (err) {
      console.error(err);
      setBadge("error", false);
    }
  });

  // helpers
  function setOut(task, data) {
    $("#out_" + task).textContent = JSON.stringify(data, null, 2);
    try { sessionStorage.setItem(SS_OUT_PREFIX + task, JSON.stringify(data)); } catch {}
  }

  function renderTable(task, rows) {
    const table = $("#table_" + task);
    if (!rows || !rows.length) {
      table.innerHTML = "";
      try { sessionStorage.removeItem(SS_ROWS_PREFIX + task); } catch {}
      return;
    }
    const cols = Object.keys(rows[0]);
    const thead =
      "<thead><tr>" +
      cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("") +
      "</tr></thead>";
    const tbody =
      "<tbody>" +
      rows
        .map(
          (r) =>
            "<tr>" +
            cols.map((c) => `<td>${escapeHtml(String(r[c]))}</td>`).join("") +
            "</tr>"
        )
        .join("") +
      "</tbody>";
    table.innerHTML = thead + tbody;
    try { sessionStorage.setItem(SS_ROWS_PREFIX + task, JSON.stringify(rows)); } catch {}
  }

  function escapeHtml(s) {
    return s.replace(/[&<>\"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
  }

  // wire actions
  $$(".panel").forEach((panel) => {
    panel.addEventListener("click", async (ev) => {
      const btn = ev.target.closest("button");
      if (!btn) return;
      const action = btn.dataset.action;
      const task = btn.dataset.task;
      if (!action || !task) return;

      try {
        if (action === "train" || action === "evaluate") {
          const json = await apiFetch(`/${action}/${task}`, { method: "POST" });
          setOut(task, json);
          renderTable(task, []);
          return;
        }

        if (action === "predict") {
          const payload = buildPayload(task);
          const json = await apiFetch(`/predict/${task}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ payload }),
          });
          setOut(task, json);
          const items = json?.result?.items || [];
          renderTable(task, items);
          return;
        }
      } catch (err) {
        console.error(err);
        setOut(task, { error: String(err) });
      }
    });
  });

  function buildPayload(task) {
    const p = {};
    if (task === "forecasting") {
      p.horizon = parseInt($("#f_horizon").value, 10) || 6;
    } else if (task === "predictive") {
      p.user_id = parseInt($("#p_user").value, 10) || 1;
      p.top_n = parseInt($("#p_topn").value, 10) || 10;
    } else if (task === "classification") {
      p.user_id = parseInt($("#c_user").value, 10) || 1;
      p.movie_ids = $("#c_movieids")
        .value.split(",")
        .map((s) => s.trim())
        .filter(Boolean)
        .map(Number);
    } else if (task === "clustering") {
      p.user_id = parseInt($("#cl_user").value, 10) || 1;
      p.top_n = parseInt($("#cl_topn").value, 10) || 10;
    } else if (task === "nlp") {
      const q = $("#n_query").value.trim();
      const mid = parseInt($("#n_movie").value, 10) || 0;
      p.top_n = parseInt($("#n_topn").value, 10) || 5;
      if (q) p.query = q;
      if (mid > 0) p.movie_id = mid;
    }
    return p;
  }
})();
