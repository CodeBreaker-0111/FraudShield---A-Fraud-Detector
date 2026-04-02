// ── Nav ──────────────────────────────────────────────────────────────
document.querySelectorAll(".navbtn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".navbtn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("page-" + btn.dataset.tab).classList.add("active");
  });
});

// ── Model status ─────────────────────────────────────────────────────
async function checkModel() {
  try {
    const res  = await fetch("/api/metrics");
    const data = await res.json();
    const dot  = document.querySelector(".pill-dot");
    const txt  = document.getElementById("pillText");
    if (data.error) {
      dot.className = "pill-dot error";
      txt.textContent = "model not ready";
    } else {
      dot.className = "pill-dot ready";
      txt.textContent = "model ready · auc " + data.roc_auc;
      const aucEl = document.getElementById("aucStat");
      if (aucEl) aucEl.textContent = data.roc_auc;
      loadModelTab(data);
    }
  } catch {
    document.querySelector(".pill-dot").className = "pill-dot error";
    document.getElementById("pillText").textContent = "server offline";
  }
}

// ── Analyze ──────────────────────────────────────────────────────────
async function analyze() {
  const btn = document.getElementById("runBtn");
  const lbl = document.getElementById("runBtnLabel");
  btn.disabled = true;
  lbl.textContent = "Analyzing...";

  const payload = {
    card_type:          document.getElementById("cardType").value,
    amount:             parseFloat(document.getElementById("amount").value),
    merchant_category:  document.getElementById("merchantCat").value,
    hour:               parseInt(document.getElementById("hour").value),
    distance_from_home: parseFloat(document.getElementById("distance").value),
    txn_per_hour:       parseInt(document.getElementById("txnPerHour").value),
    avg_monthly_spend:  parseFloat(document.getElementById("avgSpend").value),
    new_device:         parseInt(document.getElementById("newDevice").value),
    international:      parseInt(document.getElementById("international").value),
  };

  try {
    const res  = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    renderResult(data);
  } catch {
    alert("Cannot reach server. Is app.py running?");
  } finally {
    btn.disabled = false;
    lbl.textContent = "Run Analysis";
  }
}

function renderResult(data) {
  document.getElementById("resultEmpty").classList.add("hidden");
  const box = document.getElementById("resultBox");
  box.classList.remove("hidden");

  const pct   = data.fraud_probability;
  const level = data.risk_level;

  // Number color
  const numEl = document.getElementById("probNumber");
  numEl.textContent = pct;
  numEl.style.color = level === "high" ? "var(--red)" : level === "medium" ? "var(--amber)" : "var(--green)";

  // Tag
  const tag = document.getElementById("riskTag");
  tag.textContent = level === "high" ? "High Risk" : level === "medium" ? "Medium Risk" : "Low Risk";
  tag.className = "risk-tag " + level;

  // Bar
  const fill = document.getElementById("probFill");
  fill.style.width = pct + "%";
  fill.style.background = level === "high" ? "var(--red)" : level === "medium" ? "var(--amber)" : "var(--green)";

  // Verdict
  document.getElementById("verdict").textContent = data.verdict;

  // Triggers
  const tEl = document.getElementById("triggers");
  tEl.innerHTML = (data.risk_factors || []).map(f => `
    <div class="trigger-item">
      <span class="trigger-label">${f.label}</span>
      <span class="trigger-val ${f.level}">${f.value}</span>
    </div>
  `).join("");
}

// ── Dashboard charts ─────────────────────────────────────────────────
function initCharts() {
  const gridColor = "rgba(255,255,255,0.05)";
  const tickColor = "#6e6e76";

  // Category chart
  const catCtx = document.getElementById("catChart");
  if (catCtx) {
    new Chart(catCtx, {
      type: "bar",
      data: {
        labels: ["Online", "ATM", "Electronics", "Jewelry", "Travel", "Gaming"],
        datasets: [
          { label: "Legit", data: [420,180,150,60,110,45], backgroundColor: "rgba(255,255,255,0.08)", borderRadius: 3 },
          { label: "Fraud", data: [12,8,6,4,3,4], backgroundColor: "rgba(255,92,92,0.6)", borderRadius: 3 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: tickColor, font: { family: "'Syne'" }, boxWidth: 10, boxHeight: 10 } } },
        scales: {
          x: { ticks: { color: tickColor }, grid: { color: gridColor } },
          y: { ticks: { color: tickColor }, grid: { color: gridColor } },
        }
      }
    });
  }

  // Hour chart
  const hourCtx = document.getElementById("hourChart");
  if (hourCtx) {
    new Chart(hourCtx, {
      type: "line",
      data: {
        labels: Array.from({length:24}, (_,i) => i+":00"),
        datasets: [{
          label: "Fraud",
          data: [2,5,8,6,4,2,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,2,2,3],
          borderColor: "rgba(255,92,92,0.8)",
          backgroundColor: "rgba(255,92,92,0.07)",
          fill: true, tension: 0.4, pointRadius: 2, pointBackgroundColor: "rgba(255,92,92,0.8)",
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: tickColor, font: { family: "'Syne'" }, boxWidth: 10 } } },
        scales: {
          x: { ticks: { color: tickColor, maxTicksLimit: 8, maxRotation: 0 }, grid: { color: gridColor } },
          y: { ticks: { color: tickColor }, grid: { color: gridColor } },
        }
      }
    });
  }

  // Table
  const rows = [
    { id:"TXN-4821", card:"Credit", amount:"₹89,000", cat:"Jewelry",     hour:"02:00", score:94, status:"blocked" },
    { id:"TXN-4819", card:"Debit",  amount:"₹12,500", cat:"ATM",         hour:"03:30", score:81, status:"blocked" },
    { id:"TXN-4815", card:"Credit", amount:"₹35,000", cat:"Electronics", hour:"23:00", score:63, status:"review"  },
    { id:"TXN-4810", card:"Debit",  amount:"₹2,200",  cat:"Gaming",      hour:"01:15", score:57, status:"review"  },
    { id:"TXN-4807", card:"Credit", amount:"₹450",    cat:"Grocery",     hour:"10:00", score:4,  status:"ok"      },
    { id:"TXN-4803", card:"Debit",  amount:"₹1,800",  cat:"Restaurant",  hour:"13:00", score:6,  status:"ok"      },
  ];
  const tbody = document.getElementById("txnBody");
  if (tbody) {
    tbody.innerHTML = rows.map(r => {
      const sc = r.status === "blocked" ? "spill-blocked" : r.status === "review" ? "spill-review" : "spill-ok";
      const sl = r.status === "blocked" ? "Blocked" : r.status === "review" ? "Review" : "OK";
      const scoreColor = r.score >= 65 ? "var(--red)" : r.score >= 30 ? "var(--amber)" : "var(--green)";
      return `<tr>
        <td class="txn-id">${r.id}</td>
        <td>${r.card}</td>
        <td>${r.amount}</td>
        <td>${r.cat}</td>
        <td style="font-family:var(--mono);font-size:12px;">${r.hour}</td>
        <td style="font-family:var(--mono);font-size:12px;color:${scoreColor}">${r.score}%</td>
        <td><span class="spill ${sc}">${sl}</span></td>
      </tr>`;
    }).join("");
  }
}

// ── Model tab ────────────────────────────────────────────────────────
function loadModelTab(data) {
  // Feature importance bars
  if (data.feature_importance) {
    const max = data.feature_importance[0].importance;
    document.getElementById("featList").innerHTML =
      data.feature_importance.slice(0, 10).map(f => `
        <div class="feat-row">
          <span class="feat-name">${f.feature}</span>
          <div class="feat-track"><div class="feat-fill" style="width:${(f.importance/max*100).toFixed(1)}%"></div></div>
          <span class="feat-val">${(f.importance*100).toFixed(1)}%</span>
        </div>
      `).join("");
  }

  // Metrics
  const cm = data.confusion_matrix || [[0,0],[0,0]];
  document.getElementById("metricsList").innerHTML = [
    ["Dataset",      data.dataset],
    ["ROC-AUC",      data.roc_auc],
    ["Avg Precision",data.avg_precision],
    ["Test samples", (data.n_test||0).toLocaleString()],
    ["True Positives (caught)",  cm[1]?.[1] ?? "—"],
    ["False Positives (wrong)",  cm[0]?.[1] ?? "—"],
    ["False Negatives (missed)", cm[1]?.[0] ?? "—"],
  ].map(([k,v]) => `<div class="m-row"><span>${k}</span><span>${v}</span></div>`).join("");
}

// ── Init ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkModel();
  initCharts();
});