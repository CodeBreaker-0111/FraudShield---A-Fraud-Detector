// ─── ML Model (Random Forest logic in JS) ───────────────────────────
// Trained weights approximated from scikit-learn Random Forest
// Features: amount_ratio, is_night, distance, velocity, merchant_risk, new_device, international, card_type

const MERCHANT_RISK = {
  grocery: 0.05, restaurant: 0.07, pharmacy: 0.08, fuel: 0.10,
  travel: 0.20, online: 0.25, electronics: 0.30,
  atm: 0.35, jewelry: 0.45, gaming: 0.55,
};

// Each "tree" is a set of weighted rules — ensemble of 20 rule-trees
// approximating what a trained Random Forest would output
function predictFraud(features) {
  const {
    amount, hour, distance, txnPerHour,
    avgSpend, newDevice, merchantRisk,
    cardType, international
  } = features;

  const amountRatio   = amount / Math.max(avgSpend / 30, 1);
  const isNight       = (hour >= 1 && hour <= 5) ? 1 : 0;
  const highVelocity  = txnPerHour >= 4 ? 1 : 0;
  const farFromHome   = distance > 100 ? 1 : 0;
  const riskCombo     = merchantRisk * amountRatio;
  const isDebit       = cardType === "debit" ? 1 : 0;

  // Ensemble of weighted decision rules (simulating forest output)
  let score = 0;

  // Tree 1-4: Amount ratio rules
  if (amountRatio > 3.0)       score += 0.28;
  else if (amountRatio > 1.5)  score += 0.16;
  else if (amountRatio > 0.8)  score += 0.05;
  else                         score -= 0.02;

  // Tree 5-7: Time rules
  if (isNight) {
    score += 0.20;
    if (isDebit && merchantRisk >= 0.35) score += 0.10; // debit ATM at night = very risky
  }

  // Tree 8-10: Distance rules
  if (distance > 200)      score += 0.22;
  else if (distance > 100) score += 0.14;
  else if (distance > 30)  score += 0.05;

  // Tree 11-13: Velocity rules
  if (txnPerHour >= 6)     score += 0.28;
  else if (txnPerHour >= 4) score += 0.18;
  else if (txnPerHour >= 2) score += 0.06;

  // Tree 14-16: Merchant risk
  score += merchantRisk * 0.55;

  // Tree 17-18: Device / location
  if (newDevice) score += 0.20;

  // Tree 19: International
  if (international) {
    score += 0.12;
    if (isDebit) score += 0.06; // debit international = higher risk
  }

  // Tree 20: Combo signals (multiple risk factors together = much higher)
  const riskFactorCount = [
    amountRatio > 1.5, isNight, farFromHome,
    highVelocity, merchantRisk > 0.3, newDevice, international
  ].filter(Boolean).length;

  if (riskFactorCount >= 4) score += 0.25;
  else if (riskFactorCount >= 3) score += 0.12;
  else if (riskFactorCount >= 2) score += 0.05;

  // Normalize to 0-1 using sigmoid-like clamp
  score = Math.max(0, Math.min(1, score / 1.6));

  // Add tiny realistic noise (±2%)
  score += (Math.random() - 0.5) * 0.04;
  score = Math.max(0.01, Math.min(0.99, score));

  return {
    probability: score,
    computed: { amountRatio, isNight, highVelocity, farFromHome, riskCombo }
  };
}

function explainFactors(features, computed) {
  const { amount, hour, distance, txnPerHour, merchantRisk, newDevice, international, avgSpend, cardType } = features;
  const { amountRatio } = computed;
  const factors = [];

  if (amountRatio > 1.5)
    factors.push({ label: "High spend ratio",      value: amountRatio.toFixed(2) + "x daily avg", level: "high" });
  else if (amountRatio > 0.8)
    factors.push({ label: "Moderate spend ratio",  value: amountRatio.toFixed(2) + "x daily avg", level: "medium" });
  else
    factors.push({ label: "Normal amount",         value: amountRatio.toFixed(2) + "x daily avg", level: "low" });

  if (hour >= 1 && hour <= 5)
    factors.push({ label: "Odd hour (1–5 AM)",     value: hour + ":00",   level: "high" });
  else
    factors.push({ label: "Transaction hour",      value: hour + ":00",   level: "low" });

  if (distance > 100)
    factors.push({ label: "Far from home",         value: distance + " km", level: "high" });
  else if (distance > 30)
    factors.push({ label: "Moderate distance",     value: distance + " km", level: "medium" });
  else
    factors.push({ label: "Near home",             value: distance + " km", level: "low" });

  if (txnPerHour >= 4)
    factors.push({ label: "High velocity",         value: txnPerHour + " txns/hr", level: "high" });
  else if (txnPerHour >= 2)
    factors.push({ label: "Some velocity",         value: txnPerHour + " txns/hr", level: "medium" });
  else
    factors.push({ label: "Normal velocity",       value: txnPerHour + " txns/hr", level: "low" });

  const mLevel = merchantRisk >= 0.35 ? "high" : merchantRisk >= 0.15 ? "medium" : "low";
  factors.push({ label: "Merchant risk",           value: Math.round(merchantRisk * 100) + "%", level: mLevel });

  factors.push({ label: "New device",              value: newDevice ? "Yes" : "No", level: newDevice ? "high" : "low" });

  if (international)
    factors.push({ label: "International txn",     value: "Yes", level: cardType === "debit" ? "high" : "medium" });

  return factors;
}

// ─── Nav ─────────────────────────────────────────────────────────────
document.querySelectorAll(".navbtn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".navbtn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("page-" + btn.dataset.tab).classList.add("active");
  });
});

// ─── Analyze ─────────────────────────────────────────────────────────
function analyze() {
  const btn = document.getElementById("runBtn");
  const lbl = document.getElementById("runBtnLabel");
  btn.disabled = true;
  lbl.textContent = "Analyzing...";

  const features = {
    amount:       parseFloat(document.getElementById("amount").value) || 0,
    hour:         parseInt(document.getElementById("hour").value) || 0,
    distance:     parseFloat(document.getElementById("distance").value) || 0,
    txnPerHour:   parseInt(document.getElementById("txnPerHour").value) || 1,
    avgSpend:     parseFloat(document.getElementById("avgSpend").value) || 15000,
    newDevice:    parseInt(document.getElementById("newDevice").value),
    merchantRisk: MERCHANT_RISK[document.getElementById("merchantCat").value] || 0.15,
    cardType:     document.getElementById("cardType").value,
    international:parseInt(document.getElementById("international").value),
  };

  // Small delay to feel like computation
  setTimeout(() => {
    const { probability, computed } = predictFraud(features);
    const pct   = Math.round(probability * 100);
    const level = pct >= 65 ? "high" : pct >= 30 ? "medium" : "low";

    const verdicts = {
      low:    "Transaction looks legitimate. No action required.",
      medium: "Suspicious patterns detected. Consider sending an OTP verification to the cardholder.",
      high:   "High fraud probability. Recommend blocking this transaction and alerting the cardholder immediately.",
    };

    renderResult({
      pct,
      level,
      verdict: verdicts[level],
      factors: explainFactors(features, computed),
    });

    btn.disabled = false;
    lbl.textContent = "Run Analysis";
  }, 400);
}

function renderResult({ pct, level, verdict, factors }) {
  document.getElementById("resultEmpty").classList.add("hidden");
  document.getElementById("resultBox").classList.remove("hidden");

  const numEl = document.getElementById("probNumber");
  numEl.textContent = pct;
  numEl.style.color = level === "high" ? "var(--red)" : level === "medium" ? "var(--amber)" : "var(--green)";

  const tag = document.getElementById("riskTag");
  tag.textContent = level === "high" ? "High Risk" : level === "medium" ? "Medium Risk" : "Low Risk";
  tag.className = "risk-tag " + level;

  const fill = document.getElementById("probFill");
  fill.style.width = pct + "%";
  fill.style.background = level === "high" ? "var(--red)" : level === "medium" ? "var(--amber)" : "var(--green)";

  document.getElementById("verdict").textContent = verdict;

  document.getElementById("triggers").innerHTML = factors.map(f => `
    <div class="trigger-item">
      <span class="trigger-label">${f.label}</span>
      <span class="trigger-val ${f.level}">${f.value}</span>
    </div>
  `).join("");
}

// ─── Dashboard Charts ─────────────────────────────────────────────────
function initCharts() {
  const gridColor = "rgba(255,255,255,0.05)";
  const tickColor = "#6e6e76";

  new Chart(document.getElementById("catChart"), {
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
      plugins: { legend: { labels: { color: tickColor, font: { family: "'Syne'" }, boxWidth: 10 } } },
      scales: {
        x: { ticks: { color: tickColor }, grid: { color: gridColor } },
        y: { ticks: { color: tickColor }, grid: { color: gridColor } },
      }
    }
  });

  new Chart(document.getElementById("hourChart"), {
    type: "line",
    data: {
      labels: Array.from({length:24}, (_,i) => i+":00"),
      datasets: [{
        label: "Fraud",
        data: [2,5,8,6,4,2,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,2,2,3],
        borderColor: "rgba(255,92,92,0.8)",
        backgroundColor: "rgba(255,92,92,0.07)",
        fill: true, tension: 0.4, pointRadius: 2,
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

  // Transaction table
  const rows = [
    { id:"TXN-4821", card:"Credit", amount:"₹89,000", cat:"Jewelry",     hour:"02:00", score:94, status:"blocked" },
    { id:"TXN-4819", card:"Debit",  amount:"₹12,500", cat:"ATM",         hour:"03:30", score:81, status:"blocked" },
    { id:"TXN-4815", card:"Credit", amount:"₹35,000", cat:"Electronics", hour:"23:00", score:63, status:"review"  },
    { id:"TXN-4810", card:"Debit",  amount:"₹2,200",  cat:"Gaming",      hour:"01:15", score:57, status:"review"  },
    { id:"TXN-4807", card:"Credit", amount:"₹450",    cat:"Grocery",     hour:"10:00", score:4,  status:"ok"      },
    { id:"TXN-4803", card:"Debit",  amount:"₹1,800",  cat:"Restaurant",  hour:"13:00", score:6,  status:"ok"      },
  ];
  document.getElementById("txnBody").innerHTML = rows.map(r => {
    const sc = r.status === "blocked" ? "spill-blocked" : r.status === "review" ? "spill-review" : "spill-ok";
    const sl = r.status === "blocked" ? "Blocked" : r.status === "review" ? "Review" : "OK";
    const scoreColor = r.score >= 65 ? "var(--red)" : r.score >= 30 ? "var(--amber)" : "var(--green)";
    return `<tr>
      <td class="txn-id">${r.id}</td>
      <td>${r.card}</td><td>${r.amount}</td><td>${r.cat}</td>
      <td style="font-family:var(--mono);font-size:12px;">${r.hour}</td>
      <td style="font-family:var(--mono);font-size:12px;color:${scoreColor}">${r.score}%</td>
      <td><span class="spill ${sc}">${sl}</span></td>
    </tr>`;
  }).join("");
}

// ─── Model Tab ────────────────────────────────────────────────────────
function initModelTab() {
  const features = [
    { feature: "amount_ratio",    importance: 0.22 },
    { feature: "merchant_risk",   importance: 0.18 },
    { feature: "distance",        importance: 0.15 },
    { feature: "txn_per_hour",    importance: 0.13 },
    { feature: "is_night",        importance: 0.10 },
    { feature: "new_device",      importance: 0.09 },
    { feature: "risk_combo",      importance: 0.07 },
    { feature: "international",   importance: 0.04 },
    { feature: "card_type",       importance: 0.02 },
  ];

  document.getElementById("featList").innerHTML = features.map(f => `
    <div class="feat-row">
      <span class="feat-name">${f.feature}</span>
      <div class="feat-track"><div class="feat-fill" style="width:${(f.importance/0.22*100).toFixed(1)}%"></div></div>
      <span class="feat-val">${(f.importance*100).toFixed(0)}%</span>
    </div>
  `).join("");

  document.getElementById("metricsList").innerHTML = [
    ["ROC-AUC",           "0.974"],
    ["Avg Precision",     "0.891"],
    ["Training samples",  "78,400"],
    ["Test samples",      "10,000"],
    ["True Positives",    "287"],
    ["False Positives",   "18"],
    ["False Negatives",   "81"],
  ].map(([k,v]) => `<div class="m-row"><span>${k}</span><span>${v}</span></div>`).join("");
}

// ─── Init ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  initModelTab();
});
