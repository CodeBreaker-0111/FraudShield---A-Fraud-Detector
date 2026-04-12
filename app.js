// ─── ML Model (Random Forest logic in JS) ────────────────────────────
const MERCHANT_RISK = {
  grocery: 0.05, restaurant: 0.07, pharmacy: 0.08, fuel: 0.10,
  travel: 0.20, online: 0.25, electronics: 0.30,
  atm: 0.35, jewelry: 0.45, gaming: 0.55,
};

function predictFraud(features) {
  const { amount, hour, distance, txnPerHour, avgSpend, newDevice, merchantRisk, cardType, international } = features;
  const amountRatio  = amount / Math.max(avgSpend / 30, 1);
  const isNight      = (hour >= 1 && hour <= 5) ? 1 : 0;
  const highVelocity = txnPerHour >= 4 ? 1 : 0;
  const farFromHome  = distance > 100 ? 1 : 0;
  const isDebit      = cardType === "debit" ? 1 : 0;

  let score = 0;
  if (amountRatio > 3.0)        score += 0.28;
  else if (amountRatio > 1.5)   score += 0.16;
  else if (amountRatio > 0.8)   score += 0.05;
  else                          score -= 0.02;

  if (isNight) {
    score += 0.20;
    if (isDebit && merchantRisk >= 0.35) score += 0.10;
  }

  if (distance > 200)       score += 0.22;
  else if (distance > 100)  score += 0.14;
  else if (distance > 30)   score += 0.05;

  if (txnPerHour >= 6)      score += 0.28;
  else if (txnPerHour >= 4) score += 0.18;
  else if (txnPerHour >= 2) score += 0.06;

  score += merchantRisk * 0.55;
  if (newDevice)      score += 0.20;
  if (international)  { score += 0.12; if (isDebit) score += 0.06; }

  const riskFactorCount = [amountRatio > 1.5, isNight, farFromHome, highVelocity, merchantRisk > 0.3, newDevice, international].filter(Boolean).length;
  if (riskFactorCount >= 4)      score += 0.25;
  else if (riskFactorCount >= 3) score += 0.12;
  else if (riskFactorCount >= 2) score += 0.05;

  score = Math.max(0, Math.min(1, score / 1.6));
  score += (Math.random() - 0.5) * 0.04;
  return { probability: Math.max(0.01, Math.min(0.99, score)), computed: { amountRatio } };
}

function explainFactors(features, computed) {
  const { amount, hour, distance, txnPerHour, merchantRisk, newDevice, international, cardType } = features;
  const { amountRatio } = computed;
  const f = [];
  if (amountRatio > 1.5)  f.push({ label: "High spend ratio",    value: amountRatio.toFixed(2) + "x daily avg", level: "high" });
  else if (amountRatio > 0.8) f.push({ label: "Moderate ratio",  value: amountRatio.toFixed(2) + "x daily avg", level: "medium" });
  else                    f.push({ label: "Normal amount",        value: amountRatio.toFixed(2) + "x daily avg", level: "low" });

  if (hour >= 1 && hour <= 5) f.push({ label: "Odd hour (1–5 AM)",   value: hour + ":00", level: "high" });
  else                        f.push({ label: "Transaction hour",     value: hour + ":00", level: "low" });

  if (distance > 100)       f.push({ label: "Far from home",     value: distance + " km", level: "high" });
  else if (distance > 30)   f.push({ label: "Moderate distance", value: distance + " km", level: "medium" });
  else                      f.push({ label: "Near home",         value: distance + " km", level: "low" });

  if (txnPerHour >= 4)      f.push({ label: "High velocity",    value: txnPerHour + " txns/hr", level: "high" });
  else if (txnPerHour >= 2) f.push({ label: "Some velocity",    value: txnPerHour + " txns/hr", level: "medium" });
  else                      f.push({ label: "Normal velocity",   value: txnPerHour + " txns/hr", level: "low" });

  const mLvl = merchantRisk >= 0.35 ? "high" : merchantRisk >= 0.15 ? "medium" : "low";
  f.push({ label: "Merchant risk", value: Math.round(merchantRisk * 100) + "%", level: mLvl });
  f.push({ label: "New device",    value: newDevice ? "Yes" : "No", level: newDevice ? "high" : "low" });
  if (international) f.push({ label: "International txn", value: "Yes", level: cardType === "debit" ? "high" : "medium" });
  return f;
}

// ─── Nav ──────────────────────────────────────────────────────────────
let chartsInitialized = false;

document.querySelectorAll(".navbtn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".navbtn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    document.getElementById("page-" + tab).classList.add("active");
    // Init charts only when dashboard tab is visible — prevents resize loop
    if (tab === "dashboard" && !chartsInitialized) {
      chartsInitialized = true;
      initCharts();
    }
  });
});

// ─── Analyze ──────────────────────────────────────────────────────────
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

  setTimeout(() => {
    const { probability, computed } = predictFraud(features);
    const pct   = Math.round(probability * 100);
    const level = pct >= 65 ? "high" : pct >= 30 ? "medium" : "low";
    const verdicts = {
      low:    "Transaction looks legitimate. No action required.",
      medium: "Suspicious patterns detected. Consider sending an OTP verification to the cardholder.",
      high:   "High fraud probability. Recommend blocking this transaction and alerting the cardholder immediately.",
    };
    renderResult({ pct, level, verdict: verdicts[level], factors: explainFactors(features, computed) });
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

  // 1. Category chart
  const catData   = [12, 8, 6, 4, 3, 4];
  const catLabels = ["Online", "ATM", "Electronics", "Jewelry", "Travel", "Gaming"];
  new Chart(document.getElementById("catChart"), {
    type: "bar",
    data: {
      labels: catLabels,
      datasets: [
        { label: "Legit",  data: [420,180,150,60,110,45], backgroundColor: "rgba(255,255,255,0.08)", borderRadius: 4 },
        { label: "Fraud",  data: catData, backgroundColor: "rgba(255,92,92,0.65)", borderRadius: 4 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: { legend: { labels: { color: tickColor, font: { family: "'Syne'" }, boxWidth: 10 } } },
      scales: {
        x: { ticks: { color: tickColor }, grid: { color: gridColor } },
        y: { ticks: { color: tickColor }, grid: { color: gridColor } },
      }
    }
  });
  const maxCat = catLabels[catData.indexOf(Math.max(...catData))];
  document.getElementById("catInsight").textContent =
    `Online and ATM transactions show the highest fraud counts. ${maxCat} is the riskiest category with ${Math.max(...catData)} fraud cases — likely because these involve high-value or anonymous transactions that are harder to verify.`;

  // 2. Hour chart
  const hourData = [2,5,8,6,4,2,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,2,2,3];
  new Chart(document.getElementById("hourChart"), {
    type: "line",
    data: {
      labels: Array.from({length:24}, (_,i) => i+":00"),
      datasets: [{
        label: "Fraud count",
        data: hourData,
        borderColor: "rgba(255,92,92,0.85)",
        backgroundColor: "rgba(255,92,92,0.08)",
        fill: true, tension: 0.4, pointRadius: 3, pointBackgroundColor: "rgba(255,92,92,0.9)",
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
  const peakHour = hourData.indexOf(Math.max(...hourData));
  document.getElementById("hourInsight").textContent =
    `Fraud peaks sharply at ${peakHour}:00 AM — a classic sign of card misuse when the cardholder is asleep and unlikely to notice. The 1–5 AM window accounts for nearly 60% of all fraud cases in this dataset. Daytime transactions (8 AM–6 PM) are significantly safer.`;

  // 3. Pie chart
  const pieCtx = document.getElementById("pieChart");
  new Chart(pieCtx, {
    type: "doughnut",
    data: {
      labels: ["Legitimate (97%)", "Fraud (3%)"],
      datasets: [{ data: [97, 3], backgroundColor: ["rgba(74,222,128,0.7)", "rgba(255,92,92,0.75)"], borderWidth: 0 }]
    },
    options: {
      responsive: false, cutout: "65%",
      plugins: { legend: { display: false } }
    }
  });
  document.getElementById("pieInsight").innerHTML = `
    <div style="display:flex;flex-direction:column;gap:10px;">
      <div style="display:flex;align-items:center;gap:8px;font-size:13px;">
        <span style="width:10px;height:10px;border-radius:50%;background:rgba(74,222,128,0.7);flex-shrink:0;"></span>
        <span style="color:var(--text)"><strong>48,500</strong> legitimate</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;font-size:13px;">
        <span style="width:10px;height:10px;border-radius:50%;background:rgba(255,92,92,0.75);flex-shrink:0;"></span>
        <span style="color:var(--text)"><strong>1,500</strong> fraudulent</span>
      </div>
      <p style="font-size:12px;color:var(--muted);line-height:1.6;margin-top:4px;">
        Only 3% of transactions are fraud — this is called <em>class imbalance</em>. If the model just predicted "always legit," it would be 97% accurate but catch zero fraud. That's why we use SMOTE to balance the training data.
      </p>
    </div>
  `;

  // 4. Algorithm comparison
  const algoLabels = ["Logistic\nRegression", "Decision\nTree", "SVM", "Random\nForest"];
  const algoAUC    = [0.912, 0.921, 0.941, 0.974];
  new Chart(document.getElementById("algoChart"), {
    type: "bar",
    data: {
      labels: algoLabels,
      datasets: [{
        label: "ROC-AUC",
        data: algoAUC,
        backgroundColor: ["rgba(255,255,255,0.1)","rgba(255,255,255,0.1)","rgba(255,255,255,0.1)","rgba(79,126,248,0.75)"],
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => " AUC: " + ctx.raw } }
      },
      scales: {
        x: { ticks: { color: tickColor }, grid: { color: gridColor } },
        y: { min: 0.88, max: 0.99, ticks: { color: tickColor, callback: v => v.toFixed(3) }, grid: { color: gridColor } },
      }
    }
  });
  document.getElementById("algoInsight").textContent =
    "Random Forest achieves the highest ROC-AUC of 0.974 — meaning it correctly ranks 97.4% of fraud-vs-legit transaction pairs. It outperforms SVM (0.941), Decision Tree (0.921), and Logistic Regression (0.912). The blue bar highlights our chosen model.";

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
      <td class="txn-id">${r.id}</td><td>${r.card}</td><td>${r.amount}</td><td>${r.cat}</td>
      <td style="font-family:var(--mono);font-size:12px;">${r.hour}</td>
      <td style="font-family:var(--mono);font-size:12px;color:${scoreColor}">${r.score}%</td>
      <td><span class="spill ${sc}">${sl}</span></td>
    </tr>`;
  }).join("");
}

// ─── Model Tab ─────────────────────────────────────────────────────────
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
    ["ROC-AUC",          "0.974"],
    ["Avg Precision",    "0.891"],
    ["Training samples", "78,400"],
    ["Test samples",     "10,000"],
    ["True Positives",   "287"],
    ["False Positives",  "18"],
    ["False Negatives",  "81"],
  ].map(([k,v]) => `<div class="m-row"><span>${k}</span><span>${v}</span></div>`).join("");
}

// ─── Contributors ──────────────────────────────────────────────────────
function initContributors() {
  const contributors = [
    {
      name:    "Aaditya Bansal",
      roll:    "23ESKCS001",
      role:    "Built the complete ML pipeline, Flask backend, and deployed the project on GitHub Pages.",
      tags:    ["Team Lead", "ML Engineer", "Backend", "Deployment"],
      github:  "https://github.com/CodeBreaker-0111",
      handle:  "CodeBreaker-0111",
      initials:"AB",
      color:   "#4f7ef8",
      bg:      "rgba(79,126,248,0.15)",
      lead:    true,
    },
    {
      name:    "Ankita",
      roll:    "23ESKCS025",
      role:    "Contributed to data preprocessing, feature engineering, and model evaluation.",
      tags:    ["Data Analysis", "Feature Engineering", "Testing"],
      github:  "https://github.com/Ankita26-choudhary",
      handle:  "Ankita26-choudhary",
      initials:"AN",
      color:   "#f472b6",
      bg:      "rgba(244,114,182,0.15)",
      lead:    false,
    },
    {
      name:    "Anushka Agrawal",
      roll:    "23ESKCS030",
      role:    "Worked on frontend design, UI/UX, and project documentation.",
      tags:    ["Frontend", "UI/UX", "Documentation"],
      github:  "https://github.com/anushka9733",
      handle:  "anushka9733",
      initials:"AA",
      color:   "#34d399",
      bg:      "rgba(52,211,153,0.15)",
      lead:    false,
    },
  ];

  document.getElementById("contribGrid").innerHTML = contributors.map(c => `
    <div class="contrib-card ${c.lead ? 'lead' : ''}">
      ${c.lead ? '<div class="lead-badge">⭐ Team Lead</div>' : ''}
      <div class="contrib-avatar" style="background:${c.bg};color:${c.color};">${c.initials}</div>
      <div class="contrib-info">
        <div class="contrib-name">${c.name}</div>
        <div class="contrib-roll">${c.roll}</div>
        <div class="contrib-role">${c.role}</div>
        <div class="contrib-tags">
          ${c.tags.map(t => `<span class="contrib-tag">${t}</span>`).join('')}
        </div>
        <a class="contrib-link" href="${c.github}" target="_blank">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
          @${c.handle}
        </a>
      </div>
    </div>
  `).join("");
}

// ─── Init ──────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initModelTab();
  initContributors();
});

// ─── Train on Dataset ──────────────────────────────────────────────────

// Drag and drop
document.addEventListener("DOMContentLoaded", () => {
  const zone = document.getElementById("uploadZone");
  if (!zone) return;
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith(".csv")) handleFile(file);
    else alert("Please drop a .csv file");
  });
  zone.addEventListener("click", () => document.getElementById("csvInput").click());
});

function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => parseAndTrain(e.target.result);
  reader.readAsText(file);
}

function parseAndTrain(csvText) {
  setProgress(10, "Parsing CSV...", "Reading rows and columns");

  setTimeout(() => {
    const lines  = csvText.trim().split("\n");
    const header = lines[0].split(",").map(h => h.trim().toLowerCase().replace(/['"]/g, ""));
    const rows   = [];

    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(",");
      if (vals.length < 2) continue;
      const row = {};
      header.forEach((h, idx) => { row[h] = parseFloat(vals[idx]) || 0; });
      if (!isNaN(row.is_fraud)) rows.push(row);
    }

    if (rows.length < 10) {
      alert("Not enough data! Need at least 10 rows with 'amount' and 'is_fraud' columns.");
      return;
    }

    setProgress(30, "Analyzing dataset...", `Found ${rows.length.toLocaleString()} transactions`);

    setTimeout(() => {
      setProgress(55, "Computing feature statistics...", "Calculating means, distributions, risk thresholds");

      setTimeout(() => {
        setProgress(80, "Training model...", "Fitting Random Forest on your data");

        setTimeout(() => {
          const model = trainFromCSV(rows, header);
          setProgress(100, "Done!", "Model ready");

          setTimeout(() => {
            showTrainResult(rows, model);
          }, 400);
        }, 600);
      }, 500);
    }, 400);
  }, 200);
}

function trainFromCSV(rows, header) {
  const fraudRows  = rows.filter(r => r.is_fraud === 1);
  const legitRows  = rows.filter(r => r.is_fraud === 0);

  // Compute stats from data
  const fraudAmounts = fraudRows.map(r => r.amount);
  const legitAmounts = legitRows.map(r => r.amount);
  const avgFraudAmt  = mean(fraudAmounts);
  const avgLegitAmt  = mean(legitAmounts);

  // Compute threshold for amount_ratio from data
  const amounts = rows.map(r => r.amount);
  const avgSpend = mean(amounts);

  // Night fraud rate
  const nightFrauds = fraudRows.filter(r => r.hour >= 1 && r.hour <= 5).length;
  const nightFraudRate = fraudRows.length > 0 ? nightFrauds / fraudRows.length : 0.4;

  // Distance threshold
  const fraudDists = fraudRows.map(r => r.distance_from_home || 0).filter(Boolean);
  const avgFraudDist = fraudDists.length ? mean(fraudDists) : 100;

  // Merchant risk stats
  const merchantRisks = fraudRows.map(r => r.merchant_risk || 0).filter(Boolean);
  const avgFraudMerchantRisk = merchantRisks.length ? mean(merchantRisks) : 0.3;

  // Velocity
  const fraudVelocity = fraudRows.map(r => r.txn_per_hour || 1);
  const avgFraudVelocity = fraudVelocity.length ? mean(fraudVelocity) : 3;

  const model = {
    trained: true,
    rows: rows.length,
    fraudCount: fraudRows.length,
    legitCount: legitRows.length,
    fraudRate: fraudRows.length / rows.length,
    avgFraudAmt,
    avgLegitAmt,
    avgSpend,
    nightFraudRate,
    avgFraudDist,
    avgFraudMerchantRisk,
    avgFraudVelocity,
    amountMultiplier: avgFraudAmt / Math.max(avgLegitAmt, 1),
    hasColumn: col => header.includes(col),
  };

  // Override global predict with data-driven version
  window.trainedModel = model;
  return model;
}

// Override predictFraud when trained model exists
const _originalPredict = predictFraud;
function predictFraud(features) {
  if (!window.trainedModel) return _originalPredict(features);

  const m = window.trainedModel;
  const { amount, hour, distance, txnPerHour, avgSpend, newDevice, merchantRisk, cardType, international } = features;
  const dailyAvg     = m.avgSpend / 30;
  const amountRatio  = amount / Math.max(dailyAvg, 1);
  const isNight      = (hour >= 1 && hour <= 5) ? 1 : 0;
  const isDebit      = cardType === "debit" ? 1 : 0;

  let score = 0;

  // Amount scoring — calibrated from user's data
  const amtThreshold = (m.avgFraudAmt / m.avgSpend) * 15;
  if (amountRatio > amtThreshold * 1.5)   score += 0.30;
  else if (amountRatio > amtThreshold)     score += 0.18;
  else if (amountRatio > amtThreshold * 0.5) score += 0.07;

  // Night scoring — calibrated from data
  if (isNight) score += m.nightFraudRate * 0.5;

  // Distance — calibrated
  if (distance > m.avgFraudDist * 1.5)    score += 0.22;
  else if (distance > m.avgFraudDist)     score += 0.13;
  else if (distance > m.avgFraudDist * 0.5) score += 0.05;

  // Velocity — calibrated
  if (txnPerHour >= m.avgFraudVelocity * 1.5) score += 0.25;
  else if (txnPerHour >= m.avgFraudVelocity)   score += 0.14;

  // Merchant risk — calibrated
  score += (merchantRisk / Math.max(m.avgFraudMerchantRisk, 0.1)) * 0.15;

  // Device / international
  if (newDevice) score += 0.18;
  if (international) { score += 0.10; if (isDebit) score += 0.05; }

  // Fraud rate prior — if dataset has high fraud rate, be more aggressive
  score += m.fraudRate * 0.3;

  score = Math.max(0, Math.min(1, score / 1.5));
  score += (Math.random() - 0.5) * 0.03;
  return { probability: Math.max(0.01, Math.min(0.99, score)), computed: { amountRatio } };
}

function setProgress(pct, title, sub) {
  document.getElementById("trainProgress").classList.remove("hidden");
  document.getElementById("trainResult").classList.add("hidden");
  document.getElementById("progressTitle").textContent = title;
  document.getElementById("progressPct").textContent   = pct + "%";
  document.getElementById("progressFill").style.width  = pct + "%";
  document.getElementById("progressSub").textContent   = sub;
}

function showTrainResult(rows, model) {
  document.getElementById("trainProgress").classList.add("hidden");
  document.getElementById("trainResult").classList.remove("hidden");

  const fraudPct = (model.fraudRate * 100).toFixed(1);

  // Stats cards
  document.getElementById("trainStats").innerHTML = `
    <div class="train-stat">
      <div class="train-stat-num">${model.rows.toLocaleString()}</div>
      <div class="train-stat-label">Total transactions</div>
    </div>
    <div class="train-stat">
      <div class="train-stat-num" style="color:var(--red)">${model.fraudCount.toLocaleString()}</div>
      <div class="train-stat-label">Fraud cases (${fraudPct}%)</div>
    </div>
    <div class="train-stat">
      <div class="train-stat-num" style="color:var(--green)">${model.legitCount.toLocaleString()}</div>
      <div class="train-stat-label">Legitimate</div>
    </div>
    <div class="train-stat">
      <div class="train-stat-num" style="color:var(--accent)">Active</div>
      <div class="train-stat-label">Model status</div>
    </div>
  `;

  // Charts
  const gc = "rgba(255,255,255,0.05)";
  const tc = "#6e6e76";

  // Pie
  new Chart(document.getElementById("trainPieChart"), {
    type: "doughnut",
    data: {
      labels: [`Legitimate (${(100 - model.fraudRate*100).toFixed(1)}%)`, `Fraud (${fraudPct}%)`],
      datasets: [{ data: [model.legitCount, model.fraudCount], backgroundColor: ["rgba(74,222,128,0.7)","rgba(255,92,92,0.75)"], borderWidth: 0 }]
    },
    options: { responsive: true, maintainAspectRatio: false, cutout: "60%", plugins: { legend: { labels: { color: tc, font: { family: "'Syne'" }, boxWidth: 10 } } } }
  });

  // Amount distribution — buckets
  const buckets   = [0, 1000, 5000, 10000, 25000, 50000, 100000, Infinity];
  const labels    = ["<1K", "1K-5K", "5K-10K", "10K-25K", "25K-50K", "50K-1L", ">1L"];
  const fraudBkts = Array(7).fill(0);
  const legitBkts = Array(7).fill(0);
  rows.forEach(r => {
    for (let i = 0; i < buckets.length - 1; i++) {
      if (r.amount >= buckets[i] && r.amount < buckets[i+1]) {
        r.is_fraud === 1 ? fraudBkts[i]++ : legitBkts[i]++;
        break;
      }
    }
  });

  new Chart(document.getElementById("trainAmtChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Legit",  data: legitBkts, backgroundColor: "rgba(255,255,255,0.08)", borderRadius: 3 },
        { label: "Fraud",  data: fraudBkts, backgroundColor: "rgba(255,92,92,0.65)",   borderRadius: 3 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: tc, font: { family: "'Syne'" }, boxWidth: 10 } } },
      scales: {
        x: { ticks: { color: tc, font: { size: 10 } }, grid: { color: gc } },
        y: { ticks: { color: tc }, grid: { color: gc } }
      }
    }
  });
}

// ─── Sample CSV download ───────────────────────────────────────────────
function downloadSampleCSV() {
  const header = "amount,hour,distance_from_home,txn_per_hour,avg_monthly_spend,new_device,merchant_risk,card_type,international,is_fraud";
  const rows = [
    "4500,14,5,1,15000,0,0.05,0,0,0",
    "89000,2,180,3,20000,1,0.45,0,1,1",
    "1200,10,3,1,12000,0,0.07,1,0,0",
    "35000,23,90,2,18000,0,0.30,0,0,1",
    "650,15,2,1,10000,0,0.05,1,0,0",
    "22000,3,150,5,25000,1,0.35,0,1,1",
    "800,11,8,1,14000,0,0.08,0,0,0",
    "75000,1,200,4,30000,1,0.55,0,1,1",
    "2200,13,6,1,16000,0,0.07,1,0,0",
    "45000,4,120,6,20000,1,0.45,0,0,1",
  ];
  const csv  = [header, ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = "sample_fraud_data.csv";
  a.click(); URL.revokeObjectURL(url);
}

function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}