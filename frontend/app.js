const DATASET_URL = "../data/transaction/transaction.csv";
const SCORES_URL = "../exp/transaction/semi_supervised/split1/split0_large_bs4_steps500/scores/transaction_scores_with_labels.csv";
const CURRENT_EXP_DIR = "exp/transaction/semi_supervised/split1/split0_large_bs4_steps500";

const tabs = [
  { id: "dashboard", label: "Dashboard", hint: "Tổng quan" },
  { id: "dataset", label: "Dataset", hint: "Dữ liệu thật" },
  { id: "training", label: "Training", hint: "Cấu hình train" },
  { id: "evaluation", label: "Evaluation", hint: "Đánh giá" },
  { id: "scores", label: "Scores", hint: "Anomaly scores" },
];

const state = {
  activeTab: "dashboard",
  dataset: [],
  scores: [],
  metrics: [],
  datasetPage: 1,
  scorePage: 1,
  datasetQuery: "",
  datasetLabel: "all",
  scoreLabel: "all",
  scoreThreshold: 0,
  dashboardThreshold: 0,
  trainConfig: {
    model: "distilgpt2",
    batchSize: 4,
    maxSteps: 500,
    evalSteps: 100,
    trainRatio: 0.75,
    learningRate: 0.00005,
    expDir: CURRENT_EXP_DIR,
  },
  evalConfig: {
    batchSize: 8,
    nPermutations: 1,
    threshold: 57.67,
  },
};

function cls(...items) {
  return items.filter(Boolean).join(" ");
}

function formatNumber(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((header, index) => {
      const raw = values[index] ?? "";
      const numeric = Number(raw);
      row[header] = raw !== "" && Number.isFinite(numeric) ? numeric : raw;
    });
    return row;
  });
}

async function fetchCsv(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Không đọc được ${url}`);
  return parseCsv(await response.text());
}

function groupStats(rows, labelKey, valueKey) {
  const groups = { 0: [], 1: [] };
  rows.forEach((row) => groups[row[labelKey]].push(Number(row[valueKey])));
  return Object.fromEntries(
    Object.entries(groups).map(([label, values]) => [
      label,
      {
        count: values.length,
        mean: values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1),
        min: Math.min(...values),
        max: Math.max(...values),
      },
    ])
  );
}

function confusionAt(threshold) {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  state.scores.forEach((row) => {
    const predicted = Number(row.anomaly_score) >= threshold ? 1 : 0;
    const actual = Number(row.is_anomaly);
    if (predicted === 1 && actual === 1) tp += 1;
    if (predicted === 1 && actual === 0) fp += 1;
    if (predicted === 0 && actual === 0) tn += 1;
    if (predicted === 0 && actual === 1) fn += 1;
  });

  const precision = tp + fp === 0 ? 1 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { threshold, tp, fp, tn, fn, precision, recall, f1 };
}

function buildMetrics() {
  if (!state.scores.length) return [];
  const values = [...new Set(state.scores.map((row) => Number(row.anomaly_score)).sort((a, b) => a - b))];
  const min = values[0];
  const max = values[values.length - 1];
  const thresholds = [Math.max(0, min - 1), ...values, max + 1];
  return thresholds.map((threshold) => confusionAt(threshold));
}

function bestMetric() {
  return state.metrics.reduce((best, item) => {
    if (item.f1 > best.f1) return item;
    if (item.f1 === best.f1 && item.fp < best.fp) return item;
    return best;
  }, state.metrics[0] || confusionAt(0));
}

function stats() {
  const datasetNormal = state.dataset.filter((row) => Number(row.is_anomaly) === 0).length;
  const datasetFraud = state.dataset.filter((row) => Number(row.is_anomaly) === 1).length;
  const scoreNormal = state.scores.filter((row) => Number(row.is_anomaly) === 0).length;
  const scoreFraud = state.scores.filter((row) => Number(row.is_anomaly) === 1).length;
  const scoreStats = groupStats(state.scores, "is_anomaly", "anomaly_score");
  return { datasetNormal, datasetFraud, scoreNormal, scoreFraud, scoreStats };
}

function renderTabs() {
  const nav = document.getElementById("tabs");
  const mobile = document.getElementById("mobile-tabs");

  nav.innerHTML = tabs
    .map((tab) => {
      const active = tab.id === state.activeTab;
      return `
        <button data-tab="${tab.id}" class="${cls(
          "w-full rounded-lg px-3 py-3 text-left transition",
          active ? "bg-base text-white" : "text-slate-600 hover:bg-slate-100 hover:text-base"
        )}">
          <span class="block text-sm font-semibold">${tab.label}</span>
          <span class="${active ? "text-slate-300" : "text-muted"} block text-xs">${tab.hint}</span>
        </button>
      `;
    })
    .join("");

  mobile.innerHTML = tabs
    .map((tab) => `<option value="${tab.id}" ${tab.id === state.activeTab ? "selected" : ""}>${tab.label}</option>`)
    .join("");
}

function shell(title, body) {
  document.getElementById("page-title").textContent = title;
  document.getElementById("app").innerHTML = body;
  renderTabs();
}

function metricCard(label, value, detail, tone = "neutral") {
  const tones = {
    neutral: "bg-white border-line",
    brand: "bg-brand-50 border-brand-100",
    mint: "bg-mint-50 border-emerald-100",
  };
  return `
    <article class="rounded-xl border ${tones[tone]} p-5 shadow-panel">
      <p class="text-sm text-muted">${label}</p>
      <p class="mt-2 text-3xl font-semibold tracking-tight">${value}</p>
      <p class="mt-2 text-sm text-slate-500">${detail}</p>
    </article>
  `;
}

function renderDashboard() {
  const s = stats();
  const best = bestMetric();
  const current = confusionAt(Number(state.dashboardThreshold || best.threshold));
  const minScore = Math.min(...state.scores.map((row) => Number(row.anomaly_score)));
  const maxScore = Math.max(...state.scores.map((row) => Number(row.anomaly_score)));
  const baseline = state.scores.length ? s.scoreFraud / state.scores.length : 0;
  shell(
    "Dashboard",
    `
      <div class="space-y-6">
        <div class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          ${metricCard("NLL threshold", formatNumber(current.threshold, 0), `Best F1 threshold: ${formatNumber(best.threshold)}`, "neutral")}
          ${metricCard("Precision", `${formatNumber(current.precision * 100, 1)}%`, `TP ${current.tp}, FP ${current.fp}`, "brand")}
          ${metricCard("Recall", `${formatNumber(current.recall * 100, 1)}%`, `FN ${current.fn}, fraud ${s.scoreFraud}`, "mint")}
          ${metricCard("F1", `${formatNumber(current.f1 * 100, 1)}%`, `Baseline random: ${formatNumber(baseline * 100, 1)}%`, "neutral")}
        </div>

        <div class="grid gap-6 xl:grid-cols-[1fr_320px]">
          <section class="rounded-xl border border-line bg-white p-6 shadow-panel">
            <div class="flex flex-col gap-5">
              <div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div>
                  <h3 class="text-lg font-semibold">Precision-Recall Curve</h3>
                  <p class="mt-1 text-sm text-muted">Mỗi điểm tương ứng một NLL threshold khác nhau. Chấm tím là threshold hiện tại.</p>
                </div>
                <button id="use-best-threshold" class="rounded-lg border border-line px-3 py-2 text-sm font-medium hover:bg-slate-50">Dùng best F1</button>
              </div>

              <div class="rounded-xl border border-line bg-slate-50 p-4">
                <div class="flex items-center justify-between gap-4 text-sm">
                  <label for="dashboard-threshold" class="font-medium">NLL threshold</label>
                  <span class="min-w-20 rounded-lg bg-white px-3 py-1 text-right font-semibold shadow-sm">${formatNumber(current.threshold)}</span>
                </div>
                <input id="dashboard-threshold" type="range" min="${minScore}" max="${maxScore}" step="0.01" value="${current.threshold}" class="mt-3 w-full accent-brand-600" />
                <div class="mt-1 flex justify-between text-xs text-muted">
                  <span>${formatNumber(minScore)}</span>
                  <span>${formatNumber(maxScore)}</span>
                </div>
              </div>

              <div class="flex flex-wrap gap-4 text-sm">
                <span class="inline-flex items-center gap-2"><span class="h-3 w-3 rounded-sm bg-mint-600"></span>PR curve (model)</span>
                <span class="inline-flex items-center gap-2"><span class="h-3 w-3 rounded-sm border border-dashed border-slate-500 bg-slate-100"></span>Baseline random</span>
                <span class="inline-flex items-center gap-2"><span class="h-3 w-3 rounded-full bg-indigo-700"></span>Threshold hiện tại</span>
              </div>

              <canvas id="pr-chart" class="h-[430px] w-full rounded-lg border border-line bg-white"></canvas>
            </div>
          </section>

          <section class="rounded-xl border border-line bg-white p-6 shadow-panel">
            <h3 class="text-lg font-semibold">Confusion matrix</h3>
            <p class="mt-1 text-sm text-muted">Theo threshold hiện tại.</p>
            <div class="mt-5 grid grid-cols-2 gap-3 text-center">
              <div class="rounded-lg bg-mint-50 p-4">
                <p class="text-sm text-muted">TP</p>
                <p class="text-3xl font-semibold text-mint-600">${current.tp}</p>
              </div>
              <div class="rounded-lg bg-red-50 p-4">
                <p class="text-sm text-muted">FP</p>
                <p class="text-3xl font-semibold text-red-600">${current.fp}</p>
              </div>
              <div class="rounded-lg bg-amber-50 p-4">
                <p class="text-sm text-muted">FN</p>
                <p class="text-3xl font-semibold text-amber-600">${current.fn}</p>
              </div>
              <div class="rounded-lg bg-slate-100 p-4">
                <p class="text-sm text-muted">TN</p>
                <p class="text-3xl font-semibold">${current.tn}</p>
              </div>
            </div>
            <div class="mt-5 space-y-2 text-sm">
              <p><span class="font-medium">Precision:</span> ${formatNumber(current.precision)}</p>
              <p><span class="font-medium">Recall:</span> ${formatNumber(current.recall)}</p>
              <p><span class="font-medium">F1:</span> ${formatNumber(current.f1)}</p>
            </div>
          </section>
        </div>

        <div class="grid gap-6 xl:grid-cols-2">
          <section class="rounded-xl border border-line bg-white p-6 shadow-panel">
            <h3 class="text-lg font-semibold">Score separation</h3>
            <div class="mt-4 space-y-4">
              <div>
                <div class="mb-2 flex justify-between text-sm"><span>Normal mean</span><span>${formatNumber(s.scoreStats[0].mean)}</span></div>
                <div class="h-3 rounded bg-slate-100"><div class="h-3 rounded bg-slate-500" style="width: 25%"></div></div>
              </div>
              <div>
                <div class="mb-2 flex justify-between text-sm"><span>Fraud mean</span><span>${formatNumber(s.scoreStats[1].mean)}</span></div>
                <div class="h-3 rounded bg-slate-100"><div class="h-3 rounded bg-brand-600" style="width: 50%"></div></div>
              </div>
              <div>
                <div class="mb-2 flex justify-between text-sm"><span>Fraud max</span><span>${formatNumber(s.scoreStats[1].max)}</span></div>
                <div class="h-3 rounded bg-slate-100"><div class="h-3 rounded bg-red-500" style="width: 100%"></div></div>
              </div>
            </div>
          </section>

          <section class="rounded-xl border border-line bg-white p-6 shadow-panel">
            <h3 class="text-lg font-semibold">Kiến trúc pipeline</h3>
            <div class="mt-4 space-y-3 text-sm">
              ${["transaction.csv", "load_data + semi-supervised split", "serialize row thành text", "fine-tune distilgpt2", "evaluate NLL anomaly score"].map((item, index) => `
                <div class="flex items-center gap-3 rounded-lg border border-line p-3">
                  <span class="flex h-7 w-7 items-center justify-center rounded-full bg-base text-xs font-semibold text-white">${index + 1}</span>
                  <span>${item}</span>
                </div>
              `).join("")}
            </div>
          </section>
        </div>
      </div>
    `
  );
  drawPrChart();
  bindDashboardEvents();
}

function filterDataset() {
  const q = state.datasetQuery.toLowerCase().trim();
  return state.dataset
    .map((row, index) => ({ ...row, row_index: index }))
    .filter((row) => state.datasetLabel === "all" || String(row.is_anomaly) === state.datasetLabel)
    .filter((row) => {
      if (!q) return true;
      return Object.values(row).some((value) => String(value).toLowerCase().includes(q));
    });
}

function renderDataset() {
  const filtered = filterDataset();
  const pageSize = 20;
  const totalPages = Math.max(Math.ceil(filtered.length / pageSize), 1);
  state.datasetPage = Math.min(state.datasetPage, totalPages);
  const start = (state.datasetPage - 1) * pageSize;
  const rows = filtered.slice(start, start + pageSize);

  shell(
    "Dataset",
    `
      <div class="space-y-6">
        <div class="grid gap-4 md:grid-cols-3">
          ${metricCard("Total rows", state.dataset.length.toLocaleString(), "Đọc trực tiếp từ transaction.csv")}
          ${metricCard("Normal rows", stats().datasetNormal.toLocaleString(), "is_anomaly = 0", "mint")}
          ${metricCard("Fraud rows", stats().datasetFraud.toLocaleString(), "is_anomaly = 1", "brand")}
        </div>

        <section class="rounded-xl border border-line bg-white p-5 shadow-panel">
          <div class="grid gap-3 md:grid-cols-[1fr_180px_auto]">
            <input id="dataset-query" value="${state.datasetQuery}" class="rounded-lg border border-line px-3 py-2 text-sm outline-none focus:border-brand-500" placeholder="Tìm transaction_id, merchant, country, device..." />
            <select id="dataset-label" class="rounded-lg border border-line px-3 py-2 text-sm">
              <option value="all" ${state.datasetLabel === "all" ? "selected" : ""}>Tất cả label</option>
              <option value="0" ${state.datasetLabel === "0" ? "selected" : ""}>Normal</option>
              <option value="1" ${state.datasetLabel === "1" ? "selected" : ""}>Fraud</option>
            </select>
            <button id="reset-dataset-filter" class="rounded-lg bg-base px-4 py-2 text-sm font-medium text-white">Reset</button>
          </div>
        </section>

        <section class="rounded-xl border border-line bg-white shadow-panel">
          <div class="border-b border-line px-5 py-4">
            <h3 class="font-semibold">Transaction rows</h3>
            <p class="mt-1 text-sm text-muted">Hiển thị ${filtered.length.toLocaleString()} dòng sau filter.</p>
          </div>
          <div class="overflow-x-auto">
            <table class="min-w-full text-left text-sm">
              <thead class="border-b border-line bg-slate-50 text-xs uppercase tracking-wide text-muted">
                <tr>
                  ${["row", "transaction_id", "user_id", "amount", "currency", "merchant", "category", "country", "device", "hour", "payment_method", "label"].map((h) => `<th class="px-4 py-3 font-semibold">${h}</th>`).join("")}
                </tr>
              </thead>
              <tbody class="divide-y divide-line">
                ${rows.map((row) => `
                  <tr class="hover:bg-slate-50">
                    <td class="px-4 py-3 font-medium">${row.row_index}</td>
                    <td class="px-4 py-3">${row.transaction_id}</td>
                    <td class="px-4 py-3">${row.user_id}</td>
                    <td class="px-4 py-3">${formatNumber(row.amount)}</td>
                    <td class="px-4 py-3">${row.currency}</td>
                    <td class="px-4 py-3">${row.merchant}</td>
                    <td class="px-4 py-3">${row.category}</td>
                    <td class="px-4 py-3">${row.country}</td>
                    <td class="px-4 py-3">${row.device}</td>
                    <td class="px-4 py-3">${row.hour}</td>
                    <td class="px-4 py-3">${row.payment_method}</td>
                    <td class="px-4 py-3">${labelBadge(row.is_anomaly)}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
          ${pagination("dataset", state.datasetPage, totalPages)}
        </section>
      </div>
    `
  );
  bindDatasetEvents();
}

function labelBadge(label) {
  const fraud = Number(label) === 1;
  return `<span class="${fraud ? "bg-red-50 text-red-700" : "bg-mint-50 text-mint-600"} rounded-full px-2.5 py-1 text-xs font-semibold">${fraud ? "Fraud" : "Normal"}</span>`;
}

function pagination(type, page, totalPages) {
  return `
    <div class="flex items-center justify-between border-t border-line px-5 py-4 text-sm">
      <span class="text-muted">Page ${page} / ${totalPages}</span>
      <div class="flex gap-2">
        <button data-page="${type}:prev" class="rounded-lg border border-line px-3 py-2 ${page === 1 ? "cursor-not-allowed opacity-40" : "hover:bg-slate-50"}">Previous</button>
        <button data-page="${type}:next" class="rounded-lg border border-line px-3 py-2 ${page === totalPages ? "cursor-not-allowed opacity-40" : "hover:bg-slate-50"}">Next</button>
      </div>
    </div>
  `;
}

function renderTraining() {
  const c = state.trainConfig;
  const command = trainCommand();
  shell(
    "Training",
    `
      <div class="grid gap-6 xl:grid-cols-[1fr_420px]">
        <section class="rounded-xl border border-line bg-white p-6 shadow-panel">
          <h3 class="text-lg font-semibold">Cấu hình train</h3>
          <div class="mt-5 grid gap-4 md:grid-cols-2">
            ${inputField("Model", "train-model", c.model)}
            ${numberField("Batch size", "train-batch", c.batchSize)}
            ${numberField("Max steps", "train-steps", c.maxSteps)}
            ${numberField("Eval steps", "train-eval-steps", c.evalSteps)}
            ${numberField("Train ratio", "train-ratio", c.trainRatio, "0.01")}
            ${numberField("Learning rate", "train-lr", c.learningRate, "0.00001")}
            <label class="md:col-span-2">
              <span class="text-sm font-medium">Experiment directory</span>
              <input id="train-exp-dir" value="${c.expDir}" class="mt-1 w-full rounded-lg border border-line px-3 py-2 text-sm outline-none focus:border-brand-500" />
            </label>
          </div>

          <div class="mt-6 flex flex-wrap gap-3">
            <button id="build-train" class="rounded-lg bg-base px-4 py-2 text-sm font-semibold text-white">Tạo command</button>
            <button id="copy-train" class="rounded-lg border border-line px-4 py-2 text-sm font-semibold hover:bg-slate-50">Copy command</button>
            <button id="run-train" class="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">Run training</button>
          </div>

          <div class="mt-5 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
            Nút Run training hiện kiểm tra backend API <code>/api/train</code>. Nếu chưa có backend, UI sẽ tạo command để bạn copy chạy trong PowerShell.
          </div>
        </section>

        <aside class="rounded-xl border border-line bg-white p-6 shadow-panel">
          <h3 class="text-lg font-semibold">Training plan</h3>
          <div class="mt-4 space-y-3 text-sm">
            <p><span class="font-medium">Train rows:</span> 810 normal only</p>
            <p><span class="font-medium">Fraud in train:</span> 0</p>
            <p><span class="font-medium">Target epochs:</span> khoảng 2.46</p>
            <p><span class="font-medium">Last train loss:</span> 0.7306</p>
            <p><span class="font-medium">Runtime:</span> khoảng 26 phút CPU</p>
          </div>
        </aside>

        <section class="rounded-xl border border-line bg-white p-6 shadow-panel xl:col-span-2">
          <div class="flex items-center justify-between gap-3">
            <h3 class="text-lg font-semibold">Generated command</h3>
            <span id="train-command-status" class="text-sm text-muted">Ready</span>
          </div>
          <pre class="mt-4 overflow-x-auto rounded-lg bg-slate-950 p-4 text-sm text-slate-100"><code id="train-command">${command}</code></pre>
        </section>
      </div>
    `
  );
  bindTrainingEvents();
}

function renderEvaluation() {
  const c = state.evalConfig;
  const at = confusionAt(Number(c.threshold));
  const command = evalCommand();
  shell(
    "Evaluation",
    `
      <div class="space-y-6">
        <div class="grid gap-4 md:grid-cols-4">
          ${metricCard("Test rows", state.scores.length, "270 normal, 30 fraud")}
          ${metricCard("Threshold", formatNumber(c.threshold), "Score >= threshold => fraud", "brand")}
          ${metricCard("Precision", formatNumber(at.precision), `TP ${at.tp}, FP ${at.fp}`, "mint")}
          ${metricCard("Recall", formatNumber(at.recall), `FN ${at.fn}, TN ${at.tn}`)}
        </div>

        <section class="rounded-xl border border-line bg-white p-6 shadow-panel">
          <h3 class="text-lg font-semibold">Cấu hình evaluate</h3>
          <div class="mt-5 grid gap-4 md:grid-cols-3">
            ${numberField("Batch size", "eval-batch", c.batchSize)}
            ${numberField("N permutations", "eval-permutations", c.nPermutations)}
            ${numberField("Threshold NLL", "eval-threshold", c.threshold, "0.01")}
          </div>
          <div class="mt-5 flex flex-wrap gap-3">
            <button id="apply-eval" class="rounded-lg bg-base px-4 py-2 text-sm font-semibold text-white">Áp dụng threshold</button>
            <button id="copy-eval" class="rounded-lg border border-line px-4 py-2 text-sm font-semibold hover:bg-slate-50">Copy command</button>
            <button id="run-eval" class="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">Run evaluate</button>
          </div>
        </section>

        <section class="rounded-xl border border-line bg-white p-6 shadow-panel">
          <h3 class="text-lg font-semibold">Generated command</h3>
          <pre class="mt-4 overflow-x-auto rounded-lg bg-slate-950 p-4 text-sm text-slate-100"><code id="eval-command">${command}</code></pre>
        </section>
      </div>
    `
  );
  bindEvaluationEvents();
}

function filterScores() {
  return state.scores
    .filter((row) => state.scoreLabel === "all" || String(row.is_anomaly) === state.scoreLabel)
    .filter((row) => Number(row.anomaly_score) >= Number(state.scoreThreshold))
    .sort((a, b) => Number(b.anomaly_score) - Number(a.anomaly_score));
}

function renderScores() {
  const filtered = filterScores();
  const pageSize = 20;
  const totalPages = Math.max(Math.ceil(filtered.length / pageSize), 1);
  state.scorePage = Math.min(state.scorePage, totalPages);
  const start = (state.scorePage - 1) * pageSize;
  const rows = filtered.slice(start, start + pageSize);

  shell(
    "Scores",
    `
      <div class="space-y-6">
        <section class="rounded-xl border border-line bg-white p-5 shadow-panel">
          <div class="grid gap-3 md:grid-cols-[180px_1fr_auto]">
            <select id="score-label" class="rounded-lg border border-line px-3 py-2 text-sm">
              <option value="all" ${state.scoreLabel === "all" ? "selected" : ""}>Tất cả label</option>
              <option value="0" ${state.scoreLabel === "0" ? "selected" : ""}>Normal</option>
              <option value="1" ${state.scoreLabel === "1" ? "selected" : ""}>Fraud</option>
            </select>
            <label class="flex items-center gap-3 rounded-lg border border-line px-3 py-2 text-sm">
              <span class="whitespace-nowrap font-medium">Threshold</span>
              <input id="score-threshold" type="range" min="0" max="180" step="1" value="${state.scoreThreshold}" class="w-full" />
              <span class="w-14 text-right">${formatNumber(state.scoreThreshold, 0)}</span>
            </label>
            <button id="reset-score-filter" class="rounded-lg bg-base px-4 py-2 text-sm font-medium text-white">Reset</button>
          </div>
        </section>

        <section class="rounded-xl border border-line bg-white shadow-panel">
          <div class="border-b border-line px-5 py-4">
            <h3 class="font-semibold">Anomaly score table</h3>
            <p class="mt-1 text-sm text-muted">Đọc từ transaction_scores_with_labels.csv, sort theo score giảm dần.</p>
          </div>
          <div class="overflow-x-auto">
            <table class="min-w-full text-left text-sm">
              <thead class="border-b border-line bg-slate-50 text-xs uppercase tracking-wide text-muted">
                <tr>
                  ${["rank", "row_index", "anomaly_score", "label", "transaction_id", "amount", "merchant", "country", "device"].map((h) => `<th class="px-4 py-3 font-semibold">${h}</th>`).join("")}
                </tr>
              </thead>
              <tbody class="divide-y divide-line">
                ${rows.map((row, index) => {
                  const source = state.dataset[row.row_index] || {};
                  return `
                    <tr class="hover:bg-slate-50">
                      <td class="px-4 py-3 font-medium">${start + index + 1}</td>
                      <td class="px-4 py-3">${row.row_index}</td>
                      <td class="px-4 py-3 font-semibold">${formatNumber(row.anomaly_score)}</td>
                      <td class="px-4 py-3">${labelBadge(row.is_anomaly)}</td>
                      <td class="px-4 py-3">${source.transaction_id ?? "-"}</td>
                      <td class="px-4 py-3">${formatNumber(source.amount)}</td>
                      <td class="px-4 py-3">${source.merchant ?? "-"}</td>
                      <td class="px-4 py-3">${source.country ?? "-"}</td>
                      <td class="px-4 py-3">${source.device ?? "-"}</td>
                    </tr>
                  `;
                }).join("")}
              </tbody>
            </table>
          </div>
          ${pagination("scores", state.scorePage, totalPages)}
        </section>
      </div>
    `
  );
  bindScoreEvents();
}

function inputField(label, id, value) {
  return `<label><span class="text-sm font-medium">${label}</span><input id="${id}" value="${value}" class="mt-1 w-full rounded-lg border border-line px-3 py-2 text-sm outline-none focus:border-brand-500" /></label>`;
}

function numberField(label, id, value, step = "1") {
  return `<label><span class="text-sm font-medium">${label}</span><input id="${id}" type="number" step="${step}" value="${value}" class="mt-1 w-full rounded-lg border border-line px-3 py-2 text-sm outline-none focus:border-brand-500" /></label>`;
}

function trainCommand() {
  const c = state.trainConfig;
  return `python train_anollm.py --dataset transaction --data_dir data --n_splits 1 --split_idx 0 --train_ratio ${c.trainRatio} --setting semi_supervised --binning standard --model ${c.model} --batch_size ${c.batchSize} --max_steps ${c.maxSteps} --eval_steps ${c.evalSteps} --lr ${c.learningRate} --exp_dir ${c.expDir}`;
}

function evalCommand() {
  const c = state.evalConfig;
  return `python evaluate_anollm.py --dataset transaction --data_dir data --n_splits 1 --split_idx 0 --train_ratio ${state.trainConfig.trainRatio} --setting semi_supervised --binning standard --model ${state.trainConfig.model} --batch_size ${c.batchSize} --n_permutations ${c.nPermutations} --exp_dir ${state.trainConfig.expDir}`;
}

async function copyText(text) {
  await navigator.clipboard.writeText(text);
  toast("Đã copy command vào clipboard");
}

function toast(message) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 2400);
}

async function callBackend(path, payload) {
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error("backend unavailable");
    toast("Đã gửi job tới backend");
  } catch {
    toast("Chưa có backend API. Command đã sẵn sàng để copy.");
  }
}

function drawPrChart() {
  const canvas = document.getElementById("pr-chart");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = rect.width * scale;
  canvas.height = rect.height * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  const width = rect.width;
  const height = rect.height;
  const pad = { left: 62, right: 30, top: 28, bottom: 62 };

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const s = stats();
  const baseline = state.scores.length ? s.scoreFraud / state.scores.length : 0;
  const current = confusionAt(Number(state.dashboardThreshold || bestMetric().threshold));
  const xFor = (recall) => pad.left + recall * (width - pad.left - pad.right);
  const yFor = (precision) => pad.top + (1 - precision) * (height - pad.top - pad.bottom);

  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();

  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#64748b";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((tick) => {
    const y = yFor(tick);
    ctx.strokeStyle = tick === 0 ? "#cbd5e1" : "#e2e8f0";
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(`${Math.round(tick * 100)}%`, pad.left - 10, y);
  });

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((tick) => {
    const x = xFor(tick);
    ctx.strokeStyle = "#e2e8f0";
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, height - pad.bottom + 6);
    ctx.stroke();
    ctx.fillStyle = "#64748b";
    ctx.fillText(`${Math.round(tick * 100)}%`, x, height - pad.bottom + 12);
  });

  ctx.save();
  ctx.translate(16, pad.top + (height - pad.top - pad.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#334155";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Precision", 0, 0);
  ctx.restore();

  ctx.fillStyle = "#334155";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("Recall", pad.left + (width - pad.left - pad.right) / 2, height - 8);

  const curve = state.metrics
    .filter((point) => point.recall > 0)
    .map((point) => ({ x: xFor(point.recall), y: yFor(point.precision), recall: point.recall, precision: point.precision }))
    .sort((a, b) => {
      if (a.recall !== b.recall) return a.recall - b.recall;
      return b.precision - a.precision;
    });

  if (curve.length) {
    ctx.fillStyle = "rgba(16, 185, 129, 0.12)";
    ctx.beginPath();
    ctx.moveTo(xFor(0), yFor(0));
    curve.forEach((point) => ctx.lineTo(point.x, point.y));
    ctx.lineTo(xFor(1), yFor(0));
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "#059669";
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    curve.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  }

  const baselineY = yFor(baseline);
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, baselineY);
  ctx.lineTo(width - pad.right, baselineY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#64748b";
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`Baseline random (${formatNumber(baseline * 100, 1)}%)`, pad.left + 8, baselineY - 6);

  const cx = xFor(current.recall);
  const cy = yFor(current.precision);
  ctx.fillStyle = "#4338ca";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(cx, cy, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  const calloutWidth = 178;
  const calloutX = Math.min(Math.max(cx - calloutWidth / 2, pad.left + 4), width - pad.right - calloutWidth);
  const calloutY = Math.max(cy - 96, pad.top + 8);
  ctx.fillStyle = "#0f172a";
  roundedRect(ctx, calloutX, calloutY, calloutWidth, 88, 8);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.font = "12px sans-serif";
  ctx.fillText(`Threshold: ${formatNumber(current.threshold)}`, calloutX + 12, calloutY + 10);
  ctx.fillText(`Precision: ${formatNumber(current.precision * 100, 1)}%`, calloutX + 12, calloutY + 30);
  ctx.fillText(`Recall: ${formatNumber(current.recall * 100, 1)}%`, calloutX + 12, calloutY + 50);
  ctx.fillText(`F1: ${formatNumber(current.f1 * 100, 1)}%`, calloutX + 12, calloutY + 70);
}

function roundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
}

function bindDatasetEvents() {
  document.getElementById("dataset-query").addEventListener("input", (event) => {
    state.datasetQuery = event.target.value;
    state.datasetPage = 1;
    renderDataset();
  });
  document.getElementById("dataset-label").addEventListener("change", (event) => {
    state.datasetLabel = event.target.value;
    state.datasetPage = 1;
    renderDataset();
  });
  document.getElementById("reset-dataset-filter").addEventListener("click", () => {
    state.datasetQuery = "";
    state.datasetLabel = "all";
    state.datasetPage = 1;
    renderDataset();
  });
}

function bindDashboardEvents() {
  document.getElementById("dashboard-threshold").addEventListener("input", (event) => {
    state.dashboardThreshold = Number(event.target.value);
    renderDashboard();
  });
  document.getElementById("use-best-threshold").addEventListener("click", () => {
    const best = bestMetric();
    state.dashboardThreshold = Number(best.threshold.toFixed(2));
    renderDashboard();
  });
}

function bindTrainingEvents() {
  const ids = {
    model: "train-model",
    batchSize: "train-batch",
    maxSteps: "train-steps",
    evalSteps: "train-eval-steps",
    trainRatio: "train-ratio",
    learningRate: "train-lr",
    expDir: "train-exp-dir",
  };
  Object.entries(ids).forEach(([key, id]) => {
    document.getElementById(id).addEventListener("input", (event) => {
      const numeric = ["batchSize", "maxSteps", "evalSteps", "trainRatio", "learningRate"].includes(key);
      state.trainConfig[key] = numeric ? Number(event.target.value) : event.target.value;
      document.getElementById("train-command").textContent = trainCommand();
    });
  });
  document.getElementById("build-train").addEventListener("click", () => toast("Command train đã được cập nhật"));
  document.getElementById("copy-train").addEventListener("click", () => copyText(trainCommand()));
  document.getElementById("run-train").addEventListener("click", () => callBackend("/api/train", { command: trainCommand(), config: state.trainConfig }));
}

function bindEvaluationEvents() {
  document.getElementById("eval-batch").addEventListener("input", (event) => {
    state.evalConfig.batchSize = Number(event.target.value);
    document.getElementById("eval-command").textContent = evalCommand();
  });
  document.getElementById("eval-permutations").addEventListener("input", (event) => {
    state.evalConfig.nPermutations = Number(event.target.value);
    document.getElementById("eval-command").textContent = evalCommand();
  });
  document.getElementById("eval-threshold").addEventListener("input", (event) => {
    state.evalConfig.threshold = Number(event.target.value);
  });
  document.getElementById("apply-eval").addEventListener("click", renderEvaluation);
  document.getElementById("copy-eval").addEventListener("click", () => copyText(evalCommand()));
  document.getElementById("run-eval").addEventListener("click", () => callBackend("/api/evaluate", { command: evalCommand(), config: state.evalConfig }));
}

function bindScoreEvents() {
  document.getElementById("score-label").addEventListener("change", (event) => {
    state.scoreLabel = event.target.value;
    state.scorePage = 1;
    renderScores();
  });
  document.getElementById("score-threshold").addEventListener("input", (event) => {
    state.scoreThreshold = Number(event.target.value);
    state.scorePage = 1;
    renderScores();
  });
  document.getElementById("reset-score-filter").addEventListener("click", () => {
    state.scoreLabel = "all";
    state.scoreThreshold = 0;
    state.scorePage = 1;
    renderScores();
  });
}

function bindGlobalEvents() {
  document.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-tab]");
    if (tab) {
      state.activeTab = tab.dataset.tab;
      render();
      return;
    }
    const pageButton = event.target.closest("[data-page]");
    if (!pageButton) return;
    const [type, direction] = pageButton.dataset.page.split(":");
    if (type === "dataset") {
      const total = Math.max(Math.ceil(filterDataset().length / 20), 1);
      state.datasetPage = Math.min(Math.max(state.datasetPage + (direction === "next" ? 1 : -1), 1), total);
      renderDataset();
    }
    if (type === "scores") {
      const total = Math.max(Math.ceil(filterScores().length / 20), 1);
      state.scorePage = Math.min(Math.max(state.scorePage + (direction === "next" ? 1 : -1), 1), total);
      renderScores();
    }
  });

  document.getElementById("mobile-tabs").addEventListener("change", (event) => {
    state.activeTab = event.target.value;
    render();
  });

  window.addEventListener("resize", () => {
    if (state.activeTab === "dashboard") drawPrChart();
  });
}

function render() {
  if (state.activeTab === "dashboard") renderDashboard();
  if (state.activeTab === "dataset") renderDataset();
  if (state.activeTab === "training") renderTraining();
  if (state.activeTab === "evaluation") renderEvaluation();
  if (state.activeTab === "scores") renderScores();
}

async function init() {
  renderTabs();
  bindGlobalEvents();
  try {
    const [dataset, scores] = await Promise.all([fetchCsv(DATASET_URL), fetchCsv(SCORES_URL)]);
    state.dataset = dataset;
    state.scores = scores;
    state.metrics = buildMetrics();
    state.scoreThreshold = Math.floor(bestMetric().threshold);
    state.evalConfig.threshold = Number(bestMetric().threshold.toFixed(2));
    state.dashboardThreshold = Number(bestMetric().threshold.toFixed(2));
    document.getElementById("data-status").textContent = "Đã tải CSV thật";
    document.getElementById("data-status").className = "rounded-full border border-emerald-200 bg-mint-50 px-3 py-1 text-sm font-medium text-mint-600";
    render();
  } catch (error) {
    document.getElementById("data-status").textContent = "Không đọc được CSV";
    shell(
      "Không tải được dữ liệu",
      `
        <section class="rounded-xl border border-red-200 bg-red-50 p-6 text-red-800">
          <h3 class="text-lg font-semibold">Frontend cần được serve qua HTTP để đọc CSV.</h3>
          <p class="mt-2 text-sm">Chạy từ root repo:</p>
          <pre class="mt-4 rounded bg-red-950 p-4 text-sm text-white"><code>python -m http.server 5173</code></pre>
          <p class="mt-4 text-sm">Sau đó mở: <code>http://localhost:5173/frontend/</code></p>
          <p class="mt-4 text-sm">Chi tiết lỗi: ${error.message}</p>
        </section>
      `
    );
  }
}

init();
