/**
 * FraudShield AI - Client Dashboard Logic
 * =======================================
 * Handles frontend interactivity, preset demo simulations, REST API communication
 * with Flask, risk score visualizations, and dynamic table rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
    console.log("[FraudShield AI] Dashboard initializing...");
    initApp();
});

let currentFilter = "all";

function initApp() {
    // Check initial system connectivity
    checkSystemHealth();
    
    // Load KPI dashboard metrics
    loadDashboardStats();

    // Load recent transaction history from SQLite
    loadTransactionHistory("all");

    // Setup interactive event listeners
    setupFormEvents();
    setupPresetButtons();
    setupTableFilters();
    setupUtilityButtons();
}

/**
 * Checks system health via /api/health
 */
async function checkSystemHealth() {
    const sysDot = document.getElementById("sys-dot");
    const sysTitle = document.getElementById("sys-title");
    const sysEngine = document.getElementById("sys-engine");

    try {
        const response = await fetch("/api/health");
        if (response.ok) {
            const data = await response.json();
            if (sysDot) {
                sysDot.className = "status-indicator-dot online";
            }
            if (sysTitle) sysTitle.textContent = "System Active";
            if (sysEngine) sysEngine.textContent = data.components?.algorithm || "Isolation Forest Online";
        } else {
            markSystemOffline();
        }
    } catch (error) {
        markSystemOffline();
    }
}

function markSystemOffline() {
    const sysDot = document.getElementById("sys-dot");
    const sysTitle = document.getElementById("sys-title");
    const sysEngine = document.getElementById("sys-engine");

    if (sysDot) sysDot.className = "status-indicator-dot offline";
    if (sysTitle) sysTitle.textContent = "Backend Offline";
    if (sysEngine) sysEngine.textContent = "Start Flask via python run.py";
}

/**
 * Loads aggregated metrics via /api/stats
 */
async function loadDashboardStats() {
    try {
        const response = await fetch("/api/stats");
        if (!response.ok) return;

        const result = await response.json();
        if (result.status === "success" && result.data) {
            const stats = result.data;

            const elTotal = document.getElementById("kpi-total-tx");
            const elAnomalies = document.getElementById("kpi-total-anomalies");
            const elRate = document.getElementById("kpi-anomaly-rate");
            const elAvgRisk = document.getElementById("kpi-avg-risk");
            const elHighRisk = document.getElementById("kpi-high-risk");

            if (elTotal) elTotal.textContent = stats.total_monitored.toLocaleString();
            if (elAnomalies) elAnomalies.textContent = stats.total_anomalies.toLocaleString();
            if (elRate) elRate.textContent = `${stats.anomaly_rate}% anomaly rate`;
            if (elAvgRisk) elAvgRisk.textContent = `${stats.avg_risk_score} / 100`;
            if (elHighRisk) elHighRisk.textContent = stats.high_risk_count.toLocaleString();
        }
    } catch (error) {
        console.warn("[FraudShield AI] Could not load dashboard stats:", error);
    }
}

/**
 * Loads transaction history via /api/transactions
 */
async function loadTransactionHistory(filter = "all") {
    const tbody = document.getElementById("table-body");
    if (!tbody) return;

    try {
        const response = await fetch(`/api/transactions?filter=${filter}&limit=40`);
        if (!response.ok) throw new Error("Failed to load records");

        const result = await response.json();
        const records = result.data || [];

        if (records.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center py-4 text-muted">
                        No transactions found for filter '${filter}'.
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = records.map(tx => {
            const isAnomaly = tx.is_anomaly === 1;
            const badgeClass = isAnomaly ? "badge-anomaly-flag" : "badge-normal-flag";
            const verdictText = isAnomaly ? "🚨 Suspected Fraud" : "✅ Normal";
            
            const riskTier = tx.risk_level || (tx.risk_score >= 70 ? "High" : tx.risk_score >= 40 ? "Medium" : "Low");
            const riskColorClass = riskTier === "High" ? "text-danger" : riskTier === "Medium" ? "text-amber" : "text-success";

            const formattedAmount = `$${parseFloat(tx.amount || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            const locText = tx.is_unusual_location === 1 ? "⚠️ Foreign / Unknown" : "Verified Local";

            return `
                <tr>
                    <td class="td-id">${escapeHtml(tx.transaction_id)}</td>
                    <td>${escapeHtml(tx.timestamp || "--")}</td>
                    <td><span class="code-tag">${escapeHtml(tx.transaction_type || "PAYMENT")}</span></td>
                    <td class="td-amount">${formattedAmount}</td>
                    <td>${tx.transaction_hour !== undefined ? tx.transaction_hour + ":00" : "--"}</td>
                    <td>${tx.transaction_frequency || 1} txn/24h</td>
                    <td>${locText}</td>
                    <td><strong class="${riskColorClass}">${tx.risk_score !== undefined ? tx.risk_score : "--"}</strong></td>
                    <td><span class="table-badge ${badgeClass}">${verdictText}</span></td>
                </tr>
            `;
        }).join("");

    } catch (error) {
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center py-4 text-danger">
                    Error loading transaction records from database: ${escapeHtml(error.message)}
                </td>
            </tr>
        `;
    }
}

/**
 * Handles transaction submission to /api/predict
 */
function setupFormEvents() {
    const form = document.getElementById("transaction-form");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const amountInput = document.getElementById("tx-amount");
        const typeInput = document.getElementById("tx-type");
        const hourInput = document.getElementById("tx-hour");
        const freqInput = document.getElementById("tx-frequency");
        const locRadio = document.querySelector('input[name="is_unusual_location"]:checked');

        const amount = parseFloat(amountInput.value);
        const hour = parseInt(hourInput.value);
        const freq = parseInt(freqInput.value);
        const isUnusual = locRadio ? parseInt(locRadio.value) : 0;
        const txType = typeInput ? typeInput.value : "PAYMENT";

        // Client-side validations
        if (isNaN(amount) || amount <= 0) {
            alert("Please enter a valid transaction amount greater than $0.00");
            amountInput.focus();
            return;
        }

        if (isNaN(hour) || hour < 0 || hour > 23) {
            alert("Transaction hour must be between 0 and 23.");
            hourInput.focus();
            return;
        }

        if (isNaN(freq) || freq < 1) {
            alert("Transaction frequency must be at least 1.");
            freqInput.focus();
            return;
        }

        // UI Loading state
        const btnSubmit = document.getElementById("btn-submit-analyze");
        const spinner = document.getElementById("btn-spinner");
        const btnText = document.getElementById("btn-text");

        if (btnSubmit) btnSubmit.disabled = true;
        if (spinner) spinner.classList.remove("d-none");
        if (btnText) btnText.textContent = "Evaluating with Isolation Forest...";

        try {
            const payload = {
                amount: amount,
                transaction_type: txType,
                transaction_hour: hour,
                transaction_frequency: freq,
                is_unusual_location: isUnusual,
                location_label: isUnusual === 1 ? "Foreign / Unknown IP" : "Home / Verified Device"
            };

            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok || result.status !== "success") {
                throw new Error(result.message || "Prediction evaluation failed");
            }

            // Display results
            renderAIResult(result);

            // Refresh table & KPIs
            loadDashboardStats();
            loadTransactionHistory(currentFilter);

        } catch (err) {
            alert(`Analysis Error: ${err.message}`);
        } finally {
            if (btnSubmit) btnSubmit.disabled = false;
            if (spinner) spinner.classList.add("d-none");
            if (btnText) btnText.textContent = "Run Isolation Forest Analysis";
        }
    });

    // Hour input dynamic hint
    const hourInput = document.getElementById("tx-hour");
    const hourHint = document.getElementById("hour-hint");
    if (hourInput && hourHint) {
        hourInput.addEventListener("input", () => {
            const h = parseInt(hourInput.value);
            if (h >= 0 && h <= 5) {
                hourHint.textContent = `${h}:00 (Overnight - High Anomaly Window)`;
                hourHint.style.color = "#f87171";
            } else if (h >= 6 && h <= 21) {
                hourHint.textContent = `${h}:00 (Normal daytime / evening business hours)`;
                hourHint.style.color = "#94a3b8";
            } else {
                hourHint.textContent = `${h}:00 (Late night hours)`;
                hourHint.style.color = "#fbbf24";
            }
        });
    }

    // Radio card visual toggles
    const locRadios = document.querySelectorAll('input[name="is_unusual_location"]');
    locRadios.forEach(radio => {
        radio.addEventListener("change", updateRadioCardStyles);
    });
    updateRadioCardStyles();
}

function updateRadioCardStyles() {
    const locNormalCard = document.getElementById("label-loc-normal");
    const locUnusualCard = document.getElementById("label-loc-unusual");
    const normalRadio = document.getElementById("loc-normal");
    const unusualRadio = document.getElementById("loc-unusual");

    if (normalRadio && normalRadio.checked) {
        if (locNormalCard) locNormalCard.classList.add("active-normal");
        if (locUnusualCard) locUnusualCard.classList.remove("active-unusual");
    } else if (unusualRadio && unusualRadio.checked) {
        if (locNormalCard) locNormalCard.classList.remove("active-normal");
        if (locUnusualCard) locUnusualCard.classList.add("active-unusual");
    }
}

/**
 * Visualizes the AI evaluation outcome
 */
function renderAIResult(data) {
    const placeholder = document.getElementById("result-placeholder");
    const displayArea = document.getElementById("result-display-area");
    const statusChip = document.getElementById("result-status-chip");

    if (placeholder) placeholder.classList.add("d-none");
    if (displayArea) displayArea.classList.remove("d-none");

    const isAnomaly = data.is_anomaly;
    const riskScore = parseFloat(data.risk_score || 0);
    const riskLevel = data.risk_level || (riskScore >= 70 ? "High" : riskScore >= 40 ? "Medium" : "Low");

    // Verdict Banner
    const banner = document.getElementById("verdict-banner");
    const verdictTitle = document.getElementById("verdict-title");
    const verdictIcon = document.getElementById("verdict-icon");
    const txIdDisplay = document.getElementById("verdict-tx-id");
    const riskBadge = document.getElementById("verdict-risk-badge");

    if (txIdDisplay) txIdDisplay.textContent = `ID: ${data.transaction_id} • $${parseFloat(data.amount).toLocaleString(undefined, { minimumFractionDigits: 2 })}`;

    if (banner) {
        banner.className = `verdict-banner banner-${riskLevel.toLowerCase()}`;
    }

    if (verdictTitle && verdictIcon && riskBadge) {
        if (riskLevel === "High") {
            verdictIcon.textContent = "🚨";
            verdictTitle.textContent = "Anomalous Transaction Detected!";
            riskBadge.className = "risk-badge badge-high";
            riskBadge.textContent = "HIGH RISK";
            if (statusChip) {
                statusChip.textContent = "Flagged Anomaly";
                statusChip.style.color = "#ef4444";
            }
        } else if (riskLevel === "Medium") {
            verdictIcon.textContent = "⚠️";
            verdictTitle.textContent = "Elevated Risk - Review Recommended";
            riskBadge.className = "risk-badge badge-medium";
            riskBadge.textContent = "MEDIUM RISK";
            if (statusChip) {
                statusChip.textContent = "Under Observation";
                statusChip.style.color = "#f59e0b";
            }
        } else {
            verdictIcon.textContent = "🛡️";
            verdictTitle.textContent = "Legitimate Transaction Verified";
            riskBadge.className = "risk-badge badge-low";
            riskBadge.textContent = "LOW RISK";
            if (statusChip) {
                statusChip.textContent = "Verified Normal";
                statusChip.style.color = "#10b981";
            }
        }
    }

    // Score meter progress bar
    const scoreVal = document.getElementById("display-risk-score");
    const progressBar = document.getElementById("score-progress-bar");

    if (scoreVal) scoreVal.textContent = riskScore.toFixed(1);
    if (progressBar) {
        progressBar.style.width = `${Math.min(100, Math.max(2, riskScore))}%`;
        progressBar.className = `progress-bar progress-${riskLevel.toLowerCase()}`;
    }

    // Mathematical metrics
    const metricDecision = document.getElementById("metric-decision-score");
    const metricTreeVerdict = document.getElementById("metric-tree-verdict");

    if (metricDecision) metricDecision.textContent = data.anomaly_decision_score !== undefined ? data.anomaly_decision_score : "--";
    if (metricTreeVerdict) {
        metricTreeVerdict.textContent = isAnomaly ? "Outlier (Isolated Early)" : "Inlier (Deep Path)";
        metricTreeVerdict.className = `metric-val ${isAnomaly ? "text-danger" : "text-success"}`;
    }

    // Explanations list
    const explanationsList = document.getElementById("explanations-list");
    if (explanationsList) {
        const explanations = data.explanations || [];
        explanationsList.innerHTML = explanations.map(exp => {
            const isDanger = exp.toLowerCase().includes("high") || 
                             exp.toLowerCase().includes("unusual") || 
                             exp.toLowerCase().includes("spike") ||
                             exp.toLowerCase().includes("mismatch");
            return `<li class="${isDanger ? 'factor-danger' : ''}">${escapeHtml(exp)}</li>`;
        }).join("");
    }

    // Disclaimer
    const disclaimerEl = document.getElementById("display-disclaimer");
    if (disclaimerEl && data.disclaimer) {
        disclaimerEl.textContent = data.disclaimer;
    }
}

/**
 * Quick scenario preset fill buttons
 */
function setupPresetButtons() {
    const btnNormal = document.getElementById("preset-normal");
    const btnShopping = document.getElementById("preset-shopping");
    const btnFraud = document.getElementById("preset-fraud");
    const btnVelocity = document.getElementById("preset-velocity");

    if (btnNormal) {
        btnNormal.addEventListener("click", () => {
            fillForm(28.50, "PAYMENT", 13, 2, 0);
        });
    }

    if (btnShopping) {
        btnShopping.addEventListener("click", () => {
            fillForm(420.00, "PAYMENT", 19, 3, 0);
        });
    }

    if (btnFraud) {
        btnFraud.addEventListener("click", () => {
            fillForm(88000.00, "TRANSFER", 3, 22, 1);
        });
    }

    if (btnVelocity) {
        btnVelocity.addEventListener("click", () => {
            fillForm(95.00, "PAYMENT", 2, 32, 1);
        });
    }
}

function fillForm(amount, type, hour, freq, isUnusual) {
    const elAmount = document.getElementById("tx-amount");
    const elType = document.getElementById("tx-type");
    const elHour = document.getElementById("tx-hour");
    const elFreq = document.getElementById("tx-frequency");
    const elLocNormal = document.getElementById("loc-normal");
    const elLocUnusual = document.getElementById("loc-unusual");

    if (elAmount) elAmount.value = amount.toFixed(2);
    if (elType) elType.value = type;
    if (elHour) {
        elHour.value = hour;
        elHour.dispatchEvent(new Event("input"));
    }
    if (elFreq) elFreq.value = freq;

    if (isUnusual === 1 && elLocUnusual) {
        elLocUnusual.checked = true;
    } else if (elLocNormal) {
        elLocNormal.checked = true;
    }

    updateRadioCardStyles();
}

/**
 * Table filter tab listeners
 */
function setupTableFilters() {
    const tabs = document.querySelectorAll(".filter-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", (e) => {
            tabs.forEach(t => t.classList.remove("active"));
            e.currentTarget.classList.add("active");
            currentFilter = e.currentTarget.getAttribute("data-filter") || "all";
            loadTransactionHistory(currentFilter);
        });
    });
}

/**
 * Utility button actions (Refresh, Current Hour)
 */
function setupUtilityButtons() {
    const btnRefresh = document.getElementById("btn-refresh-all");
    if (btnRefresh) {
        btnRefresh.addEventListener("click", () => {
            checkSystemHealth();
            loadDashboardStats();
            loadTransactionHistory(currentFilter);
        });
    }

    const btnTimeNow = document.getElementById("btn-time-now");
    const hourInput = document.getElementById("tx-hour");
    if (btnTimeNow && hourInput) {
        btnTimeNow.addEventListener("click", () => {
            const currentHour = new Date().getHours();
            hourInput.value = currentHour;
            hourInput.dispatchEvent(new Event("input"));
        });
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
