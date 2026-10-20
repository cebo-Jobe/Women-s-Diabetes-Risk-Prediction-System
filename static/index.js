// ============================================================
// STATE MANAGEMENT
// ============================================================
const state = {
    currentUser: null,
    currentTab: 'predictionTab'
};

// ============================================================
// DOM REFERENCES
// ============================================================
const DOM = {
    // Login
    loginPage: document.getElementById('loginPage'),
    mainPage: document.getElementById('mainPage'),
    loginEmail: document.getElementById('loginEmail'),
    loginPassword: document.getElementById('loginPassword'),
    loginBtn: document.getElementById('loginBtn'),
    loginError: document.getElementById('loginError'),
    regName: document.getElementById('regName'),
    regEmail: document.getElementById('regEmail'),
    regPassword: document.getElementById('regPassword'),
    registerBtn: document.getElementById('registerBtn'),
    registerError: document.getElementById('registerError'),
    authTabs: document.querySelectorAll('.auth-tab'),
    loginTab: document.getElementById('loginTab'),
    registerTab: document.getElementById('registerTab'),

    // Main
    userName: document.getElementById('userName'),
    logoutBtn: document.getElementById('logoutBtn'),
    retrainBtn: document.getElementById('retrainBtn'),
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabPanels: document.querySelectorAll('.tab-panel'),

    // Prediction
    predictBtn: document.getElementById('predictBtn'),
    calculateDpfBtn: document.getElementById('calculateDpfBtn'),
    diabetesPedigree: document.getElementById('diabetesPedigree'),
    pregnancies: document.getElementById('pregnancies'),
    glucose: document.getElementById('glucose'),
    bloodPressure: document.getElementById('bloodPressure'),
    skinThickness: document.getElementById('skinThickness'),
    insulin: document.getElementById('insulin'),
    bmi: document.getElementById('bmi'),
    age: document.getElementById('age'),
    parentDiabetes: document.getElementById('parent_diabetes'),
    grandparentDiabetes: document.getElementById('grandparent_diabetes'),
    siblingDiabetes: document.getElementById('sibling_diabetes'),
    cousinDiabetes: document.getElementById('cousin_diabetes'),

    // Results Modal
    resultsModal: document.getElementById('resultsModal'),
    resultTitle: document.getElementById('resultTitle'),
    resultPercentage: document.getElementById('resultPercentage'),
    resultHeader: document.getElementById('resultHeader'),
    resultDescription: document.getElementById('resultDescription'),
    riskBarFill: document.getElementById('riskBarFill'),
    riskBarMarker: document.getElementById('riskBarMarker'),
    resultSummary: document.getElementById('resultSummary'),
    recommendationList: document.getElementById('recommendationList'),
    closeResultsBtn: document.getElementById('closeResultsBtn'),
    closeResultsBtn2: document.getElementById('closeResultsBtn2'),

    // Performance
    accuracyValue: document.getElementById('accuracyValue'),
    precisionValue: document.getElementById('precisionValue'),
    recallValue: document.getElementById('recallValue'),
    f1Value: document.getElementById('f1Value'),
    featureImportanceChart: document.getElementById('featureImportanceChart'),
    confusionMatrixTable: document.getElementById('confusionMatrixTable'),
    confusionInterpretation: document.getElementById('confusionInterpretation'),

    // History
    historyBody: document.getElementById('historyBody'),
    refreshHistoryBtn: document.getElementById('refreshHistoryBtn'),

    // Profile
    profileName: document.getElementById('profileName'),
    profileEmail: document.getElementById('profileEmail'),
    statTotal: document.getElementById('statTotal'),
    statLow: document.getElementById('statLow'),
    statHigh: document.getElementById('statHigh'),
    statRisk: document.getElementById('statRisk'),
    exportBtn: document.getElementById('exportBtn'),

    // Loading
    loadingOverlay: document.getElementById('loadingOverlay')
};

// ============================================================
// RECOMMENDATIONS DATA
// ============================================================
const RECOMMENDATIONS = {
    low: [
        "Great news! Your diabetes risk is currently low.",
        "Continue maintaining a healthy lifestyle with regular exercise.",
        "Keep eating a balanced diet rich in vegetables, fruits, and whole grains.",
        "Stay hydrated by drinking plenty of water daily.",
        "Maintain 7-9 hours of quality sleep each night.",
        "Schedule regular health check-ups to monitor your health.",
        "Avoid smoking and limit alcohol consumption.",
        "Practice stress management techniques like meditation or yoga.",
        "Stay socially active and maintain strong relationships.",
        "Monitor your weight and maintain a healthy BMI.",
        "Keep up the good work with your current healthy habits!"
    ],
    high: [
        "Your assessment indicates a higher risk for diabetes.",
        "Please consult with a healthcare professional immediately.",
        "Consider getting an HbA1c test to confirm your diabetes status.",
        "Adopt a low-carb, high-fiber diet with portion control.",
        "Engage in at least 150 minutes of moderate exercise per week.",
        "Work towards achieving and maintaining a healthy weight.",
        "Limit sugary foods, drinks, and processed foods significantly.",
        "If prescribed, take medications as directed by your doctor.",
        "Monitor blood glucose levels regularly if recommended.",
        "Manage stress through relaxation techniques and adequate sleep.",
        "Join a diabetes education program or support group.",
        "Quit smoking and limit alcohol consumption immediately.",
        "Consider working with a nutritionist for a personalized meal plan."
    ]
};

// ============================================================
// API HELPER FUNCTIONS
// ============================================================
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method: method,
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' }
    };
    if (body) options.body = JSON.stringify(body);

    const response = await fetch(endpoint, options);
    const data = await response.json();
    return data;
}

function showLoading() { DOM.loadingOverlay.classList.remove('hidden'); }
function hideLoading() { DOM.loadingOverlay.classList.add('hidden'); }

// ============================================================
// AUTHENTICATION
// ============================================================
function switchAuthTab(tab) {
    DOM.authTabs.forEach(t => t.classList.remove('active'));
    document.querySelector(`.auth-tab[data-tab="${tab}"]`).classList.add('active');
    DOM.loginTab.classList.remove('active');
    DOM.registerTab.classList.remove('active');
    (tab === 'loginTab' ? DOM.loginTab : DOM.registerTab).classList.add('active');
}

DOM.authTabs.forEach(tab => {
    tab.addEventListener('click', () => switchAuthTab(tab.dataset.tab));
});

async function login() {
    // Lowercased client-side too, so it visibly matches what you typed
    // into Register -- the server normalizes as well, but keeping both
    // in sync avoids "why didn't autofill match" confusion.
    const email = DOM.loginEmail.value.trim().toLowerCase();
    const password = DOM.loginPassword.value.trim();

    if (!email || !password) {
        DOM.loginError.textContent = 'Please fill in all fields';
        DOM.loginError.classList.remove('hidden');
        return;
    }

    showLoading();
    DOM.loginError.classList.add('hidden');

    try {
        const result = await apiCall('/api/login', 'POST', { email, password });
        if (result.success) {
            state.currentUser = result.user;
            showMainApp();
        } else {
            DOM.loginError.textContent = result.message || 'Invalid email or password';
            DOM.loginError.classList.remove('hidden');
        }
    } catch (error) {
        DOM.loginError.textContent = 'Connection error. Please try again.';
        DOM.loginError.classList.remove('hidden');
    } finally {
        hideLoading();
    }
}

async function register() {
    const name = DOM.regName.value.trim();
    const email = DOM.regEmail.value.trim().toLowerCase();
    const password = DOM.regPassword.value.trim();

    if (!name || !email || !password) {
        DOM.registerError.textContent = 'Please fill in all fields';
        DOM.registerError.classList.remove('hidden');
        return;
    }
    if (password.length < 6) {
        DOM.registerError.textContent = 'Password must be at least 6 characters';
        DOM.registerError.classList.remove('hidden');
        return;
    }

    showLoading();
    DOM.registerError.classList.add('hidden');

    try {
        const result = await apiCall('/api/register', 'POST', { name, email, password });
        if (result.success) {
            alert('Registration successful! Please login.');
            DOM.regName.value = '';
            DOM.regEmail.value = '';
            DOM.regPassword.value = '';
            switchAuthTab('loginTab');
        } else {
            DOM.registerError.textContent = result.message || 'Registration failed';
            DOM.registerError.classList.remove('hidden');
        }
    } catch (error) {
        DOM.registerError.textContent = 'Connection error. Please try again.';
        DOM.registerError.classList.remove('hidden');
    } finally {
        hideLoading();
    }
}

DOM.loginBtn.addEventListener('click', login);
DOM.loginPassword.addEventListener('keydown', (e) => { if (e.key === 'Enter') login(); });
DOM.loginEmail.addEventListener('keydown', (e) => { if (e.key === 'Enter') login(); });

DOM.registerBtn.addEventListener('click', register);
DOM.regPassword.addEventListener('keydown', (e) => { if (e.key === 'Enter') register(); });

// ============================================================
// LOGOUT
// ============================================================
async function logout() {
    try {
        await apiCall('/api/logout', 'POST');
    } catch (error) {
        console.error('Logout error:', error);
    } finally {
        state.currentUser = null;
        showLoginPage();
    }
}
DOM.logoutBtn.addEventListener('click', logout);

// ============================================================
// PAGE SWITCHING
// ============================================================
function showLoginPage() {
    DOM.loginPage.classList.add('active');
    DOM.mainPage.classList.remove('active');
}

function showMainApp() {
    DOM.loginPage.classList.remove('active');
    DOM.mainPage.classList.add('active');
    if (state.currentUser) DOM.userName.textContent = state.currentUser.name;
    loadPerformanceData();
    loadHistory();
    loadProfile();
}

// ============================================================
// TAB SWITCHING
// ============================================================
DOM.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tabId) {
    DOM.tabBtns.forEach(b => b.classList.remove('active'));
    document.querySelector(`.tab-btn[data-tab="${tabId}"]`).classList.add('active');
    DOM.tabPanels.forEach(p => p.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    state.currentTab = tabId;

    if (tabId === 'historyTab') loadHistory();
    if (tabId === 'profileTab') loadProfile();
    if (tabId === 'performanceTab') loadPerformanceData();
}

// ============================================================
// DPF CALCULATION
// ============================================================
function calculateDPF() {
    const weights = { parent: 0.5, grandparent: 0.25, sibling: 0.5, cousin: 0.125 };
    const getValue = (id) => document.getElementById(id).value.toLowerCase() === 'yes' ? 1 : 0;

    const dpf = (
        weights.parent * getValue('parent_diabetes') +
        weights.grandparent * getValue('grandparent_diabetes') +
        weights.sibling * getValue('sibling_diabetes') +
        weights.cousin * getValue('cousin_diabetes')
    );

    DOM.diabetesPedigree.value = dpf.toFixed(3);
}

['parent_diabetes', 'grandparent_diabetes', 'sibling_diabetes', 'cousin_diabetes'].forEach(id => {
    document.getElementById(id).addEventListener('change', calculateDPF);
});
DOM.calculateDpfBtn.addEventListener('click', calculateDPF);
window.addEventListener('DOMContentLoaded', calculateDPF);

// ============================================================
// PREDICTION
// ============================================================
function familyFieldsValid() {
    // A real for-of loop, not a .forEach() -- `return` inside forEach's
    // callback only skipped that one iteration, it never actually
    // stopped makePrediction() from continuing to submit invalid data.
    const ids = ['parent_diabetes', 'grandparent_diabetes', 'sibling_diabetes', 'cousin_diabetes'];
    for (const id of ids) {
        const val = document.getElementById(id).value;
        if (!['yes', 'no'].includes(val)) {
            alert(`Please select 'Yes' or 'No' for ${id.replace('_diabetes', '')} diabetic status`);
            return false;
        }
    }
    return true;
}

async function makePrediction() {
    const userData = {
        pregnancies: parseFloat(DOM.pregnancies.value),
        glucose: parseFloat(DOM.glucose.value),
        bloodPressure: parseFloat(DOM.bloodPressure.value),
        skinThickness: parseFloat(DOM.skinThickness.value),
        insulin: parseFloat(DOM.insulin.value),
        bmi: parseFloat(DOM.bmi.value),
        diabetesPedigree: parseFloat(DOM.diabetesPedigree.value),
        age: parseFloat(DOM.age.value),
        parent_diabetes: DOM.parentDiabetes.value,
        grandparent_diabetes: DOM.grandparentDiabetes.value,
        sibling_diabetes: DOM.siblingDiabetes.value,
        cousin_diabetes: DOM.cousinDiabetes.value
    };

    for (const [key, value] of Object.entries(userData)) {
        if (key.endsWith('_diabetes')) continue;
        if (Number.isNaN(value) || value < 0) {
            alert(`Please enter a valid value for ${key}`);
            return;
        }
    }

    if (!familyFieldsValid()) return;

    if (!DOM.diabetesPedigree.value || Number.isNaN(parseFloat(DOM.diabetesPedigree.value))) {
        calculateDPF();
        userData.diabetesPedigree = parseFloat(DOM.diabetesPedigree.value) || 0;
    }

    showLoading();
    try {
        const result = await apiCall('/api/predict', 'POST', userData);
        if (result.success) {
            showResults(result);
        } else {
            alert('Prediction failed: ' + (result.message || 'Unknown error'));
        }
    } catch (error) {
        alert('Connection error. Please try again.');
        console.error(error);
    } finally {
        hideLoading();
    }
}
DOM.predictBtn.addEventListener('click', makePrediction);

// ============================================================
// RESULTS MODAL
// ============================================================
function showResults(data) {
    const { risk_percentage, risk_category, binary_prediction } = data;

    DOM.resultTitle.textContent = risk_category;
    DOM.resultPercentage.textContent = `${risk_percentage.toFixed(1)}%`;

    DOM.resultHeader.className = 'result-header';
    DOM.resultHeader.classList.add(risk_category === 'HIGH RISK' ? 'high-risk' : 'low-risk');

    DOM.resultDescription.textContent = risk_category === 'HIGH RISK'
        ? `Your diabetes risk is ${risk_percentage.toFixed(1)}%, which is considered high risk (40% or above).`
        : `Your diabetes risk is ${risk_percentage.toFixed(1)}%, which is considered low risk (below 40%).`;

    const percentage = Math.min(risk_percentage, 100);
    DOM.riskBarFill.style.width = `${percentage}%`;
    DOM.riskBarMarker.style.left = `${percentage}%`;

    DOM.resultSummary.textContent =
        `Risk Percentage: ${risk_percentage.toFixed(1)}%\nCategory: ${risk_category}\nBinary Prediction: ${binary_prediction === 1 ? 'Diabetes' : 'No Diabetes'}`;

    const recs = risk_category === 'HIGH RISK' ? RECOMMENDATIONS.high : RECOMMENDATIONS.low;
    DOM.recommendationList.innerHTML = '';
    recs.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        DOM.recommendationList.appendChild(li);
    });

    DOM.resultsModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeResults() {
    DOM.resultsModal.classList.add('hidden');
    document.body.style.overflow = '';
}
DOM.closeResultsBtn.addEventListener('click', closeResults);
DOM.closeResultsBtn2.addEventListener('click', closeResults);
DOM.resultsModal.addEventListener('click', (e) => { if (e.target === DOM.resultsModal) closeResults(); });

// ============================================================
// RETRAIN
// ============================================================
DOM.retrainBtn.addEventListener('click', async () => {
    if (!confirm('Retrain the model now? This will overwrite the currently saved model and may take a moment.')) {
        return;
    }
    showLoading();
    try {
        const result = await apiCall('/api/retrain', 'POST');
        if (result.success) {
            alert(result.message || 'Model retrained successfully.');
            loadPerformanceData();
        } else {
            alert('Retrain failed: ' + (result.message || 'Unknown error'));
        }
    } catch (error) {
        alert('Connection error while retraining.');
        console.error(error);
    } finally {
        hideLoading();
    }
});

// ============================================================
// PERFORMANCE TAB
// ============================================================
async function loadPerformanceData() {
    try {
        const metricsResult = await apiCall('/api/model/metrics');
        if (metricsResult.success) {
            const m = metricsResult.metrics;
            DOM.accuracyValue.textContent = (m.accuracy * 100).toFixed(1) + '%';
            DOM.precisionValue.textContent = (m.precision * 100).toFixed(1) + '%';
            DOM.recallValue.textContent = (m.recall * 100).toFixed(1) + '%';
            DOM.f1Value.textContent = (m.f1 * 100).toFixed(1) + '%';
            if (m.confusion_matrix) renderConfusionMatrix(m.confusion_matrix);
        }

        const impResult = await apiCall('/api/model/feature_importance');
        if (impResult.success) renderFeatureImportance(impResult);
    } catch (error) {
        console.error('Failed to load performance data:', error);
    }
}

// Feature Importance -- a plain Chart.js bar chart, no extra plugin needed.
let featureChartInstance = null;

function renderFeatureImportance(data) {
    const ctx = DOM.featureImportanceChart.getContext('2d');
    if (featureChartInstance) featureChartInstance.destroy();

    const colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];

    featureChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.features,
            datasets: [{
                label: 'Feature Importance',
                data: data.importances,
                backgroundColor: colors.slice(0, data.features.length),
                borderColor: colors.slice(0, data.features.length).map(c => c + 'cc'),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `Importance: ${(context.parsed.y * 100).toFixed(2)}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { callback: (value) => (value * 100).toFixed(0) + '%' }
                },
                x: { ticks: { font: { size: 11 } } }
            }
        }
    });
}

// Confusion Matrix -- rendered as a plain table instead of a Chart.js
// 'matrix' chart (that chart type needs the chartjs-chart-matrix
// plugin, which was never loaded, so it silently never rendered).
function renderConfusionMatrix(cm) {
    const [tn, fp] = cm[0];
    const [fn, tp] = cm[1];
    const max = Math.max(tn, fp, fn, tp, 1);

    const cellStyle = (value, isCorrect) => {
        const intensity = value / max;
        const bg = isCorrect
            ? `rgba(39, 174, 96, ${0.12 + intensity * 0.55})`
            : `rgba(231, 76, 60, ${0.12 + intensity * 0.55})`;
        return `background:${bg};`;
    };

    DOM.confusionMatrixTable.innerHTML = `
        <table class="cm-table">
            <tr><th></th><th>Predicted: No Diabetes</th><th>Predicted: Diabetes</th></tr>
            <tr>
                <th>Actual: No Diabetes</th>
                <td style="${cellStyle(tn, true)}">${tn}</td>
                <td style="${cellStyle(fp, false)}">${fp}</td>
            </tr>
            <tr>
                <th>Actual: Diabetes</th>
                <td style="${cellStyle(fn, false)}">${fn}</td>
                <td style="${cellStyle(tp, true)}">${tp}</td>
            </tr>
        </table>
    `;

    DOM.confusionInterpretation.textContent =
`Confusion Matrix Interpretation:
• True Negatives (TN): ${tn} - Correctly predicted no diabetes
• False Positives (FP): ${fp} - Incorrectly predicted diabetes
• False Negatives (FN): ${fn} - Missed diabetes cases
• True Positives (TP): ${tp} - Correctly predicted diabetes`;
}

// ============================================================
// HISTORY TAB
// ============================================================
async function loadHistory() {
    try {
        const result = await apiCall('/api/history');
        if (!result.success) {
            DOM.historyBody.innerHTML = `<tr><td colspan="8" class="no-data">${result.message || 'Error loading history'}</td></tr>`;
            return;
        }

        const history = result.history;
        DOM.historyBody.innerHTML = '';

        if (history.length === 0) {
            DOM.historyBody.innerHTML = '<tr><td colspan="8" class="no-data">No predictions yet</td></tr>';
            return;
        }

        history.forEach(record => {
            // risk_percentage / risk_category now come straight from the
            // real model (scored server-side) instead of a hardcoded
            // 60%/25% guess based only on the stored binary outcome.
            const category = record.risk_category || '--';
            const riskPct = typeof record.risk_percentage === 'number'
                ? record.risk_percentage.toFixed(1) + '%'
                : '--';

            const tr = document.createElement('tr');
            tr.className = category === 'HIGH RISK' ? 'high-risk' : 'low-risk';
            tr.innerHTML = `
                <td>${record.timestamp ? record.timestamp.substring(0, 16) : '--'}</td>
                <td>${record.pregnancies ?? 0}</td>
                <td>${record.glucose ?? 0}</td>
                <td>${record.blood_pressure ?? 0}</td>
                <td>${record.bmi ?? 0}</td>
                <td>${record.age ?? 0}</td>
                <td>${riskPct}</td>
                <td>${category}</td>
            `;
            DOM.historyBody.appendChild(tr);
        });
    } catch (error) {
        console.error('Failed to load history:', error);
        DOM.historyBody.innerHTML = '<tr><td colspan="8" class="no-data">Error loading history</td></tr>';
    }
}
DOM.refreshHistoryBtn.addEventListener('click', loadHistory);

// ============================================================
// PROFILE TAB
// ============================================================
async function loadProfile() {
    try {
        const result = await apiCall('/api/profile');
        if (result.success) {
            DOM.profileName.textContent = result.user.name;
            DOM.profileEmail.textContent = result.user.email;
            const stats = result.stats;
            DOM.statTotal.textContent = stats.total_predictions;
            DOM.statLow.textContent = stats.low_risk_count;
            DOM.statHigh.textContent = stats.high_risk_count;
            DOM.statRisk.textContent = stats.risk_percentage + '%';
        }
    } catch (error) {
        console.error('Failed to load profile:', error);
    }
}

// ============================================================
// EXPORT DATA
// ============================================================
async function exportData() {
    try {
        const result = await apiCall('/api/history');
        if (result.success && result.history.length > 0) {
            const headers = ['Date', 'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                              'Insulin', 'BMI', 'Diabetes Pedigree', 'Age', 'Predicted Outcome',
                              'Risk %', 'Risk Category'];
            const rows = result.history.map(r => [
                r.timestamp || '',
                r.pregnancies ?? 0,
                r.glucose ?? 0,
                r.blood_pressure ?? 0,
                r.skin_thickness ?? 0,
                r.insulin ?? 0,
                r.bmi ?? 0,
                r.diabetes_pedigree ?? 0,
                r.age ?? 0,
                r.predicted_outcome === 1 ? 'Diabetes' : 'No Diabetes',
                typeof r.risk_percentage === 'number' ? r.risk_percentage.toFixed(1) : '',
                r.risk_category || ''
            ]);

            let csv = headers.join(',') + '\n';
            rows.forEach(row => { csv += row.join(',') + '\n'; });

            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'diabetes_prediction_history.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } else {
            alert('No data to export');
        }
    } catch (error) {
        console.error('Export failed:', error);
        alert('Failed to export data');
    }
}
DOM.exportBtn.addEventListener('click', exportData);

// ============================================================
// CHECK SESSION ON LOAD
// ============================================================
async function checkSession() {
    try {
        const result = await apiCall('/api/check_session');
        if (result.logged_in) {
            state.currentUser = result.user;
            showMainApp();
        } else {
            showLoginPage();
        }
    } catch (error) {
        showLoginPage();
    }
}

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    DOM.pregnancies.value = 0;
    DOM.glucose.value = 100;
    DOM.bloodPressure.value = 70;
    DOM.skinThickness.value = 20;
    DOM.insulin.value = 80;
    DOM.bmi.value = 25.0;
    DOM.age.value = 30;

    DOM.parentDiabetes.value = 'no';
    DOM.grandparentDiabetes.value = 'no';
    DOM.siblingDiabetes.value = 'no';
    DOM.cousinDiabetes.value = 'no';

    calculateDPF();
    checkSession();
});

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !DOM.resultsModal.classList.contains('hidden')) {
        closeResults();
    }
});

console.log('Diabetes Risk Prediction System loaded successfully!');