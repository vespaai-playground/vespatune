document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let ws = null;
    let trials = [];
    let bestValue = null;
    let columns = [];
    let chart = null;
    let currentMetric = 'value';
    let availableMetrics = new Set(['value']);
    let isTraining = false;

    // --- DOM Elements ---
    const form = document.getElementById('train-form');
    const logsDiv = document.getElementById('ws-logs');
    const trialList = document.getElementById('trial-list');
    const statusText = document.getElementById('training-status');
    const connectionStatus = document.getElementById('connection-status');
    const trialCount = document.getElementById('trial-count');
    const bestScore = document.getElementById('best-score');
    const latestScore = document.getElementById('latest-score');
    const historyCount = document.getElementById('history-count');
    const bestParams = document.getElementById('best-params');
    const progressInfo = document.getElementById('progress-info');
    const chartPlaceholder = document.getElementById('chart-placeholder');
    const targetSelect = document.getElementById('target_columns');
    const idSelect = document.getElementById('id_column');
    const loadStudyBtn = document.getElementById('load-study-btn');
    const dbPathInput = document.getElementById('db_path');
    const stopBtn = document.getElementById('stop-btn');
    const startBtn = document.getElementById('start-btn');
    const metricSelect = document.getElementById('metric-select');

    // Modal Elements
    const modal = document.getElementById('trial-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalMetrics = document.getElementById('modal-metrics');
    const modalParams = document.getElementById('modal-params');
    const closeModal = document.querySelector('.close-modal');

    // --- File Uploads with Column Detection ---
    function setupUpload(boxId, inputId, hiddenId, statusId, isTrainFile) {
        const box = document.getElementById(boxId);
        const input = document.getElementById(inputId);
        const hidden = document.getElementById(hiddenId);
        const status = document.getElementById(statusId);

        box.addEventListener('click', () => input.click());

        input.addEventListener('change', async () => {
            if (input.files.length === 0) return;

            const file = input.files[0];
            const formData = new FormData();
            formData.append('file', file);

            status.textContent = 'Uploading...';
            box.classList.remove('uploaded');

            try {
                const resp = await fetch('/upload', { method: 'POST', body: formData });
                const data = await resp.json();

                hidden.value = data.path;
                status.textContent = file.name;
                box.classList.add('uploaded');

                // Populate column dropdowns from train file
                if (isTrainFile && data.columns) {
                    columns = data.columns;
                    populateColumnDropdowns(columns);
                }
                saveConfig(); // Save state after upload
            } catch (e) {
                status.textContent = 'Upload failed';
                console.error(e);
            }
        });
    }

    function populateColumnDropdowns(cols) {
        // Target columns (multi-select)
        targetSelect.innerHTML = '';
        cols.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            // Auto-select common target column names
            if (['target', 'label', 'y', 'class', 'outcome'].includes(col.toLowerCase())) {
                opt.selected = true;
            }
            targetSelect.appendChild(opt);
        });

        // ID column (single select with None option)
        idSelect.innerHTML = '<option value="">None</option>';
        cols.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            // Auto-select common ID column names
            if (['id', 'index', 'row_id', 'sample_id'].includes(col.toLowerCase())) {
                opt.selected = true;
            }
            idSelect.appendChild(opt);
        });
    }

    setupUpload('train-upload', 'train_file', 'train_filename', 'train_file_info', true);
    setupUpload('valid-upload', 'valid_file', 'valid_filename', 'valid_file_info', false);

    // --- Configuration Persistence ---
    function saveConfig() {
        const config = {
            train_filename: document.getElementById('train_filename').value,
            valid_filename: document.getElementById('valid_filename').value,
            train_file_info: document.getElementById('train_file_info').textContent,
            valid_file_info: document.getElementById('valid_file_info').textContent,
            target_columns: Array.from(targetSelect.selectedOptions).map(o => o.value),
            id_column: idSelect.value,
        };

        // Save other inputs
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            if (input.type !== 'file' && input.type !== 'hidden' && input.id) {
                config[input.id] = input.value;
            }
        });

        localStorage.setItem('vespatune_config', JSON.stringify(config));
    }

    async function loadConfig() {
        const configStr = localStorage.getItem('vespatune_config');
        if (!configStr) return;

        try {
            const config = JSON.parse(configStr);

            // Restore file paths
            if (config.train_filename) {
                document.getElementById('train_filename').value = config.train_filename;
                const info = document.getElementById('train_file_info');
                info.textContent = config.train_file_info || config.train_filename;
                if (config.train_filename) document.getElementById('train-upload').classList.add('uploaded');

                // Fetch valid columns for this file
                await fetchAndPopulateColumns(config.train_filename);
            }

            if (config.valid_filename) {
                document.getElementById('valid_filename').value = config.valid_filename;
                const info = document.getElementById('valid_file_info');
                info.textContent = config.valid_file_info || config.valid_filename;
                if (config.valid_filename) document.getElementById('valid-upload').classList.add('uploaded');
            }

            // Restore other inputs
            const inputs = form.querySelectorAll('input, select');
            inputs.forEach(input => {
                if (input.type !== 'file' && input.type !== 'hidden' && input.id && config[input.id] !== undefined) {
                    // Skip special selects for now
                    if (input.id !== 'target_columns' && input.id !== 'id_column') {
                        input.value = config[input.id];
                    }
                }
            });

            // Restore Dropdowns (after columns populated)
            if (config.target_columns && Array.isArray(config.target_columns)) {
                Array.from(targetSelect.options).forEach(opt => {
                    opt.selected = config.target_columns.includes(opt.value);
                });
            }
            if (config.id_column) {
                idSelect.value = config.id_column;
            }

        } catch (e) {
            console.error('Error loading config', e);
        }
    }

    async function fetchAndPopulateColumns(path) {
        try {
            const resp = await fetch(`/columns?path=${encodeURIComponent(path)}`);
            const data = await resp.json();
            if (data.columns) {
                columns = data.columns;
                populateColumnDropdowns(columns);
            }
        } catch (e) { console.error(e); }
    }

    // Attach listeners
    form.addEventListener('change', saveConfig);
    // Also save after successful upload (hooked below)

    // --- Chart.js Setup ---
    function initChart() {
        const ctx = document.getElementById('optimization-chart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Metric Value',
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#1e293b',
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                        borderColor: '#334155',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    // --- WebSocket & Training ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (isTraining) return;

        // Reset state
        resetState();

        // Get selected targets
        const selectedTargets = Array.from(targetSelect.selectedOptions).map(o => o.value);
        if (selectedTargets.length === 0) {
            appendLog('Please select at least one target column', 'error');
            return;
        }

        connectWebSocket();
        startTraining(selectedTargets);
    });

    stopBtn.addEventListener('click', async () => {
        if (!isTraining) return;
        try {
            await fetch('/stop', { method: 'POST' });
            appendLog('Stopping training...', 'warning');
            stopBtn.disabled = true;
            stopBtn.textContent = 'Stopping...';
        } catch (e) {
            console.error(e);
        }
    });

    function connectWebSocket() {
        if (ws && ws.readyState === WebSocket.OPEN) return;

        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

        ws.onopen = () => {
            connectionStatus.textContent = 'Connected';
            connectionStatus.classList.add('connected');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleMessage(data);
        };

        ws.onclose = () => {
            connectionStatus.textContent = 'Disconnected';
            connectionStatus.classList.remove('connected');
        };

        ws.onerror = (e) => {
            appendLog('WebSocket error', 'error');
            console.error(e);
        };
    }

    async function startTraining(selectedTargets) {
        const formData = new FormData(form);
        const data = {};

        for (const [key, value] of formData.entries()) {
            if (key === 'target_columns') continue;
            const input = document.getElementById(key);
            if (input && input.type === 'number') {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        }

        data.target_columns = selectedTargets.join(';');

        try {
            const resp = await fetch('/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await resp.json();
            if (!resp.ok) {
                appendLog(`Error: ${result.detail || JSON.stringify(result)}`, 'error');
                setStatus('Error', 'error');
            } else {
                setTrainingState(true);
            }
        } catch (e) {
            appendLog(`Error: ${e}`, 'error');
            setStatus('Error', 'error');
        }
    }

    function handleMessage(data) {
        switch (data.type) {
            case 'status':
                setStatus(data.message, 'running');
                appendLog(data.message, 'info');
                break;

            case 'trial_complete':
                addTrial(data);
                break;

            case 'training_finished':
                setStatus('Completed', 'completed');
                appendLog('Training completed successfully!', 'info');
                setTrainingState(false);
                break;

            case 'info':
                appendLog(data.message, 'info');
                break;

            case 'error':
                setStatus('Error', 'error');
                appendLog(`Error: ${data.message}`, 'error');
                setTrainingState(false);
                break;
        }
    }

    function setTrainingState(active) {
        isTraining = active;
        if (active) {
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
            stopBtn.disabled = false;
            stopBtn.textContent = 'Stop';
        } else {
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        }
    }

    function resetState() {
        trials = [];
        bestValue = null;
        availableMetrics = new Set(['value']);
        updateMetricDropdown();
        clearTrialList();
        clearLogs();

        if (chart) {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.update();
        } else {
            initChart(); // Ensure chart is initialized
        }

        chartPlaceholder.style.display = 'flex';
        document.getElementById('optimization-chart').style.display = 'none';

        trialCount.textContent = '0';
        bestScore.textContent = '--';
        latestScore.textContent = '--';
        historyCount.textContent = '0';
        bestParams.textContent = 'No results yet';
    }

    // --- Metric Selection ---
    metricSelect.addEventListener('change', () => {
        currentMetric = metricSelect.value;
        updateChart();
    });

    function updateMetricDropdown() {
        // Keep current selection if valid
        const current = metricSelect.value;
        metricSelect.innerHTML = '<option value="value">Objective Value</option>';

        availableMetrics.forEach(m => {
            if (m === 'value') return;
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            metricSelect.appendChild(opt);
        });

        if (availableMetrics.has(current)) {
            metricSelect.value = current;
        }
    }

    function addTrial(data) {
        // Store all metrics
        const trialData = {
            number: data.number,
            value: data.value,
            params: data.params,
            metrics: data.user_attrs || {},
            time: new Date()
        };

        // Add user_attrs keys to available metrics
        if (data.user_attrs) {
            Object.keys(data.user_attrs).forEach(k => availableMetrics.add(k));
            updateMetricDropdown();
        }

        trials.push(trialData);
        trials.sort((a, b) => a.number - b.number);

        // Update Global Best
        if (bestValue === null || data.value < bestValue) {
            bestValue = data.value;
            bestParams.textContent = JSON.stringify(data.best_params, null, 2);
            bestScore.textContent = bestValue.toFixed(6);
        }

        latestScore.textContent = data.value.toFixed(6);
        trialCount.textContent = trials.length;
        historyCount.textContent = trials.length;

        // UI Updates
        updateChart();
        addTrialToList(trialData, data.value === bestValue);
    }

    function updateChart() {
        if (!chart) initChart();

        chartPlaceholder.style.display = 'none';
        const canvas = document.getElementById('optimization-chart');
        canvas.style.display = 'block';

        chart.data.labels = trials.map(t => `#${t.number}`);

        // Map data based on selected metric
        const dataPoints = trials.map(t => {
            if (currentMetric === 'value') return t.value;
            return t.metrics?.[currentMetric] ?? null;
        });

        chart.data.datasets[0].data = dataPoints;
        chart.data.datasets[0].label = currentMetric;
        chart.update();
    }

    function addTrialToList(t, isBest) {
        // Remove placeholder
        const placeholder = trialList.querySelector('.trial-placeholder');
        if (placeholder) placeholder.remove();

        const div = document.createElement('div');
        div.className = `trial-item${isBest ? ' best' : ''}`;
        div.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="trial-number">#${t.number}</span>
                <span class="trial-params">${Object.keys(t.params).length} params</span>
            </div>
            <span class="trial-value">${t.value?.toFixed(6)}</span>
        `;

        div.addEventListener('click', () => openModal(t));

        // Prepend to list
        trialList.prepend(div);

        // Update other best markers
        if (isBest) {
            Array.from(trialList.children).forEach(child => {
                if (child !== div) child.classList.remove('best');
            });
        }
    }

    // --- Modal Handling ---
    function openModal(trial) {
        modalTitle.textContent = `Trial #${trial.number}`;
        modalParams.textContent = JSON.stringify(trial.params, null, 2);

        // Format metrics
        let metricsText = `Objective Value: ${trial.value}\n`;
        if (trial.metrics) {
            Object.entries(trial.metrics).forEach(([k, v]) => {
                metricsText += `${k}: ${v}\n`;
            });
        }
        modalMetrics.textContent = metricsText;

        modal.style.display = 'flex';
    }

    closeModal.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    window.onclick = (event) => {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    };

    // --- Utility ---
    function setStatus(message, type) {
        statusText.textContent = message;
        statusText.className = `status-text ${type}`;
    }

    function appendLog(message, type = '') {
        const div = document.createElement('div');
        div.className = `log-line ${type}`;
        div.textContent = `> ${message}`;
        logsDiv.appendChild(div);
        logsDiv.scrollTop = logsDiv.scrollHeight;
    }

    function clearTrialList() {
        trialList.innerHTML = '<div class="trial-placeholder">Trials will appear here</div>';
    }

    function clearLogs() {
        logsDiv.innerHTML = '<div class="log-line info">Waiting for training...</div>';
    }

    // --- Load Past Study ---
    loadStudyBtn.addEventListener('click', loadStudy);

    async function loadStudy() {
        const dbPath = dbPathInput.value.trim();
        if (!dbPath) {
            appendLog('Please enter a database path', 'error');
            return;
        }
        await fetchAndDisplayStudy(dbPath);
    }

    async function fetchAndDisplayStudy(dbPath) {
        try {
            const resp = await fetch(`/study/vespatune?db_path=${encodeURIComponent(dbPath)}`);
            const data = await resp.json();

            if (data.error) {
                appendLog(`Error loading study: ${data.error}`, 'error');
                return;
            }

            resetState();

            // Add all trials
            data.trials.forEach(t => {
                const trialObj = {
                    number: t.number,
                    value: t.value,
                    params: t.params,
                    user_attrs: t.user_attrs,
                    best_params: data.best_params // This is global best, but effectively same for parsing
                };
                addTrial(trialObj);
            });

            appendLog(`Loaded study with ${data.trials.length} trials`, 'info');

        } catch (e) {
            appendLog(`Error: ${e}`, 'error');
        }
    }

    // --- Auto Load Current Study ---
    async function checkCurrentStudy() {
        try {
            const resp = await fetch('/current_study');
            const data = await resp.json();

            if (data.db_path) {
                dbPathInput.value = data.db_path;
                // Auto-load existing results
                await fetchAndDisplayStudy(data.db_path);
            }

            if (data.is_training) {
                setTrainingState(true);
                setStatus('Training in progress (Resumed)', 'running');
                connectWebSocket();
            }
        } catch (e) {
            console.error(e);
        }
    }

    // Initialize
    initChart();
    connectWebSocket(); // Connect immediately to catch logs
    checkCurrentStudy();
    loadConfig(); // Restore config
    setStatus('Ready', '');
});
