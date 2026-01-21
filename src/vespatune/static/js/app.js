document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let ws = null;
    let trials = [];
    let bestValue = null;
    let columns = [];
    let chart = null;
    let currentMetric = 'value'; // default metric to plot
    let availableMetrics = new Set(['value']);
    let isTraining = false;

    // --- New State for Full App ---
    let runs = [];
    let activeRunId = null;   // The ID of the currently running process (if any)
    let selectedRunId = null; // The ID currently displayed in the main view (null = New Training)

    // --- DOM Elements ---
    const runListContainer = document.getElementById('run-list');
    const newTrainingBtn = document.getElementById('new-training-btn');
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

    const stopBtn = document.getElementById('stop-btn');
    const deleteBtn = document.getElementById('delete-btn');
    const startBtn = document.getElementById('start-btn');
    const metricSelect = document.getElementById('metric-select');

    // Summary Elements
    const summaryDiv = document.getElementById('training-summary');
    const sumTrainFile = document.getElementById('sum-train-file');
    const sumValidFile = document.getElementById('sum-valid-file');
    const sumTarget = document.getElementById('sum-target');
    const sumId = document.getElementById('sum-id');
    const sumModel = document.getElementById('sum-model');
    const sumProject = document.getElementById('sum-project');
    const sumTrials = document.getElementById('sum-trials');
    const sumTime = document.getElementById('sum-time');

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

    async function restoreFormFromConfig(config) {
        if (!config) return;

        try {
            // Restore file paths
            if (config.train_filename) {
                document.getElementById('train_filename').value = config.train_filename;
                // For secure temp files, we might not have original filename display, 
                // but we can try to show something indicative or just "Uploaded File"
                // Ideally, backend should persist original filename too. 
                // For now, let's use the path basename or just "Restored File".
                const display = config.train_filename.split('/').pop();
                document.getElementById('train_file_info').textContent = display;
                document.getElementById('train-upload').classList.add('uploaded');

                // Fetch valid columns for this file
                await fetchAndPopulateColumns(config.train_filename);
            }

            if (config.valid_filename) {
                document.getElementById('valid_filename').value = config.valid_filename;
                const display = config.valid_filename.split('/').pop();
                document.getElementById('valid_file_info').textContent = display;
                document.getElementById('valid-upload').classList.add('uploaded');
            }

            // Restore other inputs
            const inputs = form.querySelectorAll('input, select');
            inputs.forEach(input => {
                if (input.type !== 'file' && input.type !== 'hidden' && input.id && config[input.id] !== undefined) {
                    if (input.id !== 'target_columns' && input.id !== 'id_column') {
                        input.value = config[input.id];
                    }
                }
            });

            // Restore Dropdowns (after columns populated)
            // Parse target columns (stored as semicolon separated string in TrainRequest/DB)
            if (config.target_columns) {
                const targets = config.target_columns.split(';').map(s => s.trim());
                Array.from(targetSelect.options).forEach(opt => {
                    opt.selected = targets.includes(opt.value);
                });
            }
            if (config.id_column) {
                idSelect.value = config.id_column;
            }

        } catch (e) {
            console.error('Error restoring config', e);
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
        // Check if we're viewing the active run (not just if isTraining, since we might load an active run from DB)
        if (!activeRunId || selectedRunId !== activeRunId) return;

        try {
            await fetch('/stop', { method: 'POST' });
            appendLog('Stopping training...', 'warning');
            stopBtn.disabled = true;
            stopBtn.textContent = 'Stopping...';

            // Update UI state
            isTraining = false;

            // Refresh runs to get updated status
            setTimeout(async () => {
                await fetchRuns();
                // Re-select to refresh the view
                if (selectedRunId === activeRunId) {
                    selectRun(selectedRunId);
                }
            }, 1000);
        } catch (e) {
            console.error(e);
        }
    });

    deleteBtn.addEventListener('click', async () => {
        if (!selectedRunId) return;
        if (!confirm('Are you sure you want to delete this run? This action cannot be undone.')) return;

        try {
            const resp = await fetch(`/runs/${selectedRunId}`, { method: 'DELETE' });
            if (resp.ok) {
                // Remove from sidebar
                await fetchRuns();
                // Select active run or new training
                selectRun(activeRunId);
            } else {
                const data = await resp.json();
                alert(`Error deleting run: ${data.detail}`);
            }
        } catch (e) {
            console.error(e);
            alert('Failed to delete run');
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
            form.style.display = 'none';
            summaryDiv.style.display = 'block';
            updateSummary();
            stopBtn.disabled = false;
            stopBtn.textContent = 'Stop Training';
        } else {
            form.style.display = 'block';
            summaryDiv.style.display = 'none';
        }
    }

    function updateSummary() {
        const trainInfo = document.getElementById('train_file_info').textContent;
        const validInfo = document.getElementById('valid_file_info').textContent;

        // Use form values (which should be restored by now)
        const trainVal = trainInfo !== 'Click to upload' ? trainInfo : (document.getElementById('train_filename').value || '--');
        const validVal = validInfo !== 'Click to upload' ? validInfo : (document.getElementById('valid_filename').value || '--');

        sumTrainFile.textContent = trainVal;
        sumValidFile.textContent = validVal;

        let targets = Array.from(targetSelect.selectedOptions).map(o => o.value).join(', ');
        sumTarget.textContent = targets || 'None';

        // Handle ID column
        let idVal = document.getElementById('id_column').value;
        sumId.textContent = idVal || 'None';

        const modelType = document.getElementById('model_type').value || '--';
        const taskType = document.getElementById('task').value || '--';
        sumModel.textContent = `${modelType} (${taskType})`;

        sumProject.textContent = document.getElementById('project_name').value || '--';

        const numTrials = document.getElementById('num_trials').value || '--';
        sumTrials.textContent = numTrials;

        const timeLimit = document.getElementById('time_limit').value || '--';
        sumTime.textContent = timeLimit ? `${timeLimit}s` : '--';
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
        // Check if trial already exists (deduplication)
        const existingIndex = trials.findIndex(t => t.number === data.number);
        if (existingIndex !== -1) {
            // Trial already exists, skip
            return;
        }

        // Store all metrics
        const trialData = {
            number: data.number,
            value: data.value,
            params: data.params,
            metrics: data.metrics || data.user_attrs || {},  // Handle both formats
            time: new Date()
        };

        // Add user_attrs keys to available metrics
        if (trialData.metrics) {
            Object.keys(trialData.metrics).forEach(k => availableMetrics.add(k));
            updateMetricDropdown();
        }

        trials.push(trialData);
        trials.sort((a, b) => a.number - b.number);

        // Update Global Best
        if (bestValue === null || data.value < bestValue) {
            bestValue = data.value;
            // bestParams.textContent = JSON.stringify(data.best_params, null, 2); // This might not be present in trial object
            if (data.best_params) bestParams.textContent = JSON.stringify(data.best_params, null, 2);
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



    // --- Sidebar & Run Management ---

    async function fetchRuns() {
        try {
            const resp = await fetch('/runs');
            const data = await resp.json();
            runs = data.runs;
            renderSidebar();
        } catch (e) { console.error('Error fetching runs:', e); }
    }

    function renderSidebar() {
        runListContainer.innerHTML = '';
        runs.forEach(run => {
            const item = document.createElement('div');
            item.className = `run-item ${selectedRunId === run.id ? 'active' : ''}`;
            item.onclick = () => selectRun(run.id);

            // Date formatting
            const date = new Date(run.created_at).toLocaleString('en-US', {
                month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
            });

            item.innerHTML = `
                <div class="run-status-icon ${run.status}"></div>
                <div class="run-info">
                    <span class="run-id">Run #${run.id}</span>
                    <span class="run-date">${date}</span>
                </div>
            `;
            runListContainer.appendChild(item);
        });
    }

    async function selectRun(runId) {
        selectedRunId = runId;
        renderSidebar(); // Update active class

        if (runId === null) {
            // New Training Mode
            form.style.display = 'block';
            summaryDiv.style.display = 'none';
            resetState();
            setStatus('Ready', '');

            if (activeRunId) {
                setStatus(`Run #${activeRunId} is running in background`, 'warning');
            }

            // Disconnect WS if we are not viewing the active run
            // Actually, we might want to keep it connected but ignore messages?
            // Or just close it.
            if (ws) ws.close();
            stopBtn.disabled = true;
        } else {
            // Viewing a Run
            await loadRunDetails(runId);
        }
    }

    async function loadRunDetails(runId) {
        try {
            // 1. Get Details
            const resp = await fetch(`/runs/${runId}`);
            if (!resp.ok) throw new Error('Run not found');
            const run = await resp.json();

            // 2. Hydrate Form/Summary
            await restoreFormFromConfig(run.config);
            updateSummary();

            // Switch to Summary View
            form.style.display = 'none';
            summaryDiv.style.display = 'block';

            // 3. Status & Interactivity
            const isThisActive = (run.status === 'running' || run.status === 'pending');

            if (isThisActive && run.id === activeRunId) {
                // Active run - show stop, hide delete
                stopBtn.style.display = 'inline-block';
                stopBtn.disabled = false;
                stopBtn.textContent = 'Stop Training';
                deleteBtn.style.display = 'none';

                setStatus('Training in progress...', 'running');
                connectWebSocket();
            } else {
                // Historical run - hide stop, show delete
                stopBtn.style.display = 'none';
                deleteBtn.style.display = 'inline-block';

                setStatus(`Run #${runId}: ${run.status}`, run.status);
                if (ws && ws.readyState === WebSocket.OPEN && selectedRunId !== activeRunId) {
                    ws.close();
                }
            }

            // 4. Load Trials & Chart
            await fetchAndDisplayRunTrials(runId);

        } catch (e) { console.error(e); }
    }

    async function fetchAndDisplayRunTrials(runId) {
        resetState();
        try {
            const resp = await fetch(`/runs/${runId}/trials`);
            const data = await resp.json();

            if (data.trials) {
                // Collect metrics first
                data.trials.forEach(t => {
                    if (t.user_attrs) {
                        Object.keys(t.user_attrs).forEach(k => {
                            if (!k.startsWith('mlflow')) availableMetrics.add(k);
                        });
                    }
                });
                updateMetricDropdown();

                data.trials.forEach(t => {
                    const trialObj = {
                        number: t.number,
                        value: t.value,
                        params: t.params,
                        metrics: t.user_attrs || {},
                        best_params: null
                    };
                    // Manually inject 'metrics' key if not present in original addTrial logic
                    // My previous addTrial logic expected `data` from WS which had `value`, `params` etc.
                    // My `addTrial` wrapper inside `fetchAndDisplayStudy` did mapping.
                    // I should reuse `addTrial` but it pushes to `trials` global list which is fine.
                    // But `addTrial` calls `updateChart` on every push. That's slow for bulk load.
                    // Better to just push to trials and update chart once?
                    // For now reusing `addTrial` is simpler code-wise even if slower.

                    // Actually `fetchAndDisplayStudy` logic was good. Let's copy it.
                    addTrial(trialObj);
                });

                // Best Value update (addTrial handles it incrementally, but we can double check)
                if (data.best_value !== undefined) {
                    bestValue = data.best_value;
                    bestScore.textContent = bestValue ? bestValue.toFixed(6) : '--';
                }
                if (data.best_params) {
                    bestParams.textContent = JSON.stringify(data.best_params, null, 2);
                }
            }
        } catch (e) { console.error(e); }
    }

    // --- Auto Load Current Study ---
    async function checkCurrentStudy() {
        // First, fetch list of runs
        await fetchRuns();

        // Check for ACTIVE run
        try {
            const resp = await fetch('/active_run_id');
            const data = await resp.json();

            if (data.active_run_id) {
                activeRunId = data.active_run_id;
                // If we have an active run, select it by default
                selectRun(activeRunId);
            } else {
                activeRunId = null;
                // If no active run, default to "New Training"
                selectRun(null);
            }
        } catch (e) {
            console.error('Error checking status:', e);
            selectRun(null);
        }
    }

    // Initialize
    initChart();
    // connectWebSocket(); // Don't auto connect globally anymore, selectRun handles it
    checkCurrentStudy();

    newTrainingBtn.addEventListener('click', () => selectRun(null));
});
