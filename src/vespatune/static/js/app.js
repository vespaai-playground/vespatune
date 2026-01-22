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
    let runs = [];
    let activeRunId = null;
    let selectedRunId = null;

    // --- DOM Elements ---
    const runListContainer = document.getElementById('run-list');
    const newTrainingBtn = document.getElementById('new-training-btn');
    const form = document.getElementById('train-form');
    const trialList = document.getElementById('trial-list');
    const connectionStatus = document.getElementById('connection-status');
    const refreshMonitorBtn = document.getElementById('refresh-monitor-btn');
    const trialCount = document.getElementById('trial-count');
    const bestScore = document.getElementById('best-score');
    const latestScore = document.getElementById('latest-score');
    const historyCount = document.getElementById('history-count');
    const bestParams = document.getElementById('best-params');
    const chartPlaceholder = document.getElementById('chart-placeholder');
    const targetSelect = document.getElementById('target_columns');
    const idSelect = document.getElementById('id_column');
    const stopBtn = document.getElementById('stop-btn');
    const deleteBtn = document.getElementById('delete-btn');
    const metricSelect = document.getElementById('metric-select');
    const useIdColumnCheckbox = document.getElementById('use_id_column');

    // Summary Elements (in sidebar)
    const projectSummary = document.getElementById('project-summary');
    const summaryProjectName = document.getElementById('summary-project-name');
    const summaryStatus = document.getElementById('summary-status');
    const sumTrainFile = document.getElementById('sum-train-file');
    const sumValidFile = document.getElementById('sum-valid-file');
    const sumTarget = document.getElementById('sum-target');
    const sumModel = document.getElementById('sum-model');
    const sumTrials = document.getElementById('sum-trials');
    const sumTime = document.getElementById('sum-time');

    // New Project Modal
    const newProjectModal = document.getElementById('new-project-modal');
    const closeNewProjectBtn = document.getElementById('close-new-project');

    // Trial Modal
    const trialModal = document.getElementById('trial-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalMetrics = document.getElementById('modal-metrics');
    const modalParams = document.getElementById('modal-params');
    const closeTrialModalBtn = document.getElementById('close-trial-modal');

    // Artifacts Modal
    const artifactsModal = document.getElementById('artifacts-modal');
    const artifactsList = document.getElementById('artifacts-list');
    const closeArtifactsModalBtn = document.getElementById('close-artifacts-modal');
    const artifactsBtn = document.getElementById('artifacts-btn');

    // Initial state
    refreshMonitorBtn.disabled = false;
    artifactsBtn.disabled = true;

    // --- Utility Functions ---
    function generateProjectName() {
        const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
        const segments = [4, 4, 4];
        return segments.map(len =>
            Array.from({ length: len }, () => chars[Math.floor(Math.random() * chars.length)]).join('')
        ).join('-');
    }

    // --- Modal Functions ---
    function openNewProjectModal() {
        resetForm();
        newProjectModal.style.display = 'flex';
    }

    function closeNewProjectModal() {
        newProjectModal.style.display = 'none';
    }

    function resetForm() {
        form.reset();
        document.getElementById('train_filename').value = '';
        document.getElementById('valid_filename').value = '';
        document.getElementById('train_file_info').textContent = 'Train CSV';
        document.getElementById('valid_file_info').textContent = 'Valid CSV';
        document.getElementById('train-upload').classList.remove('uploaded');
        document.getElementById('valid-upload').classList.remove('uploaded');
        targetSelect.innerHTML = '<option value="" disabled>Upload CSV first</option>';
        idSelect.innerHTML = '<option value="" disabled>Select ID column</option>';
        useIdColumnCheckbox.checked = false;
        idSelect.disabled = true;
        columns = [];
        document.getElementById('project_name').value = generateProjectName();
    }

    newTrainingBtn.addEventListener('click', openNewProjectModal);
    closeNewProjectBtn.addEventListener('click', closeNewProjectModal);

    // Refresh button (in monitor header)
    refreshMonitorBtn.addEventListener('click', async () => {
        refreshMonitorBtn.style.opacity = '0.5';
        refreshMonitorBtn.disabled = true;
        await fetchRuns();
        if (selectedRunId) {
            await loadRunDetails(selectedRunId);
        }
        refreshMonitorBtn.style.opacity = '1';
        refreshMonitorBtn.disabled = false;
    });

    // Close modal on backdrop click
    newProjectModal.addEventListener('click', (e) => {
        if (e.target === newProjectModal) closeNewProjectModal();
    });

    // ID column checkbox toggle
    useIdColumnCheckbox.addEventListener('change', () => {
        idSelect.disabled = !useIdColumnCheckbox.checked;
        if (!useIdColumnCheckbox.checked) {
            idSelect.value = '';
        }
    });

    // --- File Uploads ---
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

                if (isTrainFile && data.columns) {
                    columns = data.columns;
                    populateColumnDropdowns(columns);
                }
            } catch (e) {
                status.textContent = 'Failed';
                console.error(e);
            }
        });
    }

    function populateColumnDropdowns(cols) {
        targetSelect.innerHTML = '';
        cols.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            if (['target', 'label', 'y', 'class', 'outcome'].includes(col.toLowerCase())) {
                opt.selected = true;
            }
            targetSelect.appendChild(opt);
        });

        idSelect.innerHTML = '';
        let hasAutoSelected = false;
        cols.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            // Auto-select and enable checkbox if common ID column found
            if (['id', 'index', 'row_id', 'sample_id'].includes(col.toLowerCase()) && !hasAutoSelected) {
                opt.selected = true;
                hasAutoSelected = true;
                useIdColumnCheckbox.checked = true;
                idSelect.disabled = false;
            }
            idSelect.appendChild(opt);
        });

        // If no auto-selection, ensure checkbox is unchecked and dropdown disabled
        if (!hasAutoSelected) {
            useIdColumnCheckbox.checked = false;
            idSelect.disabled = true;
        }
    }

    setupUpload('train-upload', 'train_file', 'train_filename', 'train_file_info', true);
    setupUpload('valid-upload', 'valid_file', 'valid_filename', 'valid_file_info', false);

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
                    borderColor: '#2a5446',
                    backgroundColor: 'rgba(42, 84, 70, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: '#2a5446',
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
                        backgroundColor: '#1a1a1a',
                        titleColor: '#ededed',
                        bodyColor: '#999',
                        borderColor: '#333',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#1a1a1a' },
                        ticks: { color: '#666', font: { size: 11 } }
                    },
                    y: {
                        grid: { color: '#1a1a1a' },
                        ticks: { color: '#666', font: { size: 11 } }
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

    // --- Form Submit & Training ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isTraining) return;

        const selectedTargets = Array.from(targetSelect.selectedOptions).map(o => o.value);
        if (selectedTargets.length === 0) {
            alert('Please select at least one target column');
            return;
        }

        // Close modal and start training
        closeNewProjectModal();

        // Connect WebSocket and wait for connection before starting
        await connectWebSocket();
        await startTraining(selectedTargets);
    });

    stopBtn.addEventListener('click', async () => {
        if (!activeRunId || selectedRunId !== activeRunId) return;

        try {
            await fetch('/stop', { method: 'POST' });
            stopBtn.disabled = true;
            stopBtn.textContent = 'Stopping...';
            isTraining = false;

            setTimeout(async () => {
                await fetchRuns();
                if (selectedRunId) {
                    await loadRunDetails(selectedRunId);
                }
            }, 1000);
        } catch (e) {
            console.error(e);
        }
    });

    deleteBtn.addEventListener('click', async () => {
        if (!selectedRunId) return;
        if (!confirm('Delete this run?')) return;

        try {
            const resp = await fetch(`/runs/${selectedRunId}`, { method: 'DELETE' });
            if (resp.ok) {
                await fetchRuns();
                // Select another run or hide summary
                if (runs.length > 0) {
                    selectRun(runs[0].id);
                } else {
                    selectedRunId = null;
                    projectSummary.style.display = 'none';
                    resetMonitorState();
                }
            }
        } catch (e) {
            console.error(e);
        }
    });

    function connectWebSocket() {
        return new Promise((resolve) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                resolve();
                return;
            }

            const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

            ws.onopen = () => {
                connectionStatus.classList.add('connected');
                connectionStatus.title = 'Connected';
                resolve();
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = () => {
                connectionStatus.classList.remove('connected');
                connectionStatus.title = 'Disconnected';
            };

            ws.onerror = (e) => {
                console.error(e);
                resolve(); // Resolve anyway to not block
            };
        });
    }

    async function startTraining(selectedTargets) {
        const formData = new FormData(form);
        const data = {};

        for (const [key, value] of formData.entries()) {
            if (key === 'target_columns') continue;
            if (key === 'id_column') continue; // Handle separately
            const input = document.getElementById(key);
            if (input && input.type === 'number') {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        }

        data.target_columns = selectedTargets.join(';');

        // Only include id_column if checkbox is checked
        if (useIdColumnCheckbox.checked && idSelect.value) {
            data.id_column = idSelect.value;
        } else {
            data.id_column = '';
        }

        try {
            const resp = await fetch('/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await resp.json();

            if (!resp.ok) {
                setSummaryStatus('Error', 'error');
            } else {
                isTraining = true;

                // Reset monitor state for new training
                resetMonitorState();
                artifactsBtn.disabled = false;
                setSummaryStatus('Running', 'running');

                // Fetch runs to get the new run
                await fetchRuns();

                // Find the newest run and set it as active/selected without full reload
                if (runs.length > 0) {
                    const newestRun = runs[0];
                    activeRunId = newestRun.id;
                    selectedRunId = newestRun.id;
                    renderSidebar();

                    // Show summary with form data (don't fetch from server to avoid reset)
                    projectSummary.style.display = 'block';
                    summaryProjectName.textContent = data.project_name || 'Project';
                    sumTrainFile.textContent = document.getElementById('train_file_info').textContent;
                    sumValidFile.textContent = document.getElementById('valid_file_info').textContent;
                    sumTarget.textContent = data.target_columns.replace(/;/g, ', ');
                    sumModel.textContent = `${data.model_type} (${data.task})`;
                    sumTrials.textContent = data.num_trials;
                    sumTime.textContent = `${data.time_limit}s`;

                    // Show stop button, hide delete
                    stopBtn.style.display = 'inline-block';
                    stopBtn.disabled = false;
                    stopBtn.textContent = 'Stop';
                    deleteBtn.style.display = 'none';
                }
            }
        } catch (e) {
            console.error(e);
        }
    }

    function handleMessage(data) {
        switch (data.type) {
            case 'status':
                setSummaryStatus(data.message, 'running');
                break;
            case 'trial_complete':
                addTrial(data);
                break;
            case 'training_finished':
                setSummaryStatus('Completed', 'completed');
                isTraining = false;
                fetchRuns(); // Refresh to update status
                break;
            case 'info':
                break;
            case 'error':
                setSummaryStatus('Error', 'error');
                isTraining = false;
                break;
        }
    }

    function updateSummary(config) {
        if (!config) return;

        summaryProjectName.textContent = config.project_name || 'Project';

        const trainFile = config.train_filename ? config.train_filename.split('/').pop() : '--';
        const validFile = config.valid_filename ? config.valid_filename.split('/').pop() : '--';

        sumTrainFile.textContent = trainFile;
        sumValidFile.textContent = validFile;
        sumTarget.textContent = config.target_columns ? config.target_columns.replace(/;/g, ', ') : '--';
        sumModel.textContent = config.model_type ? `${config.model_type} (${config.task || ''})` : '--';
        sumTrials.textContent = config.num_trials || '--';
        sumTime.textContent = config.time_limit ? `${config.time_limit}s` : '--';
    }

    function setSummaryStatus(message, type) {
        summaryStatus.textContent = message;
        summaryStatus.className = 'status-chip small';
        if (type) summaryStatus.classList.add(type);
    }

    function resetMonitorState() {
        trials = [];
        bestValue = null;
        availableMetrics = new Set(['value']);
        updateMetricDropdown();
        clearTrialList();

        if (chart) {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.update();
        } else {
            initChart();
        }

        chartPlaceholder.style.display = 'flex';
        document.getElementById('optimization-chart').style.display = 'none';

        trialCount.textContent = '0';
        bestScore.textContent = '--';
        latestScore.textContent = '--';
        historyCount.textContent = '0';
        // Only disable artifacts buttons, keep refresh enabled to allow reloading runs
        bestParams.textContent = 'No results yet';
        artifactsBtn.disabled = true;
    }

    metricSelect.addEventListener('change', () => {
        currentMetric = metricSelect.value;
        updateChart();
    });

    function updateMetricDropdown() {
        const current = metricSelect.value;
        metricSelect.innerHTML = '<option value="value">objective</option>';

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
        const existingIndex = trials.findIndex(t => t.number === data.number);
        if (existingIndex !== -1) return;

        const trialData = {
            number: data.number,
            value: data.value,
            params: data.params,
            metrics: data.metrics || data.user_attrs || {},
            time: new Date()
        };

        if (trialData.metrics) {
            Object.keys(trialData.metrics).forEach(k => availableMetrics.add(k));
            updateMetricDropdown();
        }

        trials.push(trialData);
        trials.sort((a, b) => a.number - b.number);

        if (bestValue === null || data.value < bestValue) {
            bestValue = data.value;
            if (data.best_params) bestParams.textContent = JSON.stringify(data.best_params, null, 2);
            bestScore.textContent = bestValue.toFixed(6);
        }

        latestScore.textContent = data.value.toFixed(6);
        trialCount.textContent = trials.length;
        historyCount.textContent = trials.length;

        updateChart();
        addTrialToList(trialData, data.value === bestValue);
    }

    function updateChart() {
        if (!chart) initChart();

        chartPlaceholder.style.display = 'none';
        const canvas = document.getElementById('optimization-chart');
        canvas.style.display = 'block';

        chart.data.labels = trials.map(t => `#${t.number}`);

        const dataPoints = trials.map(t => {
            if (currentMetric === 'value') return t.value;
            return t.metrics?.[currentMetric] ?? null;
        });

        chart.data.datasets[0].data = dataPoints;
        chart.data.datasets[0].label = currentMetric;
        chart.update();
    }

    function addTrialToList(t, isBest) {
        const empty = trialList.querySelector('.trial-empty');
        if (empty) empty.remove();

        const div = document.createElement('div');
        div.className = `trial-item${isBest ? ' best' : ''}`;
        div.innerHTML = `
            <span class="trial-number">#${t.number}</span>
            <span class="trial-value">${t.value?.toFixed(6)}</span>
        `;

        div.addEventListener('click', () => openTrialModal(t));
        trialList.prepend(div);

        if (isBest) {
            Array.from(trialList.children).forEach(child => {
                if (child !== div) child.classList.remove('best');
            });
        }
    }

    // --- Trial Modal ---
    function openTrialModal(trial) {
        modalTitle.textContent = `Trial #${trial.number}`;
        modalParams.textContent = JSON.stringify(trial.params, null, 2);

        let metricsText = `objective: ${trial.value}\n`;
        if (trial.metrics) {
            Object.entries(trial.metrics).forEach(([k, v]) => {
                metricsText += `${k}: ${v}\n`;
            });
        }
        modalMetrics.textContent = metricsText;

        trialModal.style.display = 'flex';
    }

    closeTrialModalBtn.addEventListener('click', () => {
        trialModal.style.display = 'none';
    });

    trialModal.addEventListener('click', (e) => {
        if (e.target === trialModal) trialModal.style.display = 'none';
    });

    // --- Artifacts Modal ---
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    async function openArtifactsModal() {
        if (!selectedRunId) return;

        artifactsList.innerHTML = '<div class="artifacts-empty">Loading...</div>';
        artifactsModal.style.display = 'flex';

        try {
            const response = await fetch(`/runs/${selectedRunId}/artifacts`);
            const data = await response.json();

            if (data.artifacts && data.artifacts.length > 0) {
                artifactsList.innerHTML = data.artifacts.map(artifact => `
                    <div class="artifact-item">
                        <div class="artifact-info">
                            <span class="artifact-name">${artifact.label}</span>
                            <span class="artifact-meta">
                                <span class="artifact-category">${artifact.category}</span>
                                ${formatFileSize(artifact.size)}
                            </span>
                        </div>
                        <button class="artifact-download" onclick="downloadArtifact(${selectedRunId}, '${artifact.path}', '${artifact.name}')">
                            Download
                        </button>
                    </div>
                `).join('');
            } else {
                artifactsList.innerHTML = '<div class="artifacts-empty">No artifacts available yet</div>';
            }
        } catch (error) {
            artifactsList.innerHTML = '<div class="artifacts-empty">Failed to load artifacts</div>';
        }
    }

    // Global function for download button onclick
    window.downloadArtifact = function (runId, path, name) {
        const url = `/runs/${runId}/artifacts/download?path=${encodeURIComponent(path)}`;
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    artifactsBtn.addEventListener('click', openArtifactsModal);

    closeArtifactsModalBtn.addEventListener('click', () => {
        artifactsModal.style.display = 'none';
    });

    artifactsModal.addEventListener('click', (e) => {
        if (e.target === artifactsModal) artifactsModal.style.display = 'none';
    });


    function clearTrialList() {
        trialList.innerHTML = '<div class="trial-empty">Trials will appear here</div>';
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
        if (runs.length === 0) {
            runListContainer.innerHTML = '<div class="run-item placeholder">No runs yet</div>';
            return;
        }
        runs.forEach(run => {
            const item = document.createElement('div');
            item.className = `run-item ${selectedRunId === run.id ? 'active' : ''}`;
            item.onclick = () => selectRun(run.id);

            const date = new Date(run.created_at).toLocaleString('en-US', {
                month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
            });

            item.innerHTML = `
                <div class="run-status-icon ${run.status}"></div>
                <div class="run-info">
                    <span class="run-id">${run.config?.project_name || `Run #${run.id}`}</span>
                    <span class="run-date">${date}</span>
                </div>
            `;
            runListContainer.appendChild(item);
        });
    }

    async function selectRun(runId) {
        selectedRunId = runId;
        renderSidebar();
        await loadRunDetails(runId);
    }

    async function loadRunDetails(runId) {
        try {
            const resp = await fetch(`/runs/${runId}`);
            if (!resp.ok) throw new Error('Run not found');
            const run = await resp.json();

            // Show summary in sidebar
            projectSummary.style.display = 'block';
            updateSummary(run.config);

            const isThisActive = (run.status === 'running' || run.status === 'pending');

            // Always enable buttons when a run is selected
            artifactsBtn.disabled = false;

            if (isThisActive) {
                activeRunId = run.id;
                stopBtn.style.display = 'inline-block';
                stopBtn.disabled = false;
                stopBtn.textContent = 'Stop';
                deleteBtn.style.display = 'none';
                setSummaryStatus('Running', 'running');
                await connectWebSocket();
            } else {
                stopBtn.style.display = 'none';
                deleteBtn.style.display = 'inline-block';
                const statusText = run.status.charAt(0).toUpperCase() + run.status.slice(1);
                setSummaryStatus(statusText, run.status);
                if (ws && ws.readyState === WebSocket.OPEN && selectedRunId !== activeRunId) {
                    ws.close();
                }
            }

            await fetchAndDisplayRunTrials(runId);
        } catch (e) { console.error(e); }
    }

    async function fetchAndDisplayRunTrials(runId) {
        resetMonitorState();
        artifactsBtn.disabled = false;
        try {
            const resp = await fetch(`/runs/${runId}/trials`);
            const data = await resp.json();

            if (data.trials && data.trials.length > 0) {
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
                    addTrial(trialObj);
                });

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

    async function checkCurrentStudy() {
        await fetchRuns();

        try {
            const resp = await fetch('/active_run_id');
            const data = await resp.json();

            if (data.active_run_id) {
                activeRunId = data.active_run_id;
                await selectRun(activeRunId);
            } else {
                activeRunId = null;
                // Select first run if exists, otherwise show empty state
                if (runs.length > 0) {
                    await selectRun(runs[0].id);
                } else {
                    projectSummary.style.display = 'none';
                }
            }
        } catch (e) {
            console.error('Error checking status:', e);
            if (runs.length > 0) {
                await selectRun(runs[0].id);
            }
        }
    }

    // Initialize
    initChart();
    checkCurrentStudy();
});
