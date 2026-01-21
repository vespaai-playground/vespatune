document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let ws = null;
    let trials = [];
    let bestValue = null;
    let columns = [];

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

    // --- WebSocket & Training ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Reset state
        trials = [];
        bestValue = null;
        updateMetrics();
        clearTrialList();
        clearLogs();
        clearChart();

        // Get selected targets
        const selectedTargets = Array.from(targetSelect.selectedOptions).map(o => o.value);
        if (selectedTargets.length === 0) {
            appendLog('Please select at least one target column', 'error');
            return;
        }

        // Connect WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

        ws.onopen = () => {
            connectionStatus.textContent = 'Connected';
            connectionStatus.classList.add('connected');
            setStatus('Starting...', 'running');
            startTraining(selectedTargets);
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
    });

    async function startTraining(selectedTargets) {
        const formData = new FormData(form);
        const data = {};

        for (const [key, value] of formData.entries()) {
            if (key === 'target_columns') continue; // Handle separately
            const input = document.getElementById(key);
            if (input && input.type === 'number') {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        }

        // Join targets with semicolon
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
                updateMetrics();
                updateChart();
                if (data.best_params) {
                    bestParams.textContent = JSON.stringify(data.best_params, null, 2);
                }
                break;

            case 'training_finished':
                setStatus('Completed', 'completed');
                appendLog('Training completed successfully!', 'info');
                loadStudyFromOutput();
                break;

            case 'error':
                setStatus('Error', 'error');
                appendLog(`Error: ${data.message}`, 'error');
                break;
        }
    }

    function addTrial(data) {
        trials.push({
            number: data.number,
            value: data.value,
            params: data.params,
            time: new Date()
        });

        if (bestValue === null || data.value < bestValue) {
            bestValue = data.value;
        }

        // Update trial list
        const isBest = data.value === bestValue;
        const div = document.createElement('div');
        div.className = `trial-item${isBest ? ' best' : ''}`;
        div.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="trial-number">#${data.number}</span>
                <span class="trial-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <span class="trial-value">${data.value?.toFixed(6)}</span>
        `;

        // Remove placeholder if exists
        const placeholder = trialList.querySelector('.trial-placeholder');
        if (placeholder) placeholder.remove();

        trialList.prepend(div);

        // Update best markers
        if (isBest) {
            trialList.querySelectorAll('.trial-item').forEach(item => {
                if (item !== div) item.classList.remove('best');
            });
        }
    }

    function updateMetrics() {
        trialCount.textContent = trials.length;
        historyCount.textContent = trials.length;

        if (bestValue !== null) {
            bestScore.textContent = bestValue.toFixed(6);
        }

        if (trials.length > 0) {
            latestScore.textContent = trials[trials.length - 1].value.toFixed(6);
        }
    }

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
        logsDiv.innerHTML = '<div class="log-line info">Starting training...</div>';
    }

    // --- Simple Chart (ASCII-style visualization) ---
    let chartData = [];

    function clearChart() {
        chartData = [];
        chartPlaceholder.style.display = 'flex';
        chartPlaceholder.textContent = 'Run training to see optimization progress';
    }

    function updateChart() {
        if (trials.length === 0) return;

        chartPlaceholder.style.display = 'none';
        chartData = trials.map(t => t.value);

        const canvas = document.getElementById('optimization-chart');
        const ctx = canvas.getContext('2d');

        // Set canvas size
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;

        // Draw chart
        drawChart(ctx, canvas.width, canvas.height);
    }

    function drawChart(ctx, width, height) {
        const padding = { top: 20, right: 20, bottom: 30, left: 50 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Clear
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);

        if (chartData.length === 0) return;

        const minVal = Math.min(...chartData);
        const maxVal = Math.max(...chartData);
        const range = maxVal - minVal || 1;

        // Draw grid lines
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (chartHeight * i / 4);
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
        }

        // Draw axis labels
        ctx.fillStyle = '#64748b';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const val = maxVal - (range * i / 4);
            const y = padding.top + (chartHeight * i / 4);
            ctx.fillText(val.toFixed(4), padding.left - 5, y + 3);
        }

        // Draw line
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        ctx.beginPath();

        chartData.forEach((val, i) => {
            const x = padding.left + (chartWidth * i / Math.max(chartData.length - 1, 1));
            const y = padding.top + chartHeight - (chartHeight * (val - minVal) / range);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        // Draw best line
        const bestY = padding.top + chartHeight - (chartHeight * (minVal - minVal) / range);
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(padding.left, bestY);
        ctx.lineTo(width - padding.right, bestY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw points
        ctx.fillStyle = '#6366f1';
        chartData.forEach((val, i) => {
            const x = padding.left + (chartWidth * i / Math.max(chartData.length - 1, 1));
            const y = padding.top + chartHeight - (chartHeight * (val - minVal) / range);

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        });

        // X-axis label
        ctx.fillStyle = '#64748b';
        ctx.textAlign = 'center';
        ctx.fillText('Trial', width / 2, height - 5);
    }

    // --- Load Past Study ---
    loadStudyBtn.addEventListener('click', loadStudy);

    async function loadStudy() {
        const dbPath = dbPathInput.value.trim();
        if (!dbPath) {
            appendLog('Please enter a database path', 'error');
            return;
        }

        try {
            const resp = await fetch(`/study/vespatune?db_path=${encodeURIComponent(dbPath)}`);
            const data = await resp.json();

            if (data.error) {
                appendLog(`Error loading study: ${data.error}`, 'error');
                return;
            }

            displayStudyResults(data);
            appendLog(`Loaded study with ${data.trials.length} trials`, 'info');

        } catch (e) {
            appendLog(`Error: ${e}`, 'error');
        }
    }

    async function loadStudyFromOutput() {
        const outputDir = document.getElementById('output_dir').value;
        if (!outputDir) return;

        const dbPath = `${outputDir}/params.db`;
        dbPathInput.value = dbPath;

        try {
            const resp = await fetch(`/study/vespatune?db_path=${encodeURIComponent(dbPath)}`);
            const data = await resp.json();

            if (!data.error) {
                displayStudyResults(data);
            }
        } catch (e) {
            console.error(e);
        }
    }

    function displayStudyResults(data) {
        // Update best params
        bestParams.textContent = JSON.stringify(data.best_params, null, 2);

        // Update metrics
        bestScore.textContent = data.best_value?.toFixed(6) || '--';

        // Populate trials from study
        trials = data.trials.map(t => ({
            number: t.number,
            value: t.value,
            params: t.params,
            time: t.datetime_complete ? new Date(t.datetime_complete) : new Date()
        }));

        // Sort by number for chart
        trials.sort((a, b) => a.number - b.number);

        trialCount.textContent = trials.length;
        historyCount.textContent = trials.length;

        if (trials.length > 0) {
            latestScore.textContent = trials[trials.length - 1].value.toFixed(6);
            bestValue = Math.min(...trials.map(t => t.value));
        }

        // Update trial list (sorted by value, best first)
        const sortedTrials = [...trials].sort((a, b) => a.value - b.value);
        trialList.innerHTML = '';

        sortedTrials.forEach((t, idx) => {
            const div = document.createElement('div');
            div.className = `trial-item${idx === 0 ? ' best' : ''}`;
            div.innerHTML = `
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span class="trial-number">#${t.number}</span>
                    <span class="trial-params" title="${JSON.stringify(t.params)}">${JSON.stringify(t.params).substring(0, 40)}...</span>
                </div>
                <span class="trial-value">${t.value?.toFixed(6)}</span>
            `;
            trialList.appendChild(div);
        });

        // Update chart
        updateChart();
    }

    // --- Initialize ---
    setStatus('Ready', '');
});
