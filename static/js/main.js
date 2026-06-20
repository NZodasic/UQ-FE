document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('video-file');
    const dropZone = document.getElementById('drop-zone');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('span');
    const loader = document.getElementById('loader');
    const resultsGrid = document.getElementById('results-grid');
    const modelSelect = document.getElementById('model-select');
    const modelStatus = document.getElementById('model-status');
    const activeModelSummary = document.getElementById('active-model-summary');
    const modelsPanel = document.getElementById('models-panel');
    const showModelsBtn = document.getElementById('show-models-btn');
    const detectionBadge = document.getElementById('detection-badge');
    const detectedModelInfo = document.getElementById('detected-model-info');
    const uqModelInfo = document.getElementById('uq-model-info');
    const xaiModelInfo = document.getElementById('xai-model-info');
    const uqBadge = document.getElementById('uq-badge');
    const xaiBadge = document.getElementById('xai-badge');
    const modelCountChip = document.getElementById('model-count-chip');
    const modelCalibrationChip = document.getElementById('model-calibration-chip');
    const headerModelName = document.getElementById('header-model-name');
    const headerAnalysisMode = document.getElementById('header-analysis-mode');

    const uqEntropy = document.getElementById('uq-entropy');
    const uqMaxVariance = document.getElementById('uq-max-variance');
    const uqMeanVariance = document.getElementById('uq-mean-variance');
    const uqTopProbability = document.getElementById('uq-top-probability');

    const xaiPeakSaliency = document.getElementById('xai-peak-saliency');
    const xaiMeanSaliency = document.getElementById('xai-mean-saliency');
    const xaiActiveArea = document.getElementById('xai-active-area');
    const xaiStrongArea = document.getElementById('xai-strong-area');

    // Override Config Controls
    const uqMethodSelect = document.getElementById('uq-method-select');
    const uqSamplesSlider = document.getElementById('uq-samples-slider');
    const uqSamplesVal = document.getElementById('uq-samples-val');
    const uqSamplesGroup = document.getElementById('uq-samples-group');
    const uqSamplesLabel = document.getElementById('uq-samples-label');
    const xaiMethodSelect = document.getElementById('xai-method-select');

    const streams = {
        'vid-original': 'original',
        'vid-detection': 'detection',
        'vid-uncertainty': 'uncertainty',
        'vid-explain': 'explain'
    };

    let availableModels = [];
    let activeModelId = null;
    let currentFilename = null;
    let predictionRequestId = 0;

    const escapeHtml = (value) => String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');

    const metric = (label, value) => {
        if (value === null || value === undefined || value === '') return '';
        return `<span><strong>${escapeHtml(label)}</strong>${escapeHtml(value)}</span>`;
    };

    const formatNumber = (value, digits = 3) => {
        const number = Number(value);
        if (!Number.isFinite(number)) return '-';
        return number.toFixed(digits);
    };

    const formatPercent = (value, digits = 1) => {
        const number = Number(value);
        if (!Number.isFinite(number)) return '-';
        return `${(number * 100).toFixed(digits)}%`;
    };

    const updateProgressBar = (barId, percent) => {
        const bar = document.getElementById(barId);
        if (bar) {
            const val = Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0;
            bar.style.width = `${val.toFixed(0)}%`;
        }
    };

    const resetAnalysisMetrics = () => {
        uqEntropy.textContent = '-';
        uqMaxVariance.textContent = '-';
        uqMeanVariance.textContent = '-';
        uqTopProbability.textContent = '-';
        xaiPeakSaliency.textContent = '-';
        xaiMeanSaliency.textContent = '-';
        xaiActiveArea.textContent = '-';
        xaiStrongArea.textContent = '-';

        updateProgressBar('uq-entropy-bar', 0);
        updateProgressBar('uq-max-variance-bar', 0);
        updateProgressBar('uq-mean-variance-bar', 0);
        updateProgressBar('uq-top-probability-bar', 0);
        updateProgressBar('xai-peak-saliency-bar', 0);
        updateProgressBar('xai-mean-saliency-bar', 0);
        updateProgressBar('xai-active-area-bar', 0);
        updateProgressBar('xai-strong-area-bar', 0);

        const confBar = document.getElementById('confidence-bar');
        if (confBar) confBar.style.width = '0%';

        const confBarWrapper = document.getElementById('confidence-bar-wrapper');
        if (confBarWrapper) confBarWrapper.classList.add('hidden');
    };

    const renderAnalysisMetrics = (data) => {
        const uncertainty = data.uncertainty || {};
        const explanation = data.explanation || {};

        uqEntropy.textContent = formatNumber(uncertainty.predictive_entropy, 3);
        uqMaxVariance.textContent = formatNumber(uncertainty.max_variance, 5);
        uqMeanVariance.textContent = formatNumber(uncertainty.mean_variance, 5);
        uqTopProbability.textContent = formatPercent(uncertainty.top_probability, 1);

        xaiPeakSaliency.textContent = formatNumber(explanation.peak_saliency, 3);
        xaiMeanSaliency.textContent = formatNumber(explanation.mean_saliency, 3);
        xaiActiveArea.textContent = formatPercent(explanation.active_area, 1);
        xaiStrongArea.textContent = formatPercent(explanation.strong_area, 1);

        // Update progress bars with tailored normalizations
        updateProgressBar('uq-entropy-bar', Math.min(100, (uncertainty.predictive_entropy / 3.0) * 100)); // normalized by max entropy ~3.0
        updateProgressBar('uq-max-variance-bar', Math.min(100, uncertainty.max_variance * 400)); // max var is 0.25
        updateProgressBar('uq-mean-variance-bar', Math.min(100, uncertainty.mean_variance * 2000)); // mean var max is 0.05
        updateProgressBar('uq-top-probability-bar', uncertainty.top_probability * 100);

        updateProgressBar('xai-peak-saliency-bar', explanation.peak_saliency * 100);
        updateProgressBar('xai-mean-saliency-bar', Math.min(100, explanation.mean_saliency * 200)); // mean is usually lower, scale by 0.5 max
        updateProgressBar('xai-active-area-bar', explanation.active_area * 100);
        updateProgressBar('xai-strong-area-bar', explanation.strong_area * 100);
    };

    const setModelStatus = (message, state = 'neutral') => {
        modelStatus.textContent = message;
        modelStatus.dataset.state = state;
    };

    const getActiveModel = () => availableModels.find((model) => model.id === activeModelId);

    const methodLabel = (method) => {
        const labels = {
            deterministic: 'Deterministic Entropy',
            mc_dropout: 'MC Dropout',
            none: 'Deterministic Entropy',
            temperature_scaling: 'Temperature Scaling',
            mc_dropout_temperature: 'MC Dropout + Temperature',
            tta: 'Test-Time Augmentation',
            gradcam: 'Grad-CAM',
            'gradcam++': 'Grad-CAM++',
            eigencam: 'EigenCAM',
            hirescam: 'HiResCAM',
            integrated_gradients: 'Integrated Gradients',
            saliency: 'Saliency Map',
            smoothgrad: 'SmoothGrad',
            uq_gradcam: 'UQ + Grad-CAM Fusion'
        };
        return labels[String(method || '').toLowerCase()] || method || '-';
    };

    const normalizeUqMethod = (method) => {
        const value = String(method || 'mc_dropout').toLowerCase();
        if (value === 'none' || value === 'no_uq') return 'deterministic';
        if (value === 'mc_dropout_temperature_scaling') return 'mc_dropout_temperature';
        return value;
    };

    const methodNeedsSamples = (method) => ['mc_dropout', 'mc_dropout_temperature', 'tta'].includes(normalizeUqMethod(method));

    const methodSampleLabel = (method) => normalizeUqMethod(method) === 'tta' ? 'TTA Samples' : 'Samples';

    const xaiLabel = (model) => {
        if (!model) return '-';
        const method = String(model.xai_method || '').toLowerCase();
        if (method === 'gradcam' && model.xai_variant) {
            return methodLabel(model.xai_variant);
        }
        return methodLabel(model.xai_method || model.xai_variant);
    };

    const uqLabel = (model) => {
        if (!model) return '-';
        const method = normalizeUqMethod(model.uq_method);
        const label = methodLabel(method);
        if (methodNeedsSamples(method)) return `${label} (${model.uq_samples || 1})`;
        if (method.includes('temperature') && model.temperature) return `${label} (T=${Number(model.temperature).toFixed(2)})`;
        return label;
    };

    const currentUqLabel = () => {
        const method = normalizeUqMethod(uqMethodSelect.value);
        const label = methodLabel(method);
        if (methodNeedsSamples(method)) return `${label} (${parseInt(uqSamplesSlider.value, 10)})`;
        const model = getActiveModel();
        if (method.includes('temperature') && model?.temperature) return `${label} (T=${Number(model.temperature).toFixed(2)})`;
        return label;
    };

    const updateMethodBadges = () => {
        const uqText = currentUqLabel();
        const xaiText = methodLabel(xaiMethodSelect.value);
        uqModelInfo.textContent = uqText;
        xaiModelInfo.textContent = xaiText;
        uqBadge.textContent = methodLabel(uqMethodSelect.value);
        xaiBadge.textContent = xaiText;
        headerAnalysisMode.textContent = `${methodLabel(uqMethodSelect.value)} + ${xaiText}`;
    };

    const getOverrideParams = () => {
        const uqMethod = normalizeUqMethod(uqMethodSelect.value);
        const uqSamples = methodNeedsSamples(uqMethod) ? parseInt(uqSamplesSlider.value, 10) : 1;
        const xaiMethodRaw = xaiMethodSelect.value;

        let xaiMethod = xaiMethodRaw;
        let xaiVariant = '';
        if (['gradcam++', 'eigencam', 'hirescam'].includes(xaiMethodRaw)) {
            xaiMethod = 'gradcam';
            xaiVariant = xaiMethodRaw;
        }

        return { uqMethod, uqSamples, xaiMethod, xaiVariant };
    };

    const populateModelSelect = () => {
        modelSelect.innerHTML = '';
        modelSelect.disabled = availableModels.length === 0;
        modelCountChip.textContent = `${availableModels.length} run${availableModels.length === 1 ? '' : 's'}`;

        if (!availableModels.length) {
            const option = document.createElement('option');
            option.textContent = 'No trained models found';
            modelSelect.appendChild(option);
            return;
        }

        for (const model of availableModels) {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.label || model.id;
            option.selected = model.id === activeModelId;
            modelSelect.appendChild(option);
        }
    };

    const populateMethodSelects = (model) => {
        const uqOptions = model?.uq_options?.length ? model.uq_options : [
            { id: 'deterministic', label: 'Deterministic Entropy' },
            { id: 'mc_dropout', label: 'MC Dropout' },
            { id: 'tta', label: 'Test-Time Augmentation' }
        ];
        const xaiOptions = model?.xai_options?.length ? model.xai_options : [
            { id: 'gradcam', label: 'Grad-CAM' },
            { id: 'saliency', label: 'Saliency Map' }
        ];

        const desiredUq = normalizeUqMethod(model?.uq_method);
        uqMethodSelect.innerHTML = '';
        for (const optionData of uqOptions) {
            const option = document.createElement('option');
            option.value = optionData.id;
            option.textContent = optionData.note ? `${optionData.label} (${optionData.note})` : optionData.label;
            option.disabled = Boolean(optionData.disabled);
            option.selected = optionData.id === desiredUq && !option.disabled;
            uqMethodSelect.appendChild(option);
        }
        if (!uqMethodSelect.value || uqMethodSelect.selectedOptions[0]?.disabled) {
            const fallback = Array.from(uqMethodSelect.options).find((option) => !option.disabled && option.value === 'mc_dropout')
                || Array.from(uqMethodSelect.options).find((option) => !option.disabled);
            if (fallback) uqMethodSelect.value = fallback.value;
        }

        const methodKey = model?.xai_variant || model?.xai_method || 'gradcam';
        xaiMethodSelect.innerHTML = '';
        for (const optionData of xaiOptions) {
            const option = document.createElement('option');
            option.value = optionData.id;
            option.textContent = optionData.label;
            xaiMethodSelect.appendChild(option);
        }
        xaiMethodSelect.value = Array.from(xaiMethodSelect.options).some((option) => option.value === methodKey)
            ? methodKey
            : 'gradcam';
    };

    const updateSamplesControl = () => {
        const method = normalizeUqMethod(uqMethodSelect.value);
        const showSamples = methodNeedsSamples(method);
        uqSamplesGroup.style.display = showSamples ? 'flex' : 'none';
        uqSamplesLabel.textContent = methodSampleLabel(method);
        if (method === 'tta') {
            uqSamplesSlider.min = '2';
            uqSamplesSlider.max = '20';
            uqSamplesSlider.step = '1';
            if (Number(uqSamplesSlider.value) > 20) uqSamplesSlider.value = '10';
            if (Number(uqSamplesSlider.value) < 2) uqSamplesSlider.value = '8';
        } else {
            uqSamplesSlider.min = '2';
            uqSamplesSlider.max = '50';
            uqSamplesSlider.step = '1';
            if (Number(uqSamplesSlider.value) < 2) uqSamplesSlider.value = '15';
        }
        uqSamplesVal.textContent = uqSamplesSlider.value;
    };

    const renderActiveModel = () => {
        const model = getActiveModel();
        if (!model) {
            activeModelSummary.innerHTML = '';
            detectionBadge.textContent = 'Model';
            detectedModelInfo.textContent = 'Awaiting video';
            uqModelInfo.textContent = '-';
            xaiModelInfo.textContent = '-';
            uqBadge.textContent = 'UQ';
            xaiBadge.textContent = 'XAI';
            headerModelName.textContent = '-';
            modelCalibrationChip.textContent = 'Calibration: -';
            resetAnalysisMetrics();
            return;
        }

        detectionBadge.textContent = model.model_label || 'Model';
        headerModelName.textContent = model.model_label || model.id;
        detectedModelInfo.textContent = currentFilename ? 'Reading prediction...' : 'Awaiting video';

        populateMethodSelects(model);
        uqSamplesSlider.value = model.uq_samples || 15;
        updateSamplesControl();
        updateMethodBadges();
        modelCalibrationChip.textContent = model.temperature_available
            ? `Calibration: T=${Number(model.temperature || 1).toFixed(2)}`
            : 'Calibration: unavailable';

        activeModelSummary.innerHTML = `
            <div class="active-model-title">${escapeHtml(model.label || model.id)}</div>
            <div class="model-metrics">
                ${metric('Backbone', model.model_label)}
                ${metric('Accuracy', model.accuracy_label)}
                ${metric('F1', model.f1_label)}
                ${metric('Val Acc', model.best_val_accuracy_label)}
                ${metric('Classes', model.num_classes)}
                ${metric('Default UQ', uqLabel(model))}
                ${metric('Default XAI', xaiLabel(model))}
                ${metric('Temperature', model.temperature ? Number(model.temperature).toFixed(2) : null)}
                ${metric('Size', model.size_mb ? `${Number(model.size_mb).toFixed(1)} MB` : null)}
            </div>
        `;
    };

    const hideModelsPanel = () => {
        modelsPanel.classList.add('hidden');
        showModelsBtn.textContent = 'Compare Runs';
    };

    const renderModelsPanel = () => {
        if (!availableModels.length) {
            modelsPanel.innerHTML = '<p class="empty-state">No model checkpoints were found.</p>';
            return;
        }

        modelsPanel.innerHTML = `
            <div class="models-panel-header">
                <div>
                    <h2>Compare Runs</h2>
                    <p>${availableModels.length} checkpoint${availableModels.length === 1 ? '' : 's'} available</p>
                </div>
                <button type="button" class="btn btn-secondary btn-sm models-panel-close">Close</button>
            </div>
            <div class="models-panel-grid">
                ${availableModels.map((model) => `
                    <article class="model-card ${model.id === activeModelId ? 'selected' : ''}">
                        <div class="model-card-header">
                            <div>
                                <h3>${escapeHtml(model.label || model.id)}</h3>
                                <p>${escapeHtml(model.source === 'training' ? model.id : model.path || '')}</p>
                            </div>
                            <button type="button" class="btn btn-secondary btn-sm" data-model-id="${escapeHtml(model.id)}">
                                ${model.id === activeModelId ? 'Selected' : 'Use'}
                            </button>
                        </div>
                        <div class="model-tags">
                            <span>${escapeHtml(model.model_label || 'Model')}</span>
                            <span>${escapeHtml(uqLabel(model))}</span>
                            <span>${escapeHtml(xaiLabel(model))}</span>
                            ${model.temperature_available ? `<span>T=${escapeHtml(Number(model.temperature || 1).toFixed(2))}</span>` : ''}
                        </div>
                        <div class="model-metrics">
                            ${metric('Classes', model.num_classes)}
                            ${metric('Accuracy', model.accuracy_label)}
                            ${metric('F1', model.f1_label)}
                            ${metric('Best Val', model.best_val_accuracy_label)}
                            ${metric('Latency', model.latency_ms ? `${Number(model.latency_ms).toFixed(1)} ms` : null)}
                            ${metric('Size', model.size_mb ? `${Number(model.size_mb).toFixed(1)} MB` : null)}
                            ${metric('Source', model.source)}
                        </div>
                    </article>
                `).join('')}
            </div>
        `;

        modelsPanel.querySelector('.models-panel-close')?.addEventListener('click', hideModelsPanel);
        modelsPanel.querySelectorAll('button[data-model-id]').forEach((button) => {
            button.addEventListener('click', () => {
                hideModelsPanel();
                selectModel(button.dataset.modelId);
            });
        });
    };

    const updateDropZoneFileName = () => {
        const file = fileInput.files?.[0];
        const dropText = document.querySelector('.drop-text');
        if (dropText) {
            dropText.textContent = file ? file.name : 'Choose file or drag here';
        }
    };

    const setDropZoneIdle = () => {
        dropZone.classList.remove('is-active');
        dropZone.style.borderColor = 'var(--border-color)';
        dropZone.style.background = 'rgba(0, 0, 0, 0.1)';
    };

    const setDropZoneActive = () => {
        dropZone.classList.add('is-active');
        dropZone.style.borderColor = 'var(--primary)';
        dropZone.style.background = 'rgba(47, 128, 237, 0.08)';
    };

    const startStreams = () => {
        if (!currentFilename) return;

        const t = Date.now();
        const { uqMethod, uqSamples, xaiMethod, xaiVariant } = getOverrideParams();

        for (const [imgId, processType] of Object.entries(streams)) {
            const imgElement = document.getElementById(imgId);

            let streamUrl = `/video_feed/${encodeURIComponent(currentFilename)}/${processType}?t=${t}`;
            if (activeModelId) {
                streamUrl += `&model_id=${encodeURIComponent(activeModelId)}`;
            }
            if (processType !== 'original') {
                if (uqMethod) streamUrl += `&uq_method=${encodeURIComponent(uqMethod)}`;
                if (uqSamples !== undefined) streamUrl += `&uq_samples=${encodeURIComponent(uqSamples)}`;
                if (xaiMethod) streamUrl += `&xai_method=${encodeURIComponent(xaiMethod)}`;
                if (xaiVariant) streamUrl += `&xai_variant=${encodeURIComponent(xaiVariant)}`;
            }
            imgElement.src = '';
            imgElement.closest('.video-wrapper')?.classList.add('has-stream');
            imgElement.src = streamUrl;
        }
    };
    const updatePredictionInfo = async () => {
        if (!currentFilename || !activeModelId) return;

        const requestId = ++predictionRequestId;
        detectedModelInfo.textContent = 'Reading prediction...';

        const confBarWrapper = document.getElementById('confidence-bar-wrapper');
        const confBar = document.getElementById('confidence-bar');
        if (confBarWrapper) confBarWrapper.classList.add('hidden');

        try {
            const { uqMethod, uqSamples, xaiMethod, xaiVariant } = getOverrideParams();

            let url = `/prediction/${encodeURIComponent(currentFilename)}?model_id=${encodeURIComponent(activeModelId)}&t=${Date.now()}`;
            if (uqMethod) url += `&uq_method=${encodeURIComponent(uqMethod)}`;
            if (uqSamples !== undefined) url += `&uq_samples=${encodeURIComponent(uqSamples)}`;
            if (xaiMethod) url += `&xai_method=${encodeURIComponent(xaiMethod)}`;
            if (xaiVariant) url += `&xai_variant=${encodeURIComponent(xaiVariant)}`;

            const res = await fetch(url);
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || 'Unable to read prediction');
            }
            if (requestId !== predictionRequestId) return;

            detectedModelInfo.textContent = `${data.class_name}: ${data.confidence_label}`;

            // Show and update confidence progress bar
            if (confBarWrapper && confBar && data.confidence !== undefined) {
                confBarWrapper.classList.remove('hidden');
                confBar.style.width = `${(data.confidence * 100).toFixed(0)}%`;
            }

            uqModelInfo.textContent = data.uq_label || uqLabel(getActiveModel());
            xaiModelInfo.textContent = data.xai_label || xaiLabel(getActiveModel());
            renderAnalysisMetrics(data);
        } catch (err) {
            console.error(err);
            if (requestId === predictionRequestId) {
                detectedModelInfo.textContent = 'Prediction unavailable';
                resetAnalysisMetrics();
            }
        }
    };

    async function selectModel(modelId) {
        if (!modelId || modelId === activeModelId) return;

        modelSelect.disabled = true;
        showModelsBtn.disabled = true;
        setModelStatus('Loading selected model...', 'loading');

        try {
            const res = await fetch(`/models/${encodeURIComponent(modelId)}/select`, {
                method: 'POST'
            });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || 'Unable to load selected model');
            }

            activeModelId = data.active_model_id;
            populateModelSelect();
            renderActiveModel();
            renderModelsPanel();
            setModelStatus('Selected model is ready.', 'ready');
            startStreams();
            updatePredictionInfo();
        } catch (err) {
            console.error(err);
            setModelStatus(err.message, 'error');
        } finally {
            modelSelect.disabled = availableModels.length === 0;
            showModelsBtn.disabled = false;
        }
    }

    async function loadModels() {
        setModelStatus('Loading trained models...', 'loading');

        try {
            const res = await fetch('/models');
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || 'Unable to list models');
            }

            availableModels = data.models || [];
            activeModelId = data.active_model_id;
            populateModelSelect();
            renderActiveModel();
            renderModelsPanel();

            if (availableModels.length) {
                setModelStatus(`${availableModels.length} model${availableModels.length === 1 ? '' : 's'} loaded from training.`, 'ready');
            } else {
                setModelStatus('No trained models found.', 'error');
            }
        } catch (err) {
            console.error(err);
            setModelStatus(err.message, 'error');
            populateModelSelect();
        }
    }

    // Parameters Change Event Listeners
    modelSelect.addEventListener('change', () => selectModel(modelSelect.value));

    uqMethodSelect.addEventListener('change', () => {
        updateSamplesControl();
        updateMethodBadges();
        startStreams();
        updatePredictionInfo();
    });

    uqSamplesSlider.addEventListener('input', () => {
        uqSamplesVal.textContent = uqSamplesSlider.value;
        updateMethodBadges();
    });

    uqSamplesSlider.addEventListener('change', () => {
        updateMethodBadges();
        startStreams();
        updatePredictionInfo();
    });

    xaiMethodSelect.addEventListener('change', () => {
        updateMethodBadges();
        startStreams();
        updatePredictionInfo();
    });

    showModelsBtn.addEventListener('click', () => {
        const isHidden = modelsPanel.classList.toggle('hidden');
        showModelsBtn.textContent = isHidden ? 'Compare Runs' : 'Hide Runs';
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modelsPanel.classList.contains('hidden')) {
            hideModelsPanel();
        }
    });

    // Upload & Video Processing
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!fileInput.files.length) return;

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('video', file);

        submitBtn.disabled = true;
        btnText.textContent = 'Uploading...';
        loader.classList.remove('hidden');

        try {
            const res = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('Failed to upload video');

            const data = await res.json();

            resultsGrid.classList.remove('hidden');
            currentFilename = data.filename;

            startStreams();
            updatePredictionInfo();

            btnText.textContent = 'Video Playing';
        } catch (err) {
            console.error(err);
            alert(`Error processing video: ${err.message}`);
            submitBtn.disabled = false;
            btnText.textContent = 'Process Video';
        } finally {
            loader.classList.add('hidden');
            setTimeout(() => {
                submitBtn.disabled = false;
                btnText.textContent = 'Process New Video';
            }, 3000);
        }
    });

    // Setup drag and drop events on the drop-zone
    fileInput.addEventListener('change', updateDropZoneFileName);

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        setDropZoneActive();
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        setDropZoneIdle();
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        setDropZoneIdle();

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            updateDropZoneFileName();
            uploadForm.dispatchEvent(new Event('submit'));
        }
    });

    loadModels();
});
