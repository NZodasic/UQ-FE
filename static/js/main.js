document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('video-file');
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
    const uqEntropy = document.getElementById('uq-entropy');
    const uqMaxVariance = document.getElementById('uq-max-variance');
    const uqMeanVariance = document.getElementById('uq-mean-variance');
    const uqTopProbability = document.getElementById('uq-top-probability');
    const xaiPeakSaliency = document.getElementById('xai-peak-saliency');
    const xaiMeanSaliency = document.getElementById('xai-mean-saliency');
    const xaiActiveArea = document.getElementById('xai-active-area');
    const xaiStrongArea = document.getElementById('xai-strong-area');

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

    const resetAnalysisMetrics = () => {
        uqEntropy.textContent = '-';
        uqMaxVariance.textContent = '-';
        uqMeanVariance.textContent = '-';
        uqTopProbability.textContent = '-';
        xaiPeakSaliency.textContent = '-';
        xaiMeanSaliency.textContent = '-';
        xaiActiveArea.textContent = '-';
        xaiStrongArea.textContent = '-';
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
    };

    const setModelStatus = (message, state = 'neutral') => {
        modelStatus.textContent = message;
        modelStatus.dataset.state = state;
    };

    const getActiveModel = () => availableModels.find((model) => model.id === activeModelId);

    const methodLabel = (method) => {
        const labels = {
            mc_dropout: 'MC Dropout',
            gradcam: 'Grad-CAM',
            'gradcam++': 'Grad-CAM++',
            eigencam: 'EigenCAM',
            hirescam: 'HiResCAM',
            integrated_gradients: 'Integrated Gradients',
            saliency: 'Saliency'
        };
        return labels[String(method || '').toLowerCase()] || method || '-';
    };

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
        const label = methodLabel(model.uq_method);
        return model.uq_samples ? `${label} (${model.uq_samples})` : label;
    };

    const populateModelSelect = () => {
        modelSelect.innerHTML = '';
        modelSelect.disabled = availableModels.length === 0;

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
            resetAnalysisMetrics();
            return;
        }

        detectionBadge.textContent = model.model_label || 'Model';
        detectedModelInfo.textContent = currentFilename ? 'Reading prediction...' : 'Awaiting video';
        uqModelInfo.textContent = uqLabel(model);
        xaiModelInfo.textContent = xaiLabel(model);
        uqBadge.textContent = methodLabel(model.uq_method);
        xaiBadge.textContent = xaiLabel(model);
        activeModelSummary.innerHTML = `
            <div class="active-model-title">${escapeHtml(model.label || model.id)}</div>
            <div class="model-metrics">
                ${metric('Accuracy', model.accuracy_label)}
                ${metric('F1', model.f1_label)}
                ${metric('Val Acc', model.best_val_accuracy_label)}
                ${metric('XAI', xaiLabel(model))}
                ${metric('UQ', uqLabel(model))}
            </div>
        `;
    };

    const renderModelsPanel = () => {
        if (!availableModels.length) {
            modelsPanel.innerHTML = '<p class="empty-state">No model checkpoints were found.</p>';
            return;
        }

        modelsPanel.innerHTML = availableModels.map((model) => `
            <article class="model-card ${model.id === activeModelId ? 'selected' : ''}">
                <div class="model-card-header">
                    <div>
                        <h3>${escapeHtml(model.label || model.id)}</h3>
                        <p>${escapeHtml(model.source === 'training' ? model.id : model.path || '')}</p>
                    </div>
                    <button type="button" data-model-id="${escapeHtml(model.id)}">
                        ${model.id === activeModelId ? 'Selected' : 'Use'}
                    </button>
                </div>
                <div class="model-metrics">
                    ${metric('Backbone', model.model_label)}
                    ${metric('Classes', model.num_classes)}
                    ${metric('Accuracy', model.accuracy_label)}
                    ${metric('F1', model.f1_label)}
                    ${metric('Best Val', model.best_val_accuracy_label)}
                    ${metric('Latency', model.latency_ms ? `${Number(model.latency_ms).toFixed(1)} ms` : null)}
                    ${metric('Size', model.size_mb ? `${Number(model.size_mb).toFixed(1)} MB` : null)}
                    ${metric('Source', model.source)}
                </div>
            </article>
        `).join('');

        modelsPanel.querySelectorAll('button[data-model-id]').forEach((button) => {
            button.addEventListener('click', () => selectModel(button.dataset.modelId));
        });
    };

    const startStreams = () => {
        if (!currentFilename) return;

        const t = Date.now();
        for (const [imgId, processType] of Object.entries(streams)) {
            const imgElement = document.getElementById(imgId);
            let streamUrl = `/video_feed/${encodeURIComponent(currentFilename)}/${processType}?t=${t}`;
            if (processType !== 'original' && activeModelId) {
                streamUrl += `&model_id=${encodeURIComponent(activeModelId)}`;
            }
            imgElement.src = '';
            imgElement.src = streamUrl;
        }
    };

    const updatePredictionInfo = async () => {
        if (!currentFilename || !activeModelId) return;

        const requestId = ++predictionRequestId;
        detectedModelInfo.textContent = 'Reading prediction...';

        try {
            const url = `/prediction/${encodeURIComponent(currentFilename)}?model_id=${encodeURIComponent(activeModelId)}&t=${Date.now()}`;
            const res = await fetch(url);
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || 'Unable to read prediction');
            }
            if (requestId !== predictionRequestId) return;

            detectedModelInfo.textContent = `${data.class_name}: ${data.confidence_label}`;
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
                setModelStatus(`${availableModels.length} model${availableModels.length === 1 ? '' : 's'} available from training outputs.`, 'ready');
            } else {
                setModelStatus('No trained models found.', 'error');
            }
        } catch (err) {
            console.error(err);
            setModelStatus(err.message, 'error');
            populateModelSelect();
        }
    }

    modelSelect.addEventListener('change', () => selectModel(modelSelect.value));

    showModelsBtn.addEventListener('click', () => {
        const isHidden = modelsPanel.classList.toggle('hidden');
        showModelsBtn.textContent = isHidden ? 'Show All' : 'Hide All';
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) return;
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('video', file);

        // UI Loading state
        submitBtn.disabled = true;
        btnText.textContent = 'Uploading...';
        loader.classList.remove('hidden');

        try {
            // Send file to FastAPI standard endpoint
            const res = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('Failed to upload video');
            
            const data = await res.json();
            
            // Show the grid
            resultsGrid.classList.remove('hidden');
            currentFilename = data.filename;
            startStreams();
            updatePredictionInfo();

            // Reset button text smoothly
            btnText.textContent = 'Video Playing';
            
        } catch (err) {
            console.error(err);
            alert(`Error processing video: ${err.message}`);
            submitBtn.disabled = false;
            btnText.textContent = 'Process Video';
        } finally {
            loader.classList.add('hidden');
            // Allow re-upload
            setTimeout(() => {
                submitBtn.disabled = false;
                btnText.textContent = 'Process New Video';
            }, 3000);
        }
    });

    // Add drag and drop features to the glass container for easy upload
    const glassContainer = document.querySelector('.glass-container');
    
    glassContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        glassContainer.style.borderColor = 'rgba(59, 130, 246, 0.5)';
        glassContainer.style.boxShadow = '0 0 30px rgba(59, 130, 246, 0.2)';
    });

    glassContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        glassContainer.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        glassContainer.style.boxShadow = '0 40px 60px -15px rgba(0, 0, 0, 0.6)';
    });

    glassContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        glassContainer.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        glassContainer.style.boxShadow = '0 40px 60px -15px rgba(0, 0, 0, 0.6)';
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            // Optionally auto submit here
        }
    });

    loadModels();
});
