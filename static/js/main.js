document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('video-file');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('span');
    const loader = document.getElementById('loader');
    const resultsGrid = document.getElementById('results-grid');

    const streams = {
        'vid-original': 'original',
        'vid-detection': 'detection',
        'vid-uncertainty': 'uncertainty',
        'vid-explain': 'explain'
    };

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

            // Attach unique timestamp to avoid cache collision if re-uploading same file name
            const t = new Date().getTime();

            // Set sources for images to begin receiving the Multipart JPEG streams
            for (const [imgId, processType] of Object.entries(streams)) {
                const imgElement = document.getElementById(imgId);
                imgElement.src = `/video_feed/${data.filename}/${processType}?t=${t}`;
            }

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
});
