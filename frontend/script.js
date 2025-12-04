// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// Emotion Icons Mapping
const EMOTION_ICONS = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'pleasant_surprise': '😲',
    'surprise': '😲',
    'sad': '😢',
    'OAF_angry': '😠',
    'OAF_disgust': '🤢',
    'OAF_Fear': '😨',
    'OAF_happy': '😊',
    'OAF_neutral': '😐',
    'OAF_Pleasant_surprise': '😲',
    'OAF_Sad': '😢',
    'YAF_angry': '😠',
    'YAF_disgust': '🤢',
    'YAF_fear': '😨',
    'YAF_happy': '😊',
    'YAF_neutral': '😐',
    'YAF_pleasant_surprised': '😲',
    'YAF_sad': '😢'
};

// State Management
let selectedFile = null;
let availableEmotions = [];

// DOM Elements
const elements = {
    statusBadge: document.getElementById('statusBadge'),
    fileInput: document.getElementById('fileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    removeBtn: document.getElementById('removeBtn'),
    fileInfo: document.getElementById('fileInfo'),
    fileName: document.getElementById('fileName'),
    audioPlayerSection: document.getElementById('audioPlayerSection'),
    audioPlayer: document.getElementById('audioPlayer'),
    resultsSection: document.getElementById('resultsSection'),
    errorSection: document.getElementById('errorSection'),
    errorMessage: document.getElementById('errorMessage'),
    retryBtn: document.getElementById('retryBtn'),
    emotionIcon: document.getElementById('emotionIcon'),
    emotionValue: document.getElementById('emotionValue'),
    modelBadge: document.getElementById('modelBadge'),
    confidenceBars: document.getElementById('confidenceBars'),
    aboutLink: document.getElementById('aboutLink'),
    apiDocsLink: document.getElementById('apiDocsLink'),
    infoModal: document.getElementById('infoModal'),
    modalClose: document.getElementById('modalClose'),
    emotionsList: document.getElementById('emotionsList')
};

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
    loadAvailableEmotions();
    setupEventListeners();
});

// Check API Health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.status === 'healthy' && data.model_loaded) {
            updateStatus('online', `Model: ${data.model_name}`);
        } else {
            updateStatus('offline', 'Model not loaded');
        }
    } catch (error) {
        updateStatus('offline', 'API offline');
        console.error('API Health Check Failed:', error);
    }
}

// Load Available Emotions
async function loadAvailableEmotions() {
    try {
        const response = await fetch(`${API_BASE_URL}/emotions`);
        const data = await response.json();
        availableEmotions = data.emotions || [];
        populateEmotionsList();
    } catch (error) {
        console.error('Failed to load emotions:', error);
    }
}

// Populate Emotions List in Modal
function populateEmotionsList() {
    if (availableEmotions.length > 0) {
        elements.emotionsList.innerHTML = availableEmotions
            .map(emotion => {
                const icon = getEmotionIcon(emotion);
                return `<li>${icon} ${formatEmotionName(emotion)}</li>`;
            })
            .join('');
    }
}

// Update Status Badge
function updateStatus(status, text) {
    elements.statusBadge.className = `status-badge ${status === 'offline' ? 'offline' : ''}`;
    elements.statusBadge.querySelector('.status-text').textContent = text;
}

// Setup Event Listeners
function setupEventListeners() {
    // File Upload
    elements.uploadBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.removeBtn.addEventListener('click', clearFile);

    // Analyze Button
    elements.analyzeBtn.addEventListener('click', analyzeAudio);

    // Retry Button
    elements.retryBtn.addEventListener('click', () => {
        hideError();
        clearFile();
    });

    // Modal
    elements.aboutLink.addEventListener('click', (e) => {
        e.preventDefault();
        showModal();
    });

    elements.apiDocsLink.addEventListener('click', (e) => {
        e.preventDefault();
        window.open(`${API_BASE_URL}/`, '_blank');
    });

    elements.modalClose.addEventListener('click', hideModal);

    window.addEventListener('click', (e) => {
        if (e.target === elements.infoModal) {
            hideModal();
        }
    });

    // Drag and Drop
    setupDragAndDrop();
}

// Handle File Selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        if (validateFile(file)) {
            selectedFile = file;
            displayFileInfo(file);
            createAudioPreview(file);
            elements.analyzeBtn.disabled = false;
            hideResults();
            hideError();
        } else {
            showError('Invalid file. Please upload a valid audio file (WAV, MP3, OGG, FLAC) under 16MB.');
        }
    }
}

// Validate File
function validateFile(file) {
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/flac'];
    const maxSize = 16 * 1024 * 1024; // 16MB

    return validTypes.includes(file.type) && file.size <= maxSize;
}

// Display File Info
function displayFileInfo(file) {
    elements.fileName.textContent = file.name;
    elements.fileInfo.style.display = 'flex';
}

// Create Audio Preview
function createAudioPreview(file) {
    const url = URL.createObjectURL(file);
    elements.audioPlayer.src = url;
    elements.audioPlayerSection.style.display = 'block';
}

// Clear File
function clearFile() {
    selectedFile = null;
    elements.fileInput.value = '';
    elements.fileInfo.style.display = 'none';
    elements.audioPlayerSection.style.display = 'none';
    elements.analyzeBtn.disabled = true;
    hideResults();
    hideError();
}

// Analyze Audio
async function analyzeAudio() {
    if (!selectedFile) return;

    // Show loading state
    showLoading();
    hideResults();
    hideError();

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Failed to analyze audio');
        }
    } catch (error) {
        showError('Failed to connect to the API. Please ensure the backend is running.');
        console.error('Analysis Error:', error);
    } finally {
        hideLoading();
    }
}

// Display Results
function displayResults(data) {
    const scores = data.confidence_scores;
    const model = data.model_used;

    // Find emotion with highest confidence score
    let highestEmotion = data.predicted_emotion;
    let highestScore = 0;

    if (scores) {
        for (const [emotion, score] of Object.entries(scores)) {
            if (score > highestScore) {
                highestScore = score;
                highestEmotion = emotion;
            }
        }
    }

    // Display emotion with highest confidence
    elements.emotionIcon.textContent = getEmotionIcon(highestEmotion);
    elements.emotionValue.textContent = formatEmotionName(highestEmotion);
    elements.modelBadge.textContent = `Model: ${model}`;

    // Display confidence scores
    displayConfidenceScores(scores);

    // Show results
    elements.resultsSection.style.display = 'block';

    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display Confidence Scores
function displayConfidenceScores(scores) {
    if (!scores) return;

    // Sort scores by value (descending)
    const sortedScores = Object.entries(scores)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 7); // Show top 7

    // Create confidence bars
    elements.confidenceBars.innerHTML = sortedScores
        .map(([emotion, score]) => {
            const percentage = (score * 100).toFixed(1);
            return `
                <div class="confidence-bar">
                    <div class="confidence-label">
                        <span class="confidence-name">${formatEmotionName(emotion)}</span>
                        <span class="confidence-percent">${percentage}%</span>
                    </div>
                    <div class="confidence-progress">
                        <div class="confidence-fill" style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        })
        .join('');
}

// Get Emotion Icon
function getEmotionIcon(emotion) {
    const normalizedEmotion = emotion.toLowerCase();

    // Try exact match
    if (EMOTION_ICONS[emotion]) {
        return EMOTION_ICONS[emotion];
    }

    // Try partial match
    for (const [key, icon] of Object.entries(EMOTION_ICONS)) {
        if (normalizedEmotion.includes(key.toLowerCase()) || key.toLowerCase().includes(normalizedEmotion)) {
            return icon;
        }
    }

    return '😊'; // Default
}

// Format Emotion Name
function formatEmotionName(emotion) {
    return emotion
        .replace(/_/g, ' ')
        .replace(/OAF|YAF/g, '')
        .trim()
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

// Show Loading State
function showLoading() {
    elements.analyzeBtn.disabled = true;
    elements.analyzeBtn.querySelector('.btn-text').style.display = 'none';
    elements.analyzeBtn.querySelector('.btn-loader').style.display = 'flex';
}

// Hide Loading State
function hideLoading() {
    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.querySelector('.btn-text').style.display = 'block';
    elements.analyzeBtn.querySelector('.btn-loader').style.display = 'none';
}

// Show Results
function hideResults() {
    elements.resultsSection.style.display = 'none';
}

// Show Error
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorSection.style.display = 'block';
    elements.errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Hide Error
function hideError() {
    elements.errorSection.style.display = 'none';
}

// Show Modal
function showModal() {
    elements.infoModal.style.display = 'block';
}

// Hide Modal
function hideModal() {
    elements.infoModal.style.display = 'none';
}

// Drag and Drop
function setupDragAndDrop() {
    const uploadCard = document.querySelector('.upload-card');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadCard.addEventListener(eventName, () => {
            uploadCard.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, () => {
            uploadCard.classList.remove('drag-over');
        }, false);
    });

    uploadCard.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                selectedFile = file;
                displayFileInfo(file);
                createAudioPreview(file);
                elements.analyzeBtn.disabled = false;
                hideResults();
                hideError();
            } else {
                showError('Invalid file. Please upload a valid audio file (WAV, MP3, OGG, FLAC) under 16MB.');
            }
        }
    }, false);
}

// Auto-refresh API status every 30 seconds
setInterval(checkAPIHealth, 30000);
