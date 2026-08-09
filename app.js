class ConfigLoader {
    static async load() {
        const response = await fetch('app-config.json');
        if (!response.ok) {
            throw new Error('Gagal memuat konfigurasi JSON');
        }
        return await response.json();
    }
}

class UIManager {
    constructor() {
        this.video = document.getElementById('video');
        this.image = document.getElementById('image-display');
        this.canvas = document.getElementById('overlay');
        this.statusText = document.getElementById('status-text');
        this.statusIndicator = document.getElementById('status-indicator');
        this.fileInput = document.getElementById('file-upload');
        this.btnUpload = document.getElementById('btn-upload');
        this.btnStartCam = document.getElementById('btn-start-cam');
        this.btnCapture = document.getElementById('btn-capture');
        this.controls = document.getElementById('controls');
    }

    bindEvents(onUpload, onCameraStart, onCapture) {
        this.btnUpload.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', onUpload);
        this.btnStartCam.addEventListener('click', onCameraStart);
        this.btnCapture.addEventListener('click', onCapture);
    }

    setStatus(text, indicatorColor = '#f59e0b') {
        this.statusText.textContent = text;
        this.statusIndicator.style.background = indicatorColor;
        this.statusIndicator.style.boxShadow = `0 0 0 10px ${indicatorColor}33`;
    }

    showControls() {
        this.controls.style.display = 'flex';
    }

    showVideo() {
        this.image.classList.add('hidden');
        this.video.classList.remove('hidden');
        this.canvas.classList.remove('hidden');
    }

    showImage() {
        this.video.classList.add('hidden');
        this.image.classList.remove('hidden');
        this.canvas.classList.remove('hidden');
    }

    showCaptureButton(show) {
        this.btnCapture.classList.toggle('hidden', !show);
    }

    clearCanvas() {
        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

class MediaService {
    constructor(ui) {
        this.ui = ui;
        this.stream = null;
    }

    async startCamera() {
        this.ui.showVideo();
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        this.stream = stream;
        this.ui.video.srcObject = stream;
        this.ui.showCaptureButton(true);
    }

    captureFrame() {
        const canvas = document.createElement('canvas');
        canvas.width = this.ui.video.videoWidth;
        canvas.height = this.ui.video.videoHeight;
        canvas.getContext('2d').drawImage(this.ui.video, 0, 0);
        return canvas.toDataURL('image/png');
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.ui.showCaptureButton(false);
        this.ui.showImage();
    }
}

class EmotionDetector {
    constructor(config, ui) {
        this.config = config;
        this.ui = ui;
        this.model = null;
    }

    async loadModels() {
        await faceapi.nets.tinyFaceDetector.loadFromUri(this.config.faceApiUrl);
        this.model = await tf.loadLayersModel(this.config.modelPath);
    }

    async analyzeImage(sourceElement) {
        this.ui.clearCanvas();
        const displaySize = {
            width: sourceElement.clientWidth,
            height: sourceElement.clientHeight
        };
        this.ui.canvas.width = displaySize.width;
        this.ui.canvas.height = displaySize.height;
        faceapi.matchDimensions(this.ui.canvas, displaySize);

        const detections = await faceapi.detectAllFaces(sourceElement, new faceapi.TinyFaceDetectorOptions());
        if (!detections.length) {
            return [];
        }

        const resized = faceapi.resizeResults(detections, displaySize);
        const context = this.ui.canvas.getContext('2d');

        return resized.map(det => {
            const box = det.box;
            context.strokeStyle = '#60a5fa';
            context.lineWidth = 3;
            context.strokeRect(box.x, box.y, box.width, box.height);
            return { box };
        });
    }

    async predictEmotion(sourceImage, box) {
        const scaleX = sourceImage.naturalWidth / sourceImage.clientWidth;
        const scaleY = sourceImage.naturalHeight / sourceImage.clientHeight;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.config.inputSize;
        tempCanvas.height = this.config.inputSize;
        const tempContext = tempCanvas.getContext('2d');
        tempContext.drawImage(
            sourceImage,
            box.x * scaleX,
            box.y * scaleY,
            box.width * scaleX,
            box.height * scaleY,
            0,
            0,
            this.config.inputSize,
            this.config.inputSize
        );

        return tf.tidy(() => {
            let tensor = tf.browser.fromPixels(tempCanvas).toFloat().div(255.0);
            if (this.model.inputs[0].shape[3] === 1) {
                tensor = tensor.mean(2).expandDims(2);
            }
            tensor = tensor.expandDims(0);
            const logits = this.model.predict(tensor);
            const predictions = logits.dataSync();
            const maxIndex = predictions.indexOf(Math.max(...predictions));
            return {
                label: this.config.emotionLabels[maxIndex] || 'Tidak Diketahui',
                score: (predictions[maxIndex] * 100).toFixed(1)
            };
        });
    }
}

class EmotionApp {
    constructor() {
        this.ui = new UIManager();
        this.mediaService = new MediaService(this.ui);
        this.detector = null;
        this.config = null;
        this.currentImage = null;
    }

    async start() {
        try {
            this.ui.setStatus('Memuat konfigurasi aplikasi...', '#fbbf24');
            this.config = await ConfigLoader.load();
            this.detector = new EmotionDetector(this.config, this.ui);
            this.ui.bindEvents(this.handleUpload.bind(this), this.handleCameraStart.bind(this), this.handleCapture.bind(this));
            this.ui.setStatus(this.config.messages.loadingFaceDetector, '#fbbf24');
            await this.detector.loadModels();
            this.ui.setStatus(this.config.messages.ready, '#22c55e');
            this.ui.showControls();
        } catch (error) {
            console.error(error);
            this.ui.setStatus(this.config?.messages?.modelLoadError || 'Terjadi kesalahan saat inisialisasi.', '#ef4444');
        }
    }

    async handleUpload(event) {
        const file = event.target.files?.[0];
        if (!file) {
            return;
        }
        this.mediaService.stopCamera();
        const reader = new FileReader();
        reader.onload = () => {
            this.displayImage(reader.result);
        };
        reader.readAsDataURL(file);
    }

    async handleCameraStart() {
        try {
            await this.mediaService.startCamera();
            this.ui.setStatus(this.config.messages.cameraActive, '#3b82f6');
        } catch (error) {
            console.error(error);
            this.ui.setStatus(this.config.messages.cameraError, '#ef4444');
        }
    }

    async handleCapture() {
        const imageData = this.mediaService.captureFrame();
        this.mediaService.stopCamera();
        this.displayImage(imageData);
    }

    displayImage(imageData) {
        this.ui.clearCanvas();
        this.ui.image.src = imageData;
        this.ui.image.onload = async () => {
            this.ui.showImage();
            this.ui.setStatus(this.config.messages.processing, '#f59e0b');
            await this.processDetection();
        };
    }

    async processDetection() {
        const faces = await this.detector.analyzeImage(this.ui.image);
        if (!faces.length) {
            this.ui.setStatus(this.config.messages.noFaceDetected, '#ef4444');
            return;
        }

        for (const face of faces) {
            const prediction = await this.detector.predictEmotion(this.ui.image, face.box);
            this.drawLabel(face.box, prediction);
        }

        this.ui.setStatus(this.config.messages.detectionComplete, '#22c55e');
    }

    drawLabel(box, prediction) {
        const ctx = this.ui.canvas.getContext('2d');
        ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
        ctx.fillRect(box.x, box.y - 32, box.width, 32);
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Inter, system-ui, sans-serif';
        ctx.fillText(`${prediction.label} · ${prediction.score}%`, box.x + 10, box.y - 10);
    }
}

window.addEventListener('load', () => {
    const app = new EmotionApp();
    app.start();
});
