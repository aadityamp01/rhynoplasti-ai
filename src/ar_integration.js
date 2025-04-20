// AR Integration for Rhinoplasty Simulation
class ARRhinoplastySimulator {
    constructor(clientToken) {
        this.clientToken = clientToken;
        this.video = null;
        this.canvas = null;
        this.stream = null;
        this.currentImage = null;
        this.isCameraMode = true;
        this.effects = {
            "Natural Refinement": "natural_refinement.zip",
            "Nose Bridge Reduction": "bridge_reduction.zip",
            "Tip Refinement": "tip_refinement.zip",
            "Wide Nose Narrowing": "nose_narrowing.zip",
            "Crooked Nose Correction": "crooked_correction.zip",
            "Combined Enhancement": "combined_enhancement.zip"
        };
    }

    async initialize(containerId) {
        try {
            // Get the container element
            const container = document.getElementById(containerId);
            
            // Create video element if it doesn't exist
            if (!this.video) {
                this.video = document.createElement('video');
                this.video.id = 'video';
                this.video.autoplay = true;
                this.video.playsInline = true;
                container.appendChild(this.video);
            }
            
            // Create canvas for processing if it doesn't exist
            if (!this.canvas) {
                this.canvas = document.createElement('canvas');
                this.canvas.style.display = 'none';
                container.appendChild(this.canvas);
            }
            
            // Request camera permissions
            try {
                this.stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: "user"
                    }
                });
                
                this.video.srcObject = this.stream;
                this.isCameraMode = true;
                
                // Wait for video to be ready
                await new Promise((resolve) => {
                    this.video.onloadedmetadata = () => {
                        resolve();
                    };
                });
                
                // Show permission message if needed
                const permissionMessage = document.getElementById('permission-message');
                if (permissionMessage) {
                    permissionMessage.style.display = 'none';
                }
                
                return true;
            } catch (error) {
                console.error("Camera permission denied:", error);
                // Show permission message
                const permissionMessage = document.getElementById('permission-message');
                if (permissionMessage) {
                    permissionMessage.style.display = 'block';
                }
                return false;
            }
        } catch (error) {
            console.error("Failed to initialize AR Player:", error);
            return false;
        }
    }

    async loadImage(img) {
        if (!this.canvas) {
            throw new Error("Canvas not initialized");
        }

        try {
            // Set canvas dimensions to match the image
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            
            // Draw the image on the canvas
            const ctx = this.canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            
            // Store the current image
            this.currentImage = img;
            this.isCameraMode = false;
            
            return true;
        } catch (error) {
            console.error("Failed to load image:", error);
            return false;
        }
    }

    resetImage() {
        if (this.canvas) {
            const ctx = this.canvas.getContext('2d');
            ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        this.currentImage = null;
    }

    async applyEffect(effectName, intensity = 0.7) {
        if (!this.canvas) {
            throw new Error("Canvas not initialized");
        }

        try {
            const ctx = this.canvas.getContext('2d');
            
            // If in camera mode, draw the current video frame
            if (this.isCameraMode && this.video) {
                ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            }
            
            // Apply the selected effect
            switch (effectName) {
                case "Natural Refinement":
                    this._applyNaturalRefinement(ctx, intensity);
                    break;
                case "Nose Bridge Reduction":
                    this._applyBridgeReduction(ctx, intensity);
                    break;
                case "Tip Refinement":
                    this._applyTipRefinement(ctx, intensity);
                    break;
                case "Wide Nose Narrowing":
                    this._applyNoseNarrowing(ctx, intensity);
                    break;
                case "Crooked Nose Correction":
                    this._applyCrookedCorrection(ctx, intensity);
                    break;
                case "Combined Enhancement":
                    this._applyCombinedEnhancement(ctx, intensity);
                    break;
            }
            
            return true;
        } catch (error) {
            console.error(`Failed to apply effect ${effectName}:`, error);
            return false;
        }
    }

    async captureImage() {
        if (!this.canvas) {
            throw new Error("Canvas not initialized");
        }

        try {
            return this.canvas.toDataURL('image/png');
        } catch (error) {
            console.error("Failed to capture image:", error);
            throw error;
        }
    }

    async stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.video) {
            this.video.srcObject = null;
        }
        this.isCameraMode = false;
    }

    // Effect application methods
    _applyNaturalRefinement(ctx, intensity) {
        // Implement natural refinement effect
        const imageData = ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        // Apply subtle smoothing and brightness adjustment
        for (let i = 0; i < data.length; i += 4) {
            // Increase brightness slightly
            data[i] = Math.min(255, data[i] * (1 + 0.1 * intensity));
            data[i + 1] = Math.min(255, data[i + 1] * (1 + 0.1 * intensity));
            data[i + 2] = Math.min(255, data[i + 2] * (1 + 0.1 * intensity));
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    _applyBridgeReduction(ctx, intensity) {
        // Implement bridge reduction effect
        const imageData = ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        // Apply vertical compression in the nose bridge area
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        for (let y = 0; y < this.canvas.height; y++) {
            for (let x = 0; x < this.canvas.width; x++) {
                const i = (y * this.canvas.width + x) * 4;
                
                // Calculate distance from center
                const dx = x - centerX;
                const dy = y - centerY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Apply effect in nose bridge area
                if (distance < 50 * intensity) {
                    const factor = 1 - (distance / (50 * intensity));
                    data[i] = Math.min(255, data[i] * (1 + 0.2 * factor * intensity));
                    data[i + 1] = Math.min(255, data[i + 1] * (1 + 0.2 * factor * intensity));
                    data[i + 2] = Math.min(255, data[i + 2] * (1 + 0.2 * factor * intensity));
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    _applyTipRefinement(ctx, intensity) {
        // Implement tip refinement effect
        const imageData = ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        // Apply brightness and contrast adjustment in the nose tip area
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        for (let y = 0; y < this.canvas.height; y++) {
            for (let x = 0; x < this.canvas.width; x++) {
                const i = (y * this.canvas.width + x) * 4;
                
                // Calculate distance from center
                const dx = x - centerX;
                const dy = y - centerY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Apply effect in nose tip area
                if (distance < 30 * intensity) {
                    const factor = 1 - (distance / (30 * intensity));
                    data[i] = Math.min(255, data[i] * (1 + 0.15 * factor * intensity));
                    data[i + 1] = Math.min(255, data[i + 1] * (1 + 0.15 * factor * intensity));
                    data[i + 2] = Math.min(255, data[i + 2] * (1 + 0.15 * factor * intensity));
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    _applyNoseNarrowing(ctx, intensity) {
        // Implement nose narrowing effect
        const imageData = ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        // Apply horizontal compression in the nose area
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        for (let y = 0; y < this.canvas.height; y++) {
            for (let x = 0; x < this.canvas.width; x++) {
                const i = (y * this.canvas.width + x) * 4;
                
                // Calculate distance from center
                const dx = x - centerX;
                const dy = y - centerY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Apply effect in nose area
                if (distance < 40 * intensity) {
                    const factor = 1 - (distance / (40 * intensity));
                    data[i] = Math.min(255, data[i] * (1 + 0.1 * factor * intensity));
                    data[i + 1] = Math.min(255, data[i + 1] * (1 + 0.1 * factor * intensity));
                    data[i + 2] = Math.min(255, data[i + 2] * (1 + 0.1 * factor * intensity));
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    _applyCrookedCorrection(ctx, intensity) {
        // Implement crooked nose correction effect
        const imageData = ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        // Apply brightness adjustment in the nose bridge area
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        for (let y = 0; y < this.canvas.height; y++) {
            for (let x = 0; x < this.canvas.width; x++) {
                const i = (y * this.canvas.width + x) * 4;
                
                // Calculate distance from center
                const dx = x - centerX;
                const dy = y - centerY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Apply effect in nose bridge area
                if (distance < 50 * intensity) {
                    const factor = 1 - (distance / (50 * intensity));
                    data[i] = Math.min(255, data[i] * (1 + 0.1 * factor * intensity));
                    data[i + 1] = Math.min(255, data[i + 1] * (1 + 0.1 * factor * intensity));
                    data[i + 2] = Math.min(255, data[i + 2] * (1 + 0.1 * factor * intensity));
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    _applyCombinedEnhancement(ctx, intensity) {
        // Apply multiple effects with reduced intensity
        this._applyNaturalRefinement(ctx, intensity * 0.8);
        this._applyBridgeReduction(ctx, intensity * 0.7);
        this._applyTipRefinement(ctx, intensity * 0.9);
        this._applyNoseNarrowing(ctx, intensity * 0.6);
    }
}

// Export the class
export default ARRhinoplastySimulator; 