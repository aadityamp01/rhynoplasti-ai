// Banuba SDK Integration for Rhinoplasty Simulation
import { BanubaPlayer } from '@banuba/webar';

class BanubaRhinoplastySimulator {
    constructor(clientToken) {
        this.clientToken = clientToken;
        this.player = null;
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
            // Initialize Banuba Player
            this.player = new BanubaPlayer({
                clientToken: this.clientToken,
                container: document.getElementById(containerId),
                width: 640,
                height: 480,
                camera: true,
                effects: Object.values(this.effects)
            });

            // Wait for player to be ready
            await this.player.ready();
            console.log("Banuba Player initialized successfully");
            return true;
        } catch (error) {
            console.error("Failed to initialize Banuba Player:", error);
            return false;
        }
    }

    async applyEffect(effectName, intensity = 0.7) {
        if (!this.player) {
            throw new Error("Banuba Player not initialized");
        }

        const effectFile = this.effects[effectName];
        if (!effectFile) {
            throw new Error(`Effect ${effectName} not found`);
        }

        try {
            // Apply the selected effect
            await this.player.applyEffect(effectFile);
            
            // Set effect intensity
            await this.player.setEffectParameter("intensity", intensity);
            
            console.log(`Applied effect: ${effectName} with intensity: ${intensity}`);
            return true;
        } catch (error) {
            console.error(`Failed to apply effect ${effectName}:`, error);
            return false;
        }
    }

    async captureImage() {
        if (!this.player) {
            throw new Error("Banuba Player not initialized");
        }

        try {
            // Capture the current frame
            const imageData = await this.player.captureImage();
            return imageData;
        } catch (error) {
            console.error("Failed to capture image:", error);
            throw error;
        }
    }

    async stop() {
        if (this.player) {
            await this.player.destroy();
            this.player = null;
        }
    }
}

// Export the class
export default BanubaRhinoplastySimulator; 