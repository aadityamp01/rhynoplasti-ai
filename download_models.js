const https = require('https');
const fs = require('fs');
const path = require('path');

// Create models directory if it doesn't exist
const modelsDir = path.join(__dirname, 'models');
if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
}

// Model files to download
const modelFiles = [
    'tiny_face_detector_model-shard1',
    'face_landmark_68_model'
].map(name => ({
    url: `https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/${name}.weights`,
    filename: `${name}.weights`
}));

// Download function
function downloadFile(url, filePath) {
    return new Promise((resolve, reject) => {
        console.log(`Downloading ${url} to ${filePath}...`);
        
        const file = fs.createWriteStream(filePath);
        
        https.get(url, (response) => {
            response.pipe(file);
            
            file.on('finish', () => {
                file.close();
                console.log(`Downloaded ${filePath}`);
                resolve();
            });
        }).on('error', (err) => {
            fs.unlink(filePath, () => {}); // Delete the file if there was an error
            reject(err);
        });
    });
}

// Download all models
async function downloadModels() {
    for (const model of modelFiles) {
        try {
            const filePath = path.join(modelsDir, model.filename);
            await downloadFile(model.url, filePath);
        } catch (error) {
            console.error(`Error downloading ${model.filename}:`, error);
        }
    }
    
    console.log('All models downloaded successfully!');
}

// Run the download
downloadModels(); 