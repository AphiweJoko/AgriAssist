from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import cohere
from dotenv import load_dotenv
import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv()


# Load environment variables
load_dotenv()


app = Flask(__name__, static_folder='../static', static_url_path='')

CORS(app)

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))



# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'frontend.html')

# Health check endpoint

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
       "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    # Get text input (from either voice or typed input)
    voice_text = request.form.get('voice_text', '').strip()
    
    # Get uploaded image file
    image = request.files.get('image')
    
    # Initialize response components
    text_response = ""
    image_analysis = ""
    
    # Process text input if provided
    if voice_text:
        text_response = generate_text_response_with_cohere(voice_text)
    
    # Process image if provided
    if image:
        try:
            # Secure filename and save temporarily
            filename = secure_filename(image.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(temp_path)
            
            # Perform image analysis
            image_analysis = analyze_image(temp_path)
            
            # If we have image analysis but no text query, generate a follow-up
            if image_analysis and not voice_text:
                text_response = generate_followup_from_image(image_analysis)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            image_analysis = f"Image analysis error: {str(e)}"
    
    # Prepare response
    response_data = {
        "status": "success",
        "text_analysis": text_response if (text_response or voice_text) else None,
        "image_analysis": image_analysis if image else None,
        "message": ""
    }
    
    # Build human-readable message
    if text_response and image_analysis:
        response_data["message"] = f"TEXT ANALYSIS:\n{text_response}\n\nIMAGE ANALYSIS:\n{image_analysis}"
    elif text_response:
        response_data["message"] = text_response
    elif image_analysis:
        response_data["message"] = image_analysis
    else:
        response_data.update({
            "status": "error",
            "message": "No valid input provided"
        })
        return jsonify(response_data), 400
    
    return jsonify(response_data)

def generate_text_response_with_cohere(query):
    """Generate plant care advice using Cohere's API"""
    try:
        prompt = f"""You are an expert botanist and plant care specialist. 
Provide detailed, professional advice for this plant issue:

Problem: {query}

Analysis:
1. Identify the most likely causes (3-5 possibilities)
2. Explain each cause in simple terms
3. Provide step-by-step solutions
4. Include preventive measures

Format your response with clear headings and bullet points."""

        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        
        return response.generations[0].text.strip()
    
    except Exception as e:
        return f"AI analysis error: {str(e)}"

def generate_followup_from_image(image_analysis):
    """Generate follow-up questions based on image analysis"""
    try:
        prompt = f"""Based on this plant image analysis:
{image_analysis}

Generate 2-3 follow-up questions the user might ask to get more specific advice about their plant's condition.
Format as a numbered list with brief explanations why each question would be helpful."""

        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=300,
            temperature=0.5
        )
        
        return "IMAGE ANALYSIS SUGGESTS THESE FOLLOW-UP QUESTIONS:\n" + response.generations[0].text.strip()
    
    except Exception as e:
        return f"Could not generate follow-up questions: {str(e)}"

def analyze_image(image_path):
    """Perform basic image analysis with OpenCV"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Could not read image file"
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate color statistics
        hue = np.mean(hsv[:,:,0])
        sat = np.mean(hsv[:,:,1])
        val = np.mean(hsv[:,:,2])
        
        # Detect issues based on color
        analysis = []
        if 40 < hue < 70 and sat > 100:  # Yellowing
            severity = min(100, int((hue - 40) / 30 * 100))
            analysis.append(f"Detected leaf yellowing ({severity}% severity)")
        elif val < 80 and sat < 50:  # Browning
            analysis.append("Detected leaf browning/drying")
        elif np.percentile(hsv[:,:,2], 10) < 40:  # Dark spots
            analysis.append("Detected dark spots on leaves")
        else:
            analysis.append("No major issues detected")
            
        # Count green pixels (rough health estimate)
        green_pixels = np.sum((hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 40))
        health_score = min(100, green_pixels / (img.shape[0] * img.shape[1]) * 300)
        analysis.append(f"Plant health score: {int(health_score)}/100")
        
        return "\n".join(analysis)
        
    except Exception as e:
        return f"Image processing failed: {str(e)}"

if __name__ == '__main__':
    print("✅ Flask server running at http://127.0.0.1:5000")
    app.run(debug=True)