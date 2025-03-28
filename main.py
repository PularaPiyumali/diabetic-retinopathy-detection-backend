from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://diabetic-retinopathy-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load the model
try:
    model = tf.keras.models.load_model('model26.h5')
    multiclass_model = tf.keras.models.load_model('model6.h5') 
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

SEVERITY_LEVELS = ["Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Image opened successfully. Mode: {image.mode}")
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            logger.info("Converted RGBA to RGB")
        
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        logger.info("Image preprocessed successfully")
        
        return image_array
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Diabetic Retinopathy Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    try:
        #Read the image file
        contents = await file.read()
        logger.info("File read successfully")
        
        #Open and preprocess the image
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Image opened successfully. Mode: {image.mode}")
        
        #Convert RGBA to RGB (if necessary)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            logger.info("Converted RGBA to RGB")
        
        #Preprocess the image
        image = image.resize((224, 224))
        image_array = np.array(image) 
        image_array = np.expand_dims(image_array, axis=0)
        logger.info("Image preprocessed successfully")
        
        #Make prediction
        predictions = model.predict(image_array)
        prediction_score = float(predictions[0][0])

        result = "Diabetic Retinopathy" if prediction_score >= 0.5 else "No Diabetic Retinopathy"
        confidence = prediction_score if prediction_score >= 0.5 else (1 - prediction_score) * 100
        
        logger.info(f"Raw prediction: {prediction_score}")
        logger.info(f"Result: {result}")
        logger.info(f"Confidence: {confidence}%")
        logger.info("Prediction made successfully")
        
        return {
            "severity": result,
            "confidence": confidence, 
        
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
def predict_dr(image_array):
    """Predict DR using binary and multiclass models"""
    try:
        # Binary prediction (DR or No DR)
        binary_prediction = model.predict(image_array)
        binary_score = float(binary_prediction[0][0])
        has_dr = binary_score >= 0.5
        
        result = {
            "hasDR": has_dr,
            "confidence": binary_score if has_dr else (1 - binary_score)
        }
        
        # If DR is detected, predict severity with multiclass model
        if has_dr:
            multiclass_prediction = multiclass_model.predict(image_array)
            severity_index = np.argmax(multiclass_prediction[0])
            severity = SEVERITY_LEVELS[severity_index]
            severity_confidence = float(multiclass_prediction[0][severity_index])
            
            result["severity"] = severity
            result["confidence"] = severity_confidence
        
        logger.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

def generate_recommendations(baseline_result, followup_result):
    """Generate recommendations based on results"""
    recommendations = []
    
    # Basic recommendations
    recommendations.append("Continue regular monitoring")
    
    # If either result shows DR
    if baseline_result["hasDR"] or followup_result["hasDR"]:
        recommendations.append("Maintain strict blood sugar control")
        
    # If follow-up shows improvement
    if baseline_result["hasDR"] and (not followup_result["hasDR"] or 
        (followup_result["hasDR"] and "severity" in baseline_result and "severity" in followup_result and
         SEVERITY_LEVELS.index(baseline_result["severity"]) > SEVERITY_LEVELS.index(followup_result["severity"]))):
        recommendations.append("Continue with current treatment plan")
        recommendations.append("Schedule next follow-up in 6 months")
    
    # If follow-up shows worsening
    elif (not baseline_result["hasDR"] and followup_result["hasDR"]) or (
        baseline_result["hasDR"] and followup_result["hasDR"] and 
        "severity" in baseline_result and "severity" in followup_result and
        SEVERITY_LEVELS.index(baseline_result["severity"]) < SEVERITY_LEVELS.index(followup_result["severity"])):
        recommendations.append("Consider treatment plan adjustment")
        recommendations.append("Schedule next follow-up in 3 months")
        recommendations.append("Consult with ophthalmologist for potential interventions")
    
    # If severe or proliferative DR in either image
    if (baseline_result["hasDR"] and "severity" in baseline_result and 
        baseline_result["severity"] in ["Severe NPDR", "Proliferative DR"]) or (
        followup_result["hasDR"] and "severity" in followup_result and 
        followup_result["severity"] in ["Severe NPDR", "Proliferative DR"]):
        recommendations.append("Immediate referral to retina specialist")
    
    return recommendations

def determine_overall_change(baseline_result, followup_result):
    """Determine overall change between baseline and follow-up"""
    if not baseline_result["hasDR"] and not followup_result["hasDR"]:
        return "No diabetic retinopathy in either image"
    
    if not baseline_result["hasDR"] and followup_result["hasDR"]:
        return f"Worsening: Development of diabetic retinopathy ({followup_result['severity']})"
    
    if baseline_result["hasDR"] and not followup_result["hasDR"]:
        return "Improvement: Diabetic retinopathy resolved"
    
    # Both have DR, compare severity
    if "severity" in baseline_result and "severity" in followup_result:
        baseline_severity_index = SEVERITY_LEVELS.index(baseline_result["severity"])
        followup_severity_index = SEVERITY_LEVELS.index(followup_result["severity"])
        
        if baseline_severity_index > followup_severity_index:
            return f"Improvement: From {baseline_result['severity']} to {followup_result['severity']}"
        elif baseline_severity_index < followup_severity_index:
            return f"Worsening: From {baseline_result['severity']} to {followup_result['severity']}"
        else:
            return f"Stable: Remained at {baseline_result['severity']}"
    
    return "Unable to determine change (missing severity information)"

# New compare endpoint
@app.post("/compare")
async def compare_images(
    baseline_file: UploadFile = File(...),
    followup_file: UploadFile = File(...)
):
    logger.info(f"Received files: {baseline_file.filename} and {followup_file.filename}")
    try:
        # Read and process baseline image
        baseline_contents = await baseline_file.read()
        baseline_array = preprocess_image(baseline_contents)
        baseline_result = predict_dr(baseline_array)
        
        # Read and process follow-up image
        followup_contents = await followup_file.read()
        followup_array = preprocess_image(followup_contents)
        followup_result = predict_dr(followup_array)
        
        # Generate comparison results
        overall_change = determine_overall_change(baseline_result, followup_result)
        recommendations = generate_recommendations(baseline_result, followup_result)
        
        result = {
            "baselineResult": baseline_result,
            "followUpResult": followup_result,
            "overallChange": overall_change,
            "recommendations": recommendations
        }
        
        logger.info("Comparison complete")
        return result
        
    except Exception as e:
        logger.error(f"Error comparing images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
