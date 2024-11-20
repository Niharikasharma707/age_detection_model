# import os
# import zipfile
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
# import tensorflow as tf
# from datasets import load_dataset

# app = FastAPI()

# try:
#     print("Loading tokenizer and model...")
#     tokenizer = AutoTokenizer.from_pretrained("basakdemirok/age_detection_tr")
#     model = TFAutoModelForSequenceClassification.from_pretrained("basakdemirok/age_detection_tr")
#     print("Model and tokenizer loaded successfully.")
# except Exception as e:
#     print(f"Error loading model or tokenizer: {e}")
#     raise HTTPException(status_code=500, detail="Error loading model or tokenizer")

# label_to_age_category = {
#     "LABEL_0": "18-24",
#     "LABEL_1": "25-34",
#     "LABEL_2": "35-44",
#     "LABEL_3": "45-54",
#     "LABEL_4": "55-64",
#     "LABEL_5": "65+"
# }

# def predict_age_category(text):
#     try:
#         inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
#         outputs = model(inputs)
#         logits = outputs.logits
#         probabilities = tf.nn.softmax(logits, axis=1).numpy().squeeze()
#         predicted_class_id = probabilities.argmax()
#         predicted_label = model.config.id2label.get(predicted_class_id, "Unknown")
#         predicted_age_category = label_to_age_category.get(predicted_label, "Unknown")
#         return {
#             "predicted_age_category": predicted_age_category,
#             "probability": float(probabilities[predicted_class_id])  # Convert to native float
#         }
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         return {
#             "predicted_age_category": "Error",
#             "probability": 0.0,
#             "error": str(e)
#         }

# @app.post("/predict_age")
# async def predict_age_from_zip(file: UploadFile = File(...)):
#     if not file.filename.endswith('.zip'):
#         raise HTTPException(status_code=400, detail="Uploaded file is not a ZIP file")

#     # Save the uploaded file to a temporary directory
#     temp_file_path = "temp_upload.zip"
#     try:
#         with open(temp_file_path, "wb") as temp_file:
#             content = await file.read()
#             temp_file.write(content)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {e}")

#     # Extract the ZIP file to a temporary directory
#     temp_dir = "temp_extracted"
#     os.makedirs(temp_dir, exist_ok=True)

#     try:
#         with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
#             zip_ref.extractall(temp_dir)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error extracting ZIP file: {e}")

#     # Process the extracted files
#     predictions = {}
#     for filename in os.listdir(temp_dir):
#         file_path = os.path.join(temp_dir, filename)
#         if os.path.isfile(file_path) and file_path.endswith('.txt'):
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     text = f.read()
#                     prediction = predict_age_category(text)
#                     predictions[filename] = prediction
#             except Exception as e:
#                 predictions[filename] = {"error": str(e)}

#     # Clean up the temporary files
#     try:
#         for filename in os.listdir(temp_dir):
#             os.remove(os.path.join(temp_dir, filename))
#         os.rmdir(temp_dir)
#         os.remove(temp_file_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error cleaning up temporary files: {e}")

#     return JSONResponse(content=predictions)

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# ==================================================================
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from io import StringIO

app = FastAPI()

try:
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("basakdemirok/age_detection_tr")
    model = TFAutoModelForSequenceClassification.from_pretrained("basakdemirok/age_detection_tr")
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise HTTPException(status_code=500, detail="Error loading model or tokenizer")

label_to_age_category = {
    "LABEL_0": "18-24",
    "LABEL_1": "25-34",
    "LABEL_2": "35-44",
    "LABEL_3": "45-54",
    "LABEL_4": "55-64",
    "LABEL_5": "65+"
}

def predict_age_category(text):
    try:

        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)

        outputs = model(inputs)
        logits = outputs.logits
    
        probabilities = tf.nn.softmax(logits, axis=1).numpy().squeeze()
        predicted_class_id = probabilities.argmax()

        predicted_label = model.config.id2label.get(predicted_class_id, "Unknown")

        predicted_age_category = label_to_age_category.get(predicted_label, "Unknown")

        if predicted_age_category == "Unknown":
            return {
                "statement": text,
                "predicted_age_category": "Error",
                "probability": 0.0
            }
        else:
            return {
                "statement": text,
                "predicted_age_category": predicted_age_category,
                "probability": float(probabilities[predicted_class_id])
            }
    except Exception as e:
   
        return {
            "statement": text,
            "predicted_age_category": "Error",
            "probability": 0.0
        }

@app.post("/predict_age_csv")
async def predict_age_from_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Uploaded file is not a CSV file")

    try:

        contents = await file.read()
        content_str = contents.decode('utf-8')
        df = pd.read_csv(StringIO(content_str), header=None)  
        
    
        print("CSV Content Preview:")
        print(df.head())

        predictions = {}
        for index, row in df.iterrows():
      
            for column_index, text in row.items():
                text = str(text).strip()  #will trim the text withnext line 
                if text:
                    prediction = predict_age_category(text)
                    predictions[f'row_{index}'] = prediction
                    break 
                else:
                 
                    predictions[f'row_{index}'] = {
                        "statement": "",
                        "predicted_age_category": "Error",
                        "probability": 0.0
                    }
                    break 

        return JSONResponse(content=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
