from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn
import os
import logging
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

def load_model():
    try:
        # Set your HF token
        hf_token = "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"
        os.environ["HUGGING_FACE_TOKEN"] = hf_token
        
        # Model IDs
        base_model_id = "meta-llama/Meta-Llama-3-8B"
        trained_model_id = "mohd010/mernat_llama_model"
        
        logger.info(f"Loading tokenizer from {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            token=hf_token,
            trust_remote_code=True
        )
        
        logger.info(f"Loading base model from {base_model_id}")
        # Configure model loading with memory optimization
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            token=hf_token,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="offload_folder",  # Folder for disk offloading
            device_map={
                "": "cpu",  # Force CPU for all components
            }
        )
        
        logger.info(f"Loading fine-tuned model from {trained_model_id}")
        model = PeftModel.from_pretrained(
            model,
            trained_model_id,
            token=hf_token,
            trust_remote_code=True,
            device_map={
                "": "cpu",  # Force CPU for all components
            }
        )
        
        model.eval()
        logger.info("Model loading completed successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

# Load the model
try:
    model, tokenizer = load_model()
    logger.info("Initial model loading successful")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        formatted_prompt = f"### Human: {request.prompt}\n### Assistant:"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=128,  # Limit new tokens
                num_return_sequences=1,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=True
            )
            
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear CPU cache
        import gc
        gc.collect()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Assistant:")[-1].strip()
        logger.info(f"Generated response: {response}")
        
        return ChatResponse(response=response)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quick-chat", response_model=ChatResponse)
async def quick_chat(request: ChatRequest):
    try:
        formatted_prompt = f"### Human: {request.prompt}\n### Assistant:"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,  # Very limited response
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Assistant:")[-1].strip()
        return ChatResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
