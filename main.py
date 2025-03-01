from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app import (
    load_pdfs_from_folder,
    get_bot_response,
    clear_chat_history,
    is_pdfs_loaded,
    get_chat_history,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class UserInput(BaseModel):
    message: str

@app.post("/ask")
async def ask_question(user_input:UserInput):
    print(f"user input: {user_input}")
    try:
        if not is_pdfs_loaded():
            load_pdfs_from_folder()
        
        user_message = user_input.message
        response = get_bot_response(user_message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_history():
    clear_chat_history()
    return {"message": "Chat history cleared."}

@app.get("/status")
async def status():
    return {
        "pdfs_loaded": is_pdfs_loaded(),
        "chat_history_length": len(get_chat_history()),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)