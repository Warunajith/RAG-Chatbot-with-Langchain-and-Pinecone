from pydantic import BaseModel

class ChatReq(BaseModel):
    question:str
    

class SessionResponse(BaseModel):
    session_token: str