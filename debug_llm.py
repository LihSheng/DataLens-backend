import os
from dotenv import load_dotenv
load_dotenv()
from app.services.vectorstore_service import get_llm
llm = get_llm()
print('LLM type:', type(llm).__name__)
print('Model:', getattr(llm, 'model_name', 'unknown'))