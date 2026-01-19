# src/rag_engine/sar_generator.py
import requests
import os
from datetime import datetime
from rag_engine.vector_store import CTAFVectorStore

class SARGenerator:
    def __init__(self, ollama_url="http://localhost:11434/api/generate"):
        self.vector_store = CTAFVectorStore()
        self.ollama_url = ollama_url

    def generate_report(self, tx_data, ml_score):
        # 1. Retrieve Regulatory Context
        query_text = f"rules for {tx_data.get('payment_method', 'unknown')} in {tx_data.get('governorate', 'unknown')}"
        context_result = self.vector_store.query(query_text, n_results=2)
        
        # Extract context from results
        context = ""
        if context_result and 'documents' in context_result:
            docs = context_result['documents']
            if docs and len(docs) > 0 and len(docs[0]) > 0:
                context = docs[0][0]  # Get the first document
        
        # 2. Automated SAR Prompting
        prompt = f"""INVESTIGATION: Analyze suspicious activity for user {tx_data.get('user_id', 'UNKNOWN')}. 
        Transaction ID: {tx_data.get('transaction_id', 'UNKNOWN')}
        ML Score: {ml_score}
        Amount: {tx_data.get('amount_tnd', 0)} TND
        Governorate: {tx_data.get('governorate', 'UNKNOWN')}
        Payment Method: {tx_data.get('payment_method', 'UNKNOWN')}
        Timestamp: {tx_data.get('timestamp', 'UNKNOWN')}
        Regulatory Context: {context if context else 'No specific regulatory context found.'}
        
        Generate a professional Suspicious Activity Report (SAR) for CTAF. Include:
        1. Executive summary of suspicious activity
        2. Risk factors observed
        3. Regulatory violations potentially involved
        4. Recommended next steps for investigation"""
        
        # 3. Local Inference (Data Privacy Compliant)
        try:
            response = requests.post(self.ollama_url, json={"model": "llama3.1", "prompt": prompt, "stream": False})
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def save_report(self, tx_data, report, ml_score):
        """Save the SAR report to file"""
        os.makedirs("./data/reports", exist_ok=True)
        
        tx_id = tx_data.get('transaction_id', 'unknown')
        filename = f"SAR_{tx_id.replace(':', '_').replace('-', '_')}.txt"
        filepath = os.path.join("./data/reports", filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SUSPICIOUS ACTIVITY REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Transaction ID: {tx_data.get('transaction_id', 'UNKNOWN')}\n")
            f.write(f"User ID: {tx_data.get('user_id', 'UNKNOWN')}\n")
            f.write(f"ML Score: {ml_score}\n")
            f.write(f"Amount: {tx_data.get('amount_tnd', 0)} TND\n")
            f.write(f"Governorate: {tx_data.get('governorate', 'UNKNOWN')}\n")
            f.write(f"Payment Method: {tx_data.get('payment_method', 'UNKNOWN')}\n")
            f.write(f"Timestamp: {tx_data.get('timestamp', 'UNKNOWN')}\n")
            f.write(f"\nREPORT:\n{report}\n")
        
        return filepath