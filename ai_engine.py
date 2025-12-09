import os
import json
from dotenv import load_dotenv

try: 
    load_dotenv() 
except: 
    pass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("❌ CRITICAL: OPENROUTER_API_KEY is missing.")

# Using Llama 3.3 70B (Excellent at professional nuance)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="meta-llama/llama-3.3-70b-instruct:free",
    temperature=0.6, # Slightly lower temp for more professional consistency
)

def clean_json_string(text_output):
    """Cleans Markdown wrappers if the AI adds them"""
    text = text_output.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# --- THE "CONSULTATIVE AUTHORITY" PROMPT ---
def run_agent(target, service, source, length_preference):
    print(f"🚀 Generating Professional Protocol: {service} -> Target: {target}")

    prompt = ChatPromptTemplate.from_template(
        """
        You are a highly expensive Strategy Consultant & AI Engineer. 
        You do not "sell"; you "advise."

        TASK:
        Write two professional B2B cold emails (Option A and Option B).

        INPUTS:
        - My Solution: {service}
        - Target Profile: {target}
        - Research Context (Icebreaker): {source}
        - Length Strategy: {length_preference}

        CRITICAL WRITING GUIDELINES (The "Expert" Protocol):
        1. **Professional Formatting:** Use standard capitalization, full stops, and proper grammar. No "text speak."
        2. **The "Gap" Technique:** Do not just state the problem. Explain the *implication* of the problem to show you understand their industry. (e.g., Don't just say "Compliance is hard." Say "The manual mapping of HS codes for compliance creates a massive liability risk.")
        3. **Unique Icebreaker:** You MUST start the email by referencing the {{source}} specifically. Connect that source material to a business insight.
        4. **Peer-to-Peer Tone:** Sound like an equal. Confident, direct, and helpful. 
        5. **No Fluff:** Do not use "I hope this email finds you well" or "I wanted to reach out." Start directly with the value or the hook.

        OUTPUT REQUIREMENTS (JSON Format):
        
        OPTION A (The Strategic Advisor):
        - Focus on the high-level business outcome (Risk, Liability, Speed).
        - Use the specific context to frame a strategic question.
        
        OPTION B (The Technical Solution):
        - Focus on the workflow (Automating the manual grunt work).
        - Be more direct about the "Mechanism" of how it works.

        Return ONLY valid JSON with keys: subject_a, body_a, style_a, subject_b, body_b, style_b.
        """
    )

    chain = prompt | llm | StrOutputParser()

    try:
        # 1. Generate Raw Text
        raw_result = chain.invoke({
            "target": target,
            "service": service,
            "source": source,
            "length_preference": length_preference
        })
        
        # 2. Clean and Parse
        cleaned_result = clean_json_string(raw_result)
        return json.loads(cleaned_result)

    except Exception as e:
        error_msg = str(e)
        return {
            "subject_a": "System Error", 
            "body_a": f"Error Details: {error_msg}. \n\nPlease check your inputs or try again.", 
            "style_a": "Failed",
            "subject_b": "System Error", 
            "body_b": "Please check Option A for details.", 
            "style_b": "Failed"
        }