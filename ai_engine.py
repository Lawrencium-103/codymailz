import os
import json
import re
import random
from dotenv import load_dotenv

try: 
    load_dotenv() 
except: 
    pass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from duckduckgo_search import DDGS

# --- CONFIGURATION ---
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("⚠️ WARNING: OPENROUTER_API_KEY is missing.")

# 1. MODEL SWITCH -> GEMINI FLASH (Speed King)
# This model is 5-10x faster than Llama 3.3 70B and excellent at JSON.
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="google/gemini-2.0-flash-001", 
    temperature=0.7,
)

def perform_deep_research(target_profile, offer_context):
    """
    Optimized Search: Fetches only 1 result to save time.
    """
    try:
        query = f"{target_profile} {offer_context} industry challenges news 2025"
        print(f"🕵️ Searching: {query}")
        
        with DDGS() as ddgs:
            # Reduced to 1 result for speed
            results = list(ddgs.text(query, max_results=1))
            
        if not results:
            return "No specific news found. Use general industry logic."
            
        summary = f"NEWS: {results[0]['title']} -> {results[0]['body']}"
        return summary

    except Exception as e:
        print(f"⚠️ Search Skipped: {e}")
        return "Search unavailable. Rely on user context."

def extract_json(text_output):
    """
    Advanced Cleaner: Strips 'Here is the JSON' text and Markdown blocks.
    """
    text = text_output.strip()
    
    # 1. Remove Markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    
    # 2. Find the first '{' and last '}'
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            return json.loads(json_str)
    except Exception as e:
        print(f"JSON Parse Failed: {e}")

    # 3. Fallback Layout if parsing fails (Prevents 'System Error')
    return {
        "subject_a": "Generated (Raw Format)",
        "body_a": text,
        "style_a": "Raw Output",
        "subject_b": "Variation B",
        "body_b": "See above.",
        "style_b": "Raw Output"
    }

def run_agent(target, service, source, length_preference, style_preferences, creativity):
    # 1. RESEARCH PHASE
    topic_anchor = f"{service} {source}"[:50] 
    research_data = perform_deep_research(target, topic_anchor)
    
    # 2. TEMPLATE SELECTION
    subject_templates = [
        "quick question about [Target Company]'s project",
        "idea to help [Target Company] boost [Metric]",
        "can we help solve [Specific Pain]?",
        "congrats on [Recent Achievement] - quick thought",
        "question re: [Industry Trend]",
        "free resource for [Specific Challenge]"
    ]
    
    hook_templates = [
        "I noticed your team recently tackled [Challenge/Project]—I admire how you're leading innovation in [Industry].",
        "I was reading about [Industry Trend] and how it could impact [Target Company] specifically.",
        "I noticed your current process doesn't mention [Compliance/Method]—companies ignoring this often face [Negative Outcome]."
    ]

    selected_subject_a = random.choice(subject_templates)
    selected_subject_b = random.choice([s for s in subject_templates if s != selected_subject_a])
    selected_hook_a = random.choice(hook_templates)
    selected_hook_b = random.choice([h for h in hook_templates if h != selected_hook_a])

    # 3. DYNAMIC TEMPERATURE UPDATE
    llm.temperature = float(creativity)

    print(f"🚀 Generating (Gemini Flash). Length: {length_preference}")

    # 4. PROMPT
    prompt = ChatPromptTemplate.from_template(
        """
        You are an Elite B2B Outreach Engineer.
        
        INPUTS:
        - Target: `{target}`
        - Offer: `{service}`
        - Context: `{source}`
        - Intel: `{research_data}`
        - Length Instruction: `{len_instruction}`
        
        INSTRUCTIONS:
        Fill the templates below. 
        
        ### OPTION A: Strategy A
        - Subject: "{subject_a}" (Fill brackets with real info)
        - Hook: "{hook_a}" (Fill brackets with Intel)
        - Body: Explain the "Transformation" (Old Way vs New Way).
        - CTA: "Are you opposed to...?"

        ### OPTION B: Insight-Asset
        - Subject: "{subject_b}"
        - Hook: "{hook_b}"
        - Body: Offer a specific asset (PDF/Calculator) related to `{service}`.
        - CTA: "Would it be a waste of time to send this?"

        **CRITICAL:** 
        1. If `{len_instruction}` says "Short", keep total words under 75 per email.
        2. If `{len_instruction}` says "Detailed", use bullet points for value.
        3. OUTPUT RAW JSON ONLY. DO NOT WRITE "Here is the output".

        FORMAT:
        {{
            "subject_a": "...", 
            "body_a": "...", 
            "style_a": "Strategic", 
            "subject_b": "...", 
            "body_b": "...", 
            "style_b": "Asset-First"
        }}
        """
    )

    # Logic for length instruction
    if "Short" in length_preference:
        len_instruction = "Short. Under 75 words. Concise."
    else:
        len_instruction = "Detailed. 100-150 words. Use bullet points."

    chain = prompt | llm | StrOutputParser()

    try:
        raw_result = chain.invoke({
            "target": target,
            "service": service,
            "source": source,
            "research_data": research_data,
            "subject_a": selected_subject_a,
            "subject_b": selected_subject_b,
            "hook_a": selected_hook_a,
            "hook_b": selected_hook_b,
            "len_instruction": len_instruction
        })
        return extract_json(raw_result)

    except Exception as e:
        return {
            "subject_a": "Error", "body_a": f"Details: {str(e)}", "style_a": "Failed",
            "subject_b": "Error", "body_b": "Retry", "style_b": "Failed"
        }