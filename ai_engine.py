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

# --- MODEL DEFINITIONS ---
# Primary: Super fast, but strict quotas
PRIMARY_MODEL = "google/gemini-2.0-flash-001"
# Backup: Slower, but generous free tier
BACKUP_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

def perform_deep_research(target_profile, offer_context):
    """
    SMART ANCHOR SEARCH: Combines 'Target' + 'Offer Keyword'.
    """
    try:
        query = f"{target_profile} {offer_context} industry challenges news 2025"
        print(f"🕵️ Anchored Search: {query}")
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
        if not results:
            return "No specific news found. Stick strictly to the user's provided context."
            
        summary = "\n".join([f"NEWS TITLE: {r['title']} -> SNIPPET: {r['body']}" for r in results])
        return summary

    except Exception as e:
        print(f"⚠️ Search Skipped: {e}")
        return "Search unavailable. Use the user's provided context strictly."

def extract_json(text_output):
    text = text_output.strip()
    try:
        # Regex to find JSON structure
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except:
        pass

    return {
        "subject_a": "Error Parsing Output",
        "body_a": text,
        "style_a": "Raw Output",
        "subject_b": "Error Parsing Output",
        "body_b": "See Option A",
        "style_b": "Raw Output"
    }

def run_agent(target, service, source, length_preference, style_preferences, creativity):
    # 1. RESEARCH PHASE
    topic_anchor = f"{service} {source}"[:50] 
    research_data = perform_deep_research(target, topic_anchor)
    
    # 2. PROVEN TEMPLATE LIBRARIES
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

    # Select random templates
    sel_subj_a = random.choice(subject_templates)
    sel_subj_b = random.choice([s for s in subject_templates if s != sel_subj_a])
    sel_hook_a = random.choice(hook_templates)
    sel_hook_b = random.choice([h for h in hook_templates if h != sel_hook_a])

    # 3. DEFINE THE PROMPT (Reused for both models)
    prompt = ChatPromptTemplate.from_template(
        """
        # STRATEGIC ROLE: ELITE B2B OUTREACH ENGINEER
        
        ## INPUTS
        - **Target:** `{target}`
        - **Offer:** `{service}`
        - **Context:** `{source}`
        - **Scraped Intel:** `{research_data}`
        - **Length Rule:** `{length_preference}`
        
        ## ***YOUR ASSIGNMENT***
        You must fill in the blanks of the specific templates below using the Scraped Intel.
        
        ### **OPTION A: The "Direct Value" Approach**
        - **Subject Line Structure:** "{subject_a}" 
          *(Replace bracketed text with real details from Intel/Target)*
        - **Icebreaker Structure:** "{hook_a}"
          *(Fill blanks with specific news/posts found in Intel)*
        - **Body:** Pivot immediately to how `{service}` solves the pain mentioned in `{source}`.
        - **CTA:** Use a "No-Oriented" question (e.g., "Are you opposed to...?").

        ### **OPTION B: The "Insight-Asset" Approach**
        - **Subject Line Structure:** "{subject_b}"
          *(Replace bracketed text with real details)*
        - **Icebreaker Structure:** "{hook_b}"
          *(Fill blanks with specific news/posts found in Intel)*
        - **Body:** Offer a specific asset (PDF, Calculator, Audit) related to `{service}`.
        - **CTA:** "Would it be a waste of time to send this over?"

        ## ***QUALITY CONTROL***
        1. **NO FAKE NAMES:** If Target is "Custom Broker", do NOT invent a company name like "Custom Cursor". Use "your brokerage".
        2. **IGNORE IRRELEVANT BRANDS:** If Intel mentions software brands unrelated to the target role, ignore them.
        3. **LOWERCASE SUBJECTS:** Ensure subject lines are lowercase.
        4. **LENGTH:** If "{length_preference}" is Short, keep it under 75 words.

        OUTPUT FORMAT (JSON ONLY):
        {{
            "subject_a": "...", "body_a": "...", "style_a": "Direct Value",
            "subject_b": "...", "body_b": "...", "style_b": "Insight-Asset"
        }}
        """
    )

    # 4. THE FALLBACK LOGIC
    try:
        # --- ATTEMPT 1: GEMINI FLASH (Speed) ---
        print(f"🚀 Attempting Primary: {PRIMARY_MODEL}")
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=PRIMARY_MODEL,
            temperature=float(creativity),
        )
        chain = prompt | llm | StrOutputParser()
        raw_result = chain.invoke({
            "target": target, "service": service, "source": source,
            "research_data": research_data, "length_preference": length_preference,
            "subject_a": sel_subj_a, "subject_b": sel_subj_b,
            "hook_a": sel_hook_a, "hook_b": sel_hook_b
        })
        return extract_json(raw_result)

    except Exception as e:
        # --- ATTEMPT 2: LLAMA 3.3 (Backup) ---
        print(f"⚠️ Primary Failed ({str(e)}). Switching to Backup: {BACKUP_MODEL}")
        try:
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                model=BACKUP_MODEL,
                temperature=float(creativity),
            )
            chain = prompt | llm | StrOutputParser()
            raw_result = chain.invoke({
                "target": target, "service": service, "source": source,
                "research_data": research_data, "length_preference": length_preference,
                "subject_a": sel_subj_a, "subject_b": sel_subj_b,
                "hook_a": sel_hook_a, "hook_b": sel_hook_b
            })
            return extract_json(raw_result)
            
        except Exception as e2:
            # --- FINAL FAILSAFE ---
            return {
                "subject_a": "System Error", 
                "body_a": f"All models busy. Please wait 10s and try again.\nError: {str(e2)}", 
                "style_a": "Failed",
                "subject_b": "Error", "body_b": "Retry", "style_b": "Failed"
            }