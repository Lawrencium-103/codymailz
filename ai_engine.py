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

def perform_deep_research(target_profile, offer_context):
    """
    SMART ANCHOR SEARCH: 
    Combines 'Target' + 'Offer Keyword' to prevent context drift.
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
    
    # 2. DEFINE LENGTH LOGIC
    # We create specific instructions based on what the user clicked
    if "Short" in length_preference:
        len_instruction = "STRICT WORD COUNT: 50-75 words. Be punchy, direct, and concise. No fluff. Get to the point immediately."
    else:
        # Detailed/Consultant Logic
        len_instruction = (
            "STRICT WORD COUNT: 100-180 words. You MUST include a 'Transformation' section. "
            "Explain the 'Current Hell' (Pain) vs 'Future Heaven' (Gain). "
            "Use a bullet list (•) to show 3 specific value points or insights."
        )

    # 3. PROVEN TEMPLATE LIBRARIES
    subject_templates = [
        "quick question about [Target Company]'s project",
        "idea to help [Target Company] boost [Metric]",
        "noticed something on your website",
        "can we help solve [Specific Pain]?",
        "congrats on [Recent Achievement] - quick thought",
        "a way to save [Target Role] time",
        "question re: [Industry Trend]",
        "free resource for [Specific Challenge]"
    ]
    
    hook_templates = [
        "I noticed your team recently tackled [Challenge/Project]—I admire how you're leading innovation in [Industry].",
        "Your recent post on [Topic]—especially the part about [Specific Detail]—really caught my attention.",
        "I was reading about [Industry Trend] and how it could impact [Target Company] specifically.",
        "I noticed your current process doesn't mention [Compliance/Method]—companies ignoring this often face [Negative Outcome]."
    ]

    selected_subject_a = random.choice(subject_templates)
    selected_subject_b = random.choice([s for s in subject_templates if s != selected_subject_a])
    
    selected_hook_a = random.choice(hook_templates)
    selected_hook_b = random.choice([h for h in hook_templates if h != selected_hook_a])

    # 4. DYNAMIC MODEL LOADING
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=float(creativity), 
    )

    print(f"🚀 Generating. Length Mode: {length_preference}")

    # 5. THE PROMPT
    prompt = ChatPromptTemplate.from_template(
        """
        # STRATEGIC ROLE: ELITE B2B OUTREACH ENGINEER
        
        ## INPUTS
        - **Target:** `{target}`
        - **Offer:** `{service}`
        - **Context:** `{source}`
        - **Scraped Intel:** `{research_data}`
        - **Length Strategy:** `{len_instruction}`
        
        ## ***YOUR ASSIGNMENT***
        You must fill in the blanks of the specific templates below using the Scraped Intel.
        
        ### **OPTION A: The Strategy Selection**
        - **Subject Line:** "{subject_a}" (Fill blanks with real details)
        - **Icebreaker:** "{hook_a}" (Fill blanks with specific news/posts found in Intel)
        - **BODY RULES:** Follow the `{len_instruction}` exactly.
        - **CTA:** Use a "No-Oriented" question (e.g., "Are you opposed to...?").
        - **INVENT A FINDING:** If research is thin, invent a plausible specific detail about {target} to fit the hook (e.g., "I saw on LinkedIn you handle automotive imports").

        ### **OPTION B: The "Insight-Asset" Approach**
        - **Subject Line:** "{subject_b}" (Fill blanks)
        - **Icebreaker:** "{hook_b}" (Fill blanks)
        - **Body:** This must be High-Value. Explain *Why* this matters. If Length is 'Detailed', give 3 bullet points on how the asset helps.
        - **Asset:** Offer a specific resource (PDF, Calculator, Audit) related to `{service}`.
        - **CTA:** "Would it be a waste of time to send this over?"

        ## ***QUALITY CONTROL***
        1. **NO FAKE NAMES:** If Target is "Custom Broker", use "your brokerage", do NOT invent "Custom Cursor".
        2. **IGNORE IRRELEVANT BRANDS:** If Intel mentions software brands unrelated to the target role, ignore them.
        3. **LOWERCASE SUBJECTS:** Keep subject lines lowercase.
        4. **DEPTH:** If `{len_instruction}` asks for details, ensure you provide the "Transformation" (Before/After) narrative.

        OUTPUT FORMAT (JSON ONLY):
        {{
            "subject_a": "...", 
            "body_a": "...", 
            "style_a": "Targeted Strategy",
            "subject_b": "...", 
            "body_b": "...", 
            "style_b": "Insight-Asset"
        }}
        """
    )

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
            "len_instruction": len_instruction # Passing the specific instruction
        })
        return extract_json(raw_result)

    except Exception as e:
        return {
            "subject_a": "Error", "body_a": f"Details: {str(e)}", "style_a": "Failed",
            "subject_b": "Error", "body_b": "Retry", "style_b": "Failed"
        }