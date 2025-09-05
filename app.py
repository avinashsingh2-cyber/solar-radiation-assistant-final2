
import streamlit as st
import pandas as pd
import re
from rapidfuzz import process, fuzz
from openai import OpenAI
import os

# --- API Key ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- Streamlit config ---
st.set_page_config(page_title="☀️ Solar Radiation Assistant", layout="wide")
st.title("☀️ Solar Radiation Assistant")

# --- Load Excel ---
EXCEL_FILE = "CombinedData.xlsx"
df = pd.read_excel(EXCEL_FILE)

# --- Normalize ---
df.columns = df.columns.str.strip()
df['Type'] = df['Type'].str.strip().str.lower()
df['State'] = df['State'].fillna("").str.strip().str.title()
df['District'] = df['District'].fillna("").str.strip().str.title()
df['Substation'] = df['Substation'].fillna("").str.strip().str.title()
df['Site'] = df['Site'].fillna("").str.strip().str.title()

# --- Utility functions ---
def fuzzy_match_best(query, choices):
    norm_choices = {c.lower(): c for c in choices if isinstance(c, str) and c.strip() != ""}
    match = process.extractOne(query.strip().lower(), norm_choices.keys(), scorer=fuzz.WRatio)
    if match:
        return norm_choices[match[0]], match[1]
    return None, 0

def extract_top_n(query):
    try:
        return int(re.findall(r"top (\d+)", query)[0])
    except:
        pass
    try:
        return int(re.findall(r"(\d+) (?:states|districts|substations|sites)", query)[0])
    except:
        pass
    if "highest" in query.lower():
        return 1
    return None  # 🔑 return None so default means "all rows"

# --- AI fallback ---
def ai_fallback(query):
    prompt = f'''
    You are a solar data assistant. The dataset has columns:
    - State, District, Substation, Site
    - SolarGIS GHI, Metonorm 8.2 GHI, Albedo
    
    User query: "{query}"
    
    If possible, interpret the query and explain what part of the dataset should be checked.
    Do not make up numbers. If exact data not found, suggest which column(s) are relevant.
    '''
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You help interpret solar dataset queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI fallback failed: {e}"

# --- Query handler ---
def answer_query(q):
    q_lower = q.strip().lower()

    # ----------- STATE-SPECIFIC QUERIES -----------
    for state in df['State'].unique():
        if state.lower() in q_lower:
            state_df = df[df['State'].str.lower() == state.lower()]

            # Just the state name → show all rows
            if q_lower.strip() == state.lower():
                return state_df[['State','District','Substation','Site','SolarGIS GHI','Metonorm 8.2 GHI','Albedo']].reset_index(drop=True)

            # 🔑 State + ghi/albedo → averages only
            if "ghi" in q_lower or "albedo" in q_lower:
                avg_vals = state_df[['SolarGIS GHI','Metonorm 8.2 GHI','Albedo']].mean(numeric_only=True).to_frame().T
                avg_vals.insert(0,"State",state)
                return avg_vals.reset_index(drop=True)

            # Substation queries within state
            if "substation" in q_lower:
                n = extract_top_n(q_lower)
                sub_df = state_df[state_df['Type']=="substation"]
                if sub_df.empty:
                    continue
                if n:  # top N
                    return sub_df.nlargest(n,"SolarGIS GHI")[['State','District','Substation','SolarGIS GHI']].reset_index(drop=True)
                else:  # all rows
                    return sub_df[['State','District','Substation','SolarGIS GHI']].reset_index(drop=True)

            # District queries within state
            if "district" in q_lower:
                dist_df = state_df[state_df['Type']=="district"]
                if dist_df.empty:
                    continue
                avg = dist_df.groupby(["State","District"])["SolarGIS GHI"].mean().reset_index()
                n = extract_top_n(q_lower)
                if n:
                    return avg.nlargest(n,"SolarGIS GHI").reset_index(drop=True)
                else:  # 🔑 return all districts
                    return avg.reset_index(drop=True)

            # Site queries within state
            if "site" in q_lower:
                site_df = state_df[state_df['Type']=="site"]
                if site_df.empty:
                    continue
                n = extract_top_n(q_lower)
                if n:
                    return site_df.nlargest(n,"SolarGIS GHI")[['State','District','Site','SolarGIS GHI']].reset_index(drop=True)
                else:  # 🔑 return all sites
                    return site_df[['State','District','Site','SolarGIS GHI']].reset_index(drop=True)

    # ----------- DISTRICT + SUBSTATION SPECIAL HANDLING (e.g., "substation in Pune") -----------
    if "substation" in q_lower and any(word in q_lower for word in df['District'].str.lower().unique()):
        for dist in df['District'].dropna().unique():
            if dist.lower() in q_lower:
                dist_df = df[(df['District'].str.lower()==dist.lower()) & (df['Type']=="substation")]
                if not dist_df.empty:
                    n = extract_top_n(q_lower)
                    if n:
                        return dist_df.nlargest(n,"SolarGIS GHI")[['State','District','Substation','SolarGIS GHI']].reset_index(drop=True)
                    else:
                        return dist_df[['State','District','Substation','SolarGIS GHI']].reset_index(drop=True)

    # ----------- GENERIC OVERALL QUERIES -----------
    if "state" in q_lower:
        state_avg = df.groupby("State")["SolarGIS GHI"].mean().reset_index()
        n = extract_top_n(q_lower)
        if n:
            return state_avg.nlargest(n,"SolarGIS GHI").reset_index(drop=True)
        else:  # 🔑 return all states if "average ghi by state"
            return state_avg.reset_index(drop=True)

    if "district" in q_lower:
        dist_df = df[df['Type']=="district"]
        avg = dist_df.groupby(["State","District"])["SolarGIS GHI"].mean().reset_index()
        n = extract_top_n(q_lower)
        if n:
            return avg.nlargest(n,"SolarGIS GHI").reset_index(drop=True)
        else:  # 🔑 all districts
            return avg.reset_index(drop=True)

    if "substation" in q_lower:
        sub_df = df[df['Type']=="substation"]
        n = extract_top_n(q_lower)
        if n:
            return sub_df.nlargest(n,"SolarGIS GHI")[['State','District','Substation','SolarGIS GHI']].reset_index(drop=True)
        else:
            return sub_df[['State','District','Substation','SolarGIS GHI']].reset_index(drop=True)

    if "site" in q_lower:
        site_df = df[df['Type']=="site"]
        n = extract_top_n(q_lower)
        if n:
            return site_df.nlargest(n,"SolarGIS GHI")[['State','District','Site','SolarGIS GHI']].reset_index(drop=True)
        else:
            return site_df[['State','District','Site','SolarGIS GHI']].reset_index(drop=True)

    # ----------- AUTO-DETECTION (names only) -----------
    best_type, best_name, best_score = None, None, 0

    name, score = fuzzy_match_best(q, df[df['Type']=="substation"]['Substation'].unique())
    if score > best_score:
        best_type, best_name, best_score = "substation", name, score

    name, score = fuzzy_match_best(q, df[df['Type']=="district"]['District'].unique())
    if score > best_score:
        best_type, best_name, best_score = "district", name, score

    name, score = fuzzy_match_best(q, df[df['Type']=="site"]['Site'].unique())
    if score > best_score:
        best_type, best_name, best_score = "site", name, score

    if best_score >= 82:
        filtered = None
        if best_type == "substation":
            filtered = df[df['Substation']==best_name]
        elif best_type == "district":
            filtered = df[df['District']==best_name]
        elif best_type == "site":
            filtered = df[df['Site']==best_name]

        if filtered is not None:
            # 🔑 If query mentions ghi/albedo → averages only
            if any(key in q_lower for key in ["ghi", "albedo", "metonorm"]):
                avg = filtered[['SolarGIS GHI','Metonorm 8.2 GHI','Albedo']].mean().to_frame().T
                avg.insert(0, best_type.title(), best_name)
                return avg.reset_index(drop=True)
            else:
                # Return all rows
                return filtered[['State','District','Substation','Site','SolarGIS GHI','Metonorm 8.2 GHI','Albedo']]

    # ----------- FALLBACK -----------
    return ai_fallback(q)

# --- Streamlit Input ---
query = st.text_input("Ask a question about the solar dataset:")
if query:
    answer = answer_query(query)
    if isinstance(answer, pd.DataFrame):
        st.dataframe(answer.reset_index(drop=True), use_container_width=True)
    else:
        st.write(answer)

