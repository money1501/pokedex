"""
PokéDex DSC — Pokémon Encounter Probability Predictor
OEP | Manish (IU2341230683) | Group 07 | CE0630 Data Science
Generation VIII (Sword/Shield)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy import stats
import math
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PokéDex DSC | Encounter Probability",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

.main { background: #0a0a0f; }

/* Type badge colors */
.type-badge {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:11px; font-weight:700; text-transform:uppercase;
    letter-spacing:1px; margin:2px;
}
.grass{background:#2d6a2d;color:#7fff7f}
.fire{background:#7a2000;color:#ff8c42}
.water{background:#003d7a;color:#5bc8f5}
.electric{background:#5a4d00;color:#f5d020}
.psychic{background:#6a0040;color:#ff69b4}
.ice{background:#004d5a;color:#adf0ff}
.dragon{background:#2d0070;color:#9b7bff}
.dark{background:#1a1a2e;color:#aaaacc}
.fairy{background:#5a003a;color:#ffb3e0}
.fighting{background:#5a1a00;color:#ff7c4d}
.poison{background:#3a0060;color:#cc66ff}
.ground{background:#4d3500;color:#d4a24c}
.rock{background:#3d3020;color:#c8b89a}
.bug{background:#2a4000;color:#88cc44}
.ghost{background:#1a003a;color:#9966cc}
.steel{background:#2a3040;color:#aabbcc}
.normal{background:#2d2d2d;color:#cccccc}
.flying{background:#1a2a4d;color:#88aaff}

/* Pokedex Card */
.poke-card {
    background: linear-gradient(145deg, #131320, #1a1a2e);
    border: 1px solid #2a2a45;
    border-radius: 16px;
    padding: 14px;
    margin: 6px 0;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.poke-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #ff0000, #3b82f6);
}
.poke-card:hover {
    border-color: #4a4a70;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
.poke-name {
    font-family:'Orbitron',sans-serif; font-size:15px; font-weight:700;
    color:#e8e8ff; margin:4px 0;
}
.poke-num { color:#5a5a8a; font-size:12px; }
.bst-bar-bg { background:#1a1a2e; border-radius:6px; height:6px; margin:4px 0; }
.bst-bar { height:6px; border-radius:6px; background:linear-gradient(90deg,#e94040,#3b82f6); }
.enc-label { font-size:11px; color:#888; }
.enc-val { font-size:18px; font-weight:700; font-family:'Orbitron',sans-serif; }

/* Section headers */
.sec-head {
    font-family:'Orbitron',sans-serif; font-size:20px; font-weight:900;
    background: linear-gradient(90deg,#e94040,#3b82f6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin: 20px 0 12px 0; letter-spacing:1px;
}
.sub-head {
    font-family:'Orbitron',sans-serif; font-size:14px; font-weight:700;
    color:#9999cc; margin:16px 0 8px 0; letter-spacing:1px;
}

/* Stat bar row */
.stat-row { display:flex; align-items:center; margin:3px 0; }
.stat-name { width:70px; font-size:11px; color:#888; text-align:right; padding-right:8px; }
.stat-bar-bg { flex:1; background:#1a1a2e; border-radius:4px; height:8px; }
.stat-val { width:35px; text-align:right; font-size:12px; font-weight:700; color:#e8e8ff; }

/* Encounter gauge */
.gauge-wrap { text-align:center; padding:12px; }
.gauge-pct {
    font-family:'Orbitron',sans-serif; font-size:36px; font-weight:900;
    background:linear-gradient(90deg,#e94040,#f59e0b,#10b981);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.gauge-bar-bg { background:#1a1a2e; border-radius:10px; height:14px; margin:8px 0; }
.gauge-bar { height:14px; border-radius:10px; }

/* KPI Card */
.kpi { background:#131320; border:1px solid #2a2a45; border-radius:12px;
        padding:16px; text-align:center; }
.kpi-val { font-family:'Orbitron',sans-serif; font-size:28px; font-weight:900;
            color:#e8e8ff; }
.kpi-label { font-size:12px; color:#6666aa; margin-top:4px; }

/* Info box */
.info-box { background:#131320; border:1px solid #2a2a45; border-left:4px solid #3b82f6;
            border-radius:8px; padding:12px 16px; margin:8px 0; }

/* Ability pill */
.ability-pill { display:inline-block; background:#1e1e35; border:1px solid #3a3a60;
                border-radius:20px; padding:4px 12px; margin:3px; font-size:12px; color:#aaaacc; }
.ability-hidden { border-color:#f59e0b; color:#f59e0b; }

/* Cluster badge */
.cluster-0 { color:#10b981; }
.cluster-1 { color:#f59e0b; }
.cluster-2 { color:#e94040; }
.cluster-3 { color:#3b82f6; }

/* Route selector */
.route-btn { background:#131320; border:1px solid #2a2a45; border-radius:8px;
             padding:8px 14px; margin:3px; cursor:pointer; font-size:12px; color:#9999cc; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

GALAR_ROUTES = [
    "Route 1","Route 2","Route 3","Route 4","Route 5","Route 6","Route 7","Route 8","Route 9","Route 10",
    "Wild Area - Motostoke Riverbank","Wild Area - East Lake Axewell","Wild Area - West Lake Axewell",
    "Wild Area - Hammerlocke Hills","Wild Area - Giant's Seat","Wild Area - Dusty Bowl",
    "Wild Area - Lake of Outrage","Glimwood Tangle","Slumbering Weald","Stony Wilderness",
    "Axew's Eye","Giant's Mirror","Giant's Cap","Bridge Field","Watchtower Ruins",
    "Rolling Fields","Dappled Grove","Turffield","Hulbury","Motostoke",
]
WEATHERS = ["Normal","Overcast","Raining","Thunderstorm","Snowing","Blizzard","Harsh Sunlight","Sandstorm","Foggy"]
TIMES = ["Day","Night"]
ENC_TYPES = ["Grass","Surfing","Fishing","Overworld"]
EVO_STAGES = [1,2,3]

TYPE_COLORS = {
    "grass":"#4caf50","fire":"#ff5722","water":"#2196f3","electric":"#ffeb3b",
    "psychic":"#e91e63","ice":"#00bcd4","dragon":"#673ab7","dark":"#455a64",
    "fairy":"#f06292","fighting":"#795548","poison":"#9c27b0","ground":"#8d6e63",
    "rock":"#9e9e9e","bug":"#8bc34a","ghost":"#5c35cc","steel":"#607d8b",
    "normal":"#aaaaaa","flying":"#5c8af7","unknown":"#888888",
}

STAT_COLORS = {
    "hp":"#e94040","attack":"#f59e0b","defense":"#10b981",
    "sp_attack":"#3b82f6","sp_defense":"#8b5cf6","speed":"#ec4899"
}

@st.cache_data(show_spinner=False)
def fetch_pokemon_batch(start_id, end_id):
    """Fetch Pokémon from PokéAPI in a range."""
    records = []
    for pid in range(start_id, end_id + 1):
        try:
            r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pid}", timeout=6)
            if r.status_code != 200:
                continue
            d = r.json()
            types = [t["type"]["name"] for t in d["types"]]
            stats_d = {s["stat"]["name"]: s["base_stat"] for s in d["stats"]}
            abilities = [a["ability"]["name"].replace("-"," ").title() for a in d["abilities"]]
            hidden_ab = next((a["ability"]["name"].replace("-"," ").title()
                              for a in d["abilities"] if a["is_hidden"]), None)
            records.append({
                "pokemon_id":  pid,
                "pokemon_name":d["name"].capitalize(),
                "type_1":      types[0] if types else "normal",
                "type_2":      types[1] if len(types) > 1 else None,
                "hp":          stats_d.get("hp", 0),
                "attack":      stats_d.get("attack", 0),
                "defense":     stats_d.get("defense", 0),
                "sp_attack":   stats_d.get("special-attack", 0),
                "sp_defense":  stats_d.get("special-defense", 0),
                "speed":       stats_d.get("speed", 0),
                "bst":         sum(stats_d.values()),
                "height":      d["height"] / 10,
                "weight":      d["weight"] / 10,
                "color":       d["species"]["url"].split("/")[-2] if "species" in d else "",
                "sprite":      d["sprites"]["front_default"],
                "abilities":   ", ".join(abilities),
                "hidden_ability": hidden_ab,
                "evolution_stage": 1 if pid % 3 == 0 else (2 if pid % 3 == 1 else 3),
                "galar_dex_no": pid - 809 if pid >= 810 else None,
            })
        except Exception:
            continue
    return records


@st.cache_data(show_spinner=False)
def load_pokemon_data():
    """Try to load Gen 8 Pokémon. Falls back to a built-in Gen 1–8 sample if offline."""
    try:
        r = requests.get("https://pokeapi.co/api/v2/pokemon/bulbasaur", timeout=4)
        if r.status_code == 200:
            # Online: fetch Gen 8 (810–905)
            records = fetch_pokemon_batch(810, 905)
            if len(records) >= 20:
                return pd.DataFrame(records)
    except Exception:
        pass

    # Offline fallback — representative Gen 8 sample
    fallback = [
        (810,"Grookey","grass",None,50,65,50,40,40,65,1),
        (812,"Rillaboom","grass",None,100,125,90,60,70,85,3),
        (813,"Scorbunny","fire",None,50,72,40,32,32,80,1),
        (815,"Cinderace","fire",None,80,116,75,65,75,119,3),
        (816,"Sobble","water",None,50,40,40,70,40,70,1),
        (818,"Inteleon","water",None,70,85,65,105,65,120,3),
        (819,"Skwovet","normal",None,70,55,55,35,35,25,1),
        (821,"Rookidee","flying",None,38,47,35,33,35,57,1),
        (823,"Corviknight","flying","steel",98,87,105,53,85,67,3),
        (824,"Blipbug","bug",None,25,30,38,40,40,42,1),
        (826,"Orbeetle","bug","psychic",60,45,110,80,120,90,3),
        (827,"Nickit","dark",None,40,28,28,47,52,50,1),
        (829,"Gossifleur","grass",None,40,40,60,40,60,10,1),
        (830,"Eldegoss","grass",None,60,50,90,80,120,60,2),
        (831,"Wooloo","normal",None,42,40,55,35,45,48,1),
        (832,"Dubwool","normal",None,72,80,100,60,90,88,2),
        (833,"Chewtle","water",None,50,64,50,38,38,44,1),
        (834,"Drednaw","water","rock",90,115,90,48,68,74,2),
        (835,"Yamper","electric",None,59,45,50,40,50,26,1),
        (836,"Boltund","electric",None,69,90,60,54,60,121,2),
        (837,"Rolycoly","rock",None,30,40,50,40,50,30,1),
        (839,"Coalossal","rock","fire",110,80,120,80,90,30,3),
        (840,"Applin","grass","dragon",40,40,80,40,40,20,1),
        (841,"Flapple","grass","dragon",70,110,80,95,60,70,2),
        (845,"Cramorant","flying","water",70,65,45,75,45,85,1),
        (846,"Arrokuda","water",None,41,63,40,40,30,66,1),
        (848,"Toxel","electric","poison",40,38,35,54,35,40,1),
        (849,"Toxtricity","electric","poison",75,98,70,114,70,75,2),
        (850,"Sizzlipede","fire","bug",50,65,45,50,50,45,1),
        (851,"Centiskorch","fire","bug",100,115,65,90,90,65,2),
        (852,"Clobbopus","fighting",None,50,68,60,50,50,32,1),
        (853,"Grapploct","fighting",None,80,118,90,70,80,42,2),
        (854,"Sinistea","ghost",None,40,45,45,74,54,50,1),
        (855,"Polteageist","ghost",None,60,65,65,134,114,70,2),
        (856,"Hatenna","psychic",None,42,30,45,56,53,39,1),
        (858,"Hatterene","psychic","fairy",57,90,95,136,103,29,3),
        (859,"Impidimp","dark","fairy",45,45,30,55,40,50,1),
        (861,"Grimmsnarl","dark","fairy",95,120,65,95,75,60,3),
        (862,"Obstagoon","dark","normal",93,90,101,60,81,95,3),
        (863,"Perrserker","steel",None,70,110,100,50,60,50,2),
        (864,"Cursola","ghost",None,60,95,50,145,85,30,2),
        (865,"Sirfetch'd","fighting",None,62,135,95,68,82,65,2),
        (866,"Mr. Rime","ice","psychic",80,85,75,110,100,70,2),
        (867,"Runerigus","ground","ghost",58,95,145,50,105,30,2),
        (868,"Milcery","fairy",None,45,40,40,50,61,34,1),
        (869,"Alcremie","fairy",None,65,60,75,110,121,64,2),
        (870,"Falinks","fighting",None,65,100,100,70,60,75,1),
        (871,"Pincurchin","electric",None,48,101,95,91,85,15,1),
        (872,"Snom","ice","bug",30,25,35,45,30,20,1),
        (873,"Frosmoth","ice","bug",70,65,60,125,90,65,2),
        (874,"Stonjourner","rock",None,100,125,135,20,20,70,1),
        (875,"Eiscue","ice",None,75,80,110,65,90,50,1),
        (876,"Indeedee","psychic","normal",60,65,55,105,95,95,1),
        (877,"Morpeko","electric","dark",58,95,58,70,58,97,1),
        (878,"Cufant","steel",None,72,80,49,40,57,40,1),
        (879,"Copperajah","steel",None,122,130,69,80,69,30,2),
        (880,"Dracozolt","electric","dragon",90,100,90,80,70,75,2),
        (881,"Arctozolt","electric","ice",90,100,90,90,80,55,2),
        (882,"Dracovish","water","dragon",90,90,100,70,80,75,2),
        (883,"Arctovish","water","ice",90,90,100,80,90,55,2),
        (884,"Duraludon","steel","dragon",70,95,115,120,50,85,2),
        (885,"Dreepy","dragon","ghost",28,60,30,40,30,82,1),
        (886,"Drakloak","dragon","ghost",68,80,50,60,50,102,2),
        (887,"Dragapult","dragon","ghost",88,120,75,100,75,142,3),
        (888,"Zacian","fairy",None,92,130,115,80,115,138,3),
        (889,"Zamazenta","fighting",None,92,130,145,80,145,138,3),
        (890,"Eternatus","poison","dragon",140,85,95,145,95,130,3),
        (891,"Kubfu","fighting",None,60,90,60,53,50,72,1),
        (892,"Urshifu","fighting","water",100,130,100,63,60,97,2),
        (893,"Zarude","dark","grass",105,120,105,70,95,105,3),
        (894,"Regieleki","electric",None,80,100,50,100,50,200,3),
        (895,"Regidrago","dragon",None,200,100,50,100,50,80,3),
        (896,"Glastrier","ice",None,100,145,130,65,110,30,3),
        (897,"Spectrier","ghost",None,100,65,60,145,80,130,3),
        (898,"Calyrex","psychic","grass",100,80,80,80,80,80,3),
    ]
    rows = []
    for row in fallback:
        pid,name,t1,t2,hp,atk,df,spa,spd,spe,evo = row
        rows.append({
            "pokemon_id":pid,"pokemon_name":name,"type_1":t1,"type_2":t2,
            "hp":hp,"attack":atk,"defense":df,"sp_attack":spa,"sp_defense":spd,"speed":spe,
            "bst":hp+atk+df+spa+spd+spe,"height":1.0,"weight":10.0,"color":"","sprite":None,
            "abilities":"Overgrow","hidden_ability":None,"evolution_stage":evo,
            "galar_dex_no":pid-809,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_encounter_table(df):
    """
    Synthetic encounter table: route × weather × time × pokemon
    P(p|l,w,t) = encounter_weight / Σ weights (slot-normalized)
    """
    np.random.seed(42)
    records = []
    rng = np.random.RandomState(42)

    # Each route has a pool of ~8-15 Pokémon with assigned weights
    for route_i, route in enumerate(GALAR_ROUTES):
        for weather in WEATHERS:
            for time in TIMES:
                # Sample a random pool from the dataset
                pool_size = rng.randint(6, 15)
                pool = df.sample(min(pool_size, len(df)), random_state=route_i + ord(weather[0]) + ord(time[0])).copy()

                # Assign encounter weights
                # Stage 1 more common, legendaries rare
                weights = []
                for _, p in pool.iterrows():
                    w = rng.randint(5, 100)
                    w *= (1.5 if p["evolution_stage"] == 1 else (0.7 if p["evolution_stage"] == 3 else 1.0))
                    # Weather bonuses for certain types
                    type_weather_bonus = {
                        "Raining": ["water","electric"], "Thunderstorm": ["electric"],
                        "Snowing": ["ice"], "Blizzard": ["ice","steel"],
                        "Harsh Sunlight": ["fire","grass"], "Sandstorm": ["rock","ground","steel"],
                        "Foggy": ["ghost","psychic"],
                    }
                    if weather in type_weather_bonus:
                        if p["type_1"] in type_weather_bonus[weather] or p["type_2"] in type_weather_bonus.get(weather, []):
                            w *= 2.0
                    # Night: ghost/dark more common
                    if time == "Night" and p["type_1"] in ["ghost","dark"]:
                        w *= 1.8
                    weights.append(max(1, int(w)))

                total_w = sum(weights)
                for (_, p), w in zip(pool.iterrows(), weights):
                    records.append({
                        "pokemon_id":    p["pokemon_id"],
                        "pokemon_name":  p["pokemon_name"],
                        "route_name":    route,
                        "weather_condition": weather,
                        "time_of_day":   time,
                        "encounter_weight": w,
                        "encounter_probability": round(w / total_w, 4),
                        "encounter_type": rng.choice(ENC_TYPES, p==[0.6,0.15,0.15,0.1])[0] if False else rng.choice(ENC_TYPES),
                        "type_1":        p["type_1"],
                        "type_2":        p["type_2"],
                        "bst":           p["bst"],
                        "evolution_stage": p["evolution_stage"],
                    })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def type_badge(t):
    if not t or t == "None" or (isinstance(t, float) and math.isnan(t)):
        return ""
    return f'<span class="type-badge {t}">{t}</span>'


def stat_bar_html(name, val, max_val=255):
    color = STAT_COLORS.get(name, "#888")
    pct = min(100, int(val / max_val * 100))
    return f"""
    <div class='stat-row'>
      <span class='stat-name'>{name.upper()}</span>
      <div class='stat-bar-bg'>
        <div style='width:{pct}%;height:8px;border-radius:4px;background:{color}'></div>
      </div>
      <span class='stat-val'>{val}</span>
    </div>"""


def encounter_gauge_html(pct):
    color = "#10b981" if pct >= 60 else ("#f59e0b" if pct >= 30 else "#e94040")
    rarity = "Common" if pct >= 60 else ("Uncommon" if pct >= 30 else "Rare")
    return f"""
    <div class='gauge-wrap'>
      <div class='gauge-pct'>{pct:.1f}%</div>
      <div style='font-size:12px;color:#888;margin-bottom:4px'>Wild Encounter Rate · {rarity}</div>
      <div class='gauge-bar-bg'>
        <div class='gauge-bar' style='width:{min(pct,100):.1f}%;background:{color}'></div>
      </div>
      <div style='font-size:10px;color:#555;margin-top:4px'>
        Simulated · based on stage, region & rarity
      </div>
    </div>"""


def compute_simulated_encounter(row, enc_df):
    """Weighted encounter rate formula from section 3.4."""
    poke_enc = enc_df[enc_df["pokemon_id"] == row["pokemon_id"]]
    if poke_enc.empty:
        return 5.0
    base_prob = poke_enc["encounter_probability"].mean()  # weight 0.50
    stage_mod = {1: 0.2, 2: 0.0, 3: -0.2}.get(row["evolution_stage"], 0)  # weight 0.20
    # weather rarity: if appears in <3 weather conditions → bonus
    weather_diversity = poke_enc["weather_condition"].nunique()
    weather_bonus = 0.1 if weather_diversity <= 3 else 0.0  # weight 0.15
    # route diversity: more routes → lower individual prob
    route_count = poke_enc["route_name"].nunique()
    route_mod = -0.05 if route_count >= 15 else 0.05  # weight 0.15

    sim = (0.50 * base_prob * 100) + (0.20 * stage_mod * 100) + (0.15 * weather_bonus * 100) + (0.15 * route_mod * 100)
    return round(np.clip(sim, 2, 95), 1)


def dark_fig():
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#131320")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a45")
    ax.tick_params(colors="#666699")
    return fig, ax


def dark_fig2(r=1, c=2, **kwargs):
    fig, axes = plt.subplots(r, c, figsize=kwargs.get("figsize", (12, 4)))
    fig.patch.set_facecolor("#0d0d1a")
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor("#131320")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a45")
        ax.tick_params(colors="#666699")
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("⚡ Loading PokéDex data..."):
    df = load_pokemon_data()
    df["type_2_str"] = df["type_2"].fillna("—")
    enc_df = build_encounter_table(df)

    # Compute simulated encounter rate for each Pokémon
    df["encounter_rate"] = df.apply(lambda r: compute_simulated_encounter(r, enc_df), axis=1)
    df["galar_only"] = df["pokemon_id"] >= 810

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div style='text-align:center;padding:12px 0'>
  <div style='font-family:Orbitron,sans-serif;font-size:20px;font-weight:900;
       background:linear-gradient(90deg,#e94040,#3b82f6);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    PokéDex DSC
  </div>
  <div style='color:#555;font-size:11px;margin-top:4px'>
    Manish · IU2341230683<br>Group 07 · CE0630
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

section = st.sidebar.radio("📍 Section", [
    "📖 Pokédex Browser",
    "🎯 Encounter Predictor",
    "🔍 Inverse Query (Pokémon → Route)",
    "📊 EDA",
    "📈 Statistical Analysis",
    "🤖 Machine Learning",
    "⚖️ Ethics",
])

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER KPIs (always visible)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='font-family:Orbitron,sans-serif;font-size:28px;font-weight:900;
     background:linear-gradient(90deg,#e94040,#3b82f6,#10b981);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
     margin-bottom:4px'>
  PokéDex DSC: Encounter Probability Predictor
</div>
<div style='color:#555;font-size:13px;margin-bottom:16px'>
  Generation VIII (Sword / Shield) · Data Science CE0630 OEP
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi"><div class="kpi-val">{len(df)}</div><div class="kpi-label">Pokémon in Dataset</div></div>', unsafe_allow_html=True)
with k2:
    galar_count = df[df["galar_only"]].shape[0]
    st.markdown(f'<div class="kpi"><div class="kpi-val">{galar_count}</div><div class="kpi-label">Galar Dex Entries</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi"><div class="kpi-val">18</div><div class="kpi-label">Distinct Types</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi"><div class="kpi-val">{len(GALAR_ROUTES)}</div><div class="kpi-label">Routes / Locations</div></div>', unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: POKÉDEX BROWSER
# ══════════════════════════════════════════════════════════════════════════════

if section == "📖 Pokédex Browser":
    st.markdown('<div class="sec-head">POKÉDEX BROWSER</div>', unsafe_allow_html=True)

    # Filters
    f1, f2, f3, f4 = st.columns([2,2,1,1])
    with f1:
        search = st.text_input("🔍 Search by name (prefix)", "")
    with f2:
        all_types = sorted(df["type_1"].dropna().unique())
        type_filter = st.multiselect("Type Filter", all_types, default=[])
    with f3:
        galar_only = st.checkbox("Galar Only", value=False)
    with f4:
        evo_stage = st.multiselect("Evo Stage", [1,2,3], default=[1,2,3])

    sort_by = st.selectbox("Sort by", ["Pokédex #","Name","BST","Encounter Rate"], index=0)

    fdf = df.copy()
    if search:
        fdf = fdf[fdf["pokemon_name"].str.lower().str.startswith(search.lower())]
    if type_filter:
        fdf = fdf[fdf["type_1"].isin(type_filter) | fdf["type_2"].isin(type_filter)]
    if galar_only:
        fdf = fdf[fdf["galar_only"]]
    if evo_stage:
        fdf = fdf[fdf["evolution_stage"].isin(evo_stage)]

    sort_map = {"Pokédex #":"pokemon_id","Name":"pokemon_name","BST":"bst","Encounter Rate":"encounter_rate"}
    fdf = fdf.sort_values(sort_map[sort_by])

    st.markdown(f"<div style='color:#555;font-size:12px;margin-bottom:8px'>Showing {len(fdf)} Pokémon</div>", unsafe_allow_html=True)

    # Card Grid — 3 per row
    cols_per_row = 3
    rows = [fdf.iloc[i:i+cols_per_row] for i in range(0, len(fdf), cols_per_row)]

    # Detail expander state
    selected_poke = st.selectbox("🔎 View detailed card for:", ["— select —"] + fdf["pokemon_name"].tolist(), index=0)

    if selected_poke != "— select —":
        prow = df[df["pokemon_name"] == selected_poke].iloc[0]
        st.markdown('<div class="sub-head">POKÉDEX CARD — DETAIL VIEW</div>', unsafe_allow_html=True)

        dc1, dc2 = st.columns([1, 1.6])

        with dc1:
            if prow["sprite"]:
                st.image(prow["sprite"], width=160)
            else:
                st.markdown(f"<div style='font-size:60px;text-align:center'>{'🔴' if prow['type_1']=='fire' else '💧' if prow['type_1']=='water' else '🌿'}</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='poke-name'>#{prow['pokemon_id']} {prow['pokemon_name']}</div>
            {type_badge(prow['type_1'])} {type_badge(prow['type_2'] if pd.notna(prow['type_2']) else None)}
            <div style='margin-top:10px;font-size:12px;color:#666'>
              📏 Height: {prow['height']} m &nbsp;|&nbsp; ⚖️ Weight: {prow['weight']} kg<br>
              🌟 Evo Stage: {prow['evolution_stage']} &nbsp;|&nbsp; 🗺️ Galar #: {int(prow['galar_dex_no']) if pd.notna(prow['galar_dex_no']) else '—'}
            </div>
            <div style='margin-top:10px'>
              <div style='font-size:11px;color:#555;margin-bottom:4px'>ABILITIES</div>
            """, unsafe_allow_html=True)

            for ab in str(prow["abilities"]).split(", "):
                is_hidden = ab.strip() == str(prow.get("hidden_ability", ""))
                cls = "ability-hidden" if is_hidden else ""
                st.markdown(f'<span class="ability-pill {cls}">{ab.strip()}</span>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(encounter_gauge_html(prow["encounter_rate"]), unsafe_allow_html=True)

        with dc2:
            st.markdown('<div class="sub-head">BASE STATS</div>', unsafe_allow_html=True)
            for stat in ["hp","attack","defense","sp_attack","sp_defense","speed"]:
                st.markdown(stat_bar_html(stat, int(prow[stat])), unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:right;font-size:13px;color:#9999cc;margin-top:6px'>BST: <b style='color:#e8e8ff'>{prow['bst']}</b></div>", unsafe_allow_html=True)

            # Radar chart
            st.markdown('<div class="sub-head">STAT RADAR</div>', unsafe_allow_html=True)
            stat_vals = [prow["hp"], prow["attack"], prow["defense"],
                         prow["sp_attack"], prow["sp_defense"], prow["speed"]]
            stat_labels = ["HP","ATK","DEF","SPA","SPD","SPE"]
            angles = np.linspace(0, 2*np.pi, len(stat_labels), endpoint=False).tolist()
            vals = stat_vals + [stat_vals[0]]
            angles += [angles[0]]

            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor("#131320")
            ax.set_facecolor("#0a0a18")
            ax.plot(angles, vals, color="#e94040", lw=2)
            ax.fill(angles, vals, color="#e94040", alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(stat_labels, color="#9999cc", size=10, fontfamily="monospace")
            ax.set_ylim(0, 200)
            ax.tick_params(colors="#333355")
            ax.spines['polar'].set_color("#2a2a45")
            ax.yaxis.set_tick_params(labelcolor="#333355", labelsize=7)
            st.pyplot(fig, use_container_width=False)

    st.markdown("---")
    st.markdown('<div class="sub-head">FULL POKÉDEX GRID</div>', unsafe_allow_html=True)

    for row_group in rows:
        cols = st.columns(cols_per_row)
        for col, (_, prow) in zip(cols, row_group.iterrows()):
            with col:
                bst_pct = min(100, int(prow["bst"] / 700 * 100))
                enc_color = "#10b981" if prow["encounter_rate"] >= 60 else ("#f59e0b" if prow["encounter_rate"] >= 30 else "#e94040")
                t1c = TYPE_COLORS.get(prow["type_1"], "#888")
                sprite_html = f'<img src="{prow["sprite"]}" style="width:56px;height:56px">' if prow["sprite"] else "🔴"

                st.markdown(f"""
                <div class='poke-card'>
                  <div style='display:flex;align-items:center;gap:10px'>
                    {sprite_html}
                    <div>
                      <div class='poke-num'>#{prow['pokemon_id']}</div>
                      <div class='poke-name' style='font-size:13px'>{prow['pokemon_name']}</div>
                      <div>{type_badge(prow['type_1'])}{type_badge(prow['type_2'] if pd.notna(prow['type_2']) else None)}</div>
                    </div>
                  </div>
                  <div style='margin-top:8px'>
                    <div style='display:flex;justify-content:space-between;font-size:11px;color:#555'>
                      <span>BST {prow['bst']}</span><span>Stage {prow['evolution_stage']}</span>
                    </div>
                    <div class='bst-bar-bg'><div class='bst-bar' style='width:{bst_pct}%'></div></div>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-top:4px'>
                      <span class='enc-label'>Encounter</span>
                      <span class='enc-val' style='color:{enc_color};font-size:16px'>{prow['encounter_rate']}%</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: ENCOUNTER PREDICTOR (Location → Pokémon)
# ══════════════════════════════════════════════════════════════════════════════

elif section == "🎯 Encounter Predictor":
    st.markdown('<div class="sec-head">ENCOUNTER PROBABILITY PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Select a <b>location</b>, <b>weather condition</b>, and <b>time of day</b> → get ranked Pokémon with encounter probabilities.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        route_sel = st.selectbox("📍 Location / Route", GALAR_ROUTES)
    with c2:
        weather_sel = st.selectbox("🌤️ Weather Condition", WEATHERS)
    with c3:
        time_sel = st.selectbox("🕐 Time of Day", TIMES)

    enc_type_filter = st.multiselect("Encounter Type", ENC_TYPES, default=ENC_TYPES)

    if st.button("⚡ Predict Encounters", type="primary"):
        mask = (
            (enc_df["route_name"] == route_sel) &
            (enc_df["weather_condition"] == weather_sel) &
            (enc_df["time_of_day"] == time_sel) &
            (enc_df["encounter_type"].isin(enc_type_filter))
        )
        result = enc_df[mask].copy()

        if result.empty:
            st.warning("No encounters found for this combination. Try adjusting filters.")
        else:
            # Normalize probabilities in this slot
            total_w = result["encounter_weight"].sum()
            result["slot_probability"] = (result["encounter_weight"] / total_w * 100).round(2)
            result = result.sort_values("slot_probability", ascending=False).head(20)

            # Merge in full pokemon data
            result = result.merge(df[["pokemon_id","sprite","bst","evolution_stage","encounter_rate"]],
                                  on="pokemon_id", how="left")

            st.markdown(f'<div class="sub-head">TOP ENCOUNTERS — {route_sel} · {weather_sel} · {time_sel}</div>', unsafe_allow_html=True)

            # Bar chart
            fig, ax = dark_fig()
            colors = [TYPE_COLORS.get(t, "#888") for t in result["type_1"]]
            ax.barh(result["pokemon_name"][::-1], result["slot_probability"][::-1], color=colors[::-1])
            ax.set_xlabel("Encounter Probability (%)", color="#666699")
            ax.set_title("Slot-Normalized Encounter Probability", color="#e8e8ff", fontsize=12)
            ax.tick_params(axis="y", colors="#9999cc", labelsize=9)
            st.pyplot(fig, use_container_width=True)

            # Cards for top 10
            st.markdown('<div class="sub-head">RANKED POKÉMON</div>', unsafe_allow_html=True)
            for i, (_, row) in enumerate(result.head(10).iterrows()):
                bar_color = "#10b981" if row["slot_probability"] >= 20 else ("#f59e0b" if row["slot_probability"] >= 10 else "#e94040")
                sprite_html = f'<img src="{row["sprite"]}" style="width:44px;height:44px">' if pd.notna(row.get("sprite")) and row["sprite"] else ""
                st.markdown(f"""
                <div class='poke-card'>
                  <div style='display:flex;align-items:center;gap:12px'>
                    <div style='font-family:Orbitron,sans-serif;font-size:20px;color:#333355;width:28px'>#{i+1}</div>
                    {sprite_html}
                    <div style='flex:1'>
                      <div class='poke-name' style='font-size:14px'>{row['pokemon_name']}</div>
                      <div>{type_badge(row['type_1'])}{type_badge(row['type_2'] if pd.notna(row['type_2']) else None)}</div>
                    </div>
                    <div style='text-align:right'>
                      <div style='font-family:Orbitron,sans-serif;font-size:22px;color:{bar_color}'>{row['slot_probability']:.1f}%</div>
                      <div style='font-size:10px;color:#555'>slot probability</div>
                    </div>
                  </div>
                  <div style='margin-top:8px'>
                    <div class='bst-bar-bg'>
                      <div style='width:{min(row["slot_probability"]*3,100):.1f}%;height:6px;border-radius:4px;background:{bar_color}'></div>
                    </div>
                    <div style='font-size:10px;color:#444;margin-top:2px'>
                      print(row.index)
                      BST: {int(row['bst'])} &nbsp;|&nbsp; Type: {row['encounter_type']} &nbsp;|&nbsp; Stage: {int(row['evolution_stage'])}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: INVERSE QUERY (Pokémon → Best Location)
# ══════════════════════════════════════════════════════════════════════════════

elif section == "🔍 Inverse Query (Pokémon → Route)":
    st.markdown('<div class="sec-head">INVERSE QUERY — FIND BEST ROUTE</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Select a <b>target Pokémon</b> → get the top routes, weather, and time that <b>maximize encounter probability</b>.</div>', unsafe_allow_html=True)

    target_poke = st.selectbox("🎯 Target Pokémon", sorted(df["pokemon_name"].tolist()))

    if st.button("🔍 Find Best Conditions", type="primary"):
        poke_enc = enc_df[enc_df["pokemon_name"] == target_poke].copy()
        if poke_enc.empty:
            st.warning("No encounter data found for this Pokémon.")
        else:
            total_per_slot = enc_df.groupby(["route_name","weather_condition","time_of_day"])["encounter_weight"].sum().reset_index()
            total_per_slot.columns = ["route_name","weather_condition","time_of_day","slot_total"]
            poke_enc = poke_enc.merge(total_per_slot, on=["route_name","weather_condition","time_of_day"])
            poke_enc["slot_prob_pct"] = (poke_enc["encounter_weight"] / poke_enc["slot_total"] * 100).round(2)
            best = poke_enc.sort_values("slot_prob_pct", ascending=False).head(10).reset_index(drop=True)

            prow = df[df["pokemon_name"] == target_poke].iloc[0]
            dc1, dc2 = st.columns([1, 2])

            with dc1:
                if prow["sprite"]:
                    st.image(prow["sprite"], width=120)
                st.markdown(f"""
                <div class='poke-name'>{prow['pokemon_name']}</div>
                {type_badge(prow['type_1'])}{type_badge(prow['type_2'] if pd.notna(prow['type_2']) else None)}
                <div style='margin-top:8px;font-size:12px;color:#666'>
                  BST: {prow['bst']} &nbsp;|&nbsp; Stage {prow['evolution_stage']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown(encounter_gauge_html(prow["encounter_rate"]), unsafe_allow_html=True)

            with dc2:
                st.markdown('<div class="sub-head">TOP CONDITIONS TO FIND THIS POKÉMON</div>', unsafe_allow_html=True)

                fig, ax = dark_fig()
                labels = [f"{r['route_name'].split('-')[-1].strip()}\n{r['weather_condition'][:5]}·{r['time_of_day'][0]}"
                          for _, r in best.iterrows()]
                colors_b = ["#10b981" if p >= 20 else ("#f59e0b" if p >= 10 else "#e94040") for p in best["slot_prob_pct"]]
                ax.bar(range(len(best)), best["slot_prob_pct"], color=colors_b)
                ax.set_xticks(range(len(best)))
                ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8, color="#9999cc")
                ax.set_ylabel("Slot Probability (%)", color="#666699")
                ax.set_title(f"Best Conditions to Encounter {target_poke}", color="#e8e8ff", fontsize=11)
                st.pyplot(fig, use_container_width=True)

                st.dataframe(
                    best[["route_name","weather_condition","time_of_day","encounter_type","slot_prob_pct"]]
                    .rename(columns={"slot_prob_pct":"Probability (%)"}),
                    use_container_width=True, hide_index=True
                )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: EDA
# ══════════════════════════════════════════════════════════════════════════════

elif section == "📊 EDA":
    st.markdown('<div class="sec-head">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)

    # Type Distribution
    st.markdown('<div class="sub-head">TYPE DISTRIBUTION</div>', unsafe_allow_html=True)
    type_counts = df["type_1"].value_counts()
    fig, ax = dark_fig()
    colors_t = [TYPE_COLORS.get(t, "#888") for t in type_counts.index]
    ax.bar(type_counts.index, type_counts.values, color=colors_t)
    ax.set_xticklabels(type_counts.index, rotation=45, ha="right", fontsize=9, color="#9999cc")
    ax.set_ylabel("Count", color="#666699")
    ax.set_title("Pokémon Count by Primary Type", color="#e8e8ff")
    st.pyplot(fig, use_container_width=True)

    # BST Histogram by Evo Stage
    st.markdown('<div class="sub-head">BST DISTRIBUTION BY EVOLUTION STAGE</div>', unsafe_allow_html=True)
    fig, ax = dark_fig()
    stage_colors = {1:"#10b981", 2:"#f59e0b", 3:"#e94040"}
    for stage in [1, 2, 3]:
        sub = df[df["evolution_stage"] == stage]["bst"]
        ax.hist(sub, bins=15, alpha=0.6, color=stage_colors[stage], label=f"Stage {stage}", edgecolor="#0d0d1a")
    ax.set_xlabel("Base Stat Total", color="#666699")
    ax.set_ylabel("Count", color="#666699")
    ax.set_title("BST by Evolution Stage", color="#e8e8ff")
    ax.legend(facecolor="#1a1a2e", labelcolor="#ccc")
    st.pyplot(fig, use_container_width=True)

    # Encounter Rate Heatmap
    st.markdown('<div class="sub-head">ENCOUNTER WEIGHT HEATMAP (Route × Weather)</div>', unsafe_allow_html=True)
    heatmap_data = enc_df.groupby(["route_name","weather_condition"])["encounter_probability"].mean().unstack(fill_value=0)
    heatmap_data = heatmap_data.iloc[:15]  # top 15 routes
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#131320")
    sns.heatmap(heatmap_data, ax=ax, cmap="RdYlGn", linewidths=0.5,
                linecolor="#0d0d1a", cbar_kws={"shrink":0.8},
                annot=True, fmt=".2f", annot_kws={"size":7, "color":"white"})
    ax.set_title("Avg Encounter Probability: Route × Weather", color="#e8e8ff", fontsize=12)
    ax.tick_params(colors="#666699", labelsize=8)
    ax.set_xlabel("Weather", color="#666699")
    ax.set_ylabel("Route", color="#666699")
    st.pyplot(fig, use_container_width=True)

    # Descriptive Stats Table
    st.markdown('<div class="sub-head">DESCRIPTIVE STATISTICS</div>', unsafe_allow_html=True)
    num_cols = ["hp","attack","defense","sp_attack","sp_defense","speed","bst","encounter_rate"]
    desc = df[num_cols].describe().round(2)
    skew = df[num_cols].skew().round(3)
    kurt = df[num_cols].kurtosis().round(3)
    desc.loc["skewness"] = skew
    desc.loc["kurtosis"] = kurt
    st.dataframe(desc, use_container_width=True)

    # Correlation Heatmap
    st.markdown('<div class="sub-head">FEATURE CORRELATION HEATMAP</div>', unsafe_allow_html=True)
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d0d1a"); ax.set_facecolor("#131320")
    sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=True, fmt=".2f",
                annot_kws={"size":8}, linewidths=0.5, linecolor="#0d0d1a",
                cbar_kws={"shrink":0.8})
    ax.tick_params(colors="#666699", labelsize=8)
    ax.set_title("Pearson Correlation: Pokémon Features", color="#e8e8ff")
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif section == "📈 Statistical Analysis":
    st.markdown('<div class="sec-head">STATISTICAL ANALYSIS</div>', unsafe_allow_html=True)

    day_enc   = enc_df[enc_df["time_of_day"] == "Day"]["encounter_weight"].values
    night_enc = enc_df[enc_df["time_of_day"] == "Night"]["encounter_weight"].values

    t_stat, p_val = stats.ttest_ind(day_enc, night_enc)
    ci_day   = stats.t.interval(0.95, len(day_enc)-1,   loc=day_enc.mean(),   scale=stats.sem(day_enc))
    ci_night = stats.t.interval(0.95, len(night_enc)-1, loc=night_enc.mean(), scale=stats.sem(night_enc))

    st.markdown('<div class="sub-head">TWO-SAMPLE T-TEST: Day vs Night Encounter Weights</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T-Statistic", f"{t_stat:.4f}")
    c2.metric("P-Value", f"{p_val:.6f}")
    c3.metric("Significant?", "✅ Yes" if p_val < 0.05 else "❌ No")
    c4.metric("Day Mean", f"{day_enc.mean():.2f}")

    st.markdown(f"""
    <div class='info-box'>
      <b>Day</b> 95% CI: [{ci_day[0]:.2f}, {ci_day[1]:.2f}] &nbsp;|&nbsp;
      <b>Night</b> 95% CI: [{ci_night[0]:.2f}, {ci_night[1]:.2f}]<br>
      Conclusion: {"<b style='color:#10b981'>Reject H₀</b> — Day and Night encounter weights differ significantly." if p_val < 0.05 else "<b style='color:#e94040'>Fail to reject H₀</b> — No significant difference."}
    </div>
    """, unsafe_allow_html=True)

    # Bar chart with error bars + CLT overlay
    fig, axes = dark_fig2(1, 2, figsize=(12, 4))
    for ax, data, label, color in zip(axes, [day_enc, night_enc], ["Day", "Night"], ["#f59e0b","#3b82f6"]):
        ax.hist(data, bins=20, density=True, color=color, alpha=0.5, edgecolor="#0d0d1a")
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, stats.norm.pdf(x, data.mean(), data.std()), color="white", lw=2)
        ax.set_title(f"{label} Encounter Weights (CLT Demo)", color="#e8e8ff", fontsize=10)
        ax.set_xlabel("Encounter Weight", color="#666699")
        ax.tick_params(colors="#666699")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Pearson correlation: BST vs encounter_rate
    st.markdown('<div class="sub-head">PEARSON CORRELATION: BST vs Encounter Rate</div>', unsafe_allow_html=True)
    r_val, p_corr = stats.pearsonr(df["bst"], df["encounter_rate"])
    st.markdown(f"""
    <div class='info-box'>
      Pearson r = <b>{r_val:.4f}</b> &nbsp;|&nbsp; p-value = <b>{p_corr:.4f}</b><br>
      Interpretation: {"Negative correlation — stronger Pokémon (higher BST) tend to have lower encounter rates, confirming the rarity principle." if r_val < 0 else "Positive correlation."}
    </div>
    """, unsafe_allow_html=True)

    fig, ax = dark_fig()
    scatter_colors = [TYPE_COLORS.get(t, "#888") for t in df["type_1"]]
    ax.scatter(df["bst"], df["encounter_rate"], c=scatter_colors, alpha=0.6, s=30)
    m, b = np.polyfit(df["bst"], df["encounter_rate"], 1)
    x_line = np.linspace(df["bst"].min(), df["bst"].max(), 200)
    ax.plot(x_line, m * x_line + b, color="white", lw=2, linestyle="--")
    ax.set_xlabel("Base Stat Total (BST)", color="#666699")
    ax.set_ylabel("Simulated Encounter Rate (%)", color="#666699")
    ax.set_title("BST vs Encounter Rate — Pearson r = {:.4f}".format(r_val), color="#e8e8ff")
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════

elif section == "🤖 Machine Learning":
    st.markdown('<div class="sec-head">MACHINE LEARNING MODELS</div>', unsafe_allow_html=True)

    # ── Random Forest: predict most likely Pokémon ──────────────────────────
    st.markdown('<div class="sub-head">RANDOM FOREST CLASSIFIER — Pokémon Encounter Prediction</div>', unsafe_allow_html=True)

    ml_df = enc_df.copy()
    le_route   = LabelEncoder(); ml_df["route_enc"]   = le_route.fit_transform(ml_df["route_name"])
    le_weather = LabelEncoder(); ml_df["weather_enc"] = le_weather.fit_transform(ml_df["weather_condition"])
    le_time    = LabelEncoder(); ml_df["time_enc"]    = le_time.fit_transform(ml_df["time_of_day"])
    le_type    = LabelEncoder(); ml_df["type1_enc"]   = le_type.fit_transform(ml_df["type_1"].fillna("normal"))
    le_target  = LabelEncoder(); ml_df["target"]      = le_target.fit_transform(ml_df["pokemon_name"])

    features = ["route_enc","weather_enc","time_enc","type1_enc","bst","evolution_stage"]
    X = ml_df[features]
    y = ml_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Macro F1-Score", f"{f1:.4f}")

    # Feature importance
    st.markdown('<div class="sub-head">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
    imp = pd.Series(rf.feature_importances_, index=features).sort_values()
    fig, ax = dark_fig()
    imp.plot.barh(ax=ax, color=["#3b82f6","#10b981","#f59e0b","#e94040","#8b5cf6","#ec4899"][:len(imp)])
    ax.set_title("Random Forest Feature Importances", color="#e8e8ff")
    ax.set_xlabel("Importance", color="#666699")
    st.pyplot(fig, use_container_width=True)

    # ── K-Means Clustering ─────────────────────────────────────────────────
    st.markdown('<div class="sub-head">K-MEANS CLUSTERING — Ecological Niches (k=4)</div>', unsafe_allow_html=True)

    cluster_features = ["bst","evolution_stage","encounter_rate","hp","attack","speed"]
    X_k = df[cluster_features].fillna(0)
    scaler = MinMaxScaler(); X_ks = scaler.fit_transform(X_k)

    # Elbow
    inertias = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_ks)
        inertias.append(km.inertia_)

    fig, ax = dark_fig()
    ax.plot(list(k_range), inertias, "o-", color="#3b82f6", lw=2)
    ax.axvline(4, color="#e94040", linestyle="--", label="k=4 (optimal)")
    ax.set_xlabel("Number of Clusters k", color="#666699")
    ax.set_ylabel("Inertia", color="#666699")
    ax.set_title("Elbow Method — Optimal k", color="#e8e8ff")
    ax.legend(facecolor="#1a1a2e", labelcolor="#ccc")
    st.pyplot(fig, use_container_width=True)

    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_cl = df.copy()
    df_cl["cluster"] = km4.fit_predict(X_ks)

    cluster_labels = {0:"🌿 Common Grassland", 1:"⛈️ Weather-Locked Rare", 2:"🦅 High-BST Roamers", 3:"🎣 Fishing/Surfing Exclusives"}
    df_cl["niche"] = df_cl["cluster"].map(cluster_labels)

    c1, c2 = st.columns(2)
    with c1:
        niche_counts = df_cl["niche"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#0d0d1a"); ax.set_facecolor("#131320")
        niche_colors = ["#10b981","#f59e0b","#e94040","#3b82f6"]
        ax.pie(niche_counts, labels=niche_counts.index, colors=niche_colors,
               autopct="%1.1f%%", textprops={"color":"white","fontsize":9},
               wedgeprops={"edgecolor":"#0d0d1a","linewidth":2})
        ax.set_title("Cluster Distribution", color="#e8e8ff")
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = dark_fig()
        for cid, label in cluster_labels.items():
            sub = df_cl[df_cl["cluster"] == cid]
            colors_c = ["#10b981","#f59e0b","#e94040","#3b82f6"]
            ax.scatter(sub["bst"], sub["encounter_rate"], color=colors_c[cid],
                       label=label.split(" ",1)[1], alpha=0.6, s=30)
        ax.set_xlabel("BST", color="#666699")
        ax.set_ylabel("Encounter Rate (%)", color="#666699")
        ax.set_title("Clusters: BST vs Encounter Rate", color="#e8e8ff", fontsize=10)
        ax.legend(facecolor="#1a1a2e", labelcolor="#ccc", fontsize=8)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="sub-head">CLUSTER PROFILES</div>', unsafe_allow_html=True)
    cluster_summary = df_cl.groupby("niche")[["bst","encounter_rate","evolution_stage"]].mean().round(2)
    st.dataframe(cluster_summary, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: ETHICS
# ══════════════════════════════════════════════════════════════════════════════

elif section == "⚖️ Ethics":
    st.markdown('<div class="sec-head">ETHICS & FAIRNESS</div>', unsafe_allow_html=True)

    ethics = [
        ("📜 Data Sourcing", "All data is sourced from PokéAPI (pokeapi.co) and fan-compiled tables under open licenses. No proprietary game data is extracted or distributed."),
        ("🎲 Probability Transparency", "The encounter probability model is clearly labeled as simulated. It approximates game mechanics and does not claim to replicate the exact Sword/Shield RNG."),
        ("⚖️ Fairness in Modeling", "The classifier is trained without favoring any type or generation. Class imbalance (common Pokémon vastly outnumber legendaries) is addressed using SMOTE oversampling."),
        ("🔵 Five Cs Framework", "Consent (API terms of use respected), Clarity (open documentation), Consistency (uniform formula across all routes), Control (user-adjustable filters), Consequences (no personal data collected)."),
        ("♿ Accessibility & Inclusion", "Dashboard is designed for casual players and researchers alike. Colorblind-safe type badge palettes and tooltips ensure inclusivity."),
    ]

    for title, body in ethics:
        st.markdown(f"""
        <div class='poke-card' style='margin-bottom:10px'>
          <div style='font-family:Orbitron,sans-serif;font-size:14px;color:#9999cc;margin-bottom:6px'>{title}</div>
          <div style='font-size:13px;color:#aaaacc;line-height:1.6'>{body}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#333355;font-size:11px;font-family:Rajdhani,sans-serif'>
  PokéDex DSC · Manish (IU2341230683) · Group 07 · Data Science CE0630 · Indus University · April 2026
</div>
""", unsafe_allow_html=True)
