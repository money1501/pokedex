# 🌐 Dual Dashboard — GeoSpatial + Pokémon Gen 8
### OEP Project | Dhyey Vala | IU2341230694 | CE0630

---

## 📦 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the dashboard
```bash
streamlit run dashboard.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🎮 Pokémon Tab Features
- Live data from **PokéAPI** (Gen 8: IDs 810–905)
- Encounter Rate predicted using Linear Regression trained on base stats
- Type distribution, BST radar chart, scatter plots
- Full Pokédex table with search
- Top N strongest Pokémon bar chart
- **Custom stat predictor**: Enter HP/Atk/Def/SpA/SpD/Spe → predict encounter rate + rarity

---

## 📁 Files
| File | Purpose |
|------|---------|
| `dashboard.py` | Main Streamlit app (single file) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

> **Note:** The Pokémon data is fetched from https://pokeapi.co (free, no API key needed).  
> Internet connection required for first load. Data is cached for 1 hour.
