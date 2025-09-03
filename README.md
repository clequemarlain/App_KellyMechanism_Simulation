# Edge Pricing & Bidding – α-fair Kelly Simulator

👉 **[Streamlit Cloud](https://app-kellymechanism-simulation.streamlit.app/)**  

A small, sharable environment to simulate α-fair Kelly games with several learning dynamics (DAQ, OGD, SBRD, NumSBRD, XL, DAH).

## Structure
```
game-sim/
  app.py                # Streamlit app to run in a browser (easy to deploy)
  requirements.txt
  src/
    game/
      __init__.py
      utils.py          # Core game logic, plotting, learning rules
      config.py         # Default configuration
      main.py           # CLI runner (python -m game.main)
```

## Local usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run CLI
PYTHONPATH=src python -m game.main

# Run Streamlit
streamlit run app.py
```

## Deploy online

### Streamlit Cloud
- Push this folder to a GitHub repo.
- On streamlit.io/cloud, create a new app, select your repo and set `Main file path` to `app.py`.
- Set Python version (3.10+), and add the `requirements.txt`.

### Hugging Face Spaces
- Create a new Space (type: **Streamlit**).
- Upload all files and choose `app.py` as the entry.
- Add `requirements.txt` to the Space. HF will build and run it.

## Notes
- The `utils.py` contains your provided methods, cleaned and packaged.
- Swap/extend learning methods inside `utils.py::GameKelly`.
- Configure defaults in `src/game/config.py`.
