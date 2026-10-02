```
text-to-sql-transformer/
├── app/
│   ├── main.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
├── model/
│   ├── embeddings.py
│   ├── attention.py
│   ├── layers.py
│   └── transformer.py
├── scripts/
│   ├── data_prep.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── check_starter.py
│   ├── train.py
│   ├── decode.py
│   └── evaluate_model.py
├── results/
├── requirements.txt
└── README.md
```
Run these commands in sequence for recreation:
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/download_data.sh
./scripts/download_data.sh
python3 -m scripts.data_prep
python3 -m scripts.tokenizer
python3 -m scripts.check_starter
```