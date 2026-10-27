# Домашнее задание 1, рекомендательные системы

**Студент:** Данияров Алихан

**Датасет:** Amazon Reviews 2023, категория Video_Games, <https://amazon-reviews-2023.github.io/>

Основной ноутбук: [`experiments.ipynb`](experiments.ipynb), модели в `models/`, общие функции в `utils/`.

## Как запустить

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
curl -L -o data/raw/Video_Games.jsonl.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz
curl -L -o data/raw/meta_Video_Games.jsonl \
  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Video_Games.jsonl
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb
```