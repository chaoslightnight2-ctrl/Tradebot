# Sontrade Tradebot

Dashboard olmadan GitHub/terminal üzerinde çalışacak Alpaca paper trading ve backtest projesi.

> Uyarı: Bu proje yatırım tavsiyesi değildir. Backtest sonuçları canlı performans garantisi vermez. Gerçek para kullanmadan önce uzun süre paper trading ile doğrula.

## Özellikler

- Alpaca paper trading desteği
- YFinance veya Alpaca veri kaynağı ile eğitim/backtest
- Long/short skor modeli
- ATR tabanlı stop-loss ve take-profit simülasyonu
- Dashboard zorunluluğu yok
- GitHub Actions ile syntax/test kontrolü

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` dosyasına kendi paper keylerini yaz:

```env
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
```

## Komutlar

```bash
python sontrade_bot.py train
python sontrade_bot.py backtest
python sontrade_bot.py optimize-thresholds
python sontrade_bot.py score
python sontrade_bot.py trade
python sontrade_bot.py loop
```

## Ücretsiz veri modu

Backtest için API key istemeden denemek istersen:

```env
SONTRADE_DATA_PROVIDER=yfinance
ALPACA_SYMBOLS=AAPL
SONTRADE_LOOKBACK_DAYS=2500
```

Paper trade için Alpaca paper key gerekir.

## Güvenlik

API keyler public repoya hardcoded eklenmez. `.env` dosyası `.gitignore` içindedir. GitHub Actions kullanacaksan keyleri `Settings > Secrets and variables > Actions` bölümünden secret olarak ekle.
