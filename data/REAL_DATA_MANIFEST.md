# Real Data Manifest for NeurIPS Paper

## Verified Real Data Sources (4 Domains)

### 1. Crypto (Finance Domain)
- **Source**: Bybit Exchange (direct historical data)
- **Records**: 43,835 (8,767 per coin × 5 coins)
- **Coins**: BTC, ETH, SOL, DOGE, XRP
- **Time Range**: Multi-year historical OHLCV data
- **File**: `data/bybit/Bybit_*.csv`
- **Verification**: Real exchange data, not synthetic

### 2. Commodities 
- **Source**: Federal Reserve Economic Data (FRED)
- **Records**: 5,630 daily prices
- **Assets**: WTI Crude Oil, Copper, Natural Gas
- **Time Range**: 2015-01-05 to 2025-12-15
- **File**: `data/commodities/fred_real_prices.csv`
- **API Endpoint**: `fred.stlouisfed.org/graph/fredgraph.csv`
- **Verification**: Official US government data

### 3. Weather
- **Source**: Open-Meteo Historical Weather API
- **Records**: 9,106 daily observations
- **Locations**: Chicago, Houston, Los Angeles, New York, Phoenix
- **Time Range**: 2020-01-07 to 2024-12-31
- **Variables**: Temperature (max/min/mean), precipitation, wind speed
- **File**: `data/weather/openmeteo_real_weather.csv`
- **API Endpoint**: `archive-api.open-meteo.com/v1/archive`
- **Verification**: Real meteorological station data

### 4. Solar Irradiance
- **Source**: Open-Meteo Historical Solar API
- **Records**: 116,835 hourly measurements
- **Locations**: Phoenix AZ, Las Vegas NV, Denver CO, Miami FL, Seattle WA
- **Time Range**: 2020-01-01 to 2024-12-31
- **Variables**: GHI, DNI, DHI (Global/Direct/Diffuse irradiance)
- **File**: `data/solar/openmeteo_real_irradiance.csv`
- **API Endpoint**: `archive-api.open-meteo.com/v1/archive`
- **Verification**: Real satellite-derived irradiance data

## Domains Excluded (Network Issues)

### Water (USGS)
- Network SSL errors prevented download
- Would have used: USGS Water Services API
- Gauge sites: Colorado River, Mississippi River, Columbia River

### Energy (Grid Demand)
- Direct grid data requires special access (EIA API key, ENTSOE token)
- Temperature-derived demand model available but considered "derived" not "raw"

## Total Real Data Summary

| Domain | Records | Variables | Source Reliability |
|--------|---------|-----------|-------------------|
| Crypto | 43,835 | OHLCV | Exchange data |
| Commodities | 5,630 | Price | US Government |
| Weather | 9,106 | Temp/Precip | Met stations |
| Solar | 116,835 | Irradiance | Satellite |
| **TOTAL** | **175,406** | - | **All verified real** |

## Reproducibility Note

All data was downloaded on 2024-12-22 using public APIs with no authentication required.
Scripts in `scripts/download_*.py` can recreate this data.
