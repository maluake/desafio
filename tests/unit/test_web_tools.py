import httpx
import respx

from app.config import Settings
from app.tools.web import WebTools


@respx.mock
def test_weather():
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
        return_value=httpx.Response(200, json={"results": [{"name": "Porto Alegre", "latitude": -30.03, "longitude": -51.23, "admin1": "RS", "country": "Brasil", "timezone": "America/Sao_Paulo"}]})
    )
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json={"daily": {"weather_code": [61], "temperature_2m_max": [24.0], "temperature_2m_min": [15.0], "precipitation_probability_max": [70], "precipitation_sum": [5.2], "wind_speed_10m_max": [20.0]}})
    )
    out = WebTools(Settings()).get_weather("Porto Alegre", "tomorrow")
    assert out["condition"] == "light rain" and out["temp_max_c"] == 24.0


@respx.mock
def test_fx_with_fallback():
    respx.get("https://api.frankfurter.dev/v1/latest").mock(return_value=httpx.Response(503))
    respx.get("https://economia.awesomeapi.com.br/json/last/EUR-BRL").mock(
        return_value=httpx.Response(200, json={"EURBRL": {"bid": "6.10", "ask": "6.12", "create_date": "2026-10-15 10:00:00"}})
    )
    out = WebTools(Settings()).get_exchange_rate("eur")
    assert out["rate"] == 6.10 and "AwesomeAPI" in out["source"]


@respx.mock
def test_fx_primary():
    respx.get("https://api.frankfurter.dev/v1/latest").mock(
        return_value=httpx.Response(200, json={"rates": {"BRL": 6.2}, "date": "2026-10-15"})
    )
    assert WebTools(Settings()).get_exchange_rate("EUR")["rate"] == 6.2
