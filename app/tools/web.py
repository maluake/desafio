"""External, keyless (by default) information tools used by the Knowledge Agent.

* ``web_search``  - DuckDuckGo via ``ddgs`` (no key) or Tavily (``TAVILY_API_KEY``).
* ``get_weather`` - Open-Meteo geocoding + daily forecast (no key).
* ``get_exchange_rate`` - Frankfurter (ECB reference rates) with AwesomeAPI fallback (no key).

Every function returns plain JSON-serialisable dicts and raises on failure; the
``ToolExecutor`` converts exceptions into ``is_error`` tool results for the model.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx

from app.config import Settings

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "heavy rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


class WebTools:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.http = client or httpx.Client(timeout=settings.http_timeout_s, follow_redirects=True)

    # ---------------------------------------------------------------- web search
    def web_search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        provider = self.settings.web_search_provider
        max_results = max(1, min(int(max_results), 8))
        if provider == "none":
            raise RuntimeError("web search is disabled")
        if provider == "tavily" and self.settings.tavily_api_key:
            r = self.http.post(
                "https://api.tavily.com/search",
                json={"api_key": self.settings.tavily_api_key, "query": query, "max_results": max_results},
            )
            r.raise_for_status()
            results = [
                {"title": i.get("title"), "url": i.get("url"), "snippet": i.get("content", "")[:500]}
                for i in r.json().get("results", [])
            ]
        else:
            from ddgs import DDGS

            with DDGS(timeout=int(self.settings.http_timeout_s)) as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
            results = [{"title": i.get("title"), "url": i.get("href"), "snippet": i.get("body", "")[:500]} for i in raw]
        return {"query": query, "provider": provider, "results": results}

    # ---------------------------------------------------------------- weather
    def get_weather(self, city: str, day: str | None = None) -> dict[str, Any]:
        """``day`` = 'today' | 'tomorrow' | ISO date. Defaults to 'tomorrow'."""
        geo = self.http.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "pt", "format": "json"},
        )
        geo.raise_for_status()
        hits = geo.json().get("results") or []
        if not hits:
            raise ValueError(f"city not found: {city}")
        place = hits[0]
        target = _resolve_day(day)
        fc = self.http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
                "timezone": place.get("timezone", "auto"),
                "start_date": target.isoformat(),
                "end_date": target.isoformat(),
            },
        )
        fc.raise_for_status()
        d = fc.json()["daily"]
        code = d["weather_code"][0]
        return {
            "city": place["name"],
            "region": place.get("admin1"),
            "country": place.get("country"),
            "date": target.isoformat(),
            "condition": WEATHER_CODES.get(code, f"code {code}"),
            "temp_min_c": d["temperature_2m_min"][0],
            "temp_max_c": d["temperature_2m_max"][0],
            "precipitation_probability_pct": d["precipitation_probability_max"][0],
            "precipitation_mm": d["precipitation_sum"][0],
            "wind_max_kmh": d["wind_speed_10m_max"][0],
            "source": "Open-Meteo (open-meteo.com)",
        }

    # ---------------------------------------------------------------- FX
    def get_exchange_rate(self, base: str, quote: str = "BRL") -> dict[str, Any]:
        base, quote = base.upper()[:3], quote.upper()[:3]
        try:
            r = self.http.get("https://api.frankfurter.dev/v1/latest", params={"base": base, "symbols": quote})
            r.raise_for_status()
            data = r.json()
            return {
                "base": base,
                "quote": quote,
                "rate": data["rates"][quote],
                "date": data["date"],
                "source": "Frankfurter / European Central Bank reference rate",
            }
        except (httpx.HTTPError, KeyError):
            r = self.http.get(f"https://economia.awesomeapi.com.br/json/last/{base}-{quote}")
            r.raise_for_status()
            item = r.json()[f"{base}{quote}"]
            return {
                "base": base,
                "quote": quote,
                "rate": float(item["bid"]),
                "ask": float(item["ask"]),
                "date": item["create_date"],
                "source": "AwesomeAPI (commercial rate, bid)",
            }


def _resolve_day(day: str | None) -> date:
    today = date.today()
    if not day or day.lower() in {"tomorrow", "amanha", "amanhã"}:
        return today + timedelta(days=1)
    if day.lower() in {"today", "hoje"}:
        return today
    return date.fromisoformat(day)
