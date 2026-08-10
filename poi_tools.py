import time
import requests
import streamlit as st

from feedback import feedback_boost_map


HEADERS = {
    "User-Agent": "TripAgent/1.0 (student project; Shahin M.)"
}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


INTEREST_TO_TAGS = {
    "food": [
        ("amenity", "restaurant|cafe")
    ],

    "museums": [
        ("tourism", "museum|gallery")
    ],

    "history": [
        (
            "historic",
            "monument|memorial|castle|ruins|archaeological_site"
        ),
        ("tourism", "museum")
    ],

    "outdoors": [
        ("leisure", "park|garden|nature_reserve"),
        ("natural", "peak|beach|waterfall")
    ],

    "shopping": [
        (
            "shop",
            "mall|department_store|clothes|books|gift"
        )
    ],

    "nightlife": [
        ("amenity", "bar|pub|nightclub")
    ],

    "architecture": [
        ("historic", "building|castle|monument"),
        ("tourism", "attraction")
    ],

    "religious sites": [
        (
            "amenity",
            "place_of_worship"
        )
    ],

    "scenic views": [
        ("tourism", "viewpoint"),
        ("natural", "peak")
    ],

    "local culture": [
        ("amenity", "marketplace"),
        ("tourism", "attraction"),
        ("historic", "monument|memorial")
    ]
}


def request_with_retry(
    method,
    url,
    retries=3,
    timeout=20,
    **kwargs
):
    last_error = None

    for attempt in range(retries):
        try:
            response = requests.request(
                method,
                url,
                timeout=timeout,
                **kwargs
            )

            if response.status_code == 429:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response

        except requests.Timeout as error:
            last_error = error

        except requests.RequestException as error:
            last_error = error

        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"API request failed after {retries} attempts: {last_error}"
    )


@st.cache_data(ttl=3600)
def geocode_city(city):
    if not city or not city.strip():
        raise ValueError(
            "City name cannot be empty."
        )

    params = {
        "q": city.strip(),
        "format": "jsonv2",
        "limit": 1
    }

    response = request_with_retry(
        "GET",
        NOMINATIM_URL,
        params=params,
        headers=HEADERS,
        timeout=15
    )

    try:
        data = response.json()

    except requests.exceptions.JSONDecodeError:
        raise ValueError(
            "Nominatim returned an invalid response."
        )

    if not isinstance(data, list):
        raise ValueError(
            "Nominatim returned an unexpected response format."
        )

    if not data:
        return None

    location = data[0]

    if (
        "lat" not in location
        or "lon" not in location
    ):
        raise ValueError(
            "Nominatim response did not contain coordinates."
        )

    try:
        lat = float(location["lat"])
        lon = float(location["lon"])

    except (TypeError, ValueError):
        raise ValueError(
            "Nominatim returned invalid coordinates."
        )

    return {
        "name": location.get(
            "display_name",
            city
        ),
        "lat": lat,
        "lon": lon
    }


def build_overpass_query(
    lat,
    lon,
    interest,
    radius=5000
):
    tags = INTEREST_TO_TAGS.get(
        interest.lower()
    )

    if not tags:
        raise ValueError(
            f"Unsupported interest: {interest}"
        )

    query_parts = []

    for key, values in tags:
        query_parts.append(
            f'nwr["{key}"~"{values}"]'
            f'(around:{radius},{lat},{lon});'
        )

    query = f"""
    [out:json][timeout:25];

    (
        {"".join(query_parts)}
    );

    out center tags;
    """

    return query


@st.cache_data(ttl=1800)
def search_pois(
    city,
    interest,
    radius=5000
):
    location = geocode_city(city)

    if location is None:
        raise ValueError(
            f"Could not find a location matching '{city}'."
        )

    query = build_overpass_query(
        location["lat"],
        location["lon"],
        interest,
        radius
    )

    response = request_with_retry(
        "POST",
        OVERPASS_URL,
        data={"data": query},
        headers=HEADERS,
        timeout=30
    )

    try:
        data = response.json()

    except requests.exceptions.JSONDecodeError:
        raise ValueError(
            "OpenStreetMap returned an invalid response."
        )

    if not isinstance(data, dict):
        raise ValueError(
            "OpenStreetMap returned an unexpected response format."
        )

    elements = data.get("elements")

    if elements is None:
        raise ValueError(
            "OpenStreetMap response did not contain POI data."
        )

    if not isinstance(elements, list):
        raise ValueError(
            "OpenStreetMap POI data was malformed."
        )

    pois = []

    for element in elements:

        if not isinstance(element, dict):
            continue

        tags = element.get("tags", {})

        if not isinstance(tags, dict):
            continue

        name = (
            tags.get("name:en")
            or tags.get("int_name")
            or tags.get("name")
        )

        if not name:
            continue

        poi_lat = element.get("lat")
        poi_lon = element.get("lon")

        if poi_lat is None or poi_lon is None:
            center = element.get("center", {})

            if isinstance(center, dict):
                poi_lat = center.get("lat")
                poi_lon = center.get("lon")

        if poi_lat is None or poi_lon is None:
            continue

        element_type = element.get("type")
        element_id = element.get("id")

        if element_type is None or element_id is None:
            continue

        try:
            poi_lat = float(poi_lat)
            poi_lon = float(poi_lon)

        except (TypeError, ValueError):
            continue

        poi_id = f"{element_type}/{element_id}"

        pois.append({
            "poi_id": poi_id,
            "name": name,
            "category": interest,
            "lat": poi_lat,
            "lon": poi_lon,
            "url": (
                f"https://www.openstreetmap.org/"
                f"{element_type}/{element_id}"
            ),
            "_base_score": 1.0
        })

    boosts = feedback_boost_map(city)

    for poi in pois:
        boost = boosts.get(
            poi["poi_id"],
            0.0
        )

        poi["_feedback_boost"] = boost
        poi["_score"] = (
            poi["_base_score"]
            + boost
        )

    pois.sort(
        key=lambda poi: poi["_score"],
        reverse=True
    )

    return pois[:50]