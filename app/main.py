import logging
import math
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .engine import Locations

CSV_PATH = Path(__file__).resolve().parent.parent / "locations.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("proximity")

app = FastAPI(title="Proximity Search API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

locations = Locations(CSV_PATH)
log.info("loaded %d locations", len(locations.ids))

# shows a file upload box on /docs
UPLOAD_FORM = {
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["lat", "long", "cat", "rad", "link"],
                    "properties": {
                        "lat": {"type": "number"},
                        "long": {"type": "number"},
                        "cat": {"type": "string"},
                        "rad": {"type": "number"},
                        "link": {"type": "string", "format": "binary"},
                    },
                }
            }
        }
    }
}


@lru_cache(maxsize=8)
def road_network(data):
    return locations.load_links(data.decode("utf-8-sig", errors="replace"))


async def read_request(request):
    fields = dict(request.query_params)
    link = None
    content_type = request.headers.get("content-type", "")
    if content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        form = await request.form()
        for key, value in form.multi_items():
            if hasattr(value, "read"):
                if link is None or key == "link":
                    link = await value.read()
            else:
                fields[key] = value
    if link is None and fields.get("link"):
        link = fields["link"].encode()  # text sent instead of a file
    return fields, link


def number(fields, name):
    if fields.get(name) in (None, ""):
        raise HTTPException(422, f"{name} is required")
    try:
        return float(fields[name])
    except ValueError:
        raise HTTPException(422, f"{name} must be a number")


def answer(fields, link):
    lat, lon, rad = number(fields, "lat"), number(fields, "long"), number(fields, "rad")
    cat = fields.get("cat", "").strip().lower()
    if not cat:
        raise HTTPException(422, "cat is required")
    if cat not in locations.categories:
        raise HTTPException(400, f"unknown category '{cat}', use one of: {', '.join(locations.categories)}")
    if not all(math.isfinite(x) for x in (lat, lon, rad)) or rad < 0:
        raise HTTPException(400, "lat, long and rad must be finite numbers and rad cannot be negative")

    hits, start = None, None
    if link:
        try:
            hits, start = locations.nearest_by_road(road_network(link), lat, lon, cat, rad)
        except Exception as e:
            log.warning("could not use the link file (%s), falling back to Manhattan distance", e)
    if hits is None:
        hits = locations.nearest_manhattan(lat, lon, cat, rad)

    ids = [i for i, _ in hits]
    log.info("search lat=%s long=%s cat=%s rad=%s roads=%s -> %s", lat, lon, cat, rad, start is not None, ids)
    if str(fields.get("verbose", "")).lower() not in ("1", "true", "yes"):
        return ids
    return {
        "ids": ids,
        "used_roads": start is not None,
        "start": start,
        "results": [{"id": i, "distance": d} for i, d in hits],
    }


@app.get("/search", include_in_schema=False)
@app.get("/search/")
async def search_get(request: Request):
    fields, link = await read_request(request)
    return answer(fields, link)


@app.post("/search", include_in_schema=False)
@app.post("/search/", openapi_extra=UPLOAD_FORM)
async def search_post(request: Request):
    fields, link = await read_request(request)
    return answer(fields, link)


@app.get("/")
def home():
    return {
        "message": "Proximity Search API",
        "usage": "POST /search/ with lat, long, cat, rad and the link file (see /docs)",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
