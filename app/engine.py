import csv
import heapq
import math

import numpy as np
from scipy.spatial import cKDTree


class Locations:
    def __init__(self, csv_path):
        ids, lats, lons, cats = [], [], [], []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                ids.append(int(row["ID"]))
                lats.append(float(row["Latitude"]))
                lons.append(float(row["Longitude"]))
                cats.append(row["Category"].strip().lower())

        self.ids = np.array(ids)
        self.lat = np.array(lats)
        self.lon = np.array(lons)
        self.cat = np.array(cats)
        self.categories = sorted(set(cats))

        self.tree = cKDTree(np.column_stack((self.lat, self.lon)))
        gaps, _ = self.tree.query(self.tree.data, k=2)
        self.spacing = float(np.median(gaps[:, 1]))

    def in_radius(self, lat, lon, cat, rad):
        dx = np.abs(self.lat - lat)
        dy = np.abs(self.lon - lon)
        inside = (self.cat == cat) & (np.sqrt(dx * dx + dy * dy) <= rad)
        return np.flatnonzero(inside), dx + dy

    def nearest_manhattan(self, lat, lon, cat, rad, k=10):
        rows, manhattan = self.in_radius(lat, lon, cat, rad)
        order = np.lexsort((self.ids[rows], manhattan[rows]))[:k]
        return [(int(self.ids[r]), float(manhattan[r])) for r in rows[order]]

    def load_links(self, text):
        points = []
        for line in text.splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) != 4:
                continue
            try:
                points.append([float(x) for x in parts])
            except ValueError:
                continue
        if not points:
            raise ValueError("no links found in the file")

        p = np.array(points)
        a = self.find(p[:, 1], p[:, 0])  # longitude comes first in the file
        b = self.find(p[:, 3], p[:, 2])
        keep = (a >= 0) & (b >= 0) & (a != b)
        if not keep.any():
            raise ValueError("none of the links matches a location")
        a, b = a[keep], b[keep]

        dx = self.lat[a] - self.lat[b]
        dy = self.lon[a] - self.lon[b]
        length = np.sqrt(dx * dx + dy * dy)

        graph = [[] for _ in range(len(self.ids))]
        for u, v, w in zip(a.tolist(), b.tolist(), length.tolist()):
            graph[u].append((v, w))
            graph[v].append((u, w))
        return graph

    def find(self, lat, lon):
        dist, rows = self.tree.query(np.column_stack((lat, lon)))
        return np.where(dist < self.spacing / 4, rows, -1)

    def start_row(self, graph, lat, lon):
        dist = np.sqrt((self.lat - lat) ** 2 + (self.lon - lon) ** 2)
        has_road = np.array([len(roads) > 0 for roads in graph])
        dist[~has_road] = np.inf
        best = np.flatnonzero(dist == dist.min())
        return int(best[np.argmin(self.ids[best])])

    def nearest_by_road(self, graph, lat, lon, cat, rad, k=10):
        rows, manhattan = self.in_radius(lat, lon, cat, rad)
        valid = set(rows.tolist())
        start = self.start_row(graph, lat, lon)

        dist = {start: 0.0}
        done = set()
        heap = [(0.0, start)]
        found = []
        limit = math.inf
        while heap and len(found) < len(valid):
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            if d > limit:
                break
            done.add(u)
            if u in valid:
                found.append((d, int(self.ids[u])))
                if len(found) == k:
                    limit = d
            for v, w in graph[u]:
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(heap, (d + w, v))

        found.sort()  # by distance, then by ID
        hits = [(i, d) for d, i in found[:k]]
        if len(hits) < k:
            # fill up with unreachable ones, closest by Manhattan distance first
            rest = sorted((manhattan[u], int(self.ids[u])) for u in valid if u not in done)
            hits += [(i, None) for _, i in rest[:k - len(hits)]]
        return hits, int(self.ids[start])
