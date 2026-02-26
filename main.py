#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import logging
import re
import string
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtSvg, QtWidgets

APP_TITLE = "VJ Desktop — Подбор насосов"  # Заголовок окна приложения
GOOGLE_SHEET_ID = "1vnEhN44pmiwOdMpY8_wJc4pzGTtWNncSey6hrLoboXo"  # ID Google таблицы
INSTALLATION_GIDS = [217699851, 1623760841]  # Листы установок
PUMP_GIDS = [1779550847, 983928362]  # Листы насосов

WINDOW_MIN_SIZE = (1240, 780)  # Минимальный размер главного окна
PARAM_PANEL_WIDTH = 420  # Фиксированная ширина правой панели параметров
SEARCH_MIN_CHARS = 3  # Минимум символов для запуска поиска
SEARCH_LIMIT = 100  # Максимум позиций в выпадающем поиске
SVG_TEMPLATE_PATH = Path("hfs_small_low.svg")  # Путь к SVG шаблону чертежа
CATALOG_CACHE_PATH = Path("catalog_cache.json")  # Локальный cache каталога

PARAM_FONT_SIZE_PT = 9.5  # Размер шрифта значений в блоке параметров
GRAPH_WIDTH = 760  # Фиксированная ширина области графиков
GRAPH_HEIGHT = 540  # Фиксированная высота области графиков
MID_INFO_BOX_WIDTH_RATIO = 0.96  # Относительная ширина инфобокса между графиками
FACT_WRAP_AFTER = 90  # Позиция переноса строк в блоке "Рабочая точка"
SYSTEM_WRAP_AFTER = 100  # Позиция переноса строк в блоке "Параметры системы"

SOURCE_LAYOUTS = {
    217699851: {"type": "HFS_CRV", "name_col": "A", "article_col": "AE", "motor_power_col": "Z"},
    1623760841: {"type": "HFS_CRV", "name_col": "A", "article_col": "AE", "motor_power_col": "Z"},
    1779550847: {"type": "CRV", "name_col": "BZ", "article_col": "B", "motor_power_col": "BU"},
    983928362: {"type": "CRV", "name_col": "BZ", "article_col": "B", "motor_power_col": "BU"},
}

VISIBLE_PARAMS_BY_TYPE: dict[str, dict[str, str]] = {
    "HFS_CRV": {
        "Артикул": "Артикул",
        "Наименование": "Наименование",
        "DN основных коллекторов": "DN",
        "Кол-во насосов": "К",
    },
    "CRV": {
        "Артикул": "Артикул",
        "Наименование": "Наименование",
        "Мощность двигателя": "P2N",
        "DN": "DN",
    },
    "DEFAULT": {"Артикул": "Артикул", "Наименование": "Наименование"},
}

FLUID_DEFAULTS = {"name": "Вода", "temp_c": "20", "density": "998.3", "viscosity": "1"}

COLORS = {
    "q_h": "#1f4e79",
    "q_eta": "#2e7d32",
    "q_p": "#5e35b1",
    "q_npsh": "#8e24aa",
    "pipeline": "#c62828",
    "intersection": "#e60000",
    "workpoint": "#7f7f7f",
}

POLYNOMIAL_DEGREES = {"q_h": 3, "q_eta": 3, "q_p": 3, "q_npsh": 3}  # Порядки полиномов для каждой кривой
RIGHT_AXIS_VISIBLE_RATIO = 0.4  # Видимая доля правой оси по высоте
WORK_ZONE_LEFT_BEP_FACTOR = 0.7  # Левая граница рабочей зоны от BEP
WORK_ZONE_RIGHT_BEP_FACTOR = 1.2  # Правая граница рабочей зоны от BEP
LINE_WIDTH_WORKZONE = 3.0  # Толщина линии в рабочей зоне
LINE_WIDTH_OUTSIDE = 1.0  # Толщина линии вне рабочей зоны
LINE_WIDTH_SECONDARY = 1.0  # Толщина линий неактивных насосов

ENABLE_VERBOSE_LOGS = True  # Подробный вывод логов в консоль
LOG_LEVEL = logging.DEBUG  # Уровень логирования


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("vj_desktop")
    if logger.handlers:
        return logger
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logger.setLevel(LOG_LEVEL if ENABLE_VERBOSE_LOGS else logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


LOGGER = setup_logging()


def excel_col_to_index(col: str) -> int:
    c = col.strip().upper()
    n = 0
    for ch in c:
        if ch in string.ascii_uppercase:
            n = n * 26 + (ord(ch) - 64)
    return max(0, n - 1)

def q_factor(unit: str) -> float:
    return {"м3/ч": 1.0, "л/с": 1.0 / 3.6, "л/мин": 1000.0 / 60.0}.get(unit, 1.0)


def h_factor(unit: str) -> float:
    # conversion from meters head for water-like fluid
    return {"м": 1.0, "бар": 0.0980665, "кПа": 9.80665, "МПа": 0.00980665}.get(unit, 1.0)


def round_up_nice(max_val: float) -> tuple[float, float]:
    if max_val <= 0:
        return 1.0, 0.2
    if max_val < 1:
        step = 0.1 if max_val <= 0.5 else 0.2
    elif max_val < 10:
        step = 1.0 if max_val <= 6 else 2.0
    else:
        # 1/2/5 * 10^n
        n = 10 ** int(np.floor(np.log10(max_val)))
        for m in (1, 2, 5, 10):
            step = m * n / 5
            if max_val / step <= 8:
                break
    vmax = np.ceil(max_val / step) * step
    return float(vmax), float(step)



@dataclass
class CatalogItem:
    key: str
    article: str
    name: str
    equipment_type: str
    source_gid: int
    params: dict[str, str]
    by_col_letter: dict[str, str]
    curves: dict[str, list[tuple[float, float]]]
    main_pumps_default: int
    pump_model: str


class MemoryCatalog:
    def __init__(self) -> None:
        self.items_by_key: dict[str, CatalogItem] = {}
        self._pump_curve_index: dict[str, dict[str, list[tuple[float, float]]]] = {}

    @staticmethod
    def _norm(v: Any) -> str:
        return re.sub(r"\s+", " ", str(v).replace("\ufeff", "")).strip()

    @classmethod
    def _norm_cf(cls, v: Any) -> str:
        return cls._norm(v).casefold()

    def _csv_url(self, gid: int) -> str:
        return f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={gid}"

    def _fetch_gid(self, gid: int) -> pd.DataFrame:
        resp = requests.get(self._csv_url(gid), timeout=30)
        resp.raise_for_status()
        try:
            txt = resp.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            txt = resp.text
        df = pd.read_csv(io.StringIO(txt)).fillna("")
        df.columns = [self._norm(c) if self._norm(c) else f"COL_{i}" for i, c in enumerate(df.columns)]
        return df

    def _save_cache(self) -> None:
        data = []
        for item in self.items_by_key.values():
            data.append(
                {
                    "key": item.key,
                    "article": item.article,
                    "name": item.name,
                    "equipment_type": item.equipment_type,
                    "source_gid": item.source_gid,
                    "params": item.params,
                    "by_col_letter": item.by_col_letter,
                    "curves": item.curves,
                    "main_pumps_default": item.main_pumps_default,
                    "pump_model": item.pump_model,
                }
            )
        CATALOG_CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _load_cache(self) -> None:
        self.items_by_key.clear()
        arr = json.loads(CATALOG_CACHE_PATH.read_text(encoding="utf-8"))
        for d in arr:
            self.items_by_key[d["key"]] = CatalogItem(
                key=d["key"],
                article=d["article"],
                name=d["name"],
                equipment_type=d["equipment_type"],
                source_gid=d["source_gid"],
                params=d["params"],
                by_col_letter=d["by_col_letter"],
                curves={k: [tuple(p) for p in v] for k, v in d["curves"].items()},
                main_pumps_default=d.get("main_pumps_default", 1),
                pump_model=d.get("pump_model", ""),
            )

    def load(self, force_refresh: bool = False) -> None:
        if CATALOG_CACHE_PATH.exists() and not force_refresh:
            self._load_cache()
            LOGGER.info("Catalog from cache: %s", len(self.items_by_key))
            return

        self.items_by_key.clear()
        self._pump_curve_index.clear()

        for gid in PUMP_GIDS + INSTALLATION_GIDS:
            layout = SOURCE_LAYOUTS[gid]
            self._parse_df(self._fetch_gid(gid), gid, layout)

        self._save_cache()
        LOGGER.info("Catalog from google: %s", len(self.items_by_key))

    def _parse_curve_points(self, raw: dict[str, str]) -> dict[str, list[tuple[float, float]]]:
        def arr(prefixes: list[str]) -> list[float]:
            vals = []
            for i in range(1, 11):
                val = None
                for p in prefixes:
                    key = f"{p}{i}"
                    if key in raw and raw[key] != "":
                        val = raw[key]
                        break
                try:
                    vals.append(float(str(val).replace(",", "."))) if val is not None else vals.append(np.nan)
                except ValueError:
                    vals.append(np.nan)
            return vals

        q = arr(["Q"])
        h = arr(["H"])
        p = arr(["P"])
        n = arr(["NPSH", "NPSHr", "NPSHR"])
        e = arr(["КПД", "ETA", "Eta"])
        if len(e):
            e[0] = 0.0

        def pair(x: list[float], y: list[float]) -> list[tuple[float, float]]:
            return [(a, b) for a, b in zip(x, y) if not np.isnan(a) and not np.isnan(b)]

        curves = {"q_h": pair(q, h), "q_eta": pair(q, e), "q_p": pair(q, p), "q_npsh": pair(q, n)}
        if not curves["q_h"]:
            qd = np.linspace(5, 50, 10)
            curves = {
                "q_h": list(zip(qd, 40 - 0.5 * qd)),
                "q_eta": list(zip(qd, np.maximum(0, 20 + 1.2 * qd - 0.015 * qd**2))),
                "q_p": list(zip(qd, 2 + 0.08 * qd)),
                "q_npsh": list(zip(qd, 1.5 + 0.03 * qd)),
            }
        return curves

    def _parse_df(self, df: pd.DataFrame, gid: int, layout: dict[str, str]) -> None:
        cols = list(df.columns)
        ni = excel_col_to_index(layout["name_col"])
        ai = excel_col_to_index(layout["article_col"])
        if ni >= len(cols) or ai >= len(cols):
            return

        for _, row in df.iterrows():
            raw = {self._norm(k): self._norm(v) for k, v in row.to_dict().items()}
            article = self._norm(row.iloc[ai])
            name = self._norm(row.iloc[ni])
            if not article and not name:
                continue
            raw.setdefault("Артикул", article)
            raw.setdefault("Наименование", name)

            bycol = {layout["name_col"]: name, layout["article_col"]: article}
            main_default = 1
            pump_model = ""

            mi = excel_col_to_index(layout["motor_power_col"])
            if mi < len(cols):
                bycol[layout["motor_power_col"]] = self._norm(row.iloc[mi])

            if layout["type"] == "HFS_CRV":
                bi = excel_col_to_index("B")
                xi = excel_col_to_index("X")
                vi = excel_col_to_index("V")
                if bi < len(cols):
                    pump_model = self._norm(row.iloc[bi])
                if xi < len(cols):
                    xv = self._norm(row.iloc[xi])
                    bycol["X"] = xv
                    try:
                        main_default = max(1, int(float(xv.replace(",", "."))))
                    except ValueError:
                        pass
                if vi < len(cols):
                    dn = self._norm(row.iloc[vi])
                    bycol["V"] = dn
                    if dn:
                        raw["DN"] = dn

            curves = self._parse_curve_points(raw)
            if layout["type"] == "CRV":
                self._pump_curve_index[self._norm_cf(name)] = curves
            if layout["type"] == "HFS_CRV" and pump_model:
                pm = self._norm_cf(pump_model)
                matched = self._pump_curve_index.get(pm)
                if matched is None:
                    matched = next((v for k, v in self._pump_curve_index.items() if pm in k or k in pm), None)
                if matched is not None:
                    curves = matched

            key = self._norm(f"{article} {name}")
            self.items_by_key[key] = CatalogItem(
                key=key,
                article=article,
                name=name,
                equipment_type=layout["type"],
                source_gid=gid,
                params=raw,
                by_col_letter=bycol,
                curves=curves,
                main_pumps_default=main_default,
                pump_model=pump_model,
            )

    def search_keys(self, query: str, limit: int = SEARCH_LIMIT) -> list[str]:
        q = self._norm_cf(query)
        toks = [t for t in q.split() if t]
        out = []
        for k in self.items_by_key:
            kk = self._norm_cf(k)
            if all(t in kk for t in toks):
                out.append(k)
        return sorted(out)[:limit]

    def get(self, key: str) -> CatalogItem | None:
        return self.items_by_key.get(key)


class PlotWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(facecolor="white", layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

        self._hover = []
        self._annot = None
        self._td = None
        self._bd = None
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("axes_leave_event", self._clear_hover)

    @staticmethod
    def _poly(points: list[tuple[float, float]], deg: int) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([p[0] for p in points], dtype=float)
        y = np.array([p[1] for p in points], dtype=float)
        z = np.polyfit(x, y, min(deg, max(1, len(points) - 1)))
        p = np.poly1d(z)
        xx = np.linspace(0, max(float(np.max(x)), 1.0), 260)
        yy = np.maximum(0, p(xx))
        return xx, yy

    @staticmethod
    def _zones(x: np.ndarray, y: np.ndarray, l: float, r: float):
        a = x < l
        b = (x >= l) & (x <= r)
        c = x > r
        return (x[a], y[a]), (x[b], y[b]), (x[c], y[c])

    @staticmethod
    def _format_right_axis(axr, y_max: float):
        visible_max, step = round_up_nice(y_max * 1.2)
        total_max = visible_max / max(RIGHT_AXIS_VISIBLE_RATIO, 0.05)
        axr.set_ylim(0, total_max)
        ticks = np.arange(0, total_max + step * 0.5, step)
        axr.set_yticks(ticks)
        axr.set_yticklabels([f"{t:g}" if t <= visible_max + 1e-9 else "" for t in ticks])

    def render(self, item: CatalogItem, main_count: int, reserve_count: int, show_reserve: bool, q_work: float | None, h_work: float | None, h_static: float | None, fluid: dict[str, str], q_unit: str, h_unit: str) -> None:
        self.figure.clear()
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1.0, 0.26, 1.0], hspace=0.02)
        at = self.figure.add_subplot(gs[0, 0])
        am = self.figure.add_subplot(gs[1, 0])
        ab = self.figure.add_subplot(gs[2, 0])
        atr = at.twinx()
        abr = ab.twinx()
        am.axis("off")
        for a in (at, atr, ab, abr, am):
            a.set_facecolor("white")

        qh_x, qh_y = self._poly(item.curves["q_h"], POLYNOMIAL_DEGREES["q_h"])
        qf = q_factor(q_unit)
        hf = h_factor(h_unit)
        qe_x, qe_y = self._poly(item.curves["q_eta"], POLYNOMIAL_DEGREES["q_eta"])
        qp_x, qp_y = self._poly(item.curves["q_p"], POLYNOMIAL_DEGREES["q_p"])
        qn_x, qn_y = self._poly(item.curves["q_npsh"], POLYNOMIAL_DEGREES["q_npsh"])
        qh_x, qe_x, qp_x, qn_x = qh_x * qf, qe_x * qf, qp_x * qf, qn_x * qf
        qh_y, qn_y = qh_y * hf, qn_y * hf
        if len(qe_y):
            qe_y[0] = 0

        active = max(1, main_count)
        total = max(1, main_count + (reserve_count if show_reserve else 0))
        q_bep = float(qe_x[np.argmax(qe_y)]) if len(qe_x) else 0
        wz_l = WORK_ZONE_LEFT_BEP_FACTOR * q_bep * active
        wz_r = WORK_ZONE_RIGHT_BEP_FACTOR * q_bep * active

        mcol = SOURCE_LAYOUTS.get(item.source_gid, {}).get("motor_power_col", "")
        try:
            mp = float(str(item.by_col_letter.get(mcol, "")).replace(",", ".")) if mcol else None
        except ValueError:
            mp = None
        if mp is not None:
            qa = qp_x * active
            mask = qp_y <= mp
            if np.any(mask):
                wz_r = min(wz_r, float(np.max(qa[mask])))

        for i in range(1, total + 1):
            act = i == active
            sx = i
            if act:
                for ax, x, y, c in ((at, qh_x * sx, qh_y, COLORS["q_h"]), (atr, qe_x * sx, qe_y, COLORS["q_eta"]), (ab, qp_x * sx, qp_y, COLORS["q_p"]), (abr, qn_x * sx, qn_y, COLORS["q_npsh"])):
                    (xl, yl), (xm, ym), (xr, yr) = self._zones(x, y, wz_l, wz_r)
                    if len(xl): ax.plot(xl, yl, color=c, linewidth=LINE_WIDTH_OUTSIDE)
                    if len(xm): ax.plot(xm, ym, color=c, linewidth=LINE_WIDTH_WORKZONE)
                    if len(xr): ax.plot(xr, yr, color=c, linewidth=LINE_WIDTH_OUTSIDE)
            else:
                at.plot(qh_x * sx, qh_y, color=COLORS["q_h"], linewidth=LINE_WIDTH_SECONDARY, alpha=0.7)
                atr.plot(qe_x * sx, qe_y, color=COLORS["q_eta"], linewidth=LINE_WIDTH_SECONDARY, alpha=0.7)
                ab.plot(qp_x * sx, qp_y, color=COLORS["q_p"], linewidth=LINE_WIDTH_SECONDARY, alpha=0.7)
                abr.plot(qn_x * sx, qn_y, color=COLORS["q_npsh"], linewidth=LINE_WIDTH_SECONDARY, alpha=0.7)

        for a in (at, ab):
            a.set_xlim(0, max(1, float(np.max(qh_x) * total)))
            a.set_ylim(bottom=0)
            a.margins(x=0, y=0)
            a.grid(True, color="#dcdcdc", linewidth=0.6)

        h_max, _ = round_up_nice(float(np.max(qh_y)) * 1.2)
        p_max, _ = round_up_nice(float(np.max(qp_y)) * 1.2)
        at.set_ylim(0, h_max)
        ab.set_ylim(0, p_max)
        self._format_right_axis(atr, float(np.max(qe_y)))
        self._format_right_axis(abr, float(np.max(qn_y)))

        at.text(0.0, 1.04, f"H [{h_unit}]", transform=at.transAxes, ha="left", va="bottom", fontsize=9)
        at.text(1.0, 1.04, "КПД [%]", transform=at.transAxes, ha="right", va="bottom", fontsize=9)
        ab.text(0.0, 1.04, "P2 [кВт]", transform=ab.transAxes, ha="left", va="bottom", fontsize=9)
        ab.text(1.0, 1.04, f"NPSH [{h_unit}]", transform=ab.transAxes, ha="right", va="bottom", fontsize=9)
        at.set_xlabel(f"Q [{q_unit}]", loc="right")
        ab.set_xlabel(f"Q [{q_unit}]", loc="right")

        chunks = [f"Жидкость={fluid['name']}", f"t={fluid['temp_c']}°C", f"ρ={fluid['density']} кг/м³", f"ν={fluid['viscosity']} сСт"]
        if q_work is not None: chunks.insert(0, f"Qзад={q_work}")
        if h_work is not None: chunks.insert(1, f"Hзад={h_work}")
        if h_static is not None: chunks.insert(2, f"Hст={h_static}")

        fact = ""
        if q_work is not None and h_work is not None:
            k = (h_work - (h_static or 0.0)) / max(q_work**2, 1e-9)
            qp = np.linspace(0, max(float(np.max(qh_x) * active), q_work), 280)
            hp = (h_static or 0.0) + k * qp**2
            ha = np.interp(qp, qh_x * active, qh_y, left=np.nan, right=np.nan)
            idx = int(np.nanargmin(np.abs(hp - ha)))
            qf, hf = float(qp[idx]), float(hp[idx])
            p2f = float(np.interp(qf, qp_x * active, qp_y))
            ef = float(np.interp(qf, qe_x * active, qe_y))
            nf = float(np.interp(qf, qn_x * active, qn_y))
            at.plot(qp[: idx + 1], hp[: idx + 1], color=COLORS["pipeline"], linewidth=1.2)
            at.scatter([qf], [hf], color=COLORS["intersection"], s=30)
            at.scatter([q_work], [h_work], color=COLORS["workpoint"], s=28)
            fact = f"Рабочая точка:\nQ={qf:.1f} м³/ч | H={hf:.1f} м | P2={p2f:.2f} кВт | η={ef:.1f}% | NPSH={nf:.2f} м"

        info_x = (1.0 - MID_INFO_BOX_WIDTH_RATIO) / 2.0
        if fact:
            am.text(info_x, 0.66, textwrap.fill(fact, FACT_WRAP_AFTER), transform=am.transAxes, fontsize=8.7, ha="left", va="center")
        system = "Параметры системы:\n" + "; ".join(chunks)
        am.text(info_x, 0.18, textwrap.fill(system, SYSTEM_WRAP_AFTER), transform=am.transAxes, fontsize=8.2, ha="left", va="center")

        self._td = {"ax": at, "axr": atr, "x": qh_x * active, "y1": qh_y, "y2": np.interp(qh_x * active, qe_x * active, qe_y), "l1": "H", "l2": "КПД"}
        self._bd = {"ax": ab, "axr": abr, "x": qp_x * active, "y1": qp_y, "y2": np.interp(qp_x * active, qn_x * active, qn_y), "l1": "P2", "l2": "NPSH"}
        self._clear_hover(None)
        self.canvas.draw_idle()

    def _clear_hover(self, _):
        for a in self._hover:
            try: a.remove()
            except Exception: pass
        self._hover.clear()
        if self._annot is not None:
            try: self._annot.remove()
            except Exception: pass
            self._annot = None

    def _on_move(self, ev):
        if ev.xdata is None:
            return
        d = None
        if self._td and ev.inaxes in (self._td["ax"], self._td["axr"]): d = self._td
        if self._bd and ev.inaxes in (self._bd["ax"], self._bd["axr"]): d = self._bd
        if d is None:
            return
        x = float(ev.xdata)
        xx = d["x"]
        if x < float(np.min(xx)) or x > float(np.max(xx)):
            return
        y1 = float(np.interp(x, xx, d["y1"]))
        y2 = float(np.interp(x, xx, d["y2"]))
        self._clear_hover(None)
        self._hover.append(d["ax"].scatter([x], [y1], facecolors="none", edgecolors="#111", s=55, zorder=10))
        self._hover.append(d["axr"].scatter([x], [y2], facecolors="none", edgecolors="#111", s=55, zorder=10))
        txt = f"Q={x:.2f}\n{d['l1']}={y1:.2f}\n{d['l2']}={y2:.2f}"
        self._annot = d["ax"].annotate(txt, (x, y1), xytext=(14, -14), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333"), fontsize=8)
        self.canvas.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, catalog: MemoryCatalog):
        super().__init__()
        self.catalog = catalog
        self.current_key: str | None = None

        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(*WINDOW_MIN_SIZE)

        root = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(root)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)

        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Поиск: Артикул + Наименование")
        self.search_input.textChanged.connect(self.on_search)
        self.completer = QtWidgets.QCompleter([], self)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.completer.setFilterMode(QtCore.Qt.MatchContains)
        self.completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.completer.activated[str].connect(self.on_pick)
        self.search_input.setCompleter(self.completer)
        grid.addWidget(self.search_input, 0, 0, 1, 2)

        self.tabs = QtWidgets.QTabWidget()
        self.plot = PlotWidget(); self.plot.setFixedSize(GRAPH_WIDTH, GRAPH_HEIGHT)
        self.svg = QtSvg.QSvgWidget(); self.svg.setFixedSize(GRAPH_WIDTH, GRAPH_HEIGHT)
        self.tabs.addTab(self.plot, "График")
        self.tabs.addTab(self.svg, "Чертеж")
        grid.addWidget(self.tabs, 1, 0)

        units_row = QtWidgets.QHBoxLayout()
        units_row.addWidget(QtWidgets.QLabel("Q:"))
        self.q_unit = QtWidgets.QComboBox(); self.q_unit.addItems(["м3/ч", "л/с", "л/мин"])
        self.q_unit.currentTextChanged.connect(self.refresh_plot)
        units_row.addWidget(self.q_unit)
        units_row.addSpacing(10)
        units_row.addWidget(QtWidgets.QLabel("H:"))
        self.h_unit = QtWidgets.QComboBox(); self.h_unit.addItems(["м", "бар", "кПа", "МПа"])
        self.h_unit.currentTextChanged.connect(self.refresh_plot)
        units_row.addWidget(self.h_unit)
        units_row.addStretch(1)
        grid.addLayout(units_row, 2, 0)

        right = self._right_panel(); right.setFixedWidth(PARAM_PANEL_WIDTH)
        grid.addWidget(right, 1, 1)

        row = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Обновить базу")
        self.refresh_btn.clicked.connect(self.on_refresh_catalog)
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        row.addWidget(QtWidgets.QPushButton("Сформировать тех. лист"))
        grid.addLayout(row, 3, 0, 1, 2)

        self.setCentralWidget(root)

    def _right_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(box)

        w = QtWidgets.QGroupBox("Рабочая точка"); f = QtWidgets.QFormLayout(w); f.setVerticalSpacing(4)
        self.q, self.h, self.hs = QtWidgets.QLineEdit(), QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for e in (self.q, self.h, self.hs): e.textChanged.connect(self.refresh_plot)
        f.addRow("Q [м3/ч]", self.q); f.addRow("H [м]", self.h); f.addRow("Hст [м]", self.hs)
        lay.addWidget(w)

        fl = QtWidgets.QGroupBox("Рабочая жидкость"); ff = QtWidgets.QFormLayout(fl); ff.setVerticalSpacing(4)
        self.fn = QtWidgets.QLineEdit(FLUID_DEFAULTS["name"]); self.ft = QtWidgets.QLineEdit(FLUID_DEFAULTS["temp_c"])
        self.fd = QtWidgets.QLineEdit(FLUID_DEFAULTS["density"]); self.fv = QtWidgets.QLineEdit(FLUID_DEFAULTS["viscosity"])
        for e in (self.fn, self.ft, self.fd, self.fv): e.textChanged.connect(self.refresh_plot)
        ff.addRow("Наименование", self.fn); ff.addRow("Температура [°C]", self.ft); ff.addRow("Плотность [кг/м3]", self.fd); ff.addRow("Вязкость [сСт]", self.fv)
        lay.addWidget(fl)

        p = QtWidgets.QGroupBox("Параллельная работа"); pg = QtWidgets.QGridLayout(p)
        self.main = QtWidgets.QComboBox(); self.main.addItems([str(i) for i in range(1, 9)])
        self.res = QtWidgets.QComboBox(); self.res.addItems([str(i) for i in range(0, 9)])
        self.show_res = QtWidgets.QCheckBox("Показывать резервные")
        self.main.currentTextChanged.connect(self.refresh_plot); self.res.currentTextChanged.connect(self.refresh_plot); self.show_res.toggled.connect(self.refresh_plot)
        pg.addWidget(QtWidgets.QLabel("Основные"), 0, 0); pg.addWidget(self.main, 1, 0)
        pg.addWidget(QtWidgets.QLabel("Резервные"), 0, 1); pg.addWidget(self.res, 1, 1); pg.addWidget(self.show_res, 2, 1)
        lay.addWidget(p)

        lay.addWidget(QtWidgets.QLabel("Параметры"))
        self.scr = QtWidgets.QScrollArea(); self.scr.setWidgetResizable(True)
        self.pw = QtWidgets.QWidget(); self.pg = QtWidgets.QGridLayout(self.pw)
        self.pg.setHorizontalSpacing(6); self.pg.setVerticalSpacing(3)
        self.scr.setWidget(self.pw); lay.addWidget(self.scr, 1)
        return box

    def _fluid(self) -> dict[str, str]:
        return {"name": self.fn.text().strip() or FLUID_DEFAULTS["name"], "temp_c": self.ft.text().strip() or FLUID_DEFAULTS["temp_c"], "density": self.fd.text().strip() or FLUID_DEFAULTS["density"], "viscosity": self.fv.text().strip() or FLUID_DEFAULTS["viscosity"]}

    def on_search(self, text: str) -> None:
        q = text.strip()
        if len(q) < SEARCH_MIN_CHARS:
            self.completer.setModel(QtCore.QStringListModel([])); return
        keys = self.catalog.search_keys(q)
        self.completer.setModel(QtCore.QStringListModel(keys)); self.completer.setCompletionPrefix(q); self.completer.complete()

    def on_pick(self, key: str) -> None:
        self.current_key = key
        self.load_item(key)

    def on_refresh_catalog(self) -> None:
        self.refresh_btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.catalog.load(force_refresh=True)
            QtWidgets.QMessageBox.information(self, "База", "База обновлена")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.refresh_btn.setEnabled(True)

    @staticmethod
    def _f(x: str) -> float | None:
        x = x.strip().replace(",", ".")
        if not x: return None
        try: return float(x)
        except ValueError: return None

    @staticmethod
    def _render_svg(path: Path, params: dict[str, str]) -> bytes:
        if not path.exists(): return b""
        txt = path.read_text(encoding="utf-8")
        return re.sub(r"\{([^{}]+)\}", lambda m: str(params.get(m.group(1).strip(), m.group(0))), txt).encode("utf-8")

    def load_item(self, key: str) -> None:
        item = self.catalog.get(key)
        if not item: return
        while self.pg.count():
            it = self.pg.takeAt(0)
            if it and it.widget(): it.widget().deleteLater()

        self.main.setCurrentText(str(item.main_pumps_default))
        mapping = VISIBLE_PARAMS_BY_TYPE.get(item.equipment_type, VISIBLE_PARAMS_BY_TYPE["DEFAULT"])
        r = 0
        for disp, src in mapping.items():
            lv = QtWidgets.QLabel(disp); lv.setWordWrap(True); lv.setStyleSheet(f"font-size:{PARAM_FONT_SIZE_PT}pt;")
            vv = QtWidgets.QLabel(item.params.get(src, "—") or "—"); vv.setWordWrap(True); vv.setStyleSheet(f"font-size:{PARAM_FONT_SIZE_PT}pt;")
            self.pg.addWidget(lv, r, 0); self.pg.addWidget(vv, r, 1); r += 1

        svg = self._render_svg(SVG_TEMPLATE_PATH, item.params)
        if svg: self.svg.load(svg)
        self.refresh_plot()

    def refresh_plot(self) -> None:
        if not self.current_key: return
        item = self.catalog.get(self.current_key)
        if not item: return
        self.plot.render(item, int(self.main.currentText() or 1), int(self.res.currentText() or 0), self.show_res.isChecked(), self._f(self.q.text()), self._f(self.h.text()), self._f(self.hs.text()), self._fluid(), self.q_unit.currentText(), self.h_unit.currentText())


def apply_style(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QWidget { font-family: 'Segoe UI'; font-size: 10pt; background:#ffffff; }
        QMainWindow { background:#f3f3f3; }
        QLineEdit, QComboBox, QScrollArea, QTabWidget::pane { background: white; border: 1px solid #cfcfcf; border-radius: 4px; padding: 3px; }
        QGroupBox { background:white; border:1px solid #d0d0d0; border-radius:6px; margin-top:10px; padding:6px; font-weight:600; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
        QLabel { background:white; }
        QPushButton { background: #0f62fe; color: white; border: none; border-radius: 6px; padding: 7px 12px; }
        QTabBar::tab { background: #e9e9e9; border: 1px solid #d0d0d0; border-top-left-radius: 7px; border-top-right-radius: 7px; padding: 7px 12px; }
        QTabBar::tab:selected { background: white; border-bottom-color: white; }
        """
    )


def main() -> int:
    LOGGER.info("Application start")
    catalog = MemoryCatalog()
    try:
        catalog.load(force_refresh=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Catalog loading failed: %s", exc)

    app = QtWidgets.QApplication(sys.argv)
    apply_style(app)
    w = MainWindow(catalog)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
