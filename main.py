#!/usr/bin/env python3
"""VJ Desktop: стартовый прототип подбора насосов и тех. листов."""
from __future__ import annotations

import io
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

# ============================================================
# ВСЕ ОСНОВНЫЕ НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ============================================================
APP_TITLE = "VJ Desktop — Подбор насосов"
DB_PATH = Path("pump_catalog.db")
SYNC_META_PATH = Path("sync_meta.json")
GOOGLE_SHEET_ID = "1vnEhN44pmiwOdMpY8_wJc4pzGTtWNncSey6hrLoboXo"
INSTALLATION_GIDS = [217699851, 1623760841]
PUMP_GIDS = [1779550847, 983928362]

WINDOW_MIN_SIZE = (1200, 760)
PARAM_PANEL_WIDTH = 360
SEARCH_MIN_CHARS = 3

# Стиль линий и графики
COLORS = {
    "q_h": "#0f62fe",
    "q_eta": "#24a148",
    "q_p": "#8a3ffc",
    "q_npsh": "#fa4d56",
    "pipeline": "#525252",
    "intersection": "#da1e28",
    "workpoint": "#878d96",
}
LINE_WIDTH_ACTIVE = 2.8
LINE_WIDTH_SECONDARY = 1.2
POLYNOMIAL_DEGREES = {
    "q_h": 3,
    "q_eta": 4,
    "q_p": 3,
    "q_npsh": 3,
}

# Какие параметры показывать в правой панели по типу оборудования
VISIBLE_PARAMS_BY_TYPE: dict[str, list[str]] = {
    "CRV": ["Ток", "Схема пуска", "Торцевое", "Артикул", "Наименование"],
    "DEFAULT": ["Артикул", "Наименование", "Производитель", "Тип"],
}


@dataclass
class PumpItem:
    item_id: int
    article: str
    name: str
    equipment_type: str
    params: dict[str, Any]
    q_h_points: list[tuple[float, float]]
    q_eta_points: list[tuple[float, float]]
    q_p_points: list[tuple[float, float]]
    q_npsh_points: list[tuple[float, float]]


class DataRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_sheet TEXT NOT NULL,
                article TEXT,
                name TEXT,
                equipment_type TEXT,
                raw_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                curve_type TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                FOREIGN KEY(item_id) REFERENCES items(id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_items_article_name ON items(article, name)")
        self.conn.commit()

    @staticmethod
    def _csv_url(sheet_id: str, gid: int) -> str:
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    def _fetch_gid_df(self, gid: int) -> pd.DataFrame:
        url = self._csv_url(GOOGLE_SHEET_ID, gid)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))

    def sync_google_sheets(self) -> None:
        payload = {"gids": []}
        for group, gids in (("installation", INSTALLATION_GIDS), ("pump", PUMP_GIDS)):
            for gid in gids:
                df = self._fetch_gid_df(gid)
                payload["gids"].append({"group": group, "gid": gid, "rows": len(df)})
                self._upsert_sheet(df, source=f"{group}:{gid}")

        SYNC_META_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _upsert_sheet(self, df: pd.DataFrame, source: str) -> None:
        article_col = next((c for c in df.columns if "артик" in str(c).lower()), None)
        name_col = next((c for c in df.columns if "наимен" in str(c).lower()), None)
        type_col = next((c for c in df.columns if "тип" in str(c).lower()), None)

        for _, row in df.fillna("").iterrows():
            raw = {k: str(v) for k, v in row.to_dict().items()}
            article = raw.get(article_col, "") if article_col else ""
            name = raw.get(name_col, "") if name_col else ""
            equipment_type = raw.get(type_col, "DEFAULT") if type_col else "DEFAULT"
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO items(source_sheet, article, name, equipment_type, raw_json) VALUES(?,?,?,?,?)",
                (source, article, name, equipment_type or "DEFAULT", json.dumps(raw, ensure_ascii=False)),
            )
            item_id = cur.lastrowid
            self._insert_default_curves(item_id)
        self.conn.commit()

    def _insert_default_curves(self, item_id: int) -> None:
        q = np.linspace(5, 50, 10)
        h = 40 - 0.5 * q
        eta = 20 + 1.2 * q - 0.015 * q**2
        p = 2 + 0.08 * q
        npsh = 1.5 + 0.03 * q
        curve_map = {"q_h": h, "q_eta": eta, "q_p": p, "q_npsh": npsh}
        cur = self.conn.cursor()
        for curve_type, y_arr in curve_map.items():
            for x, y in zip(q, y_arr):
                cur.execute(
                    "INSERT INTO curves(item_id, curve_type, x, y) VALUES(?,?,?,?)",
                    (item_id, curve_type, float(x), float(y)),
                )

    def search_items(self, text: str, limit: int = 30) -> list[sqlite3.Row]:
        q = f"%{text}%"
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, article, name, equipment_type
            FROM items
            WHERE article LIKE ? OR name LIKE ?
            ORDER BY name
            LIMIT ?
            """,
            (q, q, limit),
        )
        return cur.fetchall()

    def get_item(self, item_id: int) -> PumpItem:
        cur = self.conn.cursor()
        row = cur.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"Item {item_id} not found")

        def read_curve(curve_type: str) -> list[tuple[float, float]]:
            pts = cur.execute(
                "SELECT x,y FROM curves WHERE item_id=? AND curve_type=? ORDER BY x",
                (item_id, curve_type),
            ).fetchall()
            return [(float(p[0]), float(p[1])) for p in pts]

        return PumpItem(
            item_id=item_id,
            article=row["article"] or "",
            name=row["name"] or "",
            equipment_type=row["equipment_type"] or "DEFAULT",
            params=json.loads(row["raw_json"]),
            q_h_points=read_curve("q_h"),
            q_eta_points=read_curve("q_eta"),
            q_p_points=read_curve("q_p"),
            q_npsh_points=read_curve("q_npsh"),
        )


class PlotWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    @staticmethod
    def _polyline(points: list[tuple[float, float]], degree: int) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        z = np.polyfit(x, y, min(degree, max(1, len(points) - 1)))
        p = np.poly1d(z)
        x_smooth = np.linspace(x.min(), x.max(), 160)
        return x_smooth, p(x_smooth)

    def render(self, item: PumpItem, main_count: int, reserve_count: int, show_reserve: bool, q_work: float | None, h_work: float | None, h_static: float | None) -> None:
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 1, hspace=0.25)
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax2 = self.figure.add_subplot(gs[1, 0])

        ax1_r = ax1.twinx()
        ax2_r = ax2.twinx()

        total_lines = main_count + (reserve_count if show_reserve else 0)
        total_lines = max(1, total_lines)

        qh_x, qh_y = self._polyline(item.q_h_points, POLYNOMIAL_DEGREES["q_h"])
        qe_x, qe_y = self._polyline(item.q_eta_points, POLYNOMIAL_DEGREES["q_eta"])
        qp_x, qp_y = self._polyline(item.q_p_points, POLYNOMIAL_DEGREES["q_p"])
        qn_x, qn_y = self._polyline(item.q_npsh_points, POLYNOMIAL_DEGREES["q_npsh"])

        active_line = max(1, main_count)
        for i in range(1, total_lines + 1):
            lw = LINE_WIDTH_ACTIVE if i == active_line else LINE_WIDTH_SECONDARY
            q_factor = i
            ax1.plot(qh_x * q_factor, qh_y, color=COLORS["q_h"], linewidth=lw, alpha=0.9)
            ax1_r.plot(qe_x * q_factor, qe_y, color=COLORS["q_eta"], linewidth=lw, alpha=0.8)
            ax2.plot(qp_x * q_factor, qp_y, color=COLORS["q_p"], linewidth=lw, alpha=0.9)
            ax2_r.plot(qn_x * q_factor, qn_y, color=COLORS["q_npsh"], linewidth=lw, alpha=0.8)

        ax1.set_ylabel("H [м]")
        ax1.set_xlabel("Q [м³/ч]")
        ax1_r.set_ylabel("КПД [%]")
        ax2.set_ylabel("P [кВт]")
        ax2.set_xlabel("Q [м³/ч]")
        ax2_r.set_ylabel("NPSH [м]")

        ax1.grid(True, alpha=0.25)
        ax2.grid(True, alpha=0.25)

        # Реализация "половинной" правой оси
        ax1_r.set_ylim(0, max(1, np.max(qe_y)) * 2)
        ax2_r.set_ylim(0, max(1, np.max(qn_y)) * 2)

        # Кривая трубопровода и рабочая точка
        if q_work is not None and h_work is not None:
            k = (h_work - (h_static or 0.0)) / max(q_work**2, 1e-6)
            q_pipeline = np.linspace(0, max(qh_x) * active_line, 200)
            h_pipeline = (h_static or 0.0) + k * q_pipeline**2

            h_active = np.interp(q_pipeline, qh_x * active_line, qh_y, left=np.nan, right=np.nan)
            delta = h_pipeline - h_active
            idx = np.nanargmin(np.abs(delta))
            q_int, h_int = float(q_pipeline[idx]), float(h_pipeline[idx])

            ax1.plot(q_pipeline[: idx + 1], h_pipeline[: idx + 1], color=COLORS["pipeline"], linestyle="--", linewidth=2)
            ax1.scatter([q_int], [h_int], color=COLORS["intersection"], s=30, zorder=6)
            ax1.scatter([q_work], [h_work], color=COLORS["workpoint"], s=30, zorder=6)

        self.canvas.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, repo: DataRepository) -> None:
        super().__init__()
        self.repo = repo
        self.current_item_id: int | None = None

        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(*WINDOW_MIN_SIZE)

        root = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(root)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)

        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Поиск по наименованию или артикулу...")
        self.search_input.textChanged.connect(self.on_search_text)
        self.completer = QtWidgets.QCompleter([], self)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.completer.activated.connect(self.on_completer_pick)
        self.search_input.setCompleter(self.completer)
        grid.addWidget(self.search_input, 0, 0, 1, 2)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setElideMode(QtCore.Qt.ElideRight)

        self.plot_widget = PlotWidget()
        self.tabs.addTab(self.plot_widget, "График")

        draw_placeholder = QtWidgets.QLabel("Габаритный чертеж (заглушка для шаблона)")
        draw_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.tabs.addTab(draw_placeholder, "Чертеж")

        grid.addWidget(self.tabs, 1, 0)

        right_panel = self._build_right_panel()
        right_panel.setFixedWidth(PARAM_PANEL_WIDTH)
        grid.addWidget(right_panel, 1, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.generate_btn = QtWidgets.QPushButton("Сформировать тех. лист")
        self.generate_btn.clicked.connect(self.on_generate_sheet)
        btn_row.addWidget(self.generate_btn)
        grid.addLayout(btn_row, 2, 0, 1, 2)

        self.setCentralWidget(root)

    def _build_right_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)

        top = QtWidgets.QGroupBox("Рабочая точка")
        top_l = QtWidgets.QFormLayout(top)
        self.q_input = QtWidgets.QLineEdit()
        self.h_input = QtWidgets.QLineEdit()
        self.hs_input = QtWidgets.QLineEdit()
        for w in (self.q_input, self.h_input, self.hs_input):
            w.textChanged.connect(self.refresh_plot)
        top_l.addRow("Q [м3/ч]", self.q_input)
        top_l.addRow("H [м]", self.h_input)
        top_l.addRow("Hст [м]", self.hs_input)
        lay.addWidget(top)

        parallel = QtWidgets.QGroupBox("Параллельная работа")
        par_l = QtWidgets.QGridLayout(parallel)
        self.main_combo = QtWidgets.QComboBox()
        self.main_combo.addItems([str(i) for i in range(1, 9)])
        self.reserve_combo = QtWidgets.QComboBox()
        self.reserve_combo.addItems([str(i) for i in range(0, 9)])
        self.show_reserve = QtWidgets.QCheckBox("Показывать резервные")
        for w in (self.main_combo, self.reserve_combo, self.show_reserve):
            if isinstance(w, QtWidgets.QComboBox):
                w.currentTextChanged.connect(self.refresh_plot)
            else:
                w.toggled.connect(self.refresh_plot)
        par_l.addWidget(QtWidgets.QLabel("Основные"), 0, 0)
        par_l.addWidget(self.main_combo, 1, 0)
        par_l.addWidget(QtWidgets.QLabel("Резервные"), 0, 1)
        par_l.addWidget(self.reserve_combo, 1, 1)
        par_l.addWidget(self.show_reserve, 2, 1)
        lay.addWidget(parallel)

        lay.addWidget(QtWidgets.QLabel("Параметры"))
        self.params_scroll = QtWidgets.QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_widget = QtWidgets.QWidget()
        self.params_form = QtWidgets.QFormLayout(self.params_widget)
        self.params_scroll.setWidget(self.params_widget)
        lay.addWidget(self.params_scroll, 1)

        return box

    def on_search_text(self, text: str) -> None:
        if len(text.strip()) < SEARCH_MIN_CHARS:
            self.completer.model().setStringList([])
            return
        rows = self.repo.search_items(text.strip())
        labels = [f"{r['id']} | {r['article']} | {r['name']}" for r in rows]
        model = QtCore.QStringListModel(labels)
        self.completer.setModel(model)
        self.completer.complete()

    def on_completer_pick(self, value: str) -> None:
        item_id = int(value.split("|", 1)[0].strip())
        self.current_item_id = item_id
        self.load_item(item_id)

    @staticmethod
    def _float_or_none(text: str) -> float | None:
        text = text.strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def load_item(self, item_id: int) -> None:
        item = self.repo.get_item(item_id)

        while self.params_form.rowCount() > 0:
            self.params_form.removeRow(0)

        keys = VISIBLE_PARAMS_BY_TYPE.get(item.equipment_type, VISIBLE_PARAMS_BY_TYPE["DEFAULT"])
        for key in keys:
            self.params_form.addRow(f"{key}:", QtWidgets.QLabel(str(item.params.get(key, "—"))))

        self.refresh_plot()

    def refresh_plot(self) -> None:
        if self.current_item_id is None:
            return
        item = self.repo.get_item(self.current_item_id)
        self.plot_widget.render(
            item=item,
            main_count=int(self.main_combo.currentText() or 1),
            reserve_count=int(self.reserve_combo.currentText() or 0),
            show_reserve=self.show_reserve.isChecked(),
            q_work=self._float_or_none(self.q_input.text()),
            h_work=self._float_or_none(self.h_input.text()),
            h_static=self._float_or_none(self.hs_input.text()),
        )

    def on_generate_sheet(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Тех. лист",
            "Генерация DOCX/PDF будет подключена следующим шагом (шаблоны и метки).",
        )


def apply_tile_style(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QWidget { font-family: 'Segoe UI'; font-size: 10.5pt; }
        QMainWindow, QWidget { background-color: #f4f4f4; color: #161616; }
        QLineEdit, QComboBox, QGroupBox, QScrollArea {
            background: white;
            border: 1px solid #c6c6c6;
            border-radius: 6px;
            padding: 4px;
        }
        QPushButton {
            background: #0f62fe;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 14px;
            font-weight: 600;
        }
        QPushButton:hover { background: #0353e9; }
        QTabBar::tab {
            background: #e8e8e8;
            border: 1px solid #d0d0d0;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: white;
            border-bottom-color: white;
        }
        """
    )


def main() -> int:
    repo = DataRepository(DB_PATH)
    try:
        repo.sync_google_sheets()
    except Exception as exc:  # noqa: BLE001 - UX предупреждение
        print(f"[WARN] Не удалось синхронизировать Google Sheets: {exc}")

    app = QtWidgets.QApplication(sys.argv)
    apply_tile_style(app)
    w = MainWindow(repo)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
