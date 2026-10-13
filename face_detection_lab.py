"""人臉偵測實驗工具：比較三組 Haar cascade 參數並保存結果。"""

from __future__ import annotations

import csv
import datetime as dt
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "實驗結果"
CSV_PATH = OUTPUT_DIR / "實驗紀錄.csv"

PRESETS = [
    {"name": "A_敏感", "scaleFactor": 1.03, "minNeighbors": 2, "minSizeRatio": 0.06},
    {"name": "B_平衡", "scaleFactor": 1.08, "minNeighbors": 3, "minSizeRatio": 0.08},
    {"name": "C_保守", "scaleFactor": 1.12, "minNeighbors": 4, "minSizeRatio": 0.10},
]

CSV_FIELDS = [
    "時間", "影像", "參數組", "scaleFactor", "minNeighbors", "minSize",
    "偵測框數", "真實人數", "TP", "FP", "FN", "耗時_ms", "光線", "姿勢",
    "遮擋", "隱私觀察", "輸出圖片", "測試情境",
]


class FaceDetectionLab:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("練習 8｜人臉偵測實驗報告")
        self.root.geometry("1240x880")
        self.root.minsize(1000, 720)
        self.image_path: Path | None = None
        self.frame_bgr = None
        self.preview_photo = None
        self.results: list[dict] = []
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        # Read the model through Python so OpenCV does not need to open a CJK Windows path.
        cascade_xml = cascade_path.read_text(encoding="utf-8")
        self._cascade_storage = cv2.FileStorage(
            cascade_xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY
        )
        self.detector = cv2.CascadeClassifier()
        self.detector.read(self._cascade_storage.getFirstTopLevelNode())
        if self.detector.empty():
            messagebox.showerror("模型載入失敗", f"找不到 OpenCV 人臉模型：\n{cascade_path}")
            root.destroy()
            return
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 18, "bold"), foreground="#b42332")
        style.configure("Section.TLabelframe.Label", font=("Microsoft JhengHei UI", 10, "bold"))
        style.configure("Treeview.Heading", font=("Microsoft JhengHei UI", 9, "bold"), padding=(3, 8))

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="練習 8｜人臉偵測實驗報告", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(
            outer,
            text="單張影像／攝影機：偵測、畫框、計數，並比較三組參數。此工具只偵測人臉位置，不辨識個人身分。",
            wraplength=970,
        ).pack(anchor="w", pady=(0, 12))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 10))
        ttk.Button(controls, text="選擇圖片…", command=self.choose_image).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="攝影機擷取一張", command=self.capture_camera).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="比較三組參數", command=self.run_comparison).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="開啟結果資料夾", command=self.open_output_folder).pack(side="right")
        self.source_label = ttk.Label(outer, text="尚未選擇影像", foreground="#666")
        self.source_label.pack(anchor="w", pady=(0, 8))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)

        preview_box = ttk.LabelFrame(left, text="影像預覽／偵測框", style="Section.TLabelframe", padding=8)
        preview_box.pack(fill="both", expand=True)
        self.preview = ttk.Label(preview_box, text="請選擇圖片或擷取攝影機畫面", anchor="center")
        self.preview.pack(fill="both", expand=True)

        notes = ttk.LabelFrame(right, text="實驗條件與觀察", style="Section.TLabelframe", padding=10)
        notes.pack(fill="x", pady=(0, 10))
        self.gt_var = tk.StringVar(value="")
        row = ttk.Frame(notes)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="真實人數（計算指標）").pack(side="left")
        ttk.Entry(row, textvariable=self.gt_var, width=8).pack(side="right")
        ttk.Label(notes, text="測試情境").pack(anchor="w", pady=(6, 0))
        self.scenario_var = tk.StringVar(value="正面")
        self.scenario_combo = ttk.Combobox(
            notes,
            textvariable=self.scenario_var,
            values=["正面", "側臉", "背光", "遮擋"],
            state="readonly",
        )
        self.scenario_combo.pack(fill="x")
        self.scenario_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._set_scenario_conditions(self.scenario_var.get()),
        )
        ttk.Label(
            notes,
            text="要比較情境差異，請使用不同的實拍照片；選單只分類結果，不會改變照片。",
            foreground="#666",
            wraplength=540,
        ).pack(anchor="w", pady=(3, 0))
        ttk.Label(notes, text="光線").pack(anchor="w", pady=(6, 0))
        self.light_var = tk.StringVar(value="正常")
        ttk.Combobox(notes, textvariable=self.light_var, values=["正常", "偏暗", "過亮", "背光", "不均勻"], state="readonly").pack(fill="x")
        ttk.Label(notes, text="姿勢").pack(anchor="w", pady=(6, 0))
        self.pose_var = tk.StringVar(value="正面")
        ttk.Combobox(notes, textvariable=self.pose_var, values=["正面", "側臉", "低頭／抬頭", "多人姿勢不同", "其他"], state="readonly").pack(fill="x")
        ttk.Label(notes, text="遮擋").pack(anchor="w", pady=(6, 0))
        self.occlusion_var = tk.StringVar(value="無")
        ttk.Combobox(notes, textvariable=self.occlusion_var, values=["無", "眼鏡", "口罩", "手／物品", "部分遮擋", "其他"], state="readonly").pack(fill="x")
        ttk.Label(notes, text="隱私觀察（例如：已取得同意／使用非辨識影像）").pack(anchor="w", pady=(6, 0))
        self.privacy_var = tk.StringVar(value="")
        ttk.Entry(notes, textvariable=self.privacy_var).pack(fill="x")

        result_box = ttk.LabelFrame(right, text="三組參數比較", style="Section.TLabelframe", padding=6)
        result_box.pack(fill="both", expand=True)
        self.count_summary = ttk.Label(
            result_box,
            text="比較完成後，這裡會顯示各組偵測到的人臉數。",
            font=("Microsoft JhengHei UI", 11, "bold"),
            foreground="#176b3a",
            wraplength=620,
            justify="left",
        )
        self.count_summary.pack(anchor="w", padx=4, pady=(2, 8))
        columns = ("scenario", "scale", "neighbors", "minsize", "actual", "tp", "fp", "fn", "ms")
        self.table = ttk.Treeview(result_box, columns=columns, show="headings", height=12)
        labels = {
            "scenario": "測試情境", "scale": "scale\nFactor", "neighbors": "min\nNeighbors",
            "minsize": "minSize", "actual": "實際\n人臉數", "tp": "TP", "fp": "FP",
            "fn": "FN", "ms": "耗時\n(ms)",
        }
        widths = {
            "scenario": 70, "scale": 70, "neighbors": 76, "minsize": 68, "actual": 74,
            "tp": 46, "fp": 46, "fn": 46, "ms": 72,
        }
        for col in columns:
            self.table.heading(col, text=labels[col])
            self.table.column(col, width=widths[col], minwidth=widths[col], anchor="center", stretch=True)
        self.table.pack(fill="both", expand=True)
        self.table.tag_configure("stripe", background="#eeeeee")
        ttk.Label(
            result_box,
            text="分析｜Precision = TP / (TP + FP)；Recall = TP / (TP + FN)",
            foreground="#555",
        ).pack(anchor="w", padx=4, pady=(8, 0))
        self.metrics_summary = ttk.Label(result_box, text="本次 Precision／Recall 尚未計算。", foreground="#555")
        self.metrics_summary.pack(anchor="w", padx=4, pady=(3, 0))
        ttk.Label(
            result_box,
            text="注意｜分母為 0 時標示「無法計算」",
            foreground="#b42332",
        ).pack(anchor="w", padx=4, pady=(3, 2))
        self.status = ttk.Label(outer, text="結果會存入「實驗結果」資料夾；每列 TP／FP／FN 依人數差計算。", foreground="#555")
        self.status.pack(anchor="w", pady=(10, 0))

    def choose_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="選擇一張影像",
            filetypes=[("影像檔", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有檔案", "*.*")],
        )
        if not filename:
            return
        image = self._read_image(Path(filename))
        if image is None:
            messagebox.showerror("讀取失敗", "無法讀取這張影像，請確認檔案格式。")
            return
        self._set_image(image, Path(filename))

    def capture_camera(self) -> None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            messagebox.showerror("攝影機無法開啟", "請確認攝影機已連接，且沒有被其他程式占用。")
            return
        ok, frame = camera.read()
        camera.release()
        if not ok or frame is None:
            messagebox.showerror("擷取失敗", "攝影機沒有回傳影像。")
            return
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._set_image(frame, Path(f"攝影機擷取_{stamp}.jpg"))

    def _set_scenario_conditions(self, scenario: str) -> None:
        conditions = {
            "正面": ("正常", "正面", "無"),
            "背光": ("背光", "正面", "無"),
            "側臉": ("正常", "側臉", "無"),
            "遮擋": ("正常", "正面", "部分遮擋"),
        }
        light, pose, occlusion = conditions.get(scenario, conditions["正面"])
        self.light_var.set(light)
        self.pose_var.set(pose)
        self.occlusion_var.set(occlusion)

    def _set_image(self, image, path: Path) -> None:
        self.frame_bgr = image.copy()
        self.image_path = path
        self.results = []
        self.source_label.configure(text=f"影像：{path.name}　（{image.shape[1]} × {image.shape[0]}）")
        self._show_preview(image)
        self.count_summary.configure(text="選擇測試情境並比較後，這裡會顯示三組偵測人數。")
        self.metrics_summary.configure(text="本次 Precision／Recall 尚未計算。")
        self.status.configure(text="影像已載入。填入真實人數與觀察條件，選擇情境後比較三組參數。")

    def _show_preview(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((630, 520), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(pil_image)
        self.preview.configure(image=self.preview_photo, text="")

    def run_comparison(self) -> None:
        if self.frame_bgr is None or self.image_path is None:
            messagebox.showinfo("需要影像", "請先選擇一張圖片，或擷取一張攝影機畫面。")
            return
        try:
            ground_truth = int(self.gt_var.get().strip())
            if ground_truth < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("人數格式錯誤", "請輸入大於或等於 0 的整數真實人數。")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gray = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_dimension = min(gray.shape[:2])
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.image_path.stem
        scenario = self.scenario_var.get()
        observation_parts = [self.light_var.get(), self.pose_var.get(), f"遮擋{self.occlusion_var.get()}"]
        if self.privacy_var.get().strip():
            observation_parts.append(f"隱私：{self.privacy_var.get().strip()}")
        observation = "／".join(observation_parts)
        rows = []
        # Replace this scenario's prior rows while preserving the other scenarios.
        for preset in PRESETS:
            item_id = f"{scenario}::{preset['name']}"
            if self.table.exists(item_id):
                self.table.delete(item_id)
        for preset in PRESETS:
            min_side = max(16, round(min_dimension * preset["minSizeRatio"]))
            min_size = (min_side, min_side)
            started = time.perf_counter()
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=preset["scaleFactor"],
                minNeighbors=preset["minNeighbors"],
                minSize=min_size,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            annotated = self.frame_bgr.copy()
            for x, y, width, height in faces:
                cv2.rectangle(annotated, (x, y), (x + width, y + height), (30, 190, 70), 2)
            count = len(faces)
            # 以人數差估計計數 TP/FP/FN；非逐框 IoU 配對指標。
            tp = min(count, ground_truth)
            fp = max(count - ground_truth, 0)
            fn = max(ground_truth - count, 0)
            output_name = f"{stamp}_{base}_{preset['name']}.jpg"
            output_path = OUTPUT_DIR / output_name
            self._write_image(output_path, annotated)
            row = {
                "時間": dt.datetime.now().isoformat(timespec="seconds"),
                "影像": self.image_path.name,
                "參數組": preset["name"],
                "scaleFactor": preset["scaleFactor"],
                "minNeighbors": preset["minNeighbors"],
                "minSize": f"{min_size[0]}x{min_size[1]}",
                "偵測框數": count,
                "真實人數": ground_truth,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "耗時_ms": round(elapsed_ms, 2),
                "光線": self.light_var.get(),
                "姿勢": self.pose_var.get(),
                "遮擋": self.occlusion_var.get(),
                "隱私觀察": self.privacy_var.get().strip(),
                "輸出圖片": output_name,
                "測試情境": scenario,
            }
            rows.append(row)
            scenario_index = ["正面", "側臉", "背光", "遮擋"].index(scenario)
            tag = ("stripe",) if scenario_index % 2 else ()
            self.table.insert(
                "",
                "end",
                iid=f"{scenario}::{preset['name']}",
                values=(
                    scenario if preset is PRESETS[0] else "",
                    f"{row['scaleFactor']:.2f}",
                    row["minNeighbors"],
                    row["minSize"],
                    ground_truth,
                    tp,
                    fp,
                    fn,
                    f"{elapsed_ms:.1f}",
                ),
                tags=tag,
            )
        self._append_csv(rows)
        self.results = rows
        self.count_summary.configure(
            text=(
                "偵測到的人臉數："
                f"A 敏感 {rows[0]['偵測框數']} 人　｜　"
                f"B 平衡 {rows[1]['偵測框數']} 人　｜　"
                f"C 保守 {rows[2]['偵測框數']} 人"
            )
        )
        def metric_text(numerator: int, denominator: int) -> str:
            return "無法計算" if denominator == 0 else f"{numerator / denominator:.2f}"

        metrics = []
        for row in rows:
            precision = metric_text(row["TP"], row["TP"] + row["FP"])
            recall = metric_text(row["TP"], row["TP"] + row["FN"])
            metrics.append(f"{row['參數組']}: P={precision}／R={recall}")
        self.metrics_summary.configure(text="Precision／Recall｜" + "　｜　".join(metrics))
        self._show_preview(self._read_image(OUTPUT_DIR / rows[1]["輸出圖片"]))
        self.status.configure(text=f"完成：已更新「{scenario}」情境，並保存 3 張標註影像與實驗紀錄。")

    @staticmethod
    def _append_csv(rows: list[dict]) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        previous_rows = []
        migrate = False
        if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
            with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as existing_file:
                reader = csv.DictReader(existing_file)
                migrate = reader.fieldnames != CSV_FIELDS
                if migrate:
                    previous_rows = list(reader)
        mode = "w" if migrate else "a"
        with CSV_PATH.open(mode, newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            if migrate or CSV_PATH.stat().st_size == 0:
                writer.writeheader()
            if previous_rows:
                writer.writerows(previous_rows)
            writer.writerows(rows)

    @staticmethod
    def _read_image(path: Path):
        """Read image paths containing non-ASCII characters on Windows."""
        encoded = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    @staticmethod
    def _write_image(path: Path, image) -> None:
        """Write image paths containing non-ASCII characters on Windows."""
        suffix = path.suffix or ".jpg"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            raise OSError(f"無法編碼輸出圖片：{path.name}")
        encoded.tofile(path)

    def open_output_folder(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.startfile(str(OUTPUT_DIR))
        except OSError as exc:
            messagebox.showerror("無法開啟資料夾", str(exc))


def main() -> None:
    root = tk.Tk()
    FaceDetectionLab(root)
    root.mainloop()


if __name__ == "__main__":
    main()
