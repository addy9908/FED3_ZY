import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# Object-Oriented Matplotlib API to prevent GUI conflicts
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import signal
from scipy.signal import find_peaks
# from scipy.stats import zscore

# =============================================================================
# CENTRAL DATA MANAGER
# =============================================================================
class DataManager:
    def __init__(self, logger=None, on_df_update=None):
        self.logger = logger
        self.on_df_update = on_df_update
        self.reset()
        
    def log(self, msg, level="INFO"):
        if self.logger:
            self.logger(msg, level)
        else:
            print(f"[{level}] {msg}")

    def reset(self):
        self.raw_fp_files = []
        self.raw_cam_files = []
        self.raw_fed_l_files = []
        self.raw_fed_r_files = []
        
        self._df_FP = pd.DataFrame()
        self.df_cam = pd.DataFrame()
        self.df_fed_L = pd.DataFrame()
        self.df_fed_R = pd.DataFrame()
        
        self.df_wheel_L_indices = pd.DataFrame()
        self.df_wheel_R_indices = pd.DataFrame()
        self.df_fed_L_indices = pd.DataFrame()
        self.df_fed_R_indices = pd.DataFrame()
        
        self.mappings = []
        self.aligned_data = {}  
        self.time_col = 'Time (s)'
        self.wheel_interval = 5.0  # NEW: Store interval for Panel 7 to use later
        
        # --- NEW TRACKING VARIABLES ---
        self.fp_base_name = "Session"
        self.output_dir = ""
        self.loaded_csv_path = ""

    @property
    def df_FP(self): 
        return self._df_FP

    def set_df_fp(self, df):
        """Setter for df_FP that automatically triggers the Global Table update."""
        self._df_FP = df
        if self.on_df_update: 
            self.on_df_update()

    @staticmethod
    def safe_read_csv(filepath, nrows=None):
        """Fast C-Engine reader with smart fallback for semicolons."""
        try:
            df = pd.read_csv(filepath, sep=',', on_bad_lines='skip', nrows=nrows) 
            if df.shape[1] <= 1:
                df = pd.read_csv(filepath, sep=';', on_bad_lines='skip', nrows=nrows)
            df = df.dropna(subset=[df.columns[0]]).dropna(axis=1, how='all')
            return df.reset_index(drop=True)
        except Exception:
            return None

    def enforce_monotonic(self, df, col_indices):
        if df is None or df.empty: return df
        if isinstance(col_indices, int): col_indices = [col_indices]
        df = df.copy()
        for c in col_indices:
            if 0 <= c < df.shape[1]:
                col_name = df.columns[c]
                s = df[col_name]
                diffs = s.diff().fillna(0)
                diffs.iloc[0] = s.iloc[0]
                df[col_name] = pd.Series(np.where(diffs < 0, s, diffs)).cumsum()
        return df

    def get_wheel_indices(self, df, col_name, interval, label):
        """Now uses col_name instead of index."""
        if df.empty or col_name not in df.columns: return pd.DataFrame()
        t_col, dio_col = self.time_col, col_name
        df_copy = df.copy()
        events = df_copy[(df_copy[dio_col].diff().fillna(0) > 0)]
        if events.empty: return pd.DataFrame(columns=[t_col, label])
        times = events[t_col].values
        
        prev_diff = np.concatenate([[np.nan], times[1:] - times[:-1]])
        next_diff = np.concatenate([times[1:] - times[:-1], [np.nan]])
        t_vals, d_vals = [], []
        for i, t in enumerate(times):
            if i == 0:
                if len(times) > 1 and next_diff[i] < interval: 
                    t_vals.append(t); d_vals.append(1)
            elif i == len(times)-1:
                if prev_diff[i] < interval: 
                    t_vals.append(t); d_vals.append(0)
            else:
                cp, cn = prev_diff[i] > interval, next_diff[i] > interval
                if cp != cn: 
                    t_vals.append(t); d_vals.append(1 if cp else 0)
        return pd.DataFrame({t_col: t_vals, label: d_vals})

    def get_fed_indices(self, df, col_name, label):
        """Now uses col_name instead of index."""
        if df.empty or col_name not in df.columns: return pd.DataFrame()
        t_col, dio_col = self.time_col, col_name
        mask = df[dio_col].diff().fillna(0) > 0
        return pd.DataFrame({t_col: df.loc[mask, t_col] - 0.200, label: 1}).reset_index(drop=True)

# =============================================================================
# PANELS 1-3 (Browser, FP, CAM)
# =============================================================================
class Panel1_Browser(ttk.Frame):
    def __init__(self, parent, data_mgr):
        super().__init__(parent); self.data_mgr = data_mgr
        ttk.Label(self, text="Step 1: Load Raw Files", font=("Helvetica", 12, "bold")).pack(pady=5)
        lf = ttk.Frame(self); lf.pack(fill="x", pady=5)
        self.make_list(lf, "FP Files", self.data_mgr.raw_fp_files, 0)
        self.make_list(lf, "CAM Files", self.data_mgr.raw_cam_files, 1)
        self.make_list(lf, "FED L", self.data_mgr.raw_fed_l_files, 2)
        self.make_list(lf, "FED R", self.data_mgr.raw_fed_r_files, 3)
        
        ttk.Label(self, text="Preview (First 10 rows):").pack(pady=5)
        tf = ttk.Frame(self); tf.pack(fill="both", expand=True, padx=10, pady=5)
        sy = ttk.Scrollbar(tf, orient="vertical"); sx = ttk.Scrollbar(tf, orient="horizontal")
        self.tree = ttk.Treeview(tf, height=10, yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.config(command=self.tree.yview); sx.config(command=self.tree.xview)
        sy.pack(side="right", fill="y"); sx.pack(side="bottom", fill="x"); self.tree.pack(side="left", fill="both", expand=True)

    def make_list(self, p, title, lst, col):
        f = ttk.Frame(p); f.grid(row=0, column=col, padx=5, sticky="n")
        ttk.Label(f, text=title).pack(); lb = tk.Listbox(f, height=10, width=35); lb.pack()
        
        def add():
            for file in filedialog.askopenfilenames(filetypes=[("CSV", "*.csv")]):
                lst.append(Path(file))
                lb.insert(tk.END, Path(file).name)
        def preview(evt):
            sel = lb.curselection()
            if sel:
                df = self.data_mgr.safe_read_csv(lst[sel[0]], nrows=10)
                if df is not None:
                    self.tree.delete(*self.tree.get_children())
                    self.tree["columns"] = list(df.columns)
                    self.tree["show"] = "headings"
                    for c in df.columns: 
                        self.tree.heading(c, text=c)
                        self.tree.column(c, width=80)
                    for _, r in df.iterrows(): 
                        self.tree.insert("", "end", values=list(r))
        lb.bind("<<ListboxSelect>>", preview)
        ttk.Button(f, text="+ Add", command=add).pack(pady=2)
        return lb

class Panel2_FPProcess(ttk.Frame):
    def __init__(self, parent, data_mgr):
        super().__init__(parent); self.data_mgr = data_mgr
        self._df_fp_raw, self.df_sig, self.df_ref_cal = None, None, None
        self._cached_files = None
        ttk.Label(self, text="Step 2: FP Processing", font=("Helvetica", 12, "bold")).pack(pady=5)
        f = ttk.Frame(self); f.pack()
        self.e470 = self.mk_ent(f, "470 Flag:", "2", 0)
        self.e415 = self.mk_ent(f, "415 Flag:", "1", 2)
        self.efilt = self.mk_ent(f, "Filtfilt Window:", "4", 4)
        self.epoly = self.mk_ent(f, "Polyfit Power:", "2", 6)
        
        bf = ttk.Frame(self); bf.pack(pady=5)
        ttk.Button(bf, text="Preview Raw", command=self.preview).pack(side=tk.LEFT, padx=10)
        ttk.Button(bf, text="Process dF/F", command=self.process).pack(side=tk.LEFT, padx=10)
        
        self.fig = Figure(figsize=(6, 6))
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self).pack(fill="x")

    def mk_ent(self, p, t, v, c):
        ttk.Label(p, text=t).grid(row=0, column=c)
        e = ttk.Entry(p, width=4); e.insert(0, v); e.grid(row=0, column=c+1, padx=10)
        return e

    def prepare(self):
        current_files = tuple(self.data_mgr.raw_fp_files)
        if not current_files: return False
        if self._cached_files != current_files:
            self.data_mgr.log("Loading & Fixing Rollovers...", "INFO")
            
            # --- NEW: Set output directory and base name from raw FP file ---
            first_file = Path(current_files[0])
            self.data_mgr.fp_base_name = first_file.stem
            self.data_mgr.output_dir = str(first_file.parent / "Output")
            self.data_mgr.loaded_csv_path = "" # Reset this because we are processing raw
            # ----------------------------------------------------------------

            dfs = [self.data_mgr.safe_read_csv(f) for f in current_files if self.data_mgr.safe_read_csv(f) is not None]
            if not dfs: return False
            df = pd.concat(dfs, ignore_index=True)
            cc, sc = 'ComputerTimestamp', 'SystemTimestamp'
            if cc in df.columns and sc in df.columns:
                dec = df[cc].diff().fillna(0) < 0
                if dec.any(): 
                    sys_diff = df[sc].diff() * 1000
                    expected = df[cc].shift(1) + sys_diff
                    df[cc] = df[cc] + np.cumsum(np.where(dec, expected - df[cc], 0))
            
            df[self.data_mgr.time_col] = df[cc] / 1000.0
            self._df_fp_raw = df; self._cached_files = current_files
            
            # Monotonic Verification
            is_mono = df[self.data_mgr.time_col].is_monotonic_increasing
            if is_mono:
                self.data_mgr.log("Time (s) column verified as monotonically increasing.", "INFO")
            else:
                self.data_mgr.log("WARNING: Time (s) column is NOT monotonically increasing!", "WARN")

        df = self._df_fp_raw; sigs = [c for c in df.columns if c.startswith("G")]
        dsig = df[df['LedState'] == int(self.e470.get())].reset_index(drop=True)
        dref = df[df['LedState'] == int(self.e415.get())].reset_index(drop=True)
        dref_cal = dref[sigs].rolling(2, min_periods=1).mean()
        ml = min(len(dsig), len(dref_cal))
        self.df_sig, self.df_ref_cal = dsig.iloc[:ml], dref_cal.iloc[:ml]
        return True

    def preview(self):
        if not self.prepare(): return
        self.ax1.clear()
        ds = max(1, len(self.df_sig)//10000)
        t = self.df_sig[self.data_mgr.time_col].iloc[::ds]
        for c in [c for c in self.df_sig.columns if c.startswith("G")]:
            self.ax1.plot(t, self.df_sig[c].iloc[::ds], label=f"{c}(470)")
            self.ax1.plot(t, self.df_ref_cal[c].iloc[::ds], label=f"{c}(415)", alpha=0.5, ls='--')
        self.ax1.legend(fontsize='x-small'); self.ax1.set_title("Raw"); self.canvas.draw()

    def process(self):
        if self.df_sig is None and not self.prepare(): return
        w, p = int(self.efilt.get()), int(self.epoly.get())
        dff = self.df_sig[[self.data_mgr.time_col]].copy()
        self.ax2.clear()
        ds = max(1, len(self.df_sig)//10000)
        for c in [c for c in self.df_sig.columns if c.startswith("G")]:
            sf = signal.filtfilt(np.ones(w)/w, 1, self.df_sig[c]) if w>1 else self.df_sig[c]
            rf = signal.filtfilt(np.ones(w)/w, 1, self.df_ref_cal[c]) if w>1 else self.df_ref_cal[c]
            fit = np.polyval(np.polyfit(rf, sf, p), rf)
            dff[c] = (sf - fit) / fit
            self.ax2.plot(dff[self.data_mgr.time_col].iloc[::ds], dff[c].iloc[::ds], label=f"{c} dF/F")
        
        self.data_mgr.set_df_fp(dff)
        self.ax2.legend(fontsize='x-small'); self.ax2.set_title("dF/F"); self.ax2.set_xlabel(self.data_mgr.time_col)
        self.fig.tight_layout(); self.canvas.draw()

class Panel3_CamProcess(ttk.Frame):
    def __init__(self, parent, data_mgr):
        super().__init__(parent); self.data_mgr = data_mgr
        ttk.Label(self, text="Step 3: CAM Wheel", font=("Helvetica", 12, "bold")).pack()
        f = ttk.Frame(self); f.pack()
        ttk.Label(f, text="Wheel_L Col:").grid(row=0,column=0); self.el = ttk.Entry(f, width=4); self.el.insert(0,"8"); self.el.grid(row=0,column=1, padx=10)
        ttk.Label(f, text="Wheel_R Col:").grid(row=0,column=2); self.er = ttk.Entry(f, width=4); self.er.insert(0,"9"); self.er.grid(row=0,column=3, padx=10)
        ttk.Label(f, text="minimal Interval(s):").grid(row=0,column=4); self.ei = ttk.Entry(f, width=4); self.ei.insert(0,"5"); self.ei.grid(row=0,column=5, padx=10)
        
        btn_frame = ttk.Frame(self); btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Process Data", command=self.proc).pack(side=tk.LEFT, padx=5)
        ttk.Label(btn_frame, text="X-Axis Unit:").pack(side=tk.LEFT, padx=5)
        self.x_axis_cb = ttk.Combobox(btn_frame, values=["sec", "min", "hour"], width=8)
        self.x_axis_cb.set("hour"); self.x_axis_cb.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Plot", command=self.plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Merge Raw Data into df_FP", command=self.merge).pack(side=tk.LEFT, padx=5)
        
        self.fig = Figure(figsize=(6, 4)); self.ax1 = self.fig.add_subplot(211); self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self).pack(fill="x")

    def proc(self):
        dfs = [self.data_mgr.safe_read_csv(f) for f in self.data_mgr.raw_cam_files if self.data_mgr.safe_read_csv(f) is not None]
        if not dfs: return
        df = pd.concat(dfs, ignore_index=True)
        fc = df.columns[0]; df[fc] = pd.to_datetime(df[fc], format='mixed')
        df = df.sort_values(by=fc).reset_index(drop=True)
        df[self.data_mgr.time_col] = (df.iloc[0,1]/1000.0) + (df[fc] - df.at[0,fc]).dt.total_seconds()
        
        cl, cr = int(self.el.get()), int(self.er.get())
        df = self.data_mgr.enforce_monotonic(df, [cl, cr])
        self.data_mgr.df_cam = df
        
        # Save interval globally for Panel 7 to use later
        self.data_mgr.wheel_interval = float(self.ei.get())
        
        self.data_mgr.df_wheel_L_indices = self.data_mgr.get_wheel_indices(df, df.columns[cl], self.data_mgr.wheel_interval, 'DIO_wheel_L')
        self.data_mgr.df_wheel_R_indices = self.data_mgr.get_wheel_indices(df, df.columns[cr], self.data_mgr.wheel_interval, 'DIO_wheel_R')
        self.data_mgr.log("CAM data processed successfully.", "INFO")
        self.plot()

    def plot(self):
        df = self.data_mgr.df_cam
        if df is None or df.empty: return
        cl, cr = int(self.el.get()), int(self.er.get())
        factor = {"sec": 1.0, "min": 1/60.0, "hour": 1/3600.0}[self.x_axis_cb.get()]
        x_scaled = df[self.data_mgr.time_col] * factor

        for ax, d_idx, col, t in zip([self.ax1, self.ax2], [self.data_mgr.df_wheel_L_indices, self.data_mgr.df_wheel_R_indices], [cl, cr], ['L', 'R']):
            ax.clear(); ax.step(x_scaled, df.iloc[:, col], color='gray')
            ax.set_title(f"Wheel {t}"); ax.set_xlabel(f"Time ({self.x_axis_cb.get()})")
            if not d_idx.empty:
                for tm, v in d_idx.itertuples(index=False): 
                    ax.axvline(tm * factor, color='b' if v==1 else 'r', alpha=0.5)
        self.fig.tight_layout(); self.canvas.draw()

    def merge(self):
        if self.data_mgr.df_FP.empty: return
        df = self.data_mgr.df_FP.copy()
        df_cam = self.data_mgr.df_cam
        cl, cr = int(self.el.get()), int(self.er.get())
        merged = False
        
        # Merge the RAW monotonic columns directly
        for col_idx, dio_name in zip([cl, cr], ['DIO_wheel_L', 'DIO_wheel_R']):
            if dio_name in df.columns:
                if not messagebox.askyesno("Overwrite?", f"Column '{dio_name}' exists. Overwrite?"):
                    continue
                df = df.drop(columns=[dio_name])
                
            raw_col = df_cam.columns[col_idx]
            d_sub = df_cam[[self.data_mgr.time_col, raw_col]].rename(columns={raw_col: dio_name})
            
            df = pd.merge_asof(df.sort_values(self.data_mgr.time_col), 
                               d_sub.sort_values(self.data_mgr.time_col), 
                               on=self.data_mgr.time_col, direction='backward')
            # Forward fill the step counts, starting with 0
            df[dio_name] = df[dio_name].ffill().fillna(0).astype(int)
            merged = True
            
        if merged:
            self.data_mgr.set_df_fp(df)
            self.data_mgr.log("Wheel raw data merged into df_FP.", "INFO")


class Panel4_FedProcess(ttk.Frame):
    def __init__(self, parent, data_mgr):
        super().__init__(parent); self.data_mgr = data_mgr
        ttk.Label(self, text="Step 4: FED Pellet", font=("Helvetica", 12, "bold")).pack()
        f = ttk.Frame(self); f.pack()
        ttk.Label(f, text="Pellet Col:").pack(side=tk.LEFT); self.ef = ttk.Entry(f, width=5); self.ef.insert(0,"3"); self.ef.pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(self); btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Process Data", command=self.proc).pack(side=tk.LEFT, padx=5)
        ttk.Label(btn_frame, text="X-Axis Unit:").pack(side=tk.LEFT, padx=5)
        self.x_axis_cb = ttk.Combobox(btn_frame, values=["sec", "min", "hour"], width=10)
        self.x_axis_cb.set("hour"); self.x_axis_cb.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Plot", command=self.plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Merge Raw Data into df_FP", command=self.merge).pack(side=tk.LEFT, padx=5)
        
        self.fig = Figure(figsize=(6, 4)); self.ax1 = self.fig.add_subplot(211); self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self).pack(fill="x")

    def _process_fed_files(self, files, label):
        dfs = [self.data_mgr.safe_read_csv(f) for f in files if self.data_mgr.safe_read_csv(f) is not None]
        if not dfs: return pd.DataFrame(), pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        fc = df.columns[0]; df[fc] = pd.to_datetime(df[fc], format='mixed')
        df = df.sort_values(by=fc).reset_index(drop=True)
        df[self.data_mgr.time_col] = (df.iloc[0,1]/1000.0) + (df[fc] - df.at[0,fc]).dt.total_seconds()
        
        c = int(self.ef.get())
        df = self.data_mgr.enforce_monotonic(df, c)
        idx = self.data_mgr.get_fed_indices(df, df.columns[c], label)
        return df, idx

    def proc(self):
        self.data_mgr.df_fed_L, self.data_mgr.df_fed_L_indices = self._process_fed_files(self.data_mgr.raw_fed_l_files, 'DIO_fed_L')
        self.data_mgr.df_fed_R, self.data_mgr.df_fed_R_indices = self._process_fed_files(self.data_mgr.raw_fed_r_files, 'DIO_fed_R')
        self.data_mgr.log("FED data processed successfully.", "INFO")
        self.plot()

    def plot(self):
        c = int(self.ef.get()); factor = {"sec": 1.0, "min": 1/60.0, "hour": 1/3600.0}[self.x_axis_cb.get()]
        def _plot_ax(df, idx, label, ax):
            if df.empty: return
            ax.clear(); ax.step(df[self.data_mgr.time_col] * factor, df.iloc[:,c])
            ax.set_title(label); ax.set_xlabel(f"Time ({self.x_axis_cb.get()})")
            # if not idx.empty:
            #     for t in idx[self.data_mgr.time_col]: ax.axvline(t * factor, color='r', alpha=0.5)
        _plot_ax(self.data_mgr.df_fed_L, self.data_mgr.df_fed_L_indices, 'DIO_fed_L', self.ax1)
        _plot_ax(self.data_mgr.df_fed_R, self.data_mgr.df_fed_R_indices, 'DIO_fed_R', self.ax2)
        self.fig.tight_layout(); self.canvas.draw()

    def merge(self):
        if self.data_mgr.df_FP.empty: return
        df = self.data_mgr.df_FP.copy()
        c = int(self.ef.get())
        merged = False
        
        # Merge the RAW monotonic columns directly
        for df_fed, dio_name in zip([self.data_mgr.df_fed_L, self.data_mgr.df_fed_R], ['DIO_fed_L', 'DIO_fed_R']):
            if df_fed.empty: continue
            if dio_name in df.columns:
                if not messagebox.askyesno("Overwrite?", f"Column '{dio_name}' exists. Overwrite?"): continue
                df = df.drop(columns=[dio_name])
                
            raw_col = df_fed.columns[c]
            d_sub = df_fed[[self.data_mgr.time_col, raw_col]].rename(columns={raw_col: dio_name})
            
            df = pd.merge_asof(df.sort_values(self.data_mgr.time_col), 
                               d_sub.sort_values(self.data_mgr.time_col), 
                               on=self.data_mgr.time_col, direction='backward')
            df[dio_name] = df[dio_name].ffill().fillna(0).astype(int)
            merged = True
            
        if merged:
            self.data_mgr.set_df_fp(df)
            self.data_mgr.log("FED raw data merged into df_FP.", "INFO")



class Panel5_CNO(ttk.Frame):
    def __init__(self, parent, data_mgr):
        super().__init__(parent); self.data_mgr = data_mgr
        ttk.Label(self, text="Step 5: Optional DIO CNO", font=("Helvetica", 12, "bold")).pack(pady=10)
        ttk.Label(self, text="Relative Time (hh:mm:ss):").pack()
        self.e = ttk.Entry(self); self.e.pack(pady=5)
        ttk.Button(self, text="Generate DIO_CNO", command=self.add).pack()

    def add(self):
        if self.data_mgr.df_FP.empty: return
        try:
            sec = pd.to_timedelta(self.e.get()).total_seconds()
            df = self.data_mgr.df_FP.copy()
            proceed_with_generation = True
            
            # Check if CNO already exists
            if 'DIO_CNO' in df.columns:
                overwrite = messagebox.askyesno("Overwrite?", "Column 'DIO_CNO' already exists. Overwrite?")
                if overwrite:
                    # User selected YES
                    df = df.drop(columns=['DIO_CNO'])
                else:
                    # User selected NO
                    self.data_mgr.log("Skipped overwriting DIO_CNO.", "INFO")
                    proceed_with_generation = False
            
            # GENERATE ONLY IF ALLOWED
            if proceed_with_generation:
                idx = (df[self.data_mgr.time_col] - sec).abs().idxmin()
                df['DIO_CNO'] = 0
                df.loc[idx:idx+9, 'DIO_CNO'] = 1
                self.data_mgr.set_df_fp(df)
                self.data_mgr.log(f"CNO added near {sec}s.", "INFO")
                
        except Exception as e: 
            self.data_mgr.log(f"CNO Error: {e}", "ERR")

class Panel6_Mapping(ttk.Frame):
    def __init__(self, parent, data_mgr):
        super().__init__(parent); self.data_mgr = data_mgr
        ttk.Label(self, text="Step 6: Signal-DIO Mapping", font=("Helvetica", 12, "bold")).pack()
        f = ttk.Frame(self); f.pack(pady=5)
        
        ttk.Label(f, text="Animal ID:").grid(row=0,column=0)
        self.eid = ttk.Entry(f); self.eid.insert(0,"M01"); self.eid.grid(row=0,column=1)
        ttk.Button(f, text="Refresh", command=self.ref).grid(row=0,column=2, padx=5)
        
        self.lsig = tk.Listbox(f, height=10, exportselection=False)
        self.lsig.grid(row=1,column=0, columnspan=2, pady=5)
        
        self.ldio = tk.Listbox(f, height=10, selectmode=tk.MULTIPLE, exportselection=False)
        self.ldio.grid(row=1,column=2, pady=5)
        
        ttk.Button(f, text="Add Mapping", command=self.add).grid(row=2, column=0, columnspan=3, pady=5)
        
        self.lmap = tk.Listbox(f, height=10, width=80)
        self.lmap.grid(row=3, column=0, columnspan=3, pady=5)
        
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=5)
        ttk.Button(btn_frame, text="Delete Selection", command=self.delete_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Mappings", command=self.save_mapping).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Mappings", command=self.load_mapping).pack(side=tk.LEFT, padx=5)

    def ref(self):
        self.lsig.delete(0, tk.END)
        self.ldio.delete(0, tk.END)
        for c in self.data_mgr.df_FP.columns:
            if str(c).startswith("G"): 
                self.lsig.insert(tk.END, c)
            elif str(c).startswith("DIO"): 
                self.ldio.insert(tk.END, c)

    def add(self):
        sel_sig = self.lsig.curselection()
        sel_dio = self.ldio.curselection()
        
        if not sel_sig or not sel_dio:
            self.data_mgr.log("Please select exactly one Signal and at least one DIO.", "WARN")
            return
            
        sig = self.lsig.get(sel_sig[0])
        dios = [self.ldio.get(i) for i in sel_dio]
        
        m = {"id": self.eid.get(), "sig": sig, "dios": dios}
        self.data_mgr.mappings.append(m)
        self.lmap.insert(tk.END, f"{m['id']} | {m['sig']} | {','.join(m['dios'])}")
        
        self.lsig.selection_clear(0, tk.END)
        self.ldio.selection_clear(0, tk.END)

    def delete_map(self):
        sel = self.lmap.curselection()
        if not sel:
            self.data_mgr.log("Please select a mapping from the list to delete.", "WARN")
            return
            
        idx = sel[0]
        del self.data_mgr.mappings[idx]
        self.lmap.delete(idx)
        self.data_mgr.log("Mapping successfully deleted.", "INFO")

    def save_mapping(self):
        if not self.data_mgr.mappings:
            self.data_mgr.log("No mappings available to save.", "WARN")
            return
            
        rows = []
        for m in self.data_mgr.mappings:
            rows.append({
                "Animal_ID": m["id"],
                "Signal_Channel": m["sig"],
                "DIO_Channels": ",".join(m["dios"])
            })
        df = pd.DataFrame(rows)
        
        out_dir = self.data_mgr.output_dir if self.data_mgr.output_dir else os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.data_mgr.fp_base_name if self.data_mgr.fp_base_name else "Session"
        
        save_path = os.path.join(out_dir, f"{base_name}_mapping_{ts}.csv")
        try:
            df.to_csv(save_path, index=False)
            self.data_mgr.log(f"Mappings successfully saved to: {save_path}", "INFO")
        except Exception as e:
            self.data_mgr.log(f"Failed to save mappings: {e}", "ERR")

    def load_mapping(self):
        f = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not f:
            return
        try:
            df = pd.read_csv(f)
            required_cols = ["Animal_ID", "Signal_Channel", "DIO_Channels"]
            if not all(col in df.columns for col in required_cols):
                self.data_mgr.log("Invalid mapping CSV. Columns must be: Animal_ID, Signal_Channel, DIO_Channels", "ERR")
                return
            
            if self.data_mgr.mappings:
                overwrite = messagebox.askyesno("Load Mappings", "Overwrite existing mappings?\n(No will append loaded mappings)")
                if overwrite:
                    self.data_mgr.mappings.clear()
                    self.lmap.delete(0, tk.END)
                
            for _, row in df.iterrows():
                aid = str(row["Animal_ID"])
                sig = str(row["Signal_Channel"])
                dios_str = str(row["DIO_Channels"])
                dios = [d.strip() for d in dios_str.split(",") if d.strip()]
                
                m = {"id": aid, "sig": sig, "dios": dios}
                self.data_mgr.mappings.append(m)
                self.lmap.insert(tk.END, f"{aid} | {sig} | {','.join(dios)}")
                
            self.data_mgr.log(f"Loaded {len(df)} mappings from {Path(f).name}", "INFO")
        except Exception as e:
            self.data_mgr.log(f"Failed to load mappings: {e}", "ERR")
            traceback.print_exc()


class Panel7_Alignment(ttk.Frame):
    def __init__(self, parent, data_mgr):
        super().__init__(parent); self.data_mgr = data_mgr
        ttk.Label(self, text="Step 7: Trial Alignment & Export", font=("Helvetica", 12, "bold")).pack(pady=5)
        f_top = ttk.Frame(self); f_top.pack(fill="x", padx=10, pady=5)
        ttk.Button(f_top, text="Get Sampling Rate (Hz)", command=self.get_sr).pack(side=tk.LEFT, padx=5)
        self.lbl_sr = ttk.Label(f_top, text="SR: -- Hz", font=("Helvetica", 10, "bold"), foreground="blue"); self.lbl_sr.pack(side=tk.LEFT, padx=5)
        ttk.Button(f_top, text="Refresh DIO Rows", command=self.refresh_rows).pack(side=tk.RIGHT, padx=5)
        
        self.row_container = ttk.Frame(self); self.row_container.pack(fill="both", expand=True, padx=10, pady=10)
        self.dio_entries = {}  

    def get_sr(self):
        if self.data_mgr.df_FP.empty: return
        dt = np.median(np.diff(self.data_mgr.df_FP[self.data_mgr.time_col]))
        if dt > 0:
            self.lbl_sr.config(text=f"SR: {1.0/dt:.2f} Hz")
            self.data_mgr.log(f"Sampling rate calculated: {1.0/dt:.2f} Hz", "INFO")

    def _get_prefix(self, dio_name):
        parts = dio_name.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else dio_name

    def refresh_rows(self):
        for widget in self.row_container.winfo_children(): widget.destroy()
        self.dio_entries.clear()
        self.get_sr()
        prefixes = {self._get_prefix(dio) for m in self.data_mgr.mappings for dio in m['dios']}
        if not prefixes:
            ttk.Label(self.row_container, text="No DIOs found. Config mappings in Step 6 first.").pack(pady=10); return
            
        for prefix in sorted(list(prefixes)):
            f_row = ttk.Frame(self.row_container); f_row.pack(fill="x", pady=3)
            ttk.Label(f_row, text=prefix, width=15, font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
            ttk.Label(f_row, text="Pre(s):").pack(side=tk.LEFT); e_pre = ttk.Entry(f_row, width=4); e_pre.insert(0, "4"); e_pre.pack(side=tk.LEFT, padx=2)
            ttk.Label(f_row, text="Post(s):").pack(side=tk.LEFT); e_post = ttk.Entry(f_row, width=4); e_post.insert(0, "10"); e_post.pack(side=tk.LEFT, padx=2)
            
            var_on, var_off = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
            ttk.Checkbutton(f_row, text="Onset", variable=var_on).pack(side=tk.LEFT, padx=2)
            ttk.Checkbutton(f_row, text="Offset", variable=var_off).pack(side=tk.LEFT, padx=2)
            
            self.dio_entries[prefix] = (e_pre, e_post, var_on, var_off)
            ttk.Button(f_row, text="Align", command=lambda p=prefix: self.align_prefix(p)).pack(side=tk.LEFT, padx=10)
            ttk.Button(f_row, text="Save CSVs", command=lambda p=prefix: self.save_prefix_csv(p)).pack(side=tk.LEFT, padx=5)

    def align_prefix(self, prefix):
        if self.data_mgr.df_FP.empty: return
        e_pre, e_post, var_on, var_off = self.dio_entries[prefix]
        pre, post, do_on, do_off = float(e_pre.get()), float(e_post.get()), var_on.get(), var_off.get()
        if not do_on and not do_off:
            self.data_mgr.log("Select at least Onset or Offset.", "WARN"); return
            
        df = self.data_mgr.df_FP; t_col = self.data_mgr.time_col
        dt = np.median(np.diff(df[t_col]))
        if dt <= 0: return
        npre, npost = int(pre/dt), int(post/dt)
        ctime = np.arange(-npre, npost) * dt
        
        count = 0
        for m in self.data_mgr.mappings:
            aid, sig = m['id'], m['sig']
            for dio in m['dios']:
                if self._get_prefix(dio) == prefix and dio in df.columns and sig in df.columns:
                    # Run dynamic extraction logic on df_FP directly!
                    if "wheel" in dio:
                        interval = getattr(self.data_mgr, 'wheel_interval', 5.0)
                        idx_df = self.data_mgr.get_wheel_indices(df, dio, interval, dio)
                        t_ons = idx_df[idx_df[dio] == 1][t_col].values
                        t_offs = idx_df[idx_df[dio] == 0][t_col].values
                    elif "fed" in dio:
                        idx_df = self.data_mgr.get_fed_indices(df, dio, dio)
                        t_ons = idx_df[t_col].values
                        t_offs = []
                    elif "CNO" in dio:
                        t_ons = df[df[dio] == 1][t_col].head(1).values
                        t_offs = []
                    else:
                        diffs = df[dio].diff().fillna(0).values
                        t_ons = df[t_col].values[np.where(diffs > 0)[0]]
                        t_offs = df[t_col].values[np.where(diffs < 0)[0]]
                    
                    # Convert event times to the closest exact row indices in df_FP
                    t_array = df[t_col].values
                    idx_ons = [np.abs(t_array - t).argmin() for t in t_ons]
                    idx_offs = [np.abs(t_array - t).argmin() for t in t_offs]
                    
                    evts_to_process = []
                    if do_on: evts_to_process.append(('onset', idx_ons))
                    if do_off: evts_to_process.append(('offset', idx_offs))
                    
                    sig_arr = df[sig].values
                    for evt_type, indices in evts_to_process:
                        epochs = []
                        for idx in indices:
                            if idx >= npre and idx + npost <= len(df):
                                epochs.append(sig_arr[idx-npre : idx+npost])
                        if epochs:
                            edf = pd.DataFrame(np.array(epochs).T, index=ctime, columns=[f'Trial_{i+1}' for i in range(len(epochs))])
                            edf.index.name = "Time(s)"
                            self.data_mgr.aligned_data[f"{aid} | {sig} | {dio} | {evt_type}"] = edf
                            count += 1
                            
        if count > 0: self.data_mgr.log(f"Aligned '{prefix}' ({count} groups). View in Heatmap.", "INFO")
        else: self.data_mgr.log(f"No valid events found for group '{prefix}'.", "WARN")

    def save_prefix_csv(self, prefix):
        out_dir = getattr(self.data_mgr, 'output_dir', None) or os.getcwd()
        base_name = getattr(self.data_mgr, 'fp_base_name', 'Session')
        os.makedirs(out_dir, exist_ok=True); ts = datetime.now().strftime("%Y%m%d_%H%M%S"); c = 0
        
        for key, edf in self.data_mgr.aligned_data.items():
            parts = [p.strip() for p in key.split("|")]
            if len(parts) >= 4 and self._get_prefix(parts[2]) == prefix:
                edf.to_csv(os.path.join(out_dir, f"{base_name}_{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_aligned_{ts}.csv"))
                c += 1
        self.data_mgr.log(f"Saved {c} CSV(s) for '{prefix}'." if c>0 else f"No data found for '{prefix}'.", "INFO" if c>0 else "WARN")


class Panel8_Heatmap(ttk.Frame):
    def __init__(self, parent, data_mgr):
        super().__init__(parent)
        self.data_mgr = data_mgr
        
        # Caching variables for AUC calculation
        self.current_t_axis = None
        self.current_mean_sig = None
        
        ttk.Label(self, text="Step 8: Event Heatmap Viewer", font=("Helvetica", 12, "bold")).pack(pady=5)
        
        # 1. Navigation Frame
        f_nav = ttk.Frame(self)
        f_nav.pack(fill="x", pady=5)
        
        ttk.Button(f_nav, text="Refresh", command=self.refresh_keys).pack(side=tk.LEFT, padx=5)
        ttk.Button(f_nav, text="<< Prev", command=self.prev_plot).pack(side=tk.LEFT, padx=5)
        self.cb_keys = ttk.Combobox(f_nav, width=45)
        self.cb_keys.pack(side=tk.LEFT, padx=5)
        self.cb_keys.bind("<<ComboboxSelected>>", lambda e: self.plot())
        ttk.Button(f_nav, text="Next >>", command=self.next_plot).pack(side=tk.LEFT, padx=5)
        
        # 2. Options Frame
        f_opt = ttk.Frame(self)
        f_opt.pack(fill="x", pady=5)
        
        ttk.Label(f_opt, text="Mode:").pack(side=tk.LEFT, padx=2)
        self.cb_mode = ttk.Combobox(f_opt, values=["Raw dF/F", "Z-Score (Baseline)", "Median Z-Score (Full Trace)"], width=20)
        self.cb_mode.set("Raw dF/F")
        self.cb_mode.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_opt, text="Base Start(s):").pack(side=tk.LEFT, padx=2)
        self.e_bstart = ttk.Entry(f_opt, width=4); self.e_bstart.insert(0, "-4"); self.e_bstart.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_opt, text="End(s):").pack(side=tk.LEFT, padx=2)
        self.e_bend = ttk.Entry(f_opt, width=4); self.e_bend.insert(0, "0"); self.e_bend.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_opt, text="Cmap:").pack(side=tk.LEFT, padx=2)
        self.cb_cmap = ttk.Combobox(f_opt, values=['viridis', 'hot', 'coolwarm', 'jet'], width=8)
        self.cb_cmap.set('viridis')
        self.cb_cmap.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(f_opt, text="Plot", command=self.plot).pack(side=tk.LEFT, padx=10)

        # 2.5 AUC Frame (Packed next to Options)
        f_auc = ttk.Frame(self)
        f_auc.pack(fill="x", pady=5)
        
        ttk.Label(f_auc, text="AUC Start(s):").pack(side=tk.LEFT, padx=2)
        self.e_auc_start = ttk.Entry(f_auc, width=5); self.e_auc_start.insert(0, "0"); self.e_auc_start.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_auc, text="End(s):").pack(side=tk.LEFT, padx=2)
        self.e_auc_end = ttk.Entry(f_auc, width=5); self.e_auc_end.insert(0, "5"); self.e_auc_end.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(f_auc, text="Calculate AUC", command=self.calculate_auc).pack(side=tk.LEFT, padx=10)
        self.lbl_auc_val = ttk.Label(f_auc, text="AUC: --", font=("Helvetica", 10, "bold"), foreground="purple")
        self.lbl_auc_val.pack(side=tk.LEFT, padx=5)

        # Create the figure and canvas objects first (required before toolbar)
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        # 3. Export Buttons (Packed BEFORE canvas)
        f_exp = ttk.Frame(self)
        f_exp.pack(fill="x", pady=5)
        ttk.Button(f_exp, text="Save Current (PNG)", command=self.save_png).pack(side=tk.LEFT, padx=10)
        ttk.Button(f_exp, text="Save All (PDF)", command=self.save_pdf).pack(side=tk.LEFT, padx=10)

        # 4. Toolbar (Packed BEFORE canvas)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(fill="x", padx=10)

        # 5. Canvas (Packed LAST with expand=True)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

    def refresh_keys(self):
        keys = list(self.data_mgr.aligned_data.keys())
        self.cb_keys['values'] = keys
        if keys and not self.cb_keys.get():
            self.cb_keys.set(keys[0])

    def prev_plot(self):
        keys = list(self.cb_keys['values'])
        if not keys: return
        curr = self.cb_keys.get()
        if curr in keys:
            idx = keys.index(curr)
            self.cb_keys.set(keys[(idx - 1) % len(keys)])
            self.plot()

    def next_plot(self):
        keys = list(self.cb_keys['values'])
        if not keys: return
        curr = self.cb_keys.get()
        if curr in keys:
            idx = keys.index(curr)
            self.cb_keys.set(keys[(idx + 1) % len(keys)])
            self.plot()        

    def _transform_data(self, df, key):
        """Applies Z-scoring on the fly based on user selection"""
        mode = self.cb_mode.get()
        df_plot = df.copy()
        
        if mode == "Z-Score (Baseline)":
            b_start, b_end = float(self.e_bstart.get()), float(self.e_bend.get())
            base_mask = (df_plot.index >= b_start) & (df_plot.index <= b_end)
            if base_mask.any():
                for col in df_plot.columns:
                    b_data = df_plot.loc[base_mask, col]
                    b_mean = b_data.mean()
                    b_std = b_data.std() if b_data.std() > 0 else 1
                    df_plot[col] = (df_plot[col] - b_mean) / b_std
                    
        elif mode == "Median Z-Score (Full Trace)":
            sig_name = key.split("|")[1].strip()
            full_trace = self.data_mgr.df_FP[sig_name]
            med = full_trace.median()
            mad = np.median(np.abs(full_trace - med))
            scale = (mad * 1.4826) if mad > 0 else (full_trace.std() or 1)
            for col in df_plot.columns:
                df_plot[col] = (df_plot[col] - med) / scale
                
        return df_plot

    def plot(self):
        key = self.cb_keys.get()
        if key not in self.data_mgr.aligned_data: return
        
        try:
            self.fig.clf()
            
            gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 0.02], wspace=0.02, hspace=0.15)
            ax_top = self.fig.add_subplot(gs[0, 0])
            cax = self.fig.add_subplot(gs[0, 1])  
            ax_bot = self.fig.add_subplot(gs[1, 0], sharex=ax_top)
            
            df_raw = self.data_mgr.aligned_data[key]
            df_plot = self._transform_data(df_raw, key)
            n_trials = df_plot.shape[1]
            t_axis = df_plot.index
            
            parts = [p.strip() for p in key.split("|")]
            aid, sig, dio, evt_type = parts[0], parts[1], parts[2], parts[3]
            
            # --- SPECIAL SINGLE EVENT / CNO LOGIC ---
            if "CNO" in dio or n_trials == 1:
                cax.axis('off')  
                df_full = self.data_mgr.df_FP
                t0 = None
                
                if "CNO" in dio:
                    evts = df_full.index[df_full[dio] == 1].tolist()
                    if evts: t0 = df_full[self.data_mgr.time_col].iloc[evts[0]]
                else:
                    diffs = df_full[dio].diff().fillna(0).values
                    evts = np.where(diffs > 0)[0] if evt_type == 'onset' else np.where(diffs < 0)[0]
                    if len(evts) > 0: t0 = df_full[self.data_mgr.time_col].iloc[evts[0]]
                
                if t0 is not None:
                    other_dios = []
                    for m in self.data_mgr.mappings:
                        if m['id'] == aid and m['sig'] == sig:
                            other_dios = [d for d in m['dios'] if d != dio and d in df_full.columns]
                            break
                            
                    if other_dios:
                        mask = (df_full[self.data_mgr.time_col] >= t0 + t_axis[0]) & \
                               (df_full[self.data_mgr.time_col] <= t0 + t_axis[-1])
                        df_win = df_full[mask]
                        t_win = df_win[self.data_mgr.time_col].values - t0
                        
                        colors = ['blue', 'green', 'orange', 'purple']
                        for i, od in enumerate(other_dios):
                            raw_vals = df_win[od].values
                            if len(raw_vals) > 0:
                                raw_vals = raw_vals - raw_vals[0]
                            ax_top.step(t_win, raw_vals, color=colors[i%len(colors)], label=od, where='post', lw=2)
                            
                        ax_top.legend(loc='upper left', fontsize='x-small')
                        ax_top.set_title(f"{dio} {evt_type.upper()} - Concurrent Raw DIOs")
                    else:
                        ax_top.set_title("No concurrent events mapped.")
                        ax_top.set_yticks([])
                else:
                    ax_top.set_title("Could not locate event in full trace.")
                
                # Bottom Line Plot
                mean_sig = df_plot.iloc[:, 0]
                ax_bot.plot(t_axis, mean_sig, 'k-', lw=1.5)
                ax_bot.axvline(0, color='r', ls='--')
                
            # --- NORMAL MULTI-TRIAL LOGIC ---
            else:
                im = ax_top.imshow(df_plot.values.T, aspect='auto', cmap=self.cb_cmap.get(), 
                                   extent=[t_axis[0], t_axis[-1], n_trials, 0], interpolation='nearest')
                ax_top.axvline(0, color='r', ls='--')
                ax_top.set_title(f"{key} ({self.cb_mode.get()})")
                ax_top.set_ylabel("Trial #")
                
                self.fig.colorbar(im, cax=cax)
                cax.set_ylabel(self.cb_mode.get())
                
                # Bottom Mean +/- SEM
                mean_sig = df_plot.mean(axis=1)
                sem_sig = df_plot.std(axis=1) / np.sqrt(n_trials)
                
                ax_bot.plot(t_axis, mean_sig, 'k-', lw=1.5)
                ax_bot.fill_between(t_axis, mean_sig - sem_sig, mean_sig + sem_sig, color='gray', alpha=0.5)
                ax_bot.axvline(0, color='r', ls='--')

            # Formatting and Alignment
            ax_top.tick_params(labelbottom=False)  
            ax_bot.set_ylabel(self.cb_mode.get())
            ax_bot.set_xlabel("Time (s)")
            
            # Cache active trace & time axis for potential AUC calculation
            self.current_t_axis = t_axis.values if hasattr(t_axis, 'values') else t_axis
            self.current_mean_sig = mean_sig.values if hasattr(mean_sig, 'values') else mean_sig
            self.lbl_auc_val.config(text="AUC: --") # Reset AUC label on new plot
            
            self.fig.align_ylabels([ax_top, ax_bot])
            self.fig.tight_layout()
            self.canvas.draw()
            self.toolbar.update()
            self.calculate_auc()
            
        except Exception as e:
            self.data_mgr.log(f"Plotting Error: {e}", "ERR")
            traceback.print_exc()

    def calculate_auc(self):
        if self.current_mean_sig is None or self.current_t_axis is None:
            self.lbl_auc_val.config(text="AUC: Plot first!")
            return
            
        try:
            start = float(self.e_auc_start.get())
            end = float(self.e_auc_end.get())
            
            # Select only indices inside user window
            mask = (self.current_t_axis >= start) & (self.current_t_axis <= end)
            if not mask.any():
                self.lbl_auc_val.config(text="AUC: Invalid bounds")
                return
                
            x_win = self.current_t_axis[mask]
            y_win = self.current_mean_sig[mask]
            
            # Trapezoidal numerical integration
            auc_val = np.trapz(y_win, x_win)
            
            self.lbl_auc_val.config(text=f"AUC: {auc_val:.4f}")
            self.data_mgr.log(f"Calculated AUC ({start}s to {end}s): {auc_val:.4f}", "INFO")
        except Exception as e:
            self.lbl_auc_val.config(text="AUC: Calculation Error")
            self.data_mgr.log(f"AUC Error: {e}", "ERR")

    def save_png(self):
        key = self.cb_keys.get()
        if not key: return
        
        out_dir = getattr(self.data_mgr, 'output_dir', None) or os.getcwd()
        base_name = getattr(self.data_mgr, 'fp_base_name', 'Session')
        
        safe_key = key.replace(" | ", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(out_dir, f"{base_name}_{safe_key}_{ts}.png")
        
        self.fig.savefig(fpath, dpi=600, bbox_inches='tight')
        self.data_mgr.log(f"Saved PNG (600 DPI) to: {Path(fpath).name}", "INFO")

    def save_pdf(self):
        if not self.data_mgr.aligned_data:
            self.data_mgr.log("No aligned data to save.", "WARN")
            return
            
        out_dir = getattr(self.data_mgr, 'output_dir', None) or os.getcwd()
        base_name = getattr(self.data_mgr, 'fp_base_name', 'Session')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(out_dir, f"{base_name}_Heatmaps_All_{ts}.pdf")
        
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            current_key = self.cb_keys.get()
            
            with PdfPages(fpath) as pdf:
                for key in self.data_mgr.aligned_data.keys():
                    self.cb_keys.set(key)
                    self.plot()  
                    pdf.savefig(self.fig, bbox_inches='tight')
            
            if current_key:
                self.cb_keys.set(current_key)
                self.plot()
                
            self.data_mgr.log(f"Saved all heatmaps to PDF: {Path(fpath).name}", "INFO")
        except Exception as e:
            self.data_mgr.log(f"PDF Save Error: {e}", "ERR")



class Panel9_Peak(ttk.Frame):
    def __init__(self, parent, data_mgr):
        super().__init__(parent)
        self.data_mgr = data_mgr
        
        # Caching variables for stats calculation
        self.current_rel_times = np.array([])
        self.current_amps = np.array([])
        self.current_n_trials = 1
        
        ttk.Label(self, text="Step 9: Event-Aligned Peak Analysis", font=("Helvetica", 12, "bold")).pack(pady=5)
        
        # 1. Navigation Frame
        f_nav = ttk.Frame(self)
        f_nav.pack(fill="x", pady=5)
        
        ttk.Button(f_nav, text="Refresh", command=self.refresh_keys).pack(side=tk.LEFT, padx=5)
        ttk.Button(f_nav, text="<< Prev", command=self.prev_plot).pack(side=tk.LEFT, padx=5)
        self.cb_keys = ttk.Combobox(f_nav, width=45)
        self.cb_keys.pack(side=tk.LEFT, padx=5)
        self.cb_keys.bind("<<ComboboxSelected>>", lambda e: self.plot_aligned())
        ttk.Button(f_nav, text="Next >>", command=self.next_plot).pack(side=tk.LEFT, padx=5)
        
        # 2. Parameters & Actions Frame
        f_opt = ttk.Frame(self)
        f_opt.pack(fill="x", pady=5)
        
        ttk.Label(f_opt, text="Prominence:").pack(side=tk.LEFT, padx=2)
        self.e_prom = ttk.Entry(f_opt, width=5)
        self.e_prom.insert(0, "0.5")
        self.e_prom.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_opt, text="Bin Size(s):").pack(side=tk.LEFT, padx=5)
        self.e_bin = ttk.Entry(f_opt, width=5)
        self.e_bin.insert(0, "1.0")
        self.e_bin.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_opt, text="Cmap:").pack(side=tk.LEFT, padx=5)
        self.cb_cmap = ttk.Combobox(f_opt, values=['viridis', 'hot', 'coolwarm', 'jet'], width=8)
        self.cb_cmap.set('viridis')
        self.cb_cmap.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(f_opt, text="1. Preview Global Peaks", command=self.preview_global).pack(side=tk.LEFT, padx=15)
        ttk.Button(f_opt, text="2. Plot Event-Aligned", command=self.plot_aligned).pack(side=tk.LEFT, padx=5)

        # 2.5 Peak Stats Calculation Frame
        f_stat = ttk.Frame(self)
        f_stat.pack(fill="x", pady=5)
        
        ttk.Label(f_stat, text="Calc Start(s):").pack(side=tk.LEFT, padx=2)
        self.e_stat_start = ttk.Entry(f_stat, width=5)
        self.e_stat_start.insert(0, "0")
        self.e_stat_start.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_stat, text="End(s):").pack(side=tk.LEFT, padx=2)
        self.e_stat_end = ttk.Entry(f_stat, width=5)
        self.e_stat_end.insert(0, "5")
        self.e_stat_end.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(f_stat, text="Calculate Freq & Amp", command=self.calculate_stats).pack(side=tk.LEFT, padx=10)
        self.lbl_stat_freq = ttk.Label(f_stat, text="Freq: -- Hz", font=("Helvetica", 10, "bold"), foreground="purple")
        self.lbl_stat_freq.pack(side=tk.LEFT, padx=5)
        self.lbl_stat_amp = ttk.Label(f_stat, text="Amp: -- Z", font=("Helvetica", 10, "bold"), foreground="purple")
        self.lbl_stat_amp.pack(side=tk.LEFT, padx=5)
        
        # Create Figure & Canvas objects first
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        
        # 3. Export Buttons (Packed BEFORE canvas)
        f_exp = ttk.Frame(self)
        f_exp.pack(fill="x", pady=5)
        ttk.Button(f_exp, text="Save Current (PNG)", command=self.save_png).pack(side=tk.LEFT, padx=10)
        ttk.Button(f_exp, text="Save All (PDF)", command=self.save_pdf).pack(side=tk.LEFT, padx=10)

        # 4. Toolbar (Packed BEFORE canvas)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(fill="x", padx=10)

        # 5. Canvas (Packed LAST with expand=True)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

    def refresh_keys(self):
        keys = list(self.data_mgr.aligned_data.keys())
        self.cb_keys['values'] = keys
        if keys and not self.cb_keys.get():
            self.cb_keys.set(keys[0])

    def prev_plot(self):
        keys = list(self.cb_keys['values'])
        if not keys: return
        curr = self.cb_keys.get()
        if curr in keys:
            idx = keys.index(curr)
            self.cb_keys.set(keys[(idx - 1) % len(keys)])
            self.plot_aligned()

    def next_plot(self):
        keys = list(self.cb_keys['values'])
        if not keys: return
        curr = self.cb_keys.get()
        if curr in keys:
            idx = keys.index(curr)
            self.cb_keys.set(keys[(idx + 1) % len(keys)])
            self.plot_aligned()

    def _get_global_peaks(self, sig_name):
        """Calculates Robust Median Z-Score over the entire session and finds peaks."""
        df = self.data_mgr.df_FP
        full_trace = df[sig_name]
        
        med = full_trace.median()
        mad = np.median(np.abs(full_trace - med))
        scale = (mad * 1.4826) if mad > 0 else (full_trace.std() or 1)
        z_sig = (full_trace - med) / scale
        
        prom = float(self.e_prom.get())
        peaks, _ = find_peaks(z_sig, prominence=prom)
        
        return df[self.data_mgr.time_col].values, z_sig.values, peaks

    def _get_t0s(self, dio_name, evt_type):
        """Extracts the exact event times from the global dataframe."""
        df = self.data_mgr.df_FP
        if dio_name not in df.columns: return []
        
        if "CNO" in dio_name:
            evts = df.index[df[dio_name] == 1].tolist()
            return [df[self.data_mgr.time_col].iloc[evts[0]]] if evts else []
            
        diffs = df[dio_name].diff().fillna(0).values
        is_monotonic = (df[dio_name].max() > 1)
        
        if is_monotonic:
            trans = np.where(diffs > 0)[0]
            idxs = trans[0::2] if evt_type == 'onset' else trans[1::2]
        else:
            idxs = np.where(diffs > 0)[0] if evt_type == 'onset' else np.where(diffs < 0)[0]
            
        return df[self.data_mgr.time_col].iloc[idxs].values

    def preview_global(self):
        key = self.cb_keys.get()
        if not key or self.data_mgr.df_FP.empty: return
        
        try:
            sig_name = key.split("|")[1].strip()
            t_vals, z_vals, peaks = self._get_global_peaks(sig_name)
            
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.plot(t_vals, z_vals, label=f"Median Z-Score ({sig_name})")
            ax.plot(t_vals[peaks], z_vals[peaks], "ro", markersize=6, label=f"Peaks (Prom>={self.e_prom.get()})")
            
            ax.set_title(f"Global Peak Preview: {sig_name}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Z-Score")
            ax.legend()
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.toolbar.update()
        except Exception as e:
            self.data_mgr.log(f"Preview Error: {e}", "ERR")
            traceback.print_exc()

    def plot_aligned(self):
        key = self.cb_keys.get()
        if not key or key not in self.data_mgr.aligned_data: return
        
        try:
            parts = [p.strip() for p in key.split("|")]
            aid, sig, dio, evt_type = parts[0], parts[1], parts[2], parts[3]
            
            # 1. Get Global Data and align event times
            t_vals, z_vals, peaks = self._get_global_peaks(sig)
            t0s = self._get_t0s(dio, evt_type)
            
            df_aligned = self.data_mgr.aligned_data[key]
            n_trials = df_aligned.shape[1]
            t_axis = df_aligned.index
            t_start, t_end = t_axis[0], t_axis[-1]
            
            # 2. Extract relative peak times and amplitudes for each trial
            all_rel_times = []
            all_amps = []
            all_trials = []
            
            single_z_window = []
            
            for trial_idx, t0 in enumerate(t0s):
                mask = (t_vals[peaks] >= t0 + t_start) & (t_vals[peaks] <= t0 + t_end)
                window_peaks = peaks[mask]
                
                rel_times = t_vals[window_peaks] - t0
                amps = z_vals[window_peaks]
                
                all_rel_times.extend(rel_times)
                all_amps.extend(amps)
                all_trials.extend([trial_idx + 1] * len(rel_times))
                
                if n_trials == 1:
                    win_mask = (t_vals >= t0 + t_start) & (t_vals <= t0 + t_end)
                    single_z_window = z_vals[win_mask]

            # Cache the arrays for stats calculation
            self.current_rel_times = np.array(all_rel_times)
            self.current_amps = np.array(all_amps)
            self.current_n_trials = n_trials
            
            self.lbl_stat_freq.config(text="Freq: -- Hz")
            self.lbl_stat_amp.config(text="Amp: -- Z")

            # 3. Setup Figure Layout
            self.fig.clf()
            gs = self.fig.add_gridspec(3, 2, width_ratios=[1, 0.03], hspace=0.15, wspace=0.02)
            
            ax_top = self.fig.add_subplot(gs[0, 0])
            cax = self.fig.add_subplot(gs[0, 1])
            ax_mid = self.fig.add_subplot(gs[1, 0], sharex=ax_top)
            ax_bot = self.fig.add_subplot(gs[2, 0], sharex=ax_top)

            # 4. Plot Top Panel (Raster or Trace)
            if "CNO" in dio or n_trials == 1:
                cax.axis('off')
                if len(single_z_window) > 0:
                    trace_len = min(len(single_z_window), len(t_axis))
                    ax_top.plot(t_axis[:trace_len], single_z_window[:trace_len], 'k-')
                    ax_top.plot(all_rel_times, all_amps, 'ro', markersize=6, label="Peaks")
                ax_top.set_ylabel("Z-Score")
                ax_top.legend(loc="upper right")
            else:
                if all_rel_times:
                    sc = ax_top.scatter(all_rel_times, all_trials, c=all_amps, 
                                        cmap=self.cb_cmap.get(), marker='|', s=150, linewidths=1.5)
                    self.fig.colorbar(sc, cax=cax, label="Z-Score Amplitude")
                else:
                    cax.axis('off')
                    ax_top.text(0.5, 0.5, "No peaks found", ha='center', va='center', transform=ax_top.transAxes)
                
                ax_top.set_ylim(n_trials + 0.5, 0.5) 
                ax_top.set_ylabel("Trial #")
            
            ax_top.axvline(0, color='r', ls='--')
            ax_top.set_title(f"Peak Analysis: {key}")
            ax_top.tick_params(labelbottom=False)

            # 5. Binning Logic for Middle & Bottom Panels
            bin_size = float(self.e_bin.get())
            bins = np.arange(t_start, t_end + bin_size, bin_size)
            bin_centers = bins[:-1] + (bin_size / 2)
            
            freqs, mean_amps = [], []
            
            for i in range(len(bins)-1):
                b_s, b_e = bins[i], bins[i+1]
                in_bin = (self.current_rel_times >= b_s) & (self.current_rel_times < b_e)
                count = np.sum(in_bin)
                
                freq = count / (n_trials * bin_size)
                freqs.append(freq)
                mean_amps.append(np.mean(self.current_amps[in_bin]) if count > 0 else 0)

            # Middle Panel: Frequency
            ax_mid.bar(bin_centers, freqs, width=bin_size*0.8, color='blue', alpha=0.6)
            ax_mid.axvline(0, color='r', ls='--')
            ax_mid.set_ylabel("Freq (Hz)")
            ax_mid.tick_params(labelbottom=False)

            # Bottom Panel: Amplitude
            ax_bot.plot(bin_centers, mean_amps, 'o-', color='orange', lw=2)
            ax_bot.axvline(0, color='r', ls='--')
            ax_bot.set_ylabel("Mean Amp (Z)")
            ax_bot.set_xlabel("Time (s)")
            ax_bot.set_xlim(t_start, t_end)

            # Format and align
            self.fig.align_ylabels([ax_top, ax_mid, ax_bot])
            self.fig.tight_layout()
            self.canvas.draw()
            self.toolbar.update()
            self.calculate_stats()
            
        except Exception as e:
            self.data_mgr.log(f"Peak Plot Error: {e}", "ERR")
            traceback.print_exc()

    def calculate_stats(self):
        if len(self.current_rel_times) == 0:
            self.lbl_stat_freq.config(text="Freq: No peaks")
            self.lbl_stat_amp.config(text="Amp: --")
            return
            
        try:
            start = float(self.e_stat_start.get())
            end = float(self.e_stat_end.get())
            
            if start >= end:
                self.data_mgr.log("Start time must be less than End time.", "WARN")
                return
                
            mask = (self.current_rel_times >= start) & (self.current_rel_times <= end)
            peaks_in_win = np.sum(mask)
            
            duration = end - start
            freq = peaks_in_win / (self.current_n_trials * duration)
            
            if peaks_in_win > 0:
                mean_amp = np.mean(self.current_amps[mask])
            else:
                mean_amp = 0.0
                
            self.lbl_stat_freq.config(text=f"Freq: {freq:.3f} Hz")
            self.lbl_stat_amp.config(text=f"Amp: {mean_amp:.3f} Z")
            self.data_mgr.log(f"Peak Stats ({start}s to {end}s) - Freq: {freq:.3f} Hz, Mean Amp: {mean_amp:.3f} Z", "INFO")
            
        except Exception as e:
            self.lbl_stat_freq.config(text="Freq: Error")
            self.lbl_stat_amp.config(text="Amp: Error")
            self.data_mgr.log(f"Peak Stats Error: {e}", "ERR")

    def save_png(self):
        key = self.cb_keys.get()
        if not key: return
        
        out_dir = getattr(self.data_mgr, 'output_dir', None) or os.getcwd()
        base_name = getattr(self.data_mgr, 'fp_base_name', 'Session')
        
        safe_key = key.replace(" | ", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(out_dir, f"{base_name}_{safe_key}_peaks_{ts}.png")
        
        self.fig.savefig(fpath, dpi=600, bbox_inches='tight')
        self.data_mgr.log(f"Saved PNG (600 DPI) to: {Path(fpath).name}", "INFO")

    def save_pdf(self):
        if not self.data_mgr.aligned_data:
            self.data_mgr.log("No aligned data to save.", "WARN")
            return
            
        out_dir = getattr(self.data_mgr, 'output_dir', None) or os.getcwd()
        base_name = getattr(self.data_mgr, 'fp_base_name', 'Session')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(out_dir, f"{base_name}_PeakAnalysis_All_{ts}.pdf")
        
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            current_key = self.cb_keys.get()
            
            with PdfPages(fpath) as pdf:
                for key in self.data_mgr.aligned_data.keys():
                    self.cb_keys.set(key)
                    self.plot_aligned()  
                    pdf.savefig(self.fig, bbox_inches='tight')
            
            if current_key:
                self.cb_keys.set(current_key)
                self.plot_aligned()
                
            self.data_mgr.log(f"Saved all peak plots to PDF: {Path(fpath).name}", "INFO")
        except Exception as e:
            self.data_mgr.log(f"PDF Save Error: {e}", "ERR")

# =============================================================================
# MAIN APP
# =============================================================================
class FPFEDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FP/FED Multi-Panel Synchronizer")
        self.geometry("1450x850")
        
        self.data_mgr = DataManager(logger=self.log_message, on_df_update=self.update_global_preview)
        ttk.Style().theme_use('clam')
        
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        lf, rf = ttk.Frame(paned), ttk.Frame(paned)
        paned.add(lf, weight=3)
        paned.add(rf, weight=1)
        
        # Tabs
        self.nb = ttk.Notebook(lf)
        self.nb.pack(fill='both', expand=True)
        self.nb.add(Panel1_Browser(self.nb, self.data_mgr), text="1. Browser")
        self.nb.add(Panel2_FPProcess(self.nb, self.data_mgr), text="2. FP")
        self.nb.add(Panel3_CamProcess(self.nb, self.data_mgr), text="3. CAM")
        self.nb.add(Panel4_FedProcess(self.nb, self.data_mgr), text="4. FED")
        self.nb.add(Panel5_CNO(self.nb, self.data_mgr), text="5. CNO")
        self.nb.add(Panel6_Mapping(self.nb, self.data_mgr), text="6. Mapping")
        self.nb.add(Panel7_Alignment(self.nb, self.data_mgr), text="7. Align")
        self.nb.add(Panel8_Heatmap(self.nb, self.data_mgr), text="8. Heatmap")
        self.nb.add(Panel9_Peak(self.nb, self.data_mgr), text="9. Peak")
        
        # Right Side Panel (Vertical Paned Layout)
        rp = ttk.PanedWindow(rf, orient=tk.VERTICAL)
        rp.pack(fill=tk.BOTH, expand=True)
        log_panel, preview_frame = ttk.Frame(rp), ttk.Frame(rp)
        rp.add(log_panel, weight=1)
        rp.add(preview_frame, weight=1)
        
        # Top: Log
        ttk.Label(log_panel, text="System Messages", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=2)
        
        self.log_text = tk.Text(log_panel, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9), state=tk.DISABLED)
        scroll = ttk.Scrollbar(log_panel, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        
        # Live Table Panel
        ttk.Label(preview_frame, text="Live df_FP", font=("", 10, "bold")).pack(anchor="w")
        self.tv = ttk.Treeview(preview_frame, height=6)
        self.tv.pack(fill="both", expand=True)
        
        bf = ttk.Frame(preview_frame)
        bf.pack(pady=5)
        ttk.Button(bf, text="Load CSV", command=self.load_merged_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Save CSV", command=self.save_merged_csv).pack(side=tk.LEFT, padx=5)
        
        self.log_message("System Ready.", "INFO")

    def log_message(self, msg, level="INFO"):
        self.log_text.config(state=tk.NORMAL)
        ts = datetime.now().strftime("%H:%M:%S")
        color = {"INFO": "#4CAF50", "WARN": "#FFC107", "ERR": "#F44336"}.get(level, "#FFF")
        
        self.log_text.tag_config(f"tag_{level}", foreground=color)
        self.log_text.tag_config("tag_time", foreground="#888888")
        
        self.log_text.insert(tk.END, f"[{ts}] ", "tag_time")
        self.log_text.insert(tk.END, f"{level}: ", f"tag_{level}")
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_global_preview(self):
        df = self.data_mgr.df_FP
        self.tv.delete(*self.tv.get_children())
        if not df.empty:
            self.tv["columns"] = list(df.columns)
            self.tv["show"] = "headings"
            for c in df.columns: 
                self.tv.heading(c, text=c)
                self.tv.column(c, width=70)
            for _, r in df.head(10).iterrows(): 
                self.tv.insert("", "end", values=list(r))

    def load_merged_csv(self):
        f = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if f: 
            df = self.data_mgr.safe_read_csv(f)
            if df is not None: 
                self.data_mgr.set_df_fp(df)
                
                # --- NEW: Track loaded CSV path info ---
                path_obj = Path(f)
                self.data_mgr.loaded_csv_path = f
                self.data_mgr.output_dir = str(path_obj.parent)
                
                # Clean up the base name (remove timestamp/suffixes if possible)
                base = path_obj.stem
                if "_merged" in base:
                    base = base.split("_merged")[0]
                self.data_mgr.fp_base_name = base
                # ---------------------------------------
                
                self.log_message(f"Loaded Merged CSV: {path_obj.name}", "INFO")

    def save_merged_csv(self):
        if self.data_mgr.df_FP.empty: 
            return
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.data_mgr.fp_base_name if self.data_mgr.fp_base_name else "Session"
        
        # 1. Determine where to save based on how the data was loaded
        if self.data_mgr.loaded_csv_path:
            # Loaded from a merged CSV -> Save in the same folder as that CSV
            out_dir = Path(self.data_mgr.loaded_csv_path).parent
        else:
            # Loaded from Raw FP -> Save in the generated "Output" folder
            out_dir = Path(self.data_mgr.output_dir) if self.data_mgr.output_dir else Path(os.getcwd())
            
        os.makedirs(out_dir, exist_ok=True)
        
        # 2. Always create a new file with the new timestamp
        save_path = os.path.join(out_dir, f"{base_name}_merged_{ts}.csv")
            
        # 3. Save the DataFrame
        try:
            self.data_mgr.df_FP.to_csv(save_path, index=False)
            self.log_message(f"Successfully saved to: {save_path}", "INFO")
            
            # Update the loaded path so any future saves in this session stay in the same place
            self.data_mgr.loaded_csv_path = save_path
        except Exception as e:
            self.log_message(f"Save Error: {e}", "ERR")
            traceback.print_exc()


if __name__ == "__main__": 
    FPFEDApp().mainloop()
