import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Hyper Parameter Tuning System")
root.option_add("*tearOff", False)
root.geometry("800x500")

# Forest theme source
root.tk.call("source", "tuneparam/gui/theme/forest-dark.tcl")
root.tk.call("source", "tuneparam/gui/theme/forest-light.tcl")

style = ttk.Style()

THEME_BG = {
    "forest-dark": "#313131",
    "forest-light": "#FFFFFF"
}

DEFAULT_FONT = ("나눔스퀘어_ac Bold", 11)

def set_theme(theme):
    style.theme_use(theme)
    root.configure(bg=THEME_BG[theme])
    style.configure(".", font=DEFAULT_FONT)
    style.configure("TButton", font=DEFAULT_FONT)
    style.configure("Accent.TButton", font=DEFAULT_FONT)
    style.configure("TLabel", font=DEFAULT_FONT)
    style.configure("TEntry", font=DEFAULT_FONT)
    style.configure("TCombobox", font=DEFAULT_FONT)
    style.configure("TNotebook", font=DEFAULT_FONT)
    style.configure("TNotebook.Tab", font=DEFAULT_FONT)
    style.configure("TLabel", font=("", 12))

def create_notebook_with_tabs(parent):
    notebook = ttk.Notebook(parent)
    tab_main = ttk.Frame(notebook, padding=30)
    tab_train = ttk.Frame(notebook, padding=30)
    tab_results = ttk.Frame(notebook, padding=30)
    notebook.add(tab_main, text="Main")
    notebook.add(tab_train, text="Train")
    notebook.add(tab_results, text="Results")
    return notebook, tab_main, tab_train, tab_results

def create_theme_buttons(parent, set_theme_callback):
    theme_frame = ttk.Frame(parent)
    ttk.Button(theme_frame, text="Dark", command=lambda: set_theme_callback("forest-dark")).pack(side="left", padx=3)
    ttk.Button(theme_frame, text="Light", command=lambda: set_theme_callback("forest-light")).pack(side="left", padx=3)
    return theme_frame
