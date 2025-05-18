from part.utils import root, style, THEME_BG, set_theme, create_notebook_with_tabs, create_theme_buttons
from part.main_tap import setup_main_tab
from part.train_tab import setup_train_tab

set_theme("forest-light")
root.option_add("*Font", '"나눔스퀘어_ac Bold" 11')

theme_frame = create_theme_buttons(root, set_theme)
theme_frame.grid(row=0, column=1, sticky="ne", padx=(0, 10), pady=(10, 0))

notebook, tab_main, tab_train, tab_results = create_notebook_with_tabs(root)
notebook.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=0)

setup_main_tab(tab_main, notebook, tab_train)
setup_train_tab(tab_train)

root.mainloop()
