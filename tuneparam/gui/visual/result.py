#
# def apply_gpt_params(tab_train):
#     """GPT 추천 하이퍼파라미터를 Entry 위젯에 적용"""
#     try:
#         tab_train.lr_entry.delete(0, tk.END)
#         tab_train.lr_entry.insert(0, "0.001")
#
#         tab_train.batch_entry.delete(0, tk.END)
#         tab_train.batch_entry.insert(0, "64")
#
#         tab_train.epochs_entry.delete(0, tk.END)
#         tab_train.epochs_entry.insert(0, "50")
#     except AttributeError:
#         print("Entry 위젯이 정
