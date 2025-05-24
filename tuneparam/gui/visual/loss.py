# 그래프용 작은 폰트 정의
GRAPH_FONT = ('Helvetica', 8)


def setup_loss_graph(canvas, colors):
    """Loss 그래프 초기 설정"""
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()

    # 배경 그리드
    for i in range(50, width - 10, 30):  # X축 시작점을 50으로 변경
        canvas.create_line(i, 10, i, height - 25, fill=colors['grid'], width=1)
    for i in range(height - 25, 10, -30):
        canvas.create_line(50, i, width - 10, i, fill=colors['grid'], width=1)

    # X축, Y축 그리기
    canvas.create_line(50, height - 25, width - 10, height - 25, width=2, fill=colors['axis'])
    canvas.create_line(50, 10, 50, height - 25, width=2, fill=colors['axis'])

    # 축 레이블
    canvas.create_text(width / 2, height - 10, text="Epoch", font=GRAPH_FONT, fill=colors['fg'])
    canvas.create_text(15, height / 2, text="Loss", font=GRAPH_FONT, angle=90, fill=colors['fg'])

    # 범례
    legend_x = width - 70
    legend_y1 = 20
    legend_y2 = 35

    canvas.create_line(legend_x, legend_y1, legend_x + 15, legend_y1, fill=colors['loss'], width=2)
    canvas.create_text(legend_x + 35, legend_y1, text="Train", anchor="w", font=GRAPH_FONT, fill=colors['fg'])
    canvas.create_line(legend_x, legend_y2, legend_x + 15, legend_y2, fill=colors['val_loss'], width=2, dash=(4, 2))
    canvas.create_text(legend_x + 35, legend_y2, text="Val", anchor="w", font=GRAPH_FONT, fill=colors['fg'])


def draw_loss_graph(canvas, training_logs, colors):
    """Loss 그래프를 epoch 기준으로 그리기 (라벨 포함, 가독성 향상)"""
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()

    if not training_logs:
        return

    # 데이터 추출
    epochs = [log.epoch for log in training_logs]
    train_losses = [log.loss for log in training_logs]
    val_losses = [log.val_loss for log in training_logs]

    min_epoch, max_epoch = min(epochs), max(epochs)
    min_loss = min(min(train_losses), min(val_losses))
    max_loss = max(max(train_losses), max(val_losses))

    graph_width = width - 60   # padding left + right
    graph_height = height - 40 # padding top + bottom

    x_scale = graph_width / max(max_epoch - min_epoch, 1)
    y_scale = graph_height / max(max_loss - min_loss, 1e-5)

    def scale_point(epoch, loss):
        x = 50 + (epoch - min_epoch) * x_scale
        y = height - 25 - (loss - min_loss) * y_scale
        return x, y

    font = ("Arial", 8)

    # --- Y축 라벨 ---
    for val in [max_loss, (max_loss + min_loss)/2, min_loss]:
        _, y = scale_point(min_epoch, val)
        canvas.create_line(45, y, 50, y, fill=colors['fg'])  # 짧은 선
        canvas.create_text(40, y, text=f"{val:.3f}", anchor="e", font=font, fill=colors['fg'])

    # --- X축 라벨 ---
    tick_interval = max(1, (max_epoch - min_epoch) // 5)
    for epoch in range(min_epoch, max_epoch + 1, tick_interval):
        x, _ = scale_point(epoch, min_loss)
        canvas.create_line(x, height - 25, x, height - 20, fill=colors['fg'])
        canvas.create_text(x, height - 15, text=str(epoch), anchor="n", font=font, fill=colors['fg'])

    # --- 선 연결 ---
    for i in range(1, len(training_logs)):
        x1, y1 = scale_point(epochs[i - 1], train_losses[i - 1])
        x2, y2 = scale_point(epochs[i], train_losses[i])
        canvas.create_line(x1, y1, x2, y2, fill=colors['loss'], width=2)

        x1v, y1v = scale_point(epochs[i - 1], val_losses[i - 1])
        x2v, y2v = scale_point(epochs[i], val_losses[i])
        canvas.create_line(x1v, y1v, x2v, y2v, fill=colors['val_loss'], width=2, dash=(4, 2))

    # --- 점 및 라벨 ---
    for epoch, loss in zip(epochs, train_losses):
        x, y = scale_point(epoch, loss)
        canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=colors['loss'], outline=colors['loss'])
        if epoch == epochs[-1] or epoch == epochs[0] or epoch % tick_interval == 0:
            canvas.create_text(x + 5, y - 10, text=f"{loss:.3f}", anchor="w", font=font, fill=colors['loss'])

    for epoch, val_loss in zip(epochs, val_losses):
        x, y = scale_point(epoch, val_loss)
        canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=colors['val_loss'], outline=colors['val_loss'])
        if epoch == epochs[-1] or epoch == epochs[0] or epoch % tick_interval == 0:
            canvas.create_text(x + 5, y + 10, text=f"{val_loss:.3f}", anchor="w", font=font, fill=colors['val_loss'])

    # --- 오른쪽 최종 Epoch 라벨 ---
    canvas.create_text(width - 10, height - 25, text=f"Epoch {max_epoch}", font=font, anchor="e", fill=colors['fg'])

    # --- Max / Min 안내 텍스트 ---
    canvas.create_text(55, 10, text=f"↑ Max Loss: {max_loss:.3f}", anchor="w", font=font, fill=colors['fg'])
    canvas.create_text(55, height - 30, text=f"↓ Min Loss: {min_loss:.3f}", anchor="w", font=font, fill=colors['fg'])


