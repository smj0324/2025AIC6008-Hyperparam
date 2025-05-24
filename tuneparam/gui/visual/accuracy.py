GRAPH_FONT = ('Helvetica', 8)


def draw_accuracy_graph(canvas, training_logs, colors):
    """Accuracy 그래프 그리기"""
    width = canvas.winfo_width()
    height = canvas.winfo_height()

    if width <= 0 or height <= 0 or not training_logs:
        return

    # 정확도 값 리스트 추출
    train_accuracies = [log.accuracy for log in training_logs]
    val_accuracies = [log.val_accuracy for log in training_logs]
    epochs = [log.epoch for log in training_logs]

    max_epoch = max(epochs) if epochs else 1
    max_acc = max(train_accuracies + val_accuracies)
    min_acc = min(train_accuracies + val_accuracies)

    # 최소 최대 스케일 보정
    acc_range = max(max_acc - min_acc, 0.01)
    graph_width = width - 60
    graph_height = height - 40

    def to_canvas_x(epoch):
        return 50 + (epoch / max_epoch) * graph_width

    def to_canvas_y(acc):
        return height - 25 - ((acc - min_acc) / acc_range) * graph_height

    # 학습 정확도 선
    for i in range(1, len(epochs)):
        canvas.create_line(
            to_canvas_x(epochs[i - 1]), to_canvas_y(train_accuracies[i - 1]),
            to_canvas_x(epochs[i]), to_canvas_y(train_accuracies[i]),
            fill=colors['acc'], width=2
        )

    # 검증 정확도 선 (점선)
    for i in range(1, len(epochs)):
        canvas.create_line(
            to_canvas_x(epochs[i - 1]), to_canvas_y(val_accuracies[i - 1]),
            to_canvas_x(epochs[i]), to_canvas_y(val_accuracies[i]),
            fill=colors['val_acc'], width=2, dash=(4, 2)
        )

def setup_accuracy_graph(canvas, colors):
    """Accuracy 그래프 초기 설정"""
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()

    # 배경 그리드
    for i in range(50, width-10, 30):  # X축 시작점을 50으로 변경
        canvas.create_line(i, 10, i, height-25, fill=colors['grid'], width=1)
    for i in range(height-25, 10, -30):
        canvas.create_line(50, i, width-10, i, fill=colors['grid'], width=1)

    # X축, Y축 그리기
    canvas.create_line(50, height-25, width-10, height-25, width=2, fill=colors['axis'])
    canvas.create_line(50, 10, 50, height-25, width=2, fill=colors['axis'])

    # 축 레이블
    canvas.create_text(width/2, height-10, text="Epoch", font=GRAPH_FONT, fill=colors['fg'])
    canvas.create_text(15, height/2, text="Accuracy", font=GRAPH_FONT, angle=90, fill=colors['fg'])

    # 범례
    legend_x = width - 70
    legend_y1 = 20
    legend_y2 = 35

    canvas.create_line(legend_x, legend_y1, legend_x + 15, legend_y1, fill=colors['acc'], width=2)
    canvas.create_text(legend_x + 35, legend_y1, text="Train", anchor="w", font=GRAPH_FONT, fill=colors['fg'])
    canvas.create_line(legend_x, legend_y2, legend_x + 15, legend_y2, fill=colors['val_acc'], width=2, dash=(4,2))
    canvas.create_text(legend_x + 35, legend_y2, text="Val", anchor="w", font=GRAPH_FONT, fill=colors['fg'])