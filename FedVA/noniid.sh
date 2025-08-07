#!/bin/bash

SESSION_NAME="noniid"

# 启动一个新的 tmux 会话（如果已存在则不重复创建）
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
    echo "🟢 Creating new tmux session: $SESSION_NAME"
    tmux new-session -d -s $SESSION_NAME
fi

# 在该会话中运行你的训练脚本
tmux send-keys -t $SESSION_NAME "source ~/.bashrc" C-m
tmux send-keys -t $SESSION_NAME "conda activate fedva" C-m   # 替换为你的conda环境名
tmux send-keys -t $SESSION_NAME "cd /home/sz1c24/FedVA" C-m
tmux send-keys -t $SESSION_NAME "python lf.py" C-m

echo "✅ Training started in tmux session: $SESSION_NAME"
echo "👉 To view progress: tmux attach -t $SESSION_NAME"
